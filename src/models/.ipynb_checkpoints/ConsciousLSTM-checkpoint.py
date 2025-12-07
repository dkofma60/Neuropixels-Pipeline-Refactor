import pickle
import numpy as np
import torch
import torch.nn as nn
import csv
import os
from data_processors.data_splitter import DataSplitter



def train_and_eval(hidden_dim, layer_dim, learning_rate, num_epochs, data_splitter, k, c, validation=True):
    
    device = data_splitter.X_train.device
    input_dim = data_splitter.X_train.shape[-1]
    output_dim = int(data_splitter.y.max() + 1)
    
    class ConsciousModule(nn.Module):
        
        def __init__(self, hidden_dim: int, k: int = 2, c: float = 0.1):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.k = k
            self.c = c
    
            mid1 = max(1, hidden_dim // 2)
            mid2 = max(1, hidden_dim // 4)
    
            self.A_net = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, mid1),
                nn.SiLU(),
                nn.Linear(mid1, mid2),
                nn.SiLU(),
                nn.Linear(mid2, hidden_dim * k)
            )
    
        def forward(self, h):
            B, d = h.shape
            A = self.A_net(h).view(B, d, self.k)
            z = torch.randn(B, self.k, device=h.device, dtype=h.dtype)
            noise = torch.bmm(A, z.unsqueeze(-1)).squeeze(-1)
    
            if not self.training or self.c == 0:
                return h
    
            return h + self.c * noise
        
        
    class ConsciousLSTM(nn.Module):
        
        def __init__(self, input_dim, hidden_dim, num_layers=1, k=2, c=0.1):
            super().__init__()

            self.input_dim  = input_dim
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.layers = nn.ModuleList()
            
            for layer_idx in range(num_layers):
                layer_input_dim = input_dim if layer_idx == 0 else hidden_dim
                self.layers.append(
                    nn.LSTMCell(layer_input_dim, hidden_dim)
                )

            self.conscious = ConsciousModule(hidden_dim, k=k, c=c)

        def forward(self, x, hx=None):
            batch_size, seq_len, _ = x.shape

            if hx is None:
                h_0 = x.new_zeros(self.num_layers, batch_size, self.hidden_dim)
                c_0 = x.new_zeros(self.num_layers, batch_size, self.hidden_dim)
            else:
                h_0, c_0 = hx

            h_t = [h_0[layer] for layer in range(self.num_layers)]
            c_t = [c_0[layer] for layer in range(self.num_layers)]
            outputs = []

            for t in range(seq_len):
                input_t = x[:, t, :]
                
                for layer_idx, cell in enumerate(self.layers):
                    h_t[layer_idx], c_t[layer_idx] = cell(
                        input_t,
                        (h_t[layer_idx], c_t[layer_idx])
                    )
                    h_t[layer_idx] = self.conscious(h_t[layer_idx])
                    input_t = h_t[layer_idx]
                    
                outputs.append(h_t[-1].unsqueeze(1))

            output = torch.cat(outputs, dim=1)
            h_n = torch.stack(h_t, dim=0)
            c_n = torch.stack(c_t, dim=0)

            return output, (h_n, c_n)
        
        
    class LSTMModelWithConsciousness(nn.Module):
        
        def __init__(self, input_dim, hidden_dim, num_layers, output_dim, k, c):
            super().__init__()
            self.rnn = ConsciousLSTM(input_dim, hidden_dim, num_layers, k=k, c=c)
            self.fc  = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            output, (h_n, c_n) = self.rnn(x)
            last_hidden = output[:, -1, :]
            logits = self.fc(last_hidden)
            return logits


    model = LSTMModelWithConsciousness(input_dim, hidden_dim, layer_dim, output_dim, k, c).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = data_splitter.train_loader
    if validation:
        val_or_test_loader = data_splitter.val_loader
    else:
        val_or_test_loader = data_splitter.test_loader
    
    best_epoch_idx = 0
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_running_loss = 0.0
        correct_train_preds = 0.0
        total_train_samples = 0.0
        for i, (features, labels) in enumerate(train_loader):
            out = model(features)
            loss = criterion(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_running_loss += loss.item()
            _, preds = torch.max(out, dim=1)
            correct_train_preds += (preds == labels).sum().item()
            total_train_samples += labels.shape[0]
    
        model.eval()
        correct_test_preds = 0.0
        total_test_samples = 0.0
        with torch.no_grad():
            for i, (features, labels) in enumerate(val_or_test_loader):
                out = model(features)
                _, preds = torch.max(out, dim=1)
                correct_test_preds += (preds == labels).sum().item()
                total_test_samples += labels.shape[0]

        train_acc = correct_train_preds / total_train_samples * 100
        acc = correct_test_preds / total_test_samples * 100
        print(f"Epoch {epoch+1}/{num_epochs} | Train Acc: {train_acc:.2f}% | Val/Test Acc: {acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            best_epoch_idx = epoch
    
    print(f"Best epoch: {best_epoch_idx} | Best Val Acc: {best_acc:.2f}%")
    return best_epoch_idx, best_acc, acc