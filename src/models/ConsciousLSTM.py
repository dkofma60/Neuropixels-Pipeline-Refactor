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
                nn.Linear(hidden_dim, mid1),  # d -> d/2
                nn.SiLU(),
                nn.Linear(mid1, mid2),        # d/2 -> d/4
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

            # Stack of LSTMCells (each layer: input_size -> hidden_size)
            self.layers = nn.ModuleList()
            for layer_idx in range(num_layers):
                layer_input_dim = input_dim if layer_idx == 0 else hidden_dim
                self.layers.append(
                    nn.LSTMCell(layer_input_dim, hidden_dim)
                )

            # Single consciousness module shared across all layers & time steps
            self.conscious = ConsciousModule(hidden_dim, k=k, c=c)

        def forward(self, x, hx=None):
            """
            x: (batch, seq_len, input_dim)   # batch_first style
            hx: optional tuple (h_0, c_0):
                h_0: (num_layers, batch, hidden_dim)
                c_0: (num_layers, batch, hidden_dim)
            Returns:
                output: (batch, seq_len, hidden_dim)
                (h_n, c_n): final states with same shapes as h_0, c_0
            """
            batch_size, seq_len, _ = x.shape

            # Init hidden/cell states if not provided
            if hx is None:
                h_0 = x.new_zeros(self.num_layers, batch_size, self.hidden_dim)
                c_0 = x.new_zeros(self.num_layers, batch_size, self.hidden_dim)
            else:
                h_0, c_0 = hx

            # Unstack initial states per layer
            h_t = [h_0[layer] for layer in range(self.num_layers)]  # each (B, H)
            c_t = [c_0[layer] for layer in range(self.num_layers)]  # each (B, H)

            outputs = []

            # time loop
            for t in range(seq_len):
                input_t = x[:, t, :]  # (B, input_dim) or (B, H) for deeper layers

                for layer_idx, cell in enumerate(self.layers):
                    # Classic LSTM step
                    h_t[layer_idx], c_t[layer_idx] = cell(
                        input_t,
                        (h_t[layer_idx], c_t[layer_idx])
                    )

                    # Apply consciousness: modify h_t before next time step
                    h_t[layer_idx] = self.conscious(h_t[layer_idx])

                    # Input to next layer at this time step is this layer's hidden
                    input_t = h_t[layer_idx]

                # Last layer's hidden state is the output at time t
                outputs.append(h_t[-1].unsqueeze(1))  # (B, 1, H)

            # Concatenate along time: (B, L, H)
            output = torch.cat(outputs, dim=1)

            # Stack final states into (num_layers, B, H)
            h_n = torch.stack(h_t, dim=0)
            c_n = torch.stack(c_t, dim=0)

            return output, (h_n, c_n)
        
        
    class LSTMModelWithConsciousness(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers, output_dim, k, c):
            super().__init__()
            self.rnn = ConsciousLSTM(input_dim, hidden_dim, num_layers, k=k, c=c)
            self.fc  = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            # x: (B, L, input_dim)
            output, (h_n, c_n) = self.rnn(x)
            # e.g. take last time step from last layer (B, H)
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
    best_val_acc = 0.0

    # Training loop
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
    
        #eval loop
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
        val_acc = correct_test_preds / total_test_samples * 100
        print(f"Epoch {epoch+1}/{num_epochs} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch_idx = epoch
    
    print(f"Best epoch: {best_epoch_idx} | Best Val Acc: {best_val_acc:.2f}%")
    return best_epoch_idx, best_val_acc