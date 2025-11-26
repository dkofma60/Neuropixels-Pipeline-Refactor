import pickle
import numpy as np
import torch
import torch.nn as nn
import csv
import os
from data_processors.data_splitter import DataSplitter



def train_and_eval(hidden_dim, layer_dim, learning_rate, num_epochs, data_splitter, validation=True):
    
    device = data_splitter.X_train.device
    input_dim = data_splitter.X_train.shape[-1]
    output_dim = int(data_splitter.y.max() + 1)
    
    class LSTMModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
            super(LSTMModel, self).__init__()
            self.hidden_dim = hidden_dim
            self.layer_dim = layer_dim
            self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            out, (hn, cn) = self.lstm(x)
            out = self.fc(out[:, -1, :])
            return out
    
    
    model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim).to(device)
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



def run_and_save_LSTM(hidden_dim, layer_dim, learning_rate, num_epochs, batch_size, time_steps, session_details_file_path, spike_trains_file_path):
    
    with open(spike_trains_file_path, 'rb') as f:
        spike_df = pickle.load(f)
    data_splitter = DataSplitter(spike_df, time_steps, batch_size)
    _,best_test_acc = train_and_eval(hidden_dim, layer_dim, learning_rate, num_epochs, data_splitter, False)
    
    session_details = {
    'session_number': spike_trains_file_path.split('_')[-2].split('_')[0],
    'bins': time_steps,
    'model_name': 'LSTM',
    'test_acc': np.round(best_test_acc, 2),
    'num_epochs': num_epochs,
    'hidden_dim': hidden_dim,
    'layer_dim': layer_dim,
    'learning_rate': learning_rate,
    'batch_size': batch_size
    }
    print(f'Best Test Accuracy: {best_test_acc}%')
    
    file_exists = os.path.isfile(session_details_file_path)
    with open(session_details_file_path, 'a', newline='') as csvfile:
        fieldnames = list(session_details.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(session_details)