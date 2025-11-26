import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit



class DataSplitter:
    def __init__(self, dataframe, time_steps, batch_size=32, device='cuda'):
        self.seq_len = time_steps
        self.batch_size = batch_size
        self.device = device
        self.dataframe = dataframe
        self.X, self.y = self._prepare_data()
        self.X = self.remove_0_columns(self.X)
        self._create_loaders()
    
    def _prepare_data(self):
        X = self.dataframe.iloc[:, :-1].values
        y = self.dataframe.iloc[:, -1].values
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        return X, y_encoded
        
    def remove_0_columns(self, X):
        std_devs = np.std(X, axis=0)
        zero_std_columns = np.where(std_devs == 0)[0]
        if len(zero_std_columns) > 0:
            X = np.delete(X, zero_std_columns, axis=1)
        return X

    def _make_sequences(self, X, y):
        n = X.shape[0] // self.seq_len
        Xs = X[:n * self.seq_len].reshape(n, self.seq_len, X.shape[1])
        ys = y[:n * self.seq_len].reshape(n, self.seq_len)[:, -1]
        return Xs, ys
        
    def _split_and_reshape(self, X, y):
        num_samples = X.shape[0] // self.seq_len
        num_features = X.shape[1]  # This should be the total number of features (N * F)
        X = X[:num_samples * self.seq_len].reshape(num_samples, self.seq_len, num_features)
        y = y[:num_samples * self.seq_len].reshape(num_samples, self.seq_len)[:, -1]
        return X, y

    def _create_loaders(self):
        X_seq, y_seq = self._make_sequences(self.X, self.y)
    
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        (train_idx, test_idx) = next(sss1.split(X_seq, y_seq))
    
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)  # 10% of TRAIN → VAL
        (tr_rel, val_rel) = next(sss2.split(X_seq[train_idx], y_seq[train_idx]))
        val_idx = train_idx[val_rel]
        train_idx = train_idx[tr_rel]
    
        X_train, y_train = X_seq[train_idx], y_seq[train_idx]
        X_val,   y_val   = X_seq[val_idx],   y_seq[val_idx]
        X_test,  y_test  = X_seq[test_idx],  y_seq[test_idx]
    
        self.X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        self.X_val   = torch.tensor(X_val,   dtype=torch.float32).to(self.device)
        self.X_test  = torch.tensor(X_test,  dtype=torch.float32).to(self.device)
        self.y_train = torch.tensor(y_train, dtype=torch.long).to(self.device)
        self.y_val   = torch.tensor(y_val,   dtype=torch.long).to(self.device)
        self.y_test  = torch.tensor(y_test,  dtype=torch.long).to(self.device)
    
        pin = False
        self.train_loader = DataLoader(list(zip(self.X_train, self.y_train)),
                                       batch_size=self.batch_size, shuffle=True, pin_memory=pin)
        self.val_loader   = DataLoader(list(zip(self.X_val, self.y_val)),
                                       batch_size=self.batch_size, shuffle=False, pin_memory=pin)
        self.test_loader  = DataLoader(list(zip(self.X_test, self.y_test)),
                                       batch_size=self.batch_size, shuffle=False, pin_memory=pin)

        print(f"X_train shape: {self.X_train.shape} | y_train: {self.y_train.shape}")
        print(f"X_val   shape: {self.X_val.shape}   | y_val:   {self.y_val.shape}")
        print(f"X_test  shape: {self.X_test.shape}  | y_test:  {self.y_test.shape}")
        print(f"(seq_len={self.seq_len}, num_features={self.X_train.shape[-1]})")

    def compute_correlation_matrix(self):
        corr_matrix = np.corrcoef(self.X, rowvar=False)
        return corr_matrix