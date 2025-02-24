import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

def load_data(raw_file, dx_file):
    raw_data = pd.read_csv(raw_file, sep=None, engine='python', encoding='utf-8-sig')
    dx_data  = pd.read_csv(dx_file, header=0, engine='python', encoding='utf-8-sig', sep='\n')
    raw_data = raw_data.drop(columns=['FID','IID','PAT','MAT','SEX','PHENOTYPE'], errors='ignore')
    dx_data.columns = dx_data.columns.str.strip()
    if 'New_Label' not in dx_data.columns:
        if dx_data.shape[1] == 1:
            dx_data.columns = ["New_Label"]
        else:
            raise ValueError("New_Label not found in dx_file.")
    return raw_data, dx_data

def split_into_windows_as_sequence(df, window_size=100):
    arr = df.values
    N, M = arr.shape
    W = M // window_size
    arr = arr[:, :W * window_size]
    seqs = []
    for i in range(W):
        st = i*window_size
        en = st+window_size
        seqs.append(arr[:, st:en])
    return np.stack(seqs, axis=1)

class SequenceSNPDataset(Dataset):
    def __init__(self, X_3d, y):
        self.X = torch.tensor(X_3d, dtype=torch.float32)
        if hasattr(y, "values"):
            self.y = torch.tensor(y.values, dtype=torch.long)
        else:
            self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
