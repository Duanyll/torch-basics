import lmdb
import torch
from torch.utils.data import Dataset
import pickle

class LMDBDataset(Dataset):
    def __init__(self, path: str):
        self.lmdb_path = path
        self.env = lmdb.open(path, readonly=True, lock=False, readahead=False, meminit=False)
        self.length = self._get_length()

    def _get_length(self):
        with self.env.begin(write=False) as txn:
            return txn.stat()['entries']
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        sample_key = f"{idx:08}"
        with self.env.begin(write=False) as txn:
            value = txn.get(sample_key.encode())
            if value is None:
                raise IndexError(f"Sample {idx} not found in LMDB.")
            sample = pickle.loads(value)
        return sample
    
    def __del__(self):
        self.env.close()