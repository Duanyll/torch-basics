import lmdb
import torch
from torch.utils.data import Dataset
import pickle


class LMDBDataset(Dataset):
    def __init__(self, path: str):
        self.lmdb_path = path
        self.env = lmdb.open(
            path, readonly=True, lock=False, readahead=False, meminit=False
        )
        self.keys = self._load_keys()

    def _load_keys(self):
        with self.env.begin(write=False) as txn:
            cursor = txn.cursor()
            return [key for key, _ in cursor]  # store raw bytes keys

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        if idx >= len(self.keys):
            raise IndexError(f"Index {idx} out of range.")
        key = self.keys[idx]
        with self.env.begin(write=False) as txn:
            value = txn.get(key)
            if value is None:
                raise KeyError(f"Key {key} not found in LMDB.")
            sample = pickle.loads(value)
        return sample

    def __del__(self):
        if hasattr(self, "env") and self.env is not None:
            self.env.close()
