import lmdb
from torch.utils.data import Dataset
import pickle


class LMDBDataset(Dataset):
    def __init__(self, path: str, db_name: str | None = None):
        self.lmdb_path = path
        self.env = lmdb.open(
            path, readonly=True, lock=False, readahead=False, meminit=False, max_dbs=16
        )
        self.db_name = db_name
        self.db = self.env.open_db(db_name.encode()) if db_name else self.env.open_db()
        self.keys = self._load_keys()

    def _load_keys(self):
        with self.env.begin() as txn:
            return list(txn.cursor(self.db).iternext(values=False))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        if idx >= len(self.keys):
            raise IndexError(f"Index {idx} out of range.")
        key = self.keys[idx]
        with self.env.begin(write=False) as txn:
            value = txn.get(key, db=self.db)
            if value is None:
                raise KeyError(f"Key {key} not found in LMDB.")
            sample = pickle.loads(value)
        return sample

    def __del__(self):
        if hasattr(self, "env") and self.env is not None:
            self.env.close()
