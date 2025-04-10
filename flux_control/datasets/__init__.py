from torch.utils.data import Dataset
from datasets import load_dataset

from .mock import MockDataset

def parse_dataset(dataset_config) -> Dataset:
    if not isinstance(dataset_config, dict):
        raise ValueError("dataset_config must be a dictionary.")
    if "type" not in dataset_config:
        raise ValueError("dataset_config must contain a 'type' key.")
    dataset_type = dataset_config.pop("type")

    # Currently, we only support the "huggingface" dataset type.
    if dataset_type == "huggingface":
        dataset = load_dataset(**dataset_config)
        if isinstance(dataset, Dataset):
            return dataset
        else:
            raise ValueError(
                "The loaded dataset is not of type Dataset. Make sure you have passed the correct parameters."
            )
    elif dataset_type == "mock":
        return MockDataset(**dataset_config)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
