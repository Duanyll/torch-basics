import torch
from .pipeline import load_all_models

if __name__ == "__main__":
    load_all_models(device="cuda")
    print("Loaded models.")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")