from typing import Any
from PIL import Image
import torch
from transformers import pipeline

pipe: Any = None


def load_segmentation_model(device: str = "cuda"):
    global pipe
    pipe = pipeline("mask-generation", model="facebook/sam-vit-large", device=device)


def unload_segmentation_model():
    global pipe
    pipe = None
    torch.cuda.empty_cache()


def generate_masks(image: torch.Tensor) -> Any:
    img = image.permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype("uint8")
    img = Image.fromarray(img)
    return pipe(img, points_per_batch=64)["masks"]
