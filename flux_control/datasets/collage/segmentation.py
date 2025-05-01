from typing import Any
from PIL import Image
import torch
import numpy as np
from transformers import pipeline

pipe: Any = None


def load_segmentation_model(device: str = "cuda"):
    global pipe
    pipe = pipeline("mask-generation", model="facebook/sam-vit-large", device=device)


def unload_segmentation_model():
    global pipe
    pipe = None
    torch.cuda.empty_cache()


def generate_masks(image: torch.Tensor, pack_result=True) -> Any:
    img = image.permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype("uint8")
    img = Image.fromarray(img)
    result = pipe(img, points_per_batch=64)["masks"] # List of ndarray
    if pack_result:
        result_np = np.array(result).astype("bool")
        result_pt = torch.from_numpy(result_np)
        return result_pt.to(image.device)
    else:
        return result
