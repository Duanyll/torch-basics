import torch
import kornia
from typing import Any

model: Any = None
model_device: Any = None

def load_dexined_model(device: str = "cuda"):
    """加载 DexiNed 模型"""
    global model, model_device
    model = kornia.filters.DexiNed(pretrained=True).to(device)
    model_device = device
    
def unload_dexined_model():
    """卸载 DexiNed 模型"""
    global model, model_device
    if model is not None:
        del model
        model = None
        torch.cuda.empty_cache()
    model_device = None
    
@torch.no_grad()
def estimate_edges(image: torch.Tensor) -> torch.Tensor:
    """
    估计图像的边缘

    :param image: 输入图像 [C, H, W]
    :return: 边缘图 [H, W]
    """
    global model_device
    if model_device is None:
        raise RuntimeError("DexiNed model is not loaded.")
    
    image = image.to(model_device).unsqueeze(0) * 255.0
    out = model(image)
    out = torch.sigmoid(out)
    return out.squeeze()