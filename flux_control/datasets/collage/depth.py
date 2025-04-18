from typing import cast, Any
import torch
import torchvision.transforms.functional as TF
from einops import rearrange


model: Any = None


def load_depth_model(device="cuda"):
    """初始化深度估计模型"""
    global model
    model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
    model = model.to(device)
    model.eval()
    
    
def unload_depth_model():
    """卸载深度估计模型"""
    global model
    if model is not None:
        del model
        model = None
        torch.cuda.empty_cache()


@torch.no_grad()
def estimate_depth(
    image: torch.Tensor,
) -> torch.Tensor:
    """
    估计图像深度图

    :param image: 输入图像 [C, H, W]
    :return: 深度图 [H, W]
    """
    
    c, h, w = image.shape
    image = rearrange(image, "c h w -> 1 c h w")
    image = torch.nn.functional.interpolate(
        image, size=(384, 384), mode="bilinear", align_corners=False
    )
    TF.normalize(image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)
    prediction = model(image)
    prediction = rearrange(prediction, "1 h w -> 1 1 h w")
    prediction = torch.nn.functional.interpolate(
        prediction, size=(h, w), mode="bilinear", align_corners=False
    )
    prediction = rearrange(prediction, "1 1 h w -> h w")
    # Scale the depth map to the range [0, 1]
    prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min())

    return prediction
