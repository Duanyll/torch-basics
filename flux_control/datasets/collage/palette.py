import torch
import kornia
import kornia.color as color
from typing import Optional
from einops import rearrange
from .kmeans import kmeans, get_cluster_centers_scatter


def extract_palette_from_masked_image(
    image: torch.Tensor,  # (3, H, W), RGB float32, range 0-1
    mask: torch.Tensor,  # (H, W), bool or {0, 1}
    max_colors: int = 8,
    min_colors: int = 3,
    delta_e_thresh: float = 15.0,
    max_pixels: int = 100000,
    device: Optional[torch.device] = None,
):
    assert image.ndim == 3 and image.shape[0] == 3
    assert mask.ndim == 2 and mask.shape == image.shape[1:]

    if device is None:
        device = image.device

    H, W = image.shape[1], image.shape[2]
    grid = kornia.utils.create_meshgrid(
        H, W, normalized_coordinates=True, device=device
    )
    grid = rearrange(grid, "1 h w c -> (h w) c")

    # 1. 提取 masked 像素
    mask_flat = mask.flatten()
    img_flat = rearrange(image, "c h w -> (h w) c")  # (H * W, 3)
    masked_pixels = img_flat[mask_flat.bool()]  # (N, 3)
    grid_pixels = grid[mask_flat.bool()]  # (N, 2)

    # Randomly sample pixels if too many
    if masked_pixels.shape[0] > max_pixels:
        indices = torch.randperm(masked_pixels.shape[0])[:max_pixels]
        masked_pixels = masked_pixels[indices]
        grid_pixels = grid_pixels[indices]

    for k in reversed(range(min_colors, max_colors + 1)):
        idx, centers = kmeans(masked_pixels, cluster_num=k)

        # 转 Lab 并判断 ΔE 距离
        # Reshape centers to match the expected format for rgb_to_lab: (B, 3, H, W)
        centers_input = centers.unsqueeze(0).permute(0, 2, 1).unsqueeze(-1)  # (1, 3, k, 1)
        lab_centers = color.rgb_to_lab(centers_input)[0, :, :, 0].permute(1, 0)  # (k, 3)
        diff = lab_centers.unsqueeze(1) - lab_centers.unsqueeze(0)
        dist = (diff**2).sum(dim=-1).sqrt()  # ΔE
        upper_triangle = dist[torch.triu(torch.ones_like(dist), diagonal=1).bool()]
        if (upper_triangle > delta_e_thresh).all():
            locations = get_cluster_centers_scatter(grid_pixels, idx, k)
            return centers, locations  # 满足距离要求

    # 最后 fallback：用 min_colors 个聚类中心
    idx, centers = kmeans(masked_pixels, cluster_num=min_colors)
    locations = get_cluster_centers_scatter(grid_pixels, idx, min_colors)
    return centers, locations


def extract_palette_from_masked_image_with_spatial(
    image: torch.Tensor,  # (3, H, W), RGB float32, range 0-1
    mask: torch.Tensor,  # (H, W), bool or {0, 1}
    max_colors: int = 8,
    min_colors: int = 3,
    delta_e_thresh: float = 15.0,
    max_pixels: int = 100000,
    spatial_weight: float = 0.3,  # 空间特征权重
    device: Optional[torch.device] = None,
):
    assert image.ndim == 3 and image.shape[0] == 3
    assert mask.ndim == 2 and mask.shape == image.shape[1:]

    if device is None:
        device = image.device

    H, W = image.shape[1], image.shape[2]
    grid = kornia.utils.create_meshgrid(
        H, W, normalized_coordinates=True, device=device
    )
    grid = rearrange(grid, "1 h w c -> (h w) c")

    mask_flat = mask.flatten()
    img_flat = rearrange(image, "c h w -> (h w) c")  # (H * W, 3)
    masked_pixels = img_flat[mask_flat.bool()]  # (N, 3)
    grid_pixels = grid[mask_flat.bool()]  # (N, 2)

    # 随机下采样（同步 RGB 和空间）
    if masked_pixels.shape[0] > max_pixels:
        indices = torch.randperm(masked_pixels.shape[0], device=device)[:max_pixels]
        masked_pixels = masked_pixels[indices]
        grid_pixels = grid_pixels[indices]

    # 拼接 RGB + XY
    features = torch.cat(
        [masked_pixels, spatial_weight * grid_pixels], dim=1  # 控制空间特征影响力
    )  # (N, 5)

    for k in reversed(range(min_colors, max_colors + 1)):
        idx, centers = kmeans(features, cluster_num=k)

        # 提取 RGB 部分用于 ΔE 计算
        rgb_centers = centers[:, :3]
        centers_input = rearrange(rgb_centers, "k c -> 1 c k 1")  # (1, 3, k, 1)
        lab_centers = color.rgb_to_lab(centers_input)[0, :, :, 0].permute(
            1, 0
        )  # (k, 3)
        diff = lab_centers.unsqueeze(1) - lab_centers.unsqueeze(0)
        dist = (diff**2).sum(dim=-1).sqrt()  # ΔE
        upper_triangle = dist[torch.triu(torch.ones_like(dist), diagonal=1).bool()]
        if (upper_triangle > delta_e_thresh).all():
            grid_pixels_unsq = grid_pixels.unsqueeze(0)  # (1, N, 2)
            idx_unsq = idx.unsqueeze(0).long()  # (1, N)
            locations = get_cluster_centers_scatter(grid_pixels_unsq, idx_unsq, k)
            return rgb_centers, locations.squeeze(0)

    # fallback
    idx, centers = kmeans(features, cluster_num=min_colors)
    rgb_centers = centers[:, :3]
    grid_pixels_unsq = grid_pixels.unsqueeze(0)
    idx_unsq = idx.unsqueeze(0).long()
    locations = get_cluster_centers_scatter(grid_pixels_unsq, idx_unsq, min_colors)
    return rgb_centers, locations.squeeze(0)


def show_color_palette(palette: torch.Tensor, show_hex: bool = True, figsize=(8, 2)):
    """
    在 Jupyter 中显示一个色卡。

    Args:
        palette (Tensor): (N, 3) RGB float tensor in [0, 1]
        show_hex (bool): 是否显示颜色值（Hex格式）
        figsize (tuple): 图像尺寸
    """
    
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    assert palette.ndim == 2 and palette.shape[1] == 3

    n_colors = palette.shape[0]
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.set_xlim(0, n_colors)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for i, color in enumerate(palette):
        rgb = color.cpu().numpy() if isinstance(color, torch.Tensor) else color
        hex_code = "#%02x%02x%02x" % tuple((rgb * 255).astype(int))
        rect = patches.Rectangle((i, 0), 1, 1, color=rgb)
        ax.add_patch(rect)
        if show_hex:
            ax.text(
                i + 0.5,
                0.5,
                hex_code,
                ha="center",
                va="center",
                fontsize=12,
                color="white" if color.mean() < 0.5 else "black",
            )

    plt.show()
