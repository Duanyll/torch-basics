from regex import W
import torch
import kornia
import kornia.color as color
from typing import Optional
from einops import rearrange
from ...utils.common import make_grid
from .config import CollageConfig
from .kmeans import kmeans, get_cluster_centers_scatter


def extract_palette_from_masked_image(
    image: torch.Tensor,  # (3, H, W), RGB float32, range 0-1
    mask: torch.Tensor,  # (H, W), bool or {0, 1}
    max_colors: int = 8,
    min_colors: int = 3,
    cfg: CollageConfig = CollageConfig(),
    device: Optional[torch.device] = None,
):
    assert image.ndim == 3 and image.shape[0] == 3
    assert mask.ndim == 2 and mask.shape == image.shape[1:]

    if device is None:
        device = image.device

    H, W = image.shape[1], image.shape[2]
    grid = make_grid(H, W, device=device)  # (2, H, W)
    grid = rearrange(grid, "c h w -> (h w) c")

    # 1. 提取 masked 像素
    mask_flat = mask.flatten()
    img_flat = rearrange(image, "c h w -> (h w) c")  # (H * W, 3)
    masked_pixels = img_flat[mask_flat.bool()]  # (N, 3)
    grid_pixels = grid[mask_flat.bool()]  # (N, 2)

    # Randomly sample pixels if too many
    if masked_pixels.shape[0] > cfg.max_cluster_samples:
        indices = torch.randperm(masked_pixels.shape[0])[: cfg.max_cluster_samples]
        masked_pixels = masked_pixels[indices]
        grid_pixels = grid_pixels[indices]

    for k in reversed(range(min_colors, max_colors + 1)):
        idx, centers = kmeans(masked_pixels, cluster_num=k)

        # 转 Lab 并判断 ΔE 距离
        # Reshape centers to match the expected format for rgb_to_lab: (B, 3, H, W)
        centers_input = (
            centers.unsqueeze(0).permute(0, 2, 1).unsqueeze(-1)
        )  # (1, 3, k, 1)
        lab_centers = color.rgb_to_lab(centers_input)[0, :, :, 0].permute(
            1, 0
        )  # (k, 3)
        diff = lab_centers.unsqueeze(1) - lab_centers.unsqueeze(0)
        dist = (diff**2).sum(dim=-1).sqrt()  # ΔE
        upper_triangle = dist[torch.triu(torch.ones_like(dist), diagonal=1).bool()]
        if (upper_triangle > cfg.delta_e_threshold).all():
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
    cfg: CollageConfig = CollageConfig(),
    device: Optional[torch.device] = None,
):
    assert image.ndim == 3 and image.shape[0] == 3
    assert mask.ndim == 2 and mask.shape == image.shape[1:]

    if device is None:
        device = image.device

    H, W = image.shape[1], image.shape[2]
    grid = make_grid(H, W, device=device)  # (2, H, W)
    grid = rearrange(grid, "c h w -> (h w) c")

    mask_flat = mask.flatten()
    img_flat = rearrange(image, "c h w -> (h w) c")  # (H * W, 3)
    masked_pixels = img_flat[mask_flat.bool()]  # (N, 3)
    grid_pixels = grid[mask_flat.bool()]  # (N, 2)

    # 随机下采样（同步 RGB 和空间）
    if masked_pixels.shape[0] > cfg.max_cluster_samples:
        indices = torch.randperm(masked_pixels.shape[0], device=device)[
            : cfg.max_cluster_samples
        ]
        masked_pixels = masked_pixels[indices]
        grid_pixels = grid_pixels[indices]

    # 拼接 RGB + XY
    features = torch.cat(
        [masked_pixels, cfg.palette_spatial_weight * grid_pixels],
        dim=1,  # 控制空间特征影响力
    )  # (N, 5)

    for k in reversed(range(min_colors, max_colors + 1)):
        idx, centers = kmeans(features, cluster_num=k, filter_nan=False)
        if torch.isnan(centers).any():
            continue

        # 提取 RGB 部分用于 ΔE 计算
        rgb_centers = centers[:, :3]
        centers_input = rearrange(rgb_centers, "k c -> 1 c k 1")  # (1, 3, k, 1)
        lab_centers = color.rgb_to_lab(centers_input)[0, :, :, 0].permute(
            1, 0
        )  # (k, 3)
        diff = lab_centers.unsqueeze(1) - lab_centers.unsqueeze(0)
        dist = (diff**2).sum(dim=-1).sqrt()  # ΔE
        upper_triangle = dist[torch.triu(torch.ones_like(dist), diagonal=1).bool()]
        if (upper_triangle > cfg.delta_e_threshold).all():
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


def encode_color_palette(image: torch.Tensor, masks, cfg: CollageConfig):
    palettes = []
    locations = []

    c, h, w = image.shape
    total_area = h * w
    for mask_data in masks:
        mask = mask_data["mask"]
        area = mask_data["area"]
        if area < cfg.palette_area_threshold * total_area:
            continue
        mask_torch = torch.tensor(mask, device=image.device)
        palette, location = extract_palette_from_masked_image_with_spatial(
            image, mask_torch, max_colors=cfg.palette_per_mask, min_colors=1, cfg=cfg
        )
        palettes.append(palette)
        locations.append(location)

    if len(palettes) == 0:
        return extract_palette_from_masked_image_with_spatial(
            image,
            torch.ones((h, w), device=image.device),
            max_colors=cfg.num_palette_fallback,
            min_colors=1,
            cfg=cfg,
        )

    palettes = torch.cat(palettes, dim=0)
    locations = torch.cat(locations, dim=0)

    return palettes, locations


def palette_downsample(image, mask=None, colors=4, spatial_weight=0.5):
    """
    Downsample the image to a palette of colors.
    :param image: (3, H, W), RGB float32, range 0-1
    :param mask: (H, W), bool or {0, 1}
    :param colors: number of colors in the palette
    :return: (3, H, W), RGB float32, range 0-1
    """

    c, h, w = image.shape

    image_flat = rearrange(image, "c h w -> (h w) c")  # (H * W, 3)
    
    if spatial_weight > 0:
        grid = make_grid(h, w, device=image.device)
        grid = rearrange(grid, "c h w -> (h w) c")
        image_flat = torch.cat(
            [image_flat, spatial_weight * grid], dim=1
        )
    
    if mask is not None:
        if torch.is_floating_point(mask):
            mask = mask > 0.5
        mask_flat = mask.flatten()
        pixels = image_flat[mask_flat.bool()]  # (N, 3)
    else:
        pixels = image_flat

    idx, centers = kmeans(pixels, cluster_num=colors)
    pixels = centers[idx]
    
    if mask is not None:
        result = torch.zeros_like(image_flat)
        result[mask_flat] = pixels
    else:
        result = pixels
        
    result = rearrange(result, "(h w) c -> c h w", h=h, w=w)
    if spatial_weight > 0:
        result = result[:c, :, :]
    return result
