import logging
import random
from typing import cast
from einops import rearrange
import torch
import torchvision
import torch.nn.functional as F
import kornia.filters as KF

from .config import CollageConfig
from .flow import compute_aggregated_flow
from .warp import forward_warp
from .palette import palette_downsample


logger = logging.getLogger(__name__)


def load_video(video_path: str):
    """
    Load video as TCHW tensor on CPU, uint8.
    """
    video, _, _ = torchvision.io.read_video(
        video_path, output_format="TCHW", pts_unit="sec", end_pts=10.0
    )
    return video


def select_frames(video: torch.Tensor, cfg: CollageConfig = CollageConfig()):
    """
    Randomly select consecutive frames from the video.
    """
    video_frames = video.shape[0]
    if video_frames < cfg.min_frames:
        return None

    num_frames = random.randint(cfg.min_frames, min(video_frames, cfg.max_frames))
    start_idx = random.randint(0, video_frames - num_frames)
    end_idx = start_idx + num_frames
    selected_frames = video[start_idx:end_idx]
    # 50% chance to flip the video
    if random.random() < cfg.chance_reverse:
        selected_frames = torch.flip(selected_frames, dims=[0])
    logging.debug(
        f"Video has {video_frames} frames, selected {num_frames} frames from {start_idx} to {end_idx}"
    )
    return selected_frames


def random_crop(
    video: torch.Tensor,
    require_portrait: bool = False,
    cfg: CollageConfig = CollageConfig(),
):
    """
    Randomly crop the video frames to a random resolution in the resolution bucket.
    Center crop and resize the video frames to the same resolution.
    """
    t, c, h, w = video.shape
    resolution_list = cfg.resolutions_720p if h < 1080 else cfg.resolutions_1080p
    resolution = random.choice(resolution_list)
    target_w, target_h = resolution
    if require_portrait:
        target_h, target_w = target_w, target_h
    logger.debug(f"Random crop to {target_h}x{target_w}")
    dest_aspect_ratio = target_w / target_h
    src_aspect_ratio = w / h
    if src_aspect_ratio > dest_aspect_ratio:
        new_w = int(h * dest_aspect_ratio)
        new_h = h
    else:
        new_w = w
        new_h = int(w / dest_aspect_ratio)
    # Randomly crop the video frames to new_h, new_w
    start_h = random.randint(0, h - new_h)
    start_w = random.randint(0, w - new_w)
    cropped_video = video[:, :, start_h : start_h + new_h, start_w : start_w + new_w]
    # Resize the video frames to target_h, target_w
    resized_video = torch.nn.functional.interpolate(
        cropped_video, size=(target_h, target_w), mode="bilinear", align_corners=False
    )
    return resized_video


def splat_lost_regions(
    src_image: torch.Tensor,
    dst_image: torch.Tensor,
    flow: torch.Tensor,
    affine_warped: torch.Tensor,
    affine_grid: torch.Tensor,
    warped_regions: torch.Tensor,
    canvas_alpha: torch.Tensor,
):
    splat_warped, splat_grid, splat_mask = forward_warp(
        src_image, dst_image, flow, grid=affine_grid, mask=(1 - warped_regions)
    )
    splat_grid = cast(torch.Tensor, splat_grid)
    combined = splat_warped * (1 - canvas_alpha) + affine_warped * canvas_alpha
    combined_grid = splat_grid * (1 - canvas_alpha) + affine_grid * canvas_alpha
    combined_mask = splat_mask * (1 - canvas_alpha) + canvas_alpha
    combined = combined.clamp(0, 1)
    combined_grid = combined_grid.clamp(-1, 1)
    combined_mask = combined_mask.clamp(0, 1)
    return combined, combined_grid, combined_mask


def try_extract_frame(
    video: torch.Tensor, device: str = "cuda", cfg: CollageConfig = CollageConfig()
):
    """
    Try to extract a frame pair from the video. The input is on CPU, and the output is on GPU.
    This is to save vram for UHD video.

    Args:
        video (torch.Tensor): Video tensor of shape (T, C, H, W), uint8.
        device (str): Device to use for computation.
        cfg (CollageConfig): Configuration object.
    Returns:
        tuple: (flow, src_frame, dst_frame) float32 tensors on device:
          - flow (torch.Tensor): Optical flow tensor of shape (2, H, W).
          - src_frame (torch.Tensor): Source frame tensor of shape (C, H, W).
          - dst_frame (torch.Tensor): Destination frame tensor of shape (C, H, W).
    """
    if cfg.frame_interval > 1:
        video = video[:: cfg.frame_interval]

    max_attempts = cfg.num_extract_attempts
    attempt = 0
    require_portrait = random.random() < cfg.chance_portrait

    while attempt < max_attempts:
        attempt += 1
        frames = select_frames(video)
        if frames is None:
            return None

        frames = frames.float() / 255.0
        frames = random_crop(frames, require_portrait)
        frames = frames.to(device)

        # 2. Estimate optical flow (GPU)
        flow, target_idx = compute_aggregated_flow(frames, cfg=cfg, device=device)
        if flow is not None:
            break
    else:
        logger.debug(f"Video has invalid optical flow after {max_attempts} attempts.")
        return None

    return flow, frames[0], frames[target_idx]


def make_confidence_map(grid_affine, grid_splat, mask_affine_tgt, mask_splat):
    grid_diff = grid_affine - grid_splat
    grid_diff = torch.norm(grid_diff, dim=0, p=2)
    grid_diff = torch.tanh(grid_diff * 10)
    # grid_diff = 1 - torch.exp(-grid_diff * 5)
    mask_bool = (mask_splat > 0.5) & (mask_affine_tgt > 0.5)
    grid_diff = grid_diff * mask_bool + ~mask_bool * 1.0
    confidence = F.avg_pool2d(
        rearrange(grid_diff, "h w -> 1 1 h w"), kernel_size=16, stride=16
    )
    confidence = rearrange(confidence, "1 1 h w -> h w")
    return confidence


def simplify_mask(mask, kernel_size=51, sigma=20.0):
    gaussian_blur = KF.GaussianBlur2d(
        kernel_size=(kernel_size, kernel_size),
        sigma=(sigma, sigma),
        border_type="reflect",
    ).to(mask.device)
    
    mask = mask.float()
    mask = rearrange(mask, "h w -> 1 1 h w")
    mask = gaussian_blur(mask)
    mask = rearrange(mask, "1 1 h w -> h w")
    mask = mask > 0.5
    
    return mask


def make_confidence_hint(grid_affine, grid_splat, mask_affine_tgt, mask_splat, tgt, foreground, cfg):
    # Confidence Map
    grid_diff = grid_affine - grid_splat
    grid_diff = torch.norm(grid_diff, dim=0, p=2)
    grid_diff = torch.tanh(grid_diff * cfg.confidence_tanh_scale)
    # grid_diff = 1 - torch.exp(-grid_diff * 5)
    mask_bool = (mask_splat > 0.5) & (mask_affine_tgt > 0.5)
    grid_diff = grid_diff * mask_bool + ~mask_bool * 1.0
    confidence = F.avg_pool2d(
        rearrange(grid_diff, "h w -> 1 1 h w"), kernel_size=16, stride=16
    )
    confidence = rearrange(confidence, "1 1 h w -> h w")

    # Hint Image
    hint_mask = simplify_mask(
        grid_diff > 0.5, kernel_size=cfg.simplify_kernel_size, sigma=cfg.simplify_sigma
    )
    hint_fg = palette_downsample(
        tgt,
        hint_mask * foreground,
        colors=cfg.palette_fg_colors,
        spatial_weight=cfg.palette_spatial_weight,
    )
    hint_bg = palette_downsample(
        tgt,
        hint_mask * (1 - foreground),
        colors=cfg.palette_bg_colors,
        spatial_weight=cfg.palette_spatial_weight,
    )
    hint = hint_bg + hint_fg

    return confidence, hint