import logging
import random
from typing import cast
import torch

from .palette import extract_palette_from_masked_image
from .warp import forward_warp


logger = logging.getLogger(__name__)


def select_frames(video: torch.Tensor, min_frames: int = 5, max_frames: int = 60):
    """
    Randomly select consecutive frames from the video.
    """
    video_frames = video.shape[0]
    if video_frames < min_frames:
        return None

    num_frames = random.randint(min_frames, min(video_frames, max_frames))
    start_idx = random.randint(0, video_frames - num_frames)
    end_idx = start_idx + num_frames
    selected_frames = video[start_idx:end_idx]
    logging.debug(f"Video has {video_frames} frames, selected {num_frames} frames from {start_idx} to {end_idx}")
    return selected_frames


# Currently, landscape only
RESOLUTIONS_720P = [
    # width, height
    (768, 768),
    (832, 704),
    (896, 640),
    (960, 576),
    (1024, 512)
]

RESOLUTIONS_1080P = [
    (1024, 1024),
    (1088, 960),
    (1152, 896),
    (1216, 832),
    (1280, 768),
    (1344, 704),
]


def random_crop(video: torch.Tensor):
    """
    Randomly crop the video frames to a random resolution in the resolution bucket.
    Center crop and resize the video frames to the same resolution.
    """
    t, c, h, w = video.shape
    resolution_list = RESOLUTIONS_720P if h < 1080 else RESOLUTIONS_1080P
    resolution = random.choice(resolution_list)
    target_h, target_w = resolution
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


def encode_color_palette(image: torch.Tensor, masks, area_threshold=0.03, num_colors=3):
    palettes = []
    locations = []

    c, h, w = image.shape
    total_area = h * w
    for mask_data in masks:
        mask = mask_data["mask"]
        area = mask_data["area"]
        if area < area_threshold * total_area:
            continue
        mask_torch = torch.tensor(mask, device=image.device)
        palette, location = extract_palette_from_masked_image(
            image, mask_torch, max_colors=num_colors, min_colors=1
        )
        palettes.append(palette)
        locations.append(location)

    if len(palettes) == 0:
        return None, None

    palettes = torch.cat(palettes, dim=0)
    locations = torch.cat(locations, dim=0)

    return palettes, locations