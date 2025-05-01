import numpy as np
import torch
import kornia
import cv2
import random
from einops import rearrange
import logging

from ...utils.common import make_grid

from .config import CollageConfig

logger = logging.getLogger(__name__)


def compute_transform_data_structured(
    flow: torch.Tensor,
    depth: torch.Tensor | None,
    masks: torch.Tensor,  # Shape: NHW, on GPU
    cfg: CollageConfig = CollageConfig(),
):
    if masks.shape[0] == 0:
        return [], []

    device = masks.device
    total_area = flow.shape[1] * flow.shape[2]
    min_area = int(cfg.min_object_area * total_area)
    max_drop_area = int(cfg.max_drop_area * total_area)
    mask_data = []

    # Compute areas on GPU
    areas = torch.sum(masks, dim=(1, 2))  # Shape: N
    valid_mask = areas > min_area
    valid_indices = torch.nonzero(valid_mask).squeeze(1)

    if valid_indices.numel() == 0:
        return [], []

    # Filter masks and areas
    valid_masks = masks[valid_indices]  # Shape: N'HW
    valid_areas = areas[valid_indices]  # Shape: N'

    # Sort by area (descending)
    sorted_indices = torch.argsort(valid_areas, descending=True)
    valid_masks = valid_masks[sorted_indices]
    valid_areas = valid_areas[sorted_indices]

    # Create mask_data list
    mask_data = [
        {"area": area.item(), "mask": mask, "remaining": mask.clone()}
        for area, mask in zip(valid_areas, valid_masks)
    ]

    # Find parent-child relationships
    for i in range(len(mask_data) - 1, -1, -1):
        current = mask_data[i]["mask"]
        for j in range(i - 1, -1, -1):
            candidate = mask_data[j]["mask"]
            intersection = torch.sum(current & candidate).item()
            if intersection > mask_data[i]["area"] * 0.9:
                mask_data[i]["parent"] = j
                mask_data[j]["has_child"] = True
                mask_data[j]["remaining"] = mask_data[j]["remaining"] & ~current
                break

    selected_masks = []
    dropped_masks = []
    for node in mask_data:
        if "parent" in node:
            if "finalized" in mask_data[node["parent"]]:
                node["finalized"] = True
                continue
        data = {"area": node["area"], "mask": node["mask"]}
        if "has_child" in node:
            rand = random.uniform(
                0,
                (
                    (cfg.chance_keep_stem + cfg.chance_split_stem)
                    if node["area"] > max_drop_area
                    else 1
                ),
            )
            if rand < cfg.chance_keep_stem:
                node["finalized"] = True
                selected_masks.append(data)
            elif rand <= cfg.chance_keep_stem + cfg.chance_split_stem:
                remain_area = torch.sum(node["remaining"]).item()
                if remain_area > min_area:
                    remain_data = {"area": remain_area, "mask": node["remaining"]}
                    if (
                        random.random() < cfg.chance_keep_leaf
                        or remain_area > max_drop_area
                    ):
                        selected_masks.append(remain_data)
                    else:
                        dropped_masks.append(remain_data)
            else:
                node["finalized"] = True
                dropped_masks.append(data)
        else:
            node["finalized"] = True
            if random.random() < cfg.chance_keep_leaf or node["area"] > max_drop_area:
                selected_masks.append(data)
            else:
                dropped_masks.append(data)

    # Compute depth and affine transform for selected masks
    for mask_dict in selected_masks:
        mask_torch = mask_dict["mask"]

        if depth is not None:
            masked_depth = depth[mask_torch]
            avg_depth = torch.mean(masked_depth).item()
            mask_dict["avg_depth"] = avg_depth

        # Sample 50 random points on GPU
        nonzero_coords = torch.nonzero(mask_torch, as_tuple=False)  # Shape: Kx2
        if len(nonzero_coords) < cfg.num_estimate_affine_samples:
            continue
        rand_indices = torch.randperm(len(nonzero_coords))[
            : cfg.num_estimate_affine_samples
        ]
        src_points = nonzero_coords[rand_indices].float()  # Shape: 50x2 (y, x)

        # Compute target points using flow
        src_y, src_x = src_points[:, 0], src_points[:, 1]
        flow_at_points = flow[:, src_y.long(), src_x.long()].permute(
            1, 0
        )  # Shape: 50x2
        tgt_points = src_points + flow_at_points

        # Transfer to CPU for cv2
        src_points_np = src_points.cpu().numpy()
        tgt_points_np = tgt_points.cpu().numpy()

        # Estimate affine transform
        affine_trans, inliers = cv2.estimateAffinePartial2D(
            src_points_np.astype(np.float32), tgt_points_np.astype(np.float32)
        )
        mask_dict["affine_trans"] = affine_trans

    logger.debug(
        "Compute transform data: %d masks selected, %d dropped",
        len(selected_masks),
        len(dropped_masks),
    )

    return selected_masks, dropped_masks


@torch.no_grad()
def apply_transforms(
    image, depth, transform_data, grid=None, cfg: CollageConfig = CollageConfig()
):
    """
    Apply transforms to the image based on transform data using GPU, maintaining CHW format.

    Args:
        image (torch.Tensor): [C, H, W] tensor on GPU, float32 (RGB channels)
        depth (torch.Tensor | None): [H, W] tensor on GPU, float32 (depth values, 0=far, 1=near)
        grid (torch.Tensor | None): [2, H, W] tensor on GPU, float32 (optional)
        transform_data (list of dicts): output from compute_transform_data, each dict contains:
            - affine_trans (numpy.ndarray): 2x3 affine transformation matrix
            - mask (torch.Tensor): [H, W] binary mask
            - avg_depth (float): average depth for sorting (0=far, 1=near)

    Returns:
        tuple: (transformed_rgb, transformed_grid, src_mask, tgt_mask)
            - transformed_rgb: [C, H, W] transformed RGB image
            - transformed_grid: [2, H, W] transformed coordinate grid
            - src_mask: [H, W] binary mask of warped regions
            - tgt_mask: [H, W] accumulated alpha mask
    """
    device = image.device
    H, W = image.shape[1:]
    C = image.shape[0]

    # Validate depth shape and convert to [1, H, W]
    if depth is not None:
        if depth.shape != (H, W):
            raise ValueError(f"Expected depth shape [H, W], got {depth.shape}")
        depth = depth.unsqueeze(0)  # [1, H, W]
    else:
        depth = torch.zeros((1, H, W), dtype=torch.float32, device=device)

    if grid is None:
        grid = make_grid(H, W, device=device)  # [2, H, W]

    # Prepare source tensor: concatenate image, grid, and depth
    src_tensor = torch.cat([image, grid, depth], dim=0)  # [C+3, H, W]
    canvas = torch.zeros((C+2, H, W), dtype=torch.float32, device=device)  # [C+2, H, W]
    tgt_mask = torch.zeros((1, H, W), dtype=torch.float32, device=device)  # [1, H, W]
    src_mask = torch.zeros((1, H, W), dtype=torch.float32, device=device)  # [1, H, W]
    z_buffer = torch.ones((1, H, W), dtype=torch.float32, device=device) * -1.0  # [1, H, W]

    # Sort by depth (far to near)
    if transform_data[0].get("avg_depth") is not None:
        sorted_indices = sorted(
            range(len(transform_data)), key=lambda i: transform_data[i]["avg_depth"]
        )
    else:
        sorted_indices = list(range(len(transform_data)))

    for idx in sorted_indices:
        data = transform_data[idx]
        affine_trans = torch.from_numpy(data["affine_trans"]).to(device).float()  # [2, 3]
        mask = data["mask"].to(device).float().unsqueeze(0)  # [1, H, W]
        
        if cfg.transform_dilate_size > 0:
            kernel = torch.ones(
                cfg.transform_dilate_size, cfg.transform_dilate_size, device=device
            )
            mask = kornia.morphology.dilation(mask.unsqueeze(0), kernel).squeeze(0)

        # Prepare alpha mask
        alpha_mask = mask * (src_mask < 0.5).float()  # [1, H, W]
        src_rgba = torch.cat([src_tensor, alpha_mask], dim=0)  # [C+4, H, W]

        # Apply affine transform using Kornia
        warp_dst = kornia.geometry.transform.warp_affine(
            src_rgba.unsqueeze(0), affine_trans[None, :2, :], (H, W)
        ).squeeze(0)  # [C+4, H, W]

        warped_mask = warp_dst[-1:, :, :]  # [1, H, W]
        warped_rgb = warp_dst[:C+2, :, :]  # [C+2, H, W]
        warped_depth = warp_dst[C+2:C+3, :, :]  # [1, H, W]

        # Update z-buffer and canvas
        good_z_region = warped_depth >= z_buffer  # [1, H, W]
        warped_mask = (warped_mask > 0.5) & good_z_region  # [1, H, W]
        warped_mask = warped_mask.float()

        # Erode mask using Kornia
        if cfg.transform_erode_size > 0:
            kernel = torch.ones(
                cfg.transform_erode_size, cfg.transform_erode_size, device=device
            )
            eroded_mask = kornia.morphology.erosion(
                warped_mask.unsqueeze(0), kernel
            ).squeeze(0)  # [1, H, W]
        else:
            eroded_mask = warped_mask

        tgt_mask = tgt_mask + eroded_mask  # [1, H, W]
        src_mask = src_mask + alpha_mask  # [1, H, W]
        canvas = canvas * (1.0 - warped_mask) + warped_mask * warped_rgb  # [C+2, H, W]
        z_buffer = z_buffer * (1.0 - warped_mask) + warped_mask * warped_depth  # [1, H, W]

    src_mask = torch.clamp(src_mask, 0.0, 1.0)
    tgt_mask = torch.clamp(tgt_mask, 0.0, 1.0)  # [1, H, W]

    # Extract results
    transformed_rgb = canvas[:C, :, :]  # [C, H, W]
    transformed_grid = canvas[C:C+2, :, :]  # [2, H, W]
    src_mask = src_mask.squeeze(0)  # [H, W]
    tgt_mask = tgt_mask.squeeze(0)  # [H, W]

    return transformed_rgb, transformed_grid, src_mask, tgt_mask