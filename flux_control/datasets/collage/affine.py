import numpy as np
import torch
import kornia
import cv2
import random
from einops import rearrange
import logging

from .config import CollageConfig

logger = logging.getLogger(__name__)


def compute_transform_data(flow, depth, masks):
    """
    Compute transform data for each mask.

    Args:
        src_image (torch.Tensor): [C, H, W] tensor, possibly on GPU
        flow (torch.Tensor): [2, H, W] tensor, possibly on GPU
        depth (torch.Tensor): [H, W] tensor, possibly on GPU
        masks (list of np.ndarray): list of [H, W] bool arrays

    Returns:
        list of dicts: each dict contains 'area', 'avg_depth', 'affine_trans'
    """
    transform_data = []

    for mask in masks:
        mask_torch = torch.from_numpy(mask)

        # Compute area
        area = torch.sum(mask_torch).item()
        if area <= 300:
            continue

        # Compute median depth (inverse depth)
        masked_depth = depth[mask_torch]
        avg_depth = torch.median(1.0 / (masked_depth + 1e-6)).item()

        # Sample 50 random points
        all_y, all_x = np.nonzero(mask)
        if len(all_y) < 50:
            continue
        rand_indices = np.random.choice(len(all_y), size=50, replace=False)
        src_points = np.array([all_x[rand_indices], all_y[rand_indices]]).T

        # Compute target points using flow
        flow_np = flow.cpu().numpy()
        tgt_points = src_points + flow_np[:, src_points[:, 1], src_points[:, 0]].T

        # Estimate affine transform on CPU
        affine_trans, inliers = cv2.estimateAffinePartial2D(
            src_points.astype(np.float32), tgt_points.astype(np.float32)
        )

        transform_data.append(
            {
                "area": area,
                "avg_depth": avg_depth,
                "affine_trans": affine_trans,
                "mask": mask,  # Store mask for use in apply_transforms
            }
        )

    return transform_data


def compute_transform_data_structured(
    flow: torch.Tensor,
    depth: torch.Tensor,
    masks: list[np.ndarray],
    cfg: CollageConfig = CollageConfig(),
):
    if len(masks) == 0:
        return [], []

    total_area = depth.shape[0] * depth.shape[1]
    min_area = int(cfg.min_object_area * total_area)
    max_drop_area = int(cfg.max_drop_area * total_area)
    mask_data = []

    for mask in masks:
        area = np.sum(mask)
        if area <= min_area:
            continue
        mask_data.append({"area": area, "mask": mask})

    mask_data.sort(key=lambda x: x["area"], reverse=True)
    for i in range(len(mask_data) - 1, -1, -1):
        current = mask_data[i]["mask"]
        for j in range(i - 1, -1, -1):
            candidate = mask_data[j]["mask"]
            if np.sum(current & candidate) > mask_data[i]["area"] * 0.9:
                mask_data[i]["parent"] = j
                mask_data[j]["has_child"] = True
                if not "remaining" in mask_data[j]:
                    mask_data[j]["remaining"] = candidate.copy()
                # mask_data[j]["remaining"] -= current
                mask_data[j]["remaining"] = np.logical_and(
                    mask_data[j]["remaining"], np.logical_not(current)
                )
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
                remain_area = np.sum(node["remaining"])
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
    flow_np = flow.cpu().numpy()
    for mask in selected_masks:
        mask_np = mask["mask"]
        mask_torch = torch.from_numpy(mask_np)

        masked_depth = depth[mask_torch]
        avg_depth = torch.mean(masked_depth).item()
        mask["avg_depth"] = avg_depth

        # Sample 50 random points
        all_y, all_x = np.nonzero(mask_np)
        rand_indices = np.random.choice(
            len(all_y), size=cfg.num_estimate_affine_samples, replace=False
        )
        src_points = np.array([all_x[rand_indices], all_y[rand_indices]]).T
        tgt_points = src_points + flow_np[:, src_points[:, 1], src_points[:, 0]].T

        # Estimate affine transform on CPU
        affine_trans, inliers = cv2.estimateAffinePartial2D(
            src_points.astype(np.float32), tgt_points.astype(np.float32)
        )
        mask["affine_trans"] = affine_trans

    logger.debug(
        "Compute transform data: %d masks selected, %d dropped",
        len(selected_masks),
        len(dropped_masks),
    )

    return selected_masks, dropped_masks


@torch.no_grad()
def apply_transforms(
    image, depth, transform_data, cfg: CollageConfig = CollageConfig()
):
    """
    Apply transforms to the image based on transform data using GPU and einops for enhanced readability.

    Args:
        image (torch.Tensor): [C, H, W] tensor on GPU, float32 (RGB channels)
        depth (torch.Tensor): [H, W] tensor on GPU, float32 (depth values, 0=far, 1=near)
        transform_data (list of dicts): output from compute_transform_data, each dict contains:
            - affine_trans (numpy.ndarray): 2x3 affine transformation matrix
            - mask (numpy.ndarray): [H, W] binary mask
            - avg_depth (float): average depth for sorting (0=far, 1=near)

    Returns:
        tuple: (transformed_rgb, transformed_grid, warped_regions, canvas_alpha)
            - transformed_rgb: [C, H, W] transformed RGB image
            - transformed_grid: [2, H, W] transformed coordinate grid
            - warped_regions: [H, W] binary mask of warped regions
            - canvas_alpha: [H, W] accumulated alpha mask
    """
    device = image.device
    H, W = image.shape[1:]
    C = image.shape[0]

    # Validate depth shape and convert to [H, W, 1]
    if depth.shape != (H, W):
        raise ValueError(f"Expected depth shape [H, W], got {depth.shape}")
    depth = depth.unsqueeze(-1)  # [H, W, 1]

    # Create coordinate grid on GPU
    grid_array = kornia.utils.create_meshgrid(
        H, W, normalized_coordinates=True, device=device
    )
    grid_array = rearrange(grid_array, "1 h w c -> h w c")  # [H, W, 2]

    # Prepare source array
    src_array = rearrange(image, "c h w -> h w c")  # [H, W, C]
    src_array = torch.cat([src_array, grid_array, depth], dim=-1)  # [H, W, C+3]
    canvas = torch.zeros(
        (H, W, C + 2), dtype=torch.float32, device=device
    )  # [H, W, C+2]
    canvas_alpha = torch.zeros((H, W, 1), dtype=torch.float32, device=device)
    warped_regions = torch.zeros((H, W, 1), dtype=torch.float32, device=device)
    z_buffer = torch.ones((H, W, 1), dtype=torch.float32, device=device) * -1.0

    # Sort by depth (far to near)
    sorted_indices = sorted(
        range(len(transform_data)), key=lambda i: transform_data[i]["avg_depth"]
    )

    for idx in sorted_indices:
        data = transform_data[idx]
        affine_trans = torch.from_numpy(data["affine_trans"]).to(device).float()
        mask = torch.from_numpy(data["mask"]).to(device).float()[..., None]

        # Prepare alpha mask
        alpha_mask = mask * (warped_regions < 0.5).float()
        src_rgba = torch.cat([src_array, alpha_mask], dim=-1)  # [H, W, C+4]
        src_rgba = rearrange(src_rgba, "h w c -> 1 c h w")  # [1, C+4, H, W]

        # Apply affine transform using Kornia
        warp_dst = kornia.geometry.transform.warp_affine(
            src_rgba, affine_trans[None, :2, :], (H, W)
        ).squeeze(
            0
        )  # [C+4, H, W]
        warp_dst = rearrange(warp_dst, "c h w -> h w c")  # [H, W, C+4]

        warped_mask = warp_dst[..., -1:]  # [H, W, 1]
        warped_rgb = warp_dst[..., : C + 2]  # [H, W, C+2]
        warped_depth = warp_dst[..., C + 2 : C + 3]  # [H, W, 1]

        # Update z-buffer and canvas
        good_z_region = warped_depth > z_buffer
        warped_mask = (warped_mask > 0.5) & good_z_region
        warped_mask = warped_mask.float()

        # Erode mask using Kornia
        kernel = torch.ones(
            cfg.transform_erode_size, cfg.transform_erode_size, device=device
        )
        eroded_mask = (
            kornia.morphology.erosion(
                rearrange(warped_mask, "h w 1 -> 1 1 h w"), kernel
            )
            .squeeze(0)
            .squeeze(0)
        )  # [H, W]
        eroded_mask = rearrange(eroded_mask, "h w -> h w 1")  # [H, W, 1]

        canvas_alpha += eroded_mask
        canvas_alpha = torch.clamp(canvas_alpha, 0.0, 1.0)
        warped_regions += alpha_mask
        canvas = canvas * (1.0 - warped_mask) + warped_mask * warped_rgb
        z_buffer = z_buffer * (1.0 - warped_mask) + warped_mask * warped_depth

    # Extract results
    transformed_rgb = rearrange(canvas[..., :C], "h w c -> c h w")  # [C, H, W]
    transformed_grid = rearrange(canvas[..., C : C + 2], "h w c -> c h w")  # [2, H, W]
    warped_regions = rearrange(warped_regions, "h w 1 -> h w")  # [H, W]
    canvas_alpha = rearrange(canvas_alpha, "h w 1 -> h w")  # [H, W]

    return transformed_rgb, transformed_grid, warped_regions, canvas_alpha
