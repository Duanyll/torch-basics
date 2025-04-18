import numpy as np
import torch
import kornia
import cv2
import random
from einops import rearrange
import logging

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
    min_area: int = 300,
    chance_keep_leaf: float = 0.9,
    chance_keep_parent: float = 0.2,
    chance_split: float = 0.5,
    estimate_affine_samples: int = 50,
):
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
        rand = random.random()
        data = {"area": node["area"], "mask": node["mask"]}
        if "has_child" in node:
            if rand < chance_keep_parent:
                node["finalized"] = True
                selected_masks.append(data)
            elif rand < chance_keep_parent + chance_split:
                remain_area = np.sum(node["remaining"])
                if remain_area > min_area:
                    remain_data = {"area": remain_area, "mask": node["remaining"]}
                    if random.random() < chance_keep_leaf:
                        selected_masks.append(remain_data)
                    else:
                        dropped_masks.append(remain_data)
            else:
                node["finalized"] = True
                dropped_masks.append(data)
        else:
            node["finalized"] = True
            if rand < chance_keep_leaf:
                selected_masks.append(data)
            else:
                dropped_masks.append(data)

    # Compute depth and affine transform for selected masks
    flow_np = flow.cpu().numpy()
    for mask in selected_masks:
        mask_np = mask["mask"]
        mask_torch = torch.from_numpy(mask_np)

        # Compute median depth (inverse depth)
        masked_depth = depth[mask_torch]
        avg_depth = torch.median(1.0 / (masked_depth + 1e-6)).item()
        mask["avg_depth"] = avg_depth

        # Sample 50 random points
        all_y, all_x = np.nonzero(mask_np)
        rand_indices = np.random.choice(
            len(all_y), size=estimate_affine_samples, replace=False
        )
        src_points = np.array([all_x[rand_indices], all_y[rand_indices]]).T
        tgt_points = src_points + flow_np[:, src_points[:, 1], src_points[:, 0]].T

        # Estimate affine transform on CPU
        affine_trans, inliers = cv2.estimateAffinePartial2D(
            src_points.astype(np.float32), tgt_points.astype(np.float32)
        )
        mask["affine_trans"] = affine_trans
        
    logger.debug("Compute transform data: %d masks selected, %d dropped", len(selected_masks), len(dropped_masks))

    return selected_masks, dropped_masks


def apply_transforms_cpu(image, transform_data):
    """
    Apply transforms to the image based on transform data.

    Args:
        image (torch.Tensor): [C, H, W] tensor
        transform_data (list of dicts): output from compute_transform_data

    Returns:
        tuple: (transformed_rgb, transformed_grid, warped_regions)
    """
    H, W = image.shape[1:]
    C = image.shape[0]

    # Create coordinate grid
    x = np.linspace(0, 1, W)
    y = np.linspace(0, 1, H)
    xv, yv = np.meshgrid(x, y)
    grid_array = np.stack([xv, yv], axis=-1)  # [H, W, 2]

    # Prepare source array
    src_array = np.concatenate(
        [image.cpu().numpy().transpose(1, 2, 0), grid_array], axis=-1
    )  # [H, W, C+2]
    canvas = np.zeros_like(src_array)
    canvas_alpha = np.zeros((H, W, 1), dtype=float)
    warped_regions = np.zeros((H, W, 1), dtype=float)
    z_buffer = np.ones((H, W, 1), dtype=float) * -1.0

    # Sort by depth (far to near)
    sorted_indices = sorted(
        range(len(transform_data)), key=lambda i: transform_data[i]["avg_depth"]
    )

    for idx in sorted_indices:
        data = transform_data[idx]
        affine_trans = data["affine_trans"]
        mask = data["mask"]

        # Prepare alpha mask
        alpha_mask = mask[..., None].astype(float) * (warped_regions < 0.5).astype(float)
        src_rgba = np.concatenate([src_array, alpha_mask], axis=-1)

        # Apply affine transform
        warp_dst = cv2.warpAffine(src_rgba, affine_trans, (W, H))
        warped_mask = warp_dst[..., -1:]  # Alpha channel
        warped_rgb = warp_dst[..., : C + 2]  # RGB + grid

        # Update z-buffer and canvas
        good_z_region = warped_rgb[..., -1:] > z_buffer
        warped_mask = np.logical_and(warped_mask > 0.5, good_z_region).astype(float)

        kernel = np.ones((3, 3), float)
        eroded_mask = cv2.erode(warped_mask, kernel)[..., None]

        canvas_alpha += eroded_mask
        warped_regions += alpha_mask
        canvas = canvas * (1.0 - warped_mask) + warped_mask * warped_rgb
        z_buffer = z_buffer * (1.0 - warped_mask) + warped_mask * warped_rgb[..., -1:]

    # Extract results
    transformed_rgb = canvas[..., :C]
    transformed_grid = canvas[..., C : C + 2]

    transformed_rgb = rearrange(
        torch.from_numpy(transformed_rgb), "h w c -> c h w"
    )  # [C, H, W]
    transformed_grid = rearrange(
        torch.from_numpy(transformed_grid), "h w c -> c h w"
    )  # [2, H, W]
    warped_regions = rearrange(
        torch.from_numpy(warped_regions), "h w 1 -> h w"
    )  # [1, H, W]

    return transformed_rgb, transformed_grid, warped_regions


@torch.no_grad()
def apply_transforms(image, transform_data):
    """
    Apply transforms to the image based on transform data using GPU and einops for enhanced readability.

    Args:
        image (torch.Tensor): [C, H, W] tensor on GPU, float32
        transform_data (list of dicts): output from compute_transform_data

    Returns:
        tuple: (transformed_rgb, transformed_grid, warped_regions, canvas_alpha)
    """
    device = image.device
    H, W = image.shape[1:]
    C = image.shape[0]

    # Create coordinate grid on GPU
    grid_array = kornia.utils.create_meshgrid(H, W, normalized_coordinates=True, device=device)
    grid_array = rearrange(grid_array, "1 h w c -> h w c")  # [H, W, 2]

    # Prepare source array
    src_array = rearrange(image, "c h w -> h w c")  # [H, W, C]
    src_array = torch.cat([src_array, grid_array], dim=-1)  # [H, W, C+2]
    canvas = torch.zeros_like(src_array)
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
        src_rgba = torch.cat([src_array, alpha_mask], dim=-1)  # [H, W, C+3]
        src_rgba = rearrange(src_rgba, "h w c -> 1 c h w")  # [1, C+3, H, W]

        # Apply affine transform using Kornia
        warp_dst = kornia.geometry.transform.warp_affine(
            src_rgba, affine_trans[None, :2, :], (H, W)
        ).squeeze(
            0
        )  # [C+3, H, W]
        warp_dst = rearrange(warp_dst, "c h w -> h w c")  # [H, W, C+3]

        warped_mask = warp_dst[..., -1:]  # [H, W, 1]
        warped_rgb = warp_dst[..., : C + 2]  # [H, W, C+2]
        warped_depth = warped_rgb[..., -1:]  # [H, W, 1]

        # Update z-buffer and canvas
        good_z_region = warped_depth > z_buffer
        warped_mask = (warped_mask > 0.5) & good_z_region
        warped_mask = warped_mask.float()

        # Erode mask using Kornia
        kernel = torch.ones(3, 3, device=device)
        eroded_mask = (
            kornia.morphology.erosion(
                rearrange(warped_mask, "h w 1 -> 1 1 h w"), kernel
            )
            .squeeze(0)
            .squeeze(0)
        )  # [H, W]
        eroded_mask = rearrange(eroded_mask, "h w -> h w 1")  # [H, W, 1]

        canvas_alpha += eroded_mask
        warped_regions += alpha_mask
        canvas = canvas * (1.0 - warped_mask) + warped_mask * warped_rgb
        z_buffer = z_buffer * (1.0 - warped_mask) + warped_mask * warped_depth

    # Extract results
    transformed_rgb = rearrange(canvas[..., :C], "h w c -> c h w")  # [C, H, W]
    transformed_grid = rearrange(canvas[..., C : C + 2], "h w c -> c h w")  # [2, H, W]
    warped_regions = rearrange(warped_regions, "h w 1 -> h w")  # [1, H, W]
    canvas_alpha = rearrange(canvas_alpha, "h w 1 -> h w")  # [1, H, W]

    return transformed_rgb, transformed_grid, warped_regions, canvas_alpha
