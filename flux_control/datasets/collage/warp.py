import torch
import torch.nn.functional as F
import einops
import kornia
import logging
from typing import Tuple, Optional

from .softsplat import softsplat
from ...utils.common import make_grid


def backwarp(tenIn: torch.Tensor, tenFlow: torch.Tensor) -> torch.Tensor:
    """
    Performs backward warping of an input tensor using an optical flow field.

    Args:
        tenIn (torch.Tensor): The input tensor to warp (NCHW format).
        tenFlow (torch.Tensor): The optical flow field (N2HW format).

    Returns:
        torch.Tensor: The warped input tensor (NCHW format).
    """
    n, c, h, w = tenIn.shape
    n_f, c_f, h_f, w_f = tenFlow.shape

    if h != h_f or w != w_f:
        raise ValueError(
            f"Input tensor shape ({h}x{w}) and flow shape ({h_f}x{w_f}) dimensions mismatch."
        )
    if n != n_f or c_f != 2:
        raise ValueError(
            f"Batch size mismatch or flow tensor does not have 2 channels."
        )

    # Generate base sampling grid (normalized to -1.0 to 1.0)
    # Using einops for potentially clearer shape manipulation, though linspace is direct
    tenHor = torch.linspace(
        start=-1.0, end=1.0, steps=w, dtype=tenFlow.dtype, device=tenFlow.device
    )
    tenVer = torch.linspace(
        start=-1.0, end=1.0, steps=h, dtype=tenFlow.dtype, device=tenFlow.device
    )

    grid_w = einops.repeat(tenHor, "w -> n c h w", n=n, c=1, h=h)
    grid_h = einops.repeat(tenVer, "h -> n c h w", n=n, c=1, w=w)

    base_grid = torch.cat([grid_w, grid_h], 1)  # Shape: N 2 H W

    # Normalize flow to match the grid's coordinate system (-1 to 1)
    # Flow values represent pixel displacement. Normalization maps displacement
    # to the normalized grid space.
    # Flow[0] is horizontal (W), Flow[1] is vertical (H)
    norm_flow = torch.stack(
        [
            tenFlow[:, 0:1, :, :] / ((w - 1.0) / 2.0),
            tenFlow[:, 1:2, :, :] / ((h - 1.0) / 2.0),
        ],
        dim=1,
    ).squeeze(
        2
    )  # Ensure shape N 2 H W

    # Combine base grid and normalized flow, then rearrange for grid_sample
    # grid_sample expects the grid in N H W 2 format
    sampling_grid = base_grid + norm_flow
    sampling_grid = einops.rearrange(sampling_grid, "n c h w -> n h w c")  # N H W 2

    # Perform the backward warping
    tenWarped = F.grid_sample(
        input=tenIn,
        grid=sampling_grid,
        mode="bilinear",
        padding_mode="zeros",  # Pixels sampled outside the input are set to zero
        align_corners=True,  # Consistent with the original code
    )
    return tenWarped


# --- Main Interface: forward_warp ---


def forward_warp(
    src_frame: torch.Tensor,
    tgt_frame: torch.Tensor,
    flow: torch.Tensor,
    grid: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Performs forward warping (softmax splatting) on a source frame,
    optionally including a coordinate grid and a mask, using an optical flow field.

    Args:
        src_frame (torch.Tensor): Source image tensor (C H W).
        tgt_frame (torch.Tensor): Target image tensor (C H W), used for metric calculation.
        flow (torch.Tensor): Optical flow tensor from source to target (2 H W).
        grid (Optional[torch.Tensor]): Coordinate grid tensor (2 H W). If provided,
                                      it will also be splatted using the flow.
        mask (Optional[torch.Tensor]): Boolean mask tensor (H W). If provided,
                                     it's converted to float and splatted.
                                     If None, a mask of ones is used.

    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]: A tuple containing:
            - splatted_rgb (torch.Tensor): The splatted RGB frame (C H W).
            - splatted_grid (torch.Tensor): The splatted coordinate grid (2 H W)
                                                    if `grid` was provided, else None.
            - splatted_mask (torch.Tensor): The splatted mask (H W), representing
                                           flow contribution/occlusion. Values are floats.
    """
    # --- Input Validation and Preparation ---
    if not (src_frame.ndim == 3 and tgt_frame.ndim == 3 and flow.ndim == 3):
        raise ValueError(
            "src_frame, tgt_frame, and flow must be 3D tensors (CHW or 2HW)."
        )
    if flow.shape[0] != 2:
        raise ValueError(f"Flow tensor must have 2 channels, got shape {flow.shape}")
    if src_frame.shape[1:] != flow.shape[1:] or tgt_frame.shape[1:] != flow.shape[1:]:
        raise ValueError("Spatial dimensions (H, W) of inputs must match.")

    device = src_frame.device
    c, h, w = src_frame.shape
    c_tgt = tgt_frame.shape[0]

    # Ensure all inputs are on the same device
    tgt_frame = tgt_frame.to(device)
    flow = flow.to(device)

    # Handle optional mask
    if mask is not None:
        if mask.ndim != 2 or mask.shape != (h, w):
            raise ValueError(
                f"Mask must be 2D tensor (HW) with shape {(h, w)}, got {mask.shape}"
            )
        # Convert boolean mask to float and add channel dimension (1 H W)
        processed_mask = mask.to(device=device, dtype=torch.float32).unsqueeze(0)
    else:
        # Create a default mask of ones (1 H W)
        processed_mask = torch.ones((1, h, w), dtype=torch.float32, device=device)

    # Handle optional grid
    if grid is not None:
        if grid.ndim != 3 or grid.shape[0] != 2 or grid.shape[1:] != (h, w):
            raise ValueError(
                f"Grid must be 3D tensor (2HW) with shape {(2, h, w)}, got {grid.shape}"
            )
        grid = grid.to(device)
    else:
        grid = make_grid(h, w, device=device)

    # --- Add Batch Dimension for Internal Processing ---
    # Use einops for clarity, equivalent to tensor.unsqueeze(0)
    src_frame_b = einops.rearrange(src_frame, "c h w -> 1 c h w")
    tgt_frame_b = einops.rearrange(tgt_frame, "c h w -> 1 c h w")
    flow_b = einops.rearrange(flow, "c h w -> 1 c h w")
    mask_b = einops.rearrange(processed_mask, "c h w -> 1 c h w")
    grid_b = einops.rearrange(grid, "c h w -> 1 c h w")

    # --- Concatenate Tensors for Splatting ---
    # Concatenate source data: [src_rgb, (optional) grid, mask]
    tenOne = torch.cat(
        [src_frame_b, grid_b, mask_b], dim=1
    )  # Shape: N (C+2+1) H W or N (C+1) H W

    # Concatenate target data for metric calculation.
    # Only the RGB part of the target is needed for the L1 metric.
    # We need placeholders for grid/mask dimensions if they exist in tenOne,
    # although they won't be used in the backwarp for metric calculation.
    tgt_list = [tgt_frame_b]
    if grid_b is not None:
        # Add zero placeholder for grid channels
        tgt_list.append(torch.zeros_like(grid_b))
    # Add zero placeholder for mask channel
    tgt_list.append(torch.zeros_like(mask_b))
    tenTwo = torch.cat(tgt_list, dim=1)  # Shape matches tenOne

    # --- Calculate Importance Metric ---
    # Metric is based on L1 loss between src and backwarped target (RGB only)
    # Using only first 3 channels (RGB) as per original 'partial=True' logic
    c_metric = min(c, c_tgt, 3)  # Use min channels up to 3 for metric

    warped_tgt_rgb = backwarp(tenIn=tenTwo[:, :c_metric, :, :], tenFlow=flow_b)
    tenMetric = F.l1_loss(
        input=tenOne[:, :c_metric, :, :], target=warped_tgt_rgb, reduction="none"
    ).mean(
        dim=1, keepdim=True
    )  # Average over channels -> N 1 H W

    # Apply scaling and clipping to the metric (hyperparameter alpha = -20.0)
    alpha = -20.0
    tenMetric = (alpha * tenMetric).clip(
        min=alpha, max=-alpha
    )  # Clip between -20 and 20

    # --- Perform Soft Splatting ---
    # tenIn: Contains [src_rgb, (optional) grid, mask]
    # tenFlow: Optical flow
    # tenMetric: Importance weights
    tenSoftmax = softsplat(
        tenIn=tenOne, tenFlow=flow_b, tenMetric=tenMetric, strMode="soft"
    )

    # --- Extract and Return Results ---
    # Remove batch dimension
    splatted_data = einops.rearrange(tenSoftmax, "1 c h w -> c h w")

    # Extract RGB
    splatted_rgb = splatted_data[:c, :, :]
    splatted_grid = splatted_data[c : c + 2, :, :]
    splatted_mask = splatted_data[c + 2, :, :]  # Shape: H W

    return splatted_rgb, splatted_grid, splatted_mask


# --- Example Usage (requires dummy data and softsplat package) ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)  # Show debug messages for example

    # Example dimensions
    C, H, W = 3, 64, 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Create dummy tensors (replace with actual data)
    src_frame_test = torch.rand(C, H, W, device=device)
    tgt_frame_test = torch.rand(C, H, W, device=device)
    # Flow typically ranges, let's simulate some movement
    flow_test = (torch.rand(2, H, W, device=device) - 0.5) * 10

    # Optional grid (e.g., representing normalized coordinates)
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing="ij",
    )
    grid_test = torch.stack((grid_x, grid_y), dim=0)  # Shape 2, H, W

    # Optional mask (e.g., a circular region)
    center_x, center_y = W // 2, H // 2
    radius_sq = (min(H, W) // 4) ** 2
    mask_y, mask_x = torch.meshgrid(
        torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij"
    )
    mask_test = (mask_x - center_x) ** 2 + (
        mask_y - center_y
    ) ** 2 < radius_sq  # HW boolean

    logging.info("--- Running forward_warp with grid and mask ---")
    try:
        splatted_rgb, splatted_grid, splatted_mask = forward_warp(
            src_frame=src_frame_test,
            tgt_frame=tgt_frame_test,
            flow=flow_test,
            grid=grid_test,
            mask=mask_test,
        )

        print(f"Splatted RGB shape: {splatted_rgb.shape}")
        if splatted_grid is not None:
            print(f"Splatted Grid shape: {splatted_grid.shape}")
        print(f"Splatted Mask shape: {splatted_mask.shape}")
        # Check output types and device
        print(
            f"Splatted RGB dtype: {splatted_rgb.dtype}, device: {splatted_rgb.device}"
        )
        if splatted_grid is not None:
            print(
                f"Splatted Grid dtype: {splatted_grid.dtype}, device: {splatted_grid.device}"
            )
        print(
            f"Splatted Mask dtype: {splatted_mask.dtype}, device: {splatted_mask.device}"
        )

        logging.info("\n--- Running forward_warp without grid and mask ---")
        splatted_rgb_no_opt, splatted_grid_no_opt, splatted_mask_no_opt = forward_warp(
            src_frame=src_frame_test,
            tgt_frame=tgt_frame_test,
            flow=flow_test,
            # grid=None, # Implicitly None
            # mask=None  # Implicitly None
        )

        print(f"Splatted RGB shape (no options): {splatted_rgb_no_opt.shape}")
        print(f"Splatted Grid (no options): {splatted_grid_no_opt}")  # Should be None
        print(
            f"Splatted Mask shape (no options): {splatted_mask_no_opt.shape}"
        )  # Should be HW

    except Exception as e:
        logging.error(
            f"An error occurred during forward_warp execution: {e}", exc_info=True
        )
        print(
            "\nError during execution. Check logs. If using placeholder softsplat, functionality is limited."
        )
