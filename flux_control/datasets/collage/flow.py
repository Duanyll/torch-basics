import math
import torch
import torch.nn.functional as F
import torchvision.models.optical_flow as flow_models
from torchvision.transforms import functional as TF
import numpy as np
import kornia
import kornia.filters as KFilters
import kornia.geometry.transform as KGeom
import logging
import time
from typing import Tuple, Optional, List, Any

logger = logging.getLogger(__name__)

# --- Constants and Global Variables ---
RAFT_INPUT_SIZE = [520, 960]  # Fixed input size for RAFT model
# RAFT_INPUT_SIZE = [368, 768] # Smaller alternative if needed

model: Any = None
model_device: Any = None
raft_transforms: Any = None

def load_raft_model(device = "cuda") -> None:
    """
    Loads the RAFT model and its weights.

    Returns:
        The loaded RAFT model.
    """
    global model, raft_transforms, model_device
    try:
        weights = flow_models.Raft_Large_Weights.DEFAULT
        raft_transforms = weights.transforms()
        model = flow_models.raft_large(weights=weights, progress=False).to(device)
        model = model.eval()
        model_device = device
        logger.info("RAFT Large model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading RAFT model: {e}")
        raise e


def unload_raft_model() -> None:
    """
    Unloads the RAFT model and its weights.
    """
    global model, raft_transforms, model_device
    model = None
    raft_transforms = None
    model_device = None
    torch.cuda.empty_cache()
    logger.info("RAFT Large model unloaded successfully.")

# --- Helper Functions ---


def _preprocess_for_raft(
    img1_bchw: torch.Tensor,
    img2_bchw: torch.Tensor,
    target_size: List[int] = RAFT_INPUT_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Resizes and transforms image batches for RAFT input.

    Args:
        img1_bchw: First image batch (B, C, H, W).
        img2_bchw: Second image batch (B, C, H, W).
        target_size: The target size [height, width] for RAFT input.

    Returns:
        Tuple containing the transformed image batches ready for RAFT.
    """
    # Resize
    img1_resized = TF.resize(img1_bchw, size=target_size, antialias=False)
    img2_resized = TF.resize(img2_bchw, size=target_size, antialias=False)

    # Apply RAFT-specific transforms
    return raft_transforms(img1_resized, img2_resized)


@torch.no_grad()
def _compute_pairwise_flow(
    img1_bchw: torch.Tensor,
    img2_bchw: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the optical flow between two image batches using RAFT.

    Args:
        img1_bchw: First image batch (B, C, H, W) on the correct device.
        img2_bchw: Second image batch (B, C, H, W) on the correct device.

    Returns:
        The computed flow field (B, 2, H, W) on the correct device,
        scaled to the original image dimensions. Returns flow for batch size 1.
    """
    if img1_bchw.shape[0] != 1 or img2_bchw.shape[0] != 1:
        logger.warning(
            f"Batch size > 1 ({img1_bchw.shape[0]}) received in _compute_pairwise_flow. Processing only the first element."
        )
        # Or handle batch > 1 if necessary, RAFT can process batches
        img1_bchw = img1_bchw[:1]
        img2_bchw = img2_bchw[:1]
        
    if model_device is None:
        load_raft_model(device=img1_bchw.device)

    B, C, H, W = img1_bchw.shape
    raft_h, raft_w = RAFT_INPUT_SIZE

    # Preprocess images for RAFT input
    img1_raft, img2_raft = _preprocess_for_raft(
        img1_bchw, img2_bchw, target_size=RAFT_INPUT_SIZE
    )

    # Compute flow using RAFT model
    list_of_flows = model(img1_raft.to(model_device), img2_raft.to(model_device))
    predicted_flow_raft = list_of_flows[-1]  # RAFT returns a list of flow predictions

    # Resize flow to original dimensions
    # align_corners=False is generally recommended for flow/sampling tasks
    predicted_flow_resized = KGeom.resize(
        predicted_flow_raft,
        size=(H, W),
        interpolation="bilinear",
        align_corners=False,
        antialias=False,  # Important for flow resizing
    )

    # Scale flow values to match original resolution
    # predicted_flow_resized has shape (B, 2, H, W)
    predicted_flow_resized[:, 0] *= W / raft_w
    predicted_flow_resized[:, 1] *= H / raft_h

    return predicted_flow_resized  # Return flow on the device it was computed on


# --- Main Interface Function ---


@torch.no_grad()
def compute_aggregated_flow(
    frames_tchw: torch.Tensor,
    device: str = "cuda",
    median_filter_kernel_size: int = 5,
    low_motion_threshold_frame5: float = 0.01,
    low_motion_threshold_final: float = 0.03,
    stable_flow_threshold: float = 0.5,
) -> Tuple[Optional[torch.Tensor], int]:
    """
    Computes aggregated optical flow across a sequence of frames.

    Args:
        frames_tchw: Input tensor of frames (T, C, H, W) on CPU or GPU.
        median_filter_kernel_size: Kernel size for the median blur applied to aggregated flow.
        low_motion_threshold_frame5: 95th percentile flow magnitude threshold at frame 5
                                     to detect static scenes early.
        low_motion_threshold_final: 95th percentile flow magnitude threshold at the end
                                    to detect static scenes.
        stable_flow_threshold: 90th percentile flow magnitude threshold. Aggregation stops
                               if flow exceeds this, returning the last stable flow.

    Returns:
        A tuple containing:
        - agg_flow: The final aggregated flow (2, H, W) on the GPU, or None if motion
                    is too low or input is invalid.
        - tgt_idx: The index of the *target* frame corresponding to the returned `agg_flow`
                   (i.e., flow from frame 0 to frame `tgt_idx`). Returns 0 if no valid
                   flow is computed.
    """
    T, C, H, W = frames_tchw.shape
    if T < 2:
        logger.warning("Need at least 2 frames to compute flow.")
        return None, 0
    
    base_size = math.sqrt(H ** 2 + W ** 2)
    low_motion_threshold_frame5 *= base_size
    low_motion_threshold_final *= base_size
    stable_flow_threshold *= base_size

    # Ensure frames are on the correct device and add batch dimension
    frames_b1tchw = frames_tchw.unsqueeze(0).to(device)  # (1, T, C, H, W)

    # Initialize aggregated flow (identity transform)
    agg_flow = torch.zeros((1, 2, H, W), device=device, dtype=frames_b1tchw.dtype)

    # Create sampling grid (once)
    grid = kornia.create_meshgrid(
        H, W, normalized_coordinates=False, device=device
    )  # (1, H, W, 2)
    # Grid needs to be (1, 2, H, W) for addition below -> (y, x) format internally? No, it's (x,y)
    # Kornia create_meshgrid returns (x, y) coordinates. Flow is usually (dx, dy).
    # Grid for grid_sample needs (x, y) format.
    # grid = grid.permute(0, 3, 1, 2) # -> (1, 2, H, W)

    best_flow = None
    final_tgt_idx = 0

    start_time = time.time()

    for t in range(T - 1):
        current_tgt_idx = t + 1

        # Get current frame pair
        img1 = frames_b1tchw[:, t, :, :, :]  # (1, C, H, W)
        img2 = frames_b1tchw[:, t + 1, :, :, :]  # (1, C, H, W)

        # 1. Compute pairwise flow (Frame t -> Frame t+1)
        pairwise_flow = _compute_pairwise_flow(img1, img2)  # (1, 2, H, W)

        # 2. Warp the pairwise flow according to the current aggregated flow
        #    Formula: flow_comp(x) = flow_agg(x) + flow_pair(x + flow_agg(x))
        #    We need to sample flow_pair at locations specified by (grid + flow_agg)

        # Calculate sampling coordinates (absolute pixel locations)
        # grid has shape (1, H, W, 2) -> (x, y)
        # agg_flow has shape (1, 2, H, W) -> (dx, dy)
        # Need to swap agg_flow dims to match grid: (1, H, W, 2)
        agg_flow_xy = agg_flow.permute(0, 2, 3, 1)  # (1, H, W, 2)
        vgrid_abs = grid + agg_flow_xy  # (1, H, W, 2) sampling locations

        # Normalize grid for grid_sample (-1 to 1)
        # align_corners=False convention: -1 maps to -0.5, +1 maps to W-0.5
        vgrid_norm = torch.zeros_like(vgrid_abs)
        vgrid_norm[..., 0] = 2.0 * vgrid_abs[..., 0] / max(W - 1, 1) - 1.0
        vgrid_norm[..., 1] = 2.0 * vgrid_abs[..., 1] / max(H - 1, 1) - 1.0

        # Sample the *pairwise_flow* at the warped locations
        # grid_sample expects input (N, C, Hi, Wi) and grid (N, Ho, Wo, 2)
        # pairwise_flow is (1, 2, H, W), vgrid_norm is (1, H, W, 2)
        flow_warped = F.grid_sample(
            input=pairwise_flow,
            grid=vgrid_norm,
            mode="bilinear",  # Smoother than 'nearest'
            padding_mode="border",  # Or 'zeros', 'reflection'
            align_corners=False,  # Match normalization
        )  # Output: (1, 2, H, W)

        # 3. Add the warped pairwise flow to the aggregated flow
        agg_flow = agg_flow + flow_warped

        # 4. Apply Median Filter for smoothing
        # kornia expects (B, C, H, W)
        if median_filter_kernel_size > 1:
            # Ensure odd kernel size
            k_size = (
                median_filter_kernel_size
                if median_filter_kernel_size % 2 != 0
                else median_filter_kernel_size + 1
            )
            smoothed_agg_flow = KFilters.median_blur(agg_flow, (k_size, k_size))
        else:
            smoothed_agg_flow = agg_flow

        # 5. Check flow magnitude conditions
        # Calculate norms on CPU
        with torch.no_grad():  # Ensure numpy conversion doesn't track gradients
            flow_norm = (
                torch.norm(smoothed_agg_flow.squeeze(0), dim=0).flatten().cpu().numpy()
            )

        if flow_norm.size == 0:  # Handle potential empty tensor case
            logger.warning(
                "Flow norm calculation resulted in empty tensor. Skipping checks."
            )
            continue

        flow_90 = np.percentile(flow_norm, 90)
        flow_95 = np.percentile(flow_norm, 95)

        # Check for low motion (early exit)
        if current_tgt_idx == 5 and flow_95 < low_motion_threshold_frame5:
            logger.debug(
                f"Low motion detected at frame 5 (95th perc. = {flow_95:.2f}). Stopping aggregation."
            )
            return None, 0
        # Check for low motion (final check if loop is about to end)
        if current_tgt_idx == T - 2 and flow_95 < low_motion_threshold_final:
            logger.debug(
                f"Low motion detected at final frame (95th perc. = {flow_95:.2f}). Stopping aggregation."
            )
            return None, 0

        # Check if flow magnitude is within the stable range
        if flow_90 <= stable_flow_threshold:
            # Update the best flow found so far
            best_flow = smoothed_agg_flow.clone()  # Store the smoothed version
            final_tgt_idx = current_tgt_idx
        else:
            # Flow magnitude exceeded threshold, stop aggregation
            logger.debug(
                f"Flow magnitude exceeded threshold (90th perc. = {flow_90:.2f} > {stable_flow_threshold}). Stopping aggregation."
            )
            break  # Exit the loop

    end_time = time.time()
    logger.debug(
        f"Aggregation finished. Total time: {end_time - start_time:.2f} seconds. Final target index: {final_tgt_idx}"
    )

    # Return the last flow that was considered stable
    if best_flow is None and final_tgt_idx > 0:
        # This can happen if the loop finished naturally without exceeding the threshold on the last step
        best_flow = smoothed_agg_flow.clone()

    if best_flow is None:
        logger.debug("No stable flow computed that met the criteria.")
        return None, 0
    else:
        best_flow = best_flow.squeeze(0)
        # Ensure output is (2, H, W)
        return best_flow.to(device), final_tgt_idx  # Keep on GPU
