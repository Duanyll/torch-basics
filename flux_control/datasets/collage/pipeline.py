import logging
import random
import os
from typing import Optional

if __name__ == "__main__":
    from rich.logging import RichHandler

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )

import torch
import torchvision
import pickle

from .affine import compute_transform_data_structured, apply_transforms
from .depth import load_depth_model, estimate_depth
from .dexined import load_dexined_model, estimate_edges
from .flow import load_raft_model, compute_aggregated_flow
from .segmentation import load_segmentation_model, generate_masks
from .hf import load_hf_pipeline, encode_latents, encode_prompt
from .video import select_frames, random_crop, encode_color_palette, splat_lost_regions

logger = logging.getLogger(__name__)


def load_all_models(device: str = "cuda"):
    """
    Load all models needed for the pipeline. Models are stored in global variables
    inside their respective modules.
    """
    load_segmentation_model(device)  # HuggingFace
    load_depth_model(device)  # torch hub
    load_dexined_model(device)  # torch hub
    load_raft_model(device)  # torch hub
    load_hf_pipeline(device)  # HuggingFace
    logger.info("All models loaded successfully.")


def process_sample(
    video_path: str,
    prompt: str,
    video_frames: Optional[torch.Tensor] = None,
    device: str = "cuda",
):
    """
    Make a sample from video.
    """

    video_name = os.path.basename(video_path)

    # 1. Load video (CPU)
    if video_frames is None:
        video, _, _ = torchvision.io.read_video(
            video_path, output_format="TCHW", pts_unit="sec"
        )
        video = video.float() / 255.0
    else:
        video = video_frames

    max_attempts = 5
    attempt = 0

    while attempt < max_attempts:
        frames = select_frames(video)
        if frames is None:
            logger.info(f"Video {video_name} has too few frames.")
            return None

        frames = frames.to(device)
        frames = random_crop(frames)

        # 2. Estimate optical flow (GPU)
        flow, target_idx = compute_aggregated_flow(frames, device=device)
        if flow is not None:
            break

    if flow is None:
        logger.info(
            f"Video {video_name} has invalid optical flow after {max_attempts} attempts."
        )
        return None

    src_frame = frames[0]
    dst_frame = frames[target_idx]

    # 3. Estimate depth (GPU)
    depth = estimate_depth(src_frame)

    # 4. Make segmentation mask (CPU + GPU)
    masks = generate_masks(src_frame)

    # 5. Compute affine transforms (CPU)
    transform, dropped_masks = compute_transform_data_structured(flow, depth, masks)

    # 6. Apply affine transforms (GPU)
    warped, grid, warped_regions, warped_alpha = apply_transforms(
        src_frame, depth, transform
    )

    # Chances to fill lost regions with splat
    if random.random() < 0.4:
        logger.debug(f"Video {video_name} splatting lost regions.")
        warped, grid, warped_alpha = splat_lost_regions(
            src_frame, dst_frame, flow, warped, grid, warped_regions, warped_alpha
        )

    # 7. Estimate edges (GPU)
    edges = estimate_edges(src_frame)

    # 8. Extract color palette (GPU)
    palettes, locations = encode_color_palette(src_frame, dropped_masks)

    # 9. Encode prompt (GPU)
    prompt_embeds, pooled_prompt_embeds = encode_prompt(prompt)

    def to_save(x):
        if isinstance(x, torch.Tensor):
            return x.to(device="cpu", dtype=torch.bfloat16)
        return x

    # 10. Encode and save latents (GPU)
    save_dict = {
        "video_name": video_name,
        "src": to_save(encode_latents(src_frame)),
        "clean_latents": to_save(encode_latents(dst_frame)),
        "collage_control_latents": to_save(encode_latents(warped)),
        "collage_grid": to_save(grid),
        "collage_alpha": to_save(warped_alpha),
        "edge_control_latents": to_save(encode_latents(edges)),
        "palettes": to_save(palettes),
        "palette_locations": to_save(locations),
        "prompt_embeds": to_save(prompt_embeds),
        "pooled_prompt_embeds": to_save(pooled_prompt_embeds),
    }

    return save_dict


if __name__ == "__main__":
    import argparse
    from rich.console import Console
    from ...utils.describe import describe

    parser = argparse.ArgumentParser(description="Process video and save latents.")
    parser.add_argument("video_path", type=str, help="Path to the video file.")
    parser.add_argument("prompt_path", type=str, help="Path to the prompt file.")
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (default: cuda)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.pkl",
        help="Output file to save the latents.",
    )

    args = parser.parse_args()
    video_path = args.video_path
    device = args.device

    console = Console()

    try:
        # Load all models
        load_all_models(device)
        with open(args.prompt_path, "r") as f:
            prompt = f.read().strip()
        # Process the video
        result = process_sample(video_path, prompt, device)
    except Exception:
        console.print_exception()
        result = None

    describe(result)
    # Save the result
    if result is not None:
        with open(args.output, "wb") as f:
            pickle.dump(result, f)
        logger.info(f"Processed video saved to {args.output}.")
    else:
        logger.info("Processing failed.")
