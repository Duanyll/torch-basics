import logging
import random
import os
from typing import Optional

from flux_control.datasets.collage.config import CollageConfig

if __name__ == "__main__":
    from rich.logging import RichHandler

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )

import torch
import pickle

from .affine import compute_transform_data_structured, apply_transforms
from .config import CollageConfig
from .depth import load_depth_model, estimate_depth
from .dexined import load_dexined_model, estimate_edges
from .flow import load_raft_model
from .segmentation import load_segmentation_model, generate_masks
from .hf import load_hf_pipeline, encode_latents, encode_prompt
from .palette import encode_color_palette, extract_palette_from_masked_image
from .video import load_video, splat_lost_regions, try_extract_frame
from .warp import forward_warp

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
    video: Optional[torch.Tensor] = None,
    cfg: CollageConfig = CollageConfig(),
    device: str = "cuda",
):
    """
    Make a sample from video.
    """

    video_name = os.path.basename(video_path)

    extract_result = try_extract_frame(
        load_video(video_path) if video is None else video, device=device, cfg=cfg
    )
    if extract_result is None:
        return None
    del video

    flow, src_frame, dst_frame = extract_result

    # 3. Estimate depth (GPU)
    depth = estimate_depth(src_frame)

    # 4. Make segmentation mask (CPU + GPU)
    masks = generate_masks(src_frame)

    # 5. Compute affine transforms (CPU)
    transform, dropped_masks = compute_transform_data_structured(
        flow, depth, masks, cfg=cfg
    )

    if random.random() < cfg.chance_pre_splat:
        warped, grid, warped_alpha = forward_warp(
            src_frame, dst_frame, flow
        )
        true_alpha_area = torch.mean(warped_alpha).item()
        did_splat = True
    else:
        warped, grid, warped_regions, warped_alpha, true_alpha_area = apply_transforms(
            src_frame, depth, transform, cfg=cfg
        )
        if random.random() < cfg.chance_post_splat:
            warped, grid, warped_alpha = splat_lost_regions(
                src_frame, dst_frame, flow, warped, grid, warped_regions, warped_alpha
            )
            did_splat = True
        else:
            did_splat = False
            
    if true_alpha_area > cfg.true_alpha_threshold:
        return None

    if did_splat:
        palettes, locations = extract_palette_from_masked_image(
            dst_frame, 1 - warped_alpha, max_colors=cfg.num_palette_fallback, cfg=cfg
        )
    else:
        palettes, locations = encode_color_palette(src_frame, dropped_masks, cfg=cfg)

    # 7. Estimate edges (GPU)
    edges = estimate_edges(dst_frame)
    if random.random() < cfg.chance_mask_edges:
        edges = edges * (1 - warped_alpha)

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
        result = process_sample(video_path, prompt, device=device)
    except Exception:
        console.print_exception()
        result = None

    describe(result, max_items=20)
    # Save the result
    if result is not None:
        with open(args.output, "wb") as f:
            pickle.dump(result, f)
        logger.info(f"Processed video saved to {args.output}.")
    else:
        logger.info("Processing failed.")
