import logging
import torch

from ...utils.common import pack_bool_tensor

from .affine import compute_transform_data_structured, apply_transforms
from .config import CollageConfig
from .depth import load_depth_model, estimate_depth
from .flow import load_raft_model
from .segmentation import load_segmentation_model, generate_masks
from .hf import load_hf_pipeline, encode_latents, encode_prompt
from .rmbg import load_rmbg_model, estimate_foreground
from .video import try_extract_frame, make_confidence_hint
from .warp import forward_warp

logger = logging.getLogger(__name__)


def load_all_models(device: str = "cuda"):
    """
    Load all models needed for the pipeline. Models are stored in global variables
    inside their respective modules.
    """
    load_segmentation_model(device)  # HuggingFace
    load_depth_model(device)  # torch hub
    load_raft_model(device)  # torch hub
    load_rmbg_model(device)  # torch hub
    load_hf_pipeline(device)  # HuggingFace
    logger.info("All models loaded successfully.")


def process_sample(video, prompt, device="cuda", cfg=CollageConfig()):
    extract_result = try_extract_frame(video, device=device, cfg=cfg)
    if extract_result is None:
        return None
    flow, src, tgt = extract_result
    splat, grid_splat, mask_splat = forward_warp(src, tgt, flow)
    masks = generate_masks(src)
    depth = estimate_depth(src)
    selected_masks, dropped_masks = compute_transform_data_structured(
        flow, depth, masks, cfg=cfg
    )
    affine, grid_affine, mask_affine_src, mask_affine_tgt = apply_transforms(
        src, depth, selected_masks, cfg=cfg
    )

    foreground = estimate_foreground(src)
    confidence, hint = make_confidence_hint(
        grid_affine, grid_splat, mask_affine_tgt, mask_splat, tgt, foreground, cfg
    )
    prompt_embeds, pooled_prompt_embeds = encode_prompt(prompt)
    save_image = lambda x: encode_latents(x).to(torch.bfloat16).cpu()
    save_mask = lambda x: pack_bool_tensor(x.cpu())

    save_data = {
        "src": save_image(src),
        "tgt": save_image(tgt),
        "splat": save_image(splat),
        "affine": save_image(affine),
        "hint": save_image(hint),
        "mask_splat": save_mask(mask_splat),
        "mask_affine": save_mask(mask_affine_src),
        "foreground": save_mask(foreground),
        "confidence": confidence.to(torch.bfloat16).cpu(),
        "prompt_embeds": prompt_embeds.cpu(),
        "pooled_prompt_embeds": pooled_prompt_embeds.cpu(),
    }

    return save_data


if __name__ == "__main__":
    import argparse
    import pickle
    from ...utils.describe import describe
    from .video import load_video

    
    parser = argparse.ArgumentParser(description="Process a video with the collage pipeline.")
    parser.add_argument("video_path", type=str, help="Path to the input video.")
    parser.add_argument("output_path", type=str, help="Path to save the output data.")
    parser.add_argument("--prompt", type=str, default="A beautiful landscape", help="Prompt for the model.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (e.g., 'cuda' or 'cpu').")
    parser.add_argument("--cfg", type=str, default=None, help="Path to the configuration file.")
    
    args = parser.parse_args()
    
    load_all_models(args.device)
    logger.info(f"All models loaded successfully.")
    video = load_video(args.video_path)
    prompt = args.prompt
    cfg = CollageConfig.from_toml(args.cfg) if args.cfg else CollageConfig()
    save_data = process_sample(video, prompt, device=args.device, cfg=cfg)
    
    describe(save_data, max_items=20)
    with open(args.output_path, "wb") as f:
        pickle.dump(save_data, f)
        size = f.tell()
    logger.info(f"Processed data size: {size / (1024 * 1024):.2f} MB")
    logger.info(f"Processed data saved to {args.output_path}.")