import logging
from einops import rearrange
import torch
import torch.nn.functional as F

from ...utils.common import (
    pack_bool_tensor,
    find_closest_resolution,
    crop_and_resize_image,
)

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


def process_video_sample(video, prompt, device="cuda", cfg=CollageConfig()):
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


def process_image_sample(
    coarse,
    mask_coarse,
    foreground,
    prompt,
    src=None,
    hint=None,
    confidence=None,
    confidence_fg=0.0,
    confidence_bg=0.1,
    cfg=CollageConfig(),
):
    _, orig_h, orig_w = coarse.shape
    size = find_closest_resolution(
        orig_h, orig_w, cfg.resolutions_1080p + cfg.resolutions_720p
    )
    h, w = size
    coarse = crop_and_resize_image(coarse, size)
    mask_coarse = crop_and_resize_image(mask_coarse, size)
    foreground = crop_and_resize_image(foreground, size)
    if src is not None:
        src = crop_and_resize_image(src, size)
    if hint is not None:
        hint = crop_and_resize_image(hint, size)
    if confidence is not None:
        conf_h, conf_w = h // 16, w // 16
        if conf_h != confidence.shape[1] or conf_w != confidence.shape[2]:
            logger.warning(
                f"Confidence map size {confidence.shape[1:]} does not match expected size {conf_h, conf_w}. Resizing."
            )
            confidence = torch.nn.functional.interpolate(
                confidence.unsqueeze(0),
                size=(conf_h, conf_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
    else:
        mask_fg = mask_coarse * foreground
        mask_bg = mask_coarse * (1 - foreground)
        confidence = F.avg_pool2d(
            rearrange(
                (1 - mask_coarse) + confidence_fg * mask_fg + confidence_bg * mask_bg,
                "h w -> 1 1 h w",
            ),
            kernel_size=16,
            stride=16,
        ).squeeze()

    prompt_embeds, pooled_prompt_embeds = encode_prompt(prompt)

    save_image = lambda x: encode_latents(x).to(torch.bfloat16).cpu()
    save_mask = lambda x: pack_bool_tensor(x.cpu())
    save_data = {
        "coarse": save_image(coarse),
        "mask_coarse": save_mask(mask_coarse),
        "foreground": save_mask(foreground),
        "confidence": confidence.to(torch.bfloat16).cpu(),
        "prompt_embeds": prompt_embeds.cpu(),
        "pooled_prompt_embeds": pooled_prompt_embeds.cpu(),
    }
    if src is not None:
        save_data["src"] = save_image(src)
    if hint is not None:
        save_data["hint"] = save_image(hint)

    return save_data


if __name__ == "__main__":
    import argparse
    import pickle
    from ...utils.describe import describe
    from .video import load_video

    parser = argparse.ArgumentParser(
        description="Process a video with the collage pipeline."
    )
    parser.add_argument("video_path", type=str, help="Path to the input video.")
    parser.add_argument("output_path", type=str, help="Path to save the output data.")
    parser.add_argument(
        "--prompt",
        type=str,
        default="A beautiful landscape",
        help="Prompt for the model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (e.g., 'cuda' or 'cpu').",
    )
    parser.add_argument(
        "--cfg", type=str, default=None, help="Path to the configuration file."
    )

    args = parser.parse_args()

    load_all_models(args.device)
    logger.info(f"All models loaded successfully.")
    video = load_video(args.video_path)
    prompt = args.prompt
    cfg = CollageConfig.from_toml(args.cfg) if args.cfg else CollageConfig()
    save_data = process_video_sample(video, prompt, device=args.device, cfg=cfg)

    describe(save_data, max_items=20)
    with open(args.output_path, "wb") as f:
        pickle.dump(save_data, f)
        size = f.tell()
    logger.info(f"Processed data size: {size / (1024 * 1024):.2f} MB")
    logger.info(f"Processed data saved to {args.output_path}.")
