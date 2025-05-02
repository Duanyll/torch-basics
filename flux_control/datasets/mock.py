from typing import Optional, Tuple, Any
import random
import torch

LATENT_CHANNELS = 16
POOLED_PROMPT_EMBED_DIM = 768
PROMPT_EMBED_LEN = 512
PROMPT_EMBED_DIM = 4096


class MockDataset(torch.utils.data.Dataset):
    def __init__(
        self, num_samples: int = 1024, pixel_resolution: Tuple[int, int] = (1024, 1024)
    ):
        self.num_samples = num_samples
        self.height, self.width = pixel_resolution

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, Any]:
        latent_h = self.height // 8
        latent_w = self.width // 8

        return {
            "clean_latents": torch.randn(
                (LATENT_CHANNELS, latent_h, latent_w), dtype=torch.bfloat16
            ),
            "control_latents": torch.randn(
                (LATENT_CHANNELS, latent_h, latent_w), dtype=torch.bfloat16
            ),
            "pooled_prompt_embeds": torch.randn(
                (POOLED_PROMPT_EMBED_DIM,), dtype=torch.bfloat16
            ),
            "prompt_embeds": torch.randn(
                (PROMPT_EMBED_LEN, PROMPT_EMBED_DIM), dtype=torch.bfloat16
            ),
        }


class MockCollageDataset(torch.utils.data.Dataset):
    def __init__(
        self, num_samples: int = 1024, pixel_resolution: Tuple[int, int] = (1152, 896)
    ):
        self.num_samples = num_samples
        self.height, self.width = pixel_resolution

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx: int) -> dict[str, Any]:
        latent_h = self.height // 8
        latent_w = self.width // 8
        
        def n(*shape):
            return torch.randn(shape, dtype=torch.bfloat16)
        
        def r(*shape):
            return torch.rand(shape, dtype=torch.bfloat16)
        
        palette_size = random.randint(1, 8)
        
        return {
            "video_name": "video.mp4",
            "clean_latents": n(LATENT_CHANNELS, latent_h, latent_w),
            "collage_control_latents": n(LATENT_CHANNELS, latent_h, latent_w),
            "collage_alpha": r(self.height, self.width),
            "edge_control_latents": n(LATENT_CHANNELS, latent_h, latent_w),
            "palettes": r(palette_size, 3),
            "palette_locations": n(palette_size, 2),
            "prompt_embeds": n(PROMPT_EMBED_LEN, PROMPT_EMBED_DIM),
            "pooled_prompt_embeds": n(POOLED_PROMPT_EMBED_DIM),
        }
        

class MockCollageDatasetV2(torch.utils.data.Dataset):
    def __init__(
        self, num_samples: int = 1024, pixel_resolution: Tuple[int, int] = (1152, 896)
    ):
        self.num_samples = num_samples
        self.height, self.width = pixel_resolution

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx: int) -> dict[str, Any]:
        latent_h = self.height // 8
        latent_w = self.width // 8
        
        def n(*shape):
            return torch.randn(shape, dtype=torch.bfloat16)
        
        def r(*shape):
            return torch.rand(shape, dtype=torch.bfloat16)
        
        return {
            "src": n(LATENT_CHANNELS, latent_h, latent_w),
            "tgt": n(LATENT_CHANNELS, latent_h, latent_w),
            "splat": n(LATENT_CHANNELS, latent_h, latent_w),
            "affine": n(LATENT_CHANNELS, latent_h, latent_w),
            "hint": n(LATENT_CHANNELS, latent_h, latent_w),
            "mask_splat": r(self.height, self.width),
            "mask_affine": r(self.height, self.width),
            "foreground": r(self.height, self.width),
            "confidence": r(self.height // 16, self.width // 16),
            "prompt_embeds": n(PROMPT_EMBED_LEN, PROMPT_EMBED_DIM),
            "pooled_prompt_embeds": n(POOLED_PROMPT_EMBED_DIM),
        }