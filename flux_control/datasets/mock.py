from typing import Optional, Tuple, Any

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
