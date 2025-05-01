from typing import Tuple
import torch
import kornia
from einops import rearrange
from collections.abc import MutableMapping

def flatten_dict(dictionary, parent_key='', separator='_'):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten_dict(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)

def deep_merge_dicts(dict1, dict2):
    """
    Recursively merge two dictionaries.
    """
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def meshgrid_to_ij(grid: torch.Tensor, h: int, w: int):
    """
    Convert a meshgrid for F.grid_sample to the subscript index [i, j].

    Args:
        grid: (N, 2) tensor with the meshgrid coordinates, ranging from -1 to 1.
        h: height of the image.
        w: width of the image.
    Returns:
        ij: (N, 2) tensor with the subscript indices.
    """
    # Ensure grid is of shape (N, 2)
    assert grid.shape[1] == 2, f"Expected grid shape (N, 2), got {grid.shape}"

    # Extract x and y coordinates
    x = grid[:, 0]  # (N,)
    y = grid[:, 1]  # (N,)

    # Convert from [-1, 1] to [0, w-1] for x and [0, h-1] for y
    i = ((y + 1) / 2) * (h - 1)  # (N,)
    j = ((x + 1) / 2) * (w - 1)  # (N,)

    # Stack to form (N, 2) tensor [i, j]
    ij = torch.stack([i, j], dim=1)

    return ij


def make_grid(h, w, device: torch.device = torch.device("cuda")):
    grid_array = kornia.utils.create_meshgrid(
        h, w, normalized_coordinates=True, device=device
    )
    grid = rearrange(grid_array, "1 h w c -> c h w")  # Shape: 2 H W
    return grid


def pack_bool_tensor(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Size]:
    if torch.is_floating_point(tensor):
        tensor = tensor > 0.5
    tensor = tensor.to(torch.bool)  # 转换为bool类型
    
    flat = tensor.flatten()
    pad_len = (8 - flat.numel() % 8) % 8  # 需要填充的位数，使其能整除8
    if pad_len > 0:
        flat = torch.cat([flat, torch.zeros(pad_len, dtype=torch.bool)])

    flat = flat.view(-1, 8)  # 每组8个bool
    packed = torch.sum(flat * (1 << torch.arange(8, dtype=torch.uint8)), dim=1)
    return packed, tensor.shape  # 返回原始形状用于解码


def unpack_bool_tensor(
    packed: torch.Tensor, original_shape: torch.Size
) -> torch.Tensor:
    unpacked = ((packed.unsqueeze(1) >> torch.arange(8)) & 1).to(torch.bool)
    unpacked = unpacked.view(-1)[: torch.tensor(original_shape).prod()]
    return unpacked.view(original_shape)
