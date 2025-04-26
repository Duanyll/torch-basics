import logging
from typing import Tuple, Optional
import torch
from torch.profiler import record_function
from einops import repeat

logger = logging.getLogger(__name__)
seed = 42


def kmeans(
    samples: torch.Tensor, cluster_num: int, cached_center=None, filter_nan: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run kmeans on samples. Result is on the same device as samples. If cached_center is not 
    None, it will be used as the initial cluster center. The returned cluster_centers may
    contain NaN values due to empty clusters. Set filter_nan = True to replace them zero.
    Args:
        samples: (sample_num, feature_dim)
        cluster_num: int
        cached_center: (cluster_num, feature_dim)
    Returns:
        cluster_idx: (sample_num)
        cluster_centers: (cluster_num, feature_dim) 
    """
    try:
        from .libKMCUDA import kmeans_cuda  # type: ignore
    except ImportError as e:
        logger.error("Fail to load kmeans operator from local path.")
        logger.exception(e)
        print(
            "Please use libKMCUDA built from https://github.com/duanyll/kmcuda. The built"
            "libKMCUDA.so file should be placed in the same directory as this file. Do not"
            "use the official libKMCUDA from pip."
        )
        raise e

    if cluster_num <= 1:
        return (
            torch.zeros(samples.shape[0], dtype=torch.long, device=samples.device),
            torch.mean(samples, dim=0, keepdim=True),
        )
    if cluster_num > samples.shape[0]:
        logger.warning(
            f"cluster_num ({cluster_num}) > sample_num ({samples.shape[0]})."
        )
        cluster_num = samples.shape[0]
    with record_function("kmeans"):
        if cached_center is None:
            idx, centers = kmeans_cuda(samples, cluster_num, seed=seed)
        else:
            idx, centers = kmeans_cuda(samples, cluster_num, cached_center, seed=seed)
    if filter_nan:
        centers = torch.nan_to_num(centers, nan=0.0, posinf=0.0, neginf=0.0)
    return idx.long(), centers


def get_cluster_centers_scatter(
    samples: torch.Tensor, cluster_indice: torch.Tensor, cluster_num: int
) -> torch.Tensor:
    """
    Args:
        samples: (batch_size, sample_num, feature_dim)
        cluster_indice: (batch_size, sample_num)
        cluster_num: int
    Returns:
        cluster_centers: (batch_size, cluster_num, feature_dim)
    """

    if samples.dim() == 2:
        samples = samples.unsqueeze(0)
        cluster_indice = cluster_indice.unsqueeze(0)
        has_batch = False
    else:
        has_batch = True

    dev = samples.device
    batch_size = samples.shape[0]
    sample_num = samples.shape[1]
    feature_dim = samples.shape[2]
    # print(cluster_indice.min(), cluster_indice.max())
    cluster_centers = torch.zeros(
        batch_size, cluster_num, feature_dim, device=dev
    ).scatter_add_(
        dim=1, index=repeat(cluster_indice, "b p -> b p s", s=feature_dim), src=samples
    )
    cluster_size = (
        torch.zeros(batch_size, cluster_num, device=dev)
        .scatter_add_(
            dim=1,
            index=cluster_indice,
            src=torch.ones(batch_size, sample_num, device=dev),
        )
        .unsqueeze_(dim=2)
    )
    cluster_size[cluster_size < 1] = 1
    cluster_centers /= cluster_size

    if not has_batch:
        cluster_centers = cluster_centers.squeeze(0)

    return cluster_centers
