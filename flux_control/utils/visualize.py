import cv2
import torch
from PIL import Image
from einops import rearrange, repeat
import numpy as np
import matplotlib.pyplot as plt

from .common import meshgrid_to_ij, make_grid

def visualize_flow(flow):
    """
    Visualize optical flow using OpenCV's color map.
    Args:
        flow (torch.Tensor): Optical flow tensor of shape (1, 2, H, W) or (2, H, W).
    Returns:
        PIL.Image: Visualized optical flow image.
    """
    if len(flow.shape) == 4:
        flow = rearrange(flow, "1 c h w -> c h w")
    flow = flow.cpu().numpy()
    flow = rearrange(flow, "c h w -> h w c")
    # Use Hue, Saturation, Value colour model
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 2] = 255

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang / np.pi / 2 * 180
    hsv[..., 1] = np.clip(mag * 255 / 24, 0, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return Image.fromarray(bgr)


def visualize_image(image):
    """
    Visualize an image tensor.
    Args:
        image (torch.Tensor): Image tensor of shape (1, C, H, W) or (C, H, W).
    Returns:
        PIL.Image: Visualized image.
    """
    if len(image.shape) == 4:
        image = rearrange(image, "1 c h w -> c h w")
    image = image.cpu().numpy()
    image = rearrange(image, "c h w -> h w c")
    return Image.fromarray((image * 255).astype(np.uint8))


def visualize_grayscale(image):
    """
    Visualize a grayscale image tensor.
    Args:
        image (torch.Tensor): Grayscale image tensor of shape (1, 1, H, W), (1, H, W) or (H, W).
    Returns:
        PIL.Image: Visualized grayscale image.
    """
    image = image.squeeze().cpu().numpy()  # [h, w]
    image = repeat(image, "h w -> h w c", c=3)
    image = (image * 255).astype(np.uint8)
    return Image.fromarray(image)


def visualize_grid(grid, mask=None, scale=100):
    """
    Visualize a grid tensor.
    Args:
        grid (torch.Tensor): Grid tensor of shape (2, H, W).
    Returns:
        PIL.Image: Visualized grid image.
    """
    _, h, w = grid.shape
    std_grid = make_grid(h, w, device=grid.device)
    delta = (grid - std_grid) * scale
    if mask is not None:
        delta = delta * mask
    return visualize_flow(delta)


def show_grayscale_colorbar(image, colormap="viridis", interpolation="bicubic"):
    """
    Show grayscale image with colorbar.
    Args:
        image (torch.Tensor): Grayscale image tensor of shape (1, 1, H, W), (1, H, W) or (H, W).
        colormap (str): Colormap to use for visualization.
    """
    image = image.squeeze().cpu().numpy()  # [h, w]
    plt.imshow(image, cmap=colormap, interpolation=interpolation)
    plt.colorbar()
    plt.axis("off")
    plt.show()


def show_palette_with_locations(image, palettes, locations):
    """
    Show image with palettes and locations.
    Args:
        image (torch.Tensor): Image tensor of shape (C, H, W).
        palettes (torch.Tensor): Palette tensor of shape (N, 3).
        locations (torch.Tensor): Locations tensor of shape (N, 2).
    """
    c, h, w = image.shape
    image = rearrange(image, "c h w -> h w c")
    image = image.cpu().numpy()
    plt.imshow(image)
    locations_ij = meshgrid_to_ij(locations, h, w)
    palettes = palettes.cpu().numpy()
    locations_ij = locations_ij.cpu().numpy()
    for i in range(palettes.shape[0]):
        palette = palettes[i]
        loc = locations_ij[i]
        plt.scatter(
            loc[1], loc[0], color=palette, s=100, marker="o", edgecolors="black"
        )
    plt.axis("off")
    plt.show()


def show_image_histogram(image, log_scale=False, show_cdf=False):
    """
    Display histogram for an RGB or grayscale image with optional CDF curve.
    Args:
        image: Tensor or array, shape (1,C,H,W), (C,H,W), (1,1,H,W), (1,H,W), or (H,W).
               Can be on GPU or CPU.
        log_scale: bool, if True, use logarithmic scale for y-axis of histogram.
        show_cdf: bool, if True, overlay cumulative distribution function (linear scale).
    """
    # Convert to numpy and handle GPU
    if isinstance(image, torch.Tensor):
        image = image.cpu().detach().numpy()

    # Remove singleton dimensions
    image = np.squeeze(image)

    # Create figure with two y-axes if CDF is requested
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Handle different image formats
    if len(image.shape) == 3:  # RGB image (C,H,W)
        channels = ["Red", "Green", "Blue"]
        for i in range(3):
            # Compute histogram
            hist, bins = np.histogram(
                image[i].ravel(), bins=256, range=(0, 1), density=True
            )
            bin_centers = (bins[:-1] + bins[1:]) / 2

            # Plot histogram
            ax1.hist(
                image[i].ravel(),
                bins=256,
                color=["r", "g", "b"][i],
                alpha=0.5,
                label=channels[i],
                range=(0, 1),
                density=True,
            )

            if show_cdf:
                # Compute and plot CDF
                cdf = np.cumsum(hist) * np.diff(bins)
                ax2 = ax1.twinx()
                ax2.plot(
                    bin_centers,
                    cdf,
                    color=["r", "g", "b"][i],
                    linestyle="--",
                    label=f"{channels[i]} CDF",
                )
                ax2.set_ylabel("CDF (Linear Scale)")
                ax2.set_ylim(0, 1)

        ax1.set_title("RGB Histogram" + (" with CDF" if show_cdf else ""))
        ax1.set_xlabel("Pixel Intensity")
        ax1.set_ylabel("Density (Log Scale)" if log_scale else "Density")
        ax1.legend(loc="upper left")
        if show_cdf:
            ax2.legend(loc="upper right")

    else:  # Grayscale image (H,W)
        # Compute histogram
        hist, bins = np.histogram(image.ravel(), bins=256, range=(0, 1), density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Plot histogram
        ax1.hist(
            image.ravel(), bins=256, color="gray", alpha=0.7, range=(0, 1), density=True
        )

        if show_cdf:
            # Compute and plot CDF
            cdf = np.cumsum(hist) * np.diff(bins)
            ax2 = ax1.twinx()
            ax2.plot(bin_centers, cdf, color="black", linestyle="--", label="CDF")
            ax2.set_ylabel("CDF (Linear Scale)")
            ax2.set_ylim(0, 1)
            ax2.legend(loc="upper right")

        ax1.set_title("Grayscale Histogram" + (" with CDF" if show_cdf else ""))
        ax1.set_xlabel("Pixel Intensity")
        ax1.set_ylabel("Density (Log Scale)" if log_scale else "Density")

    if log_scale:
        ax1.set_yscale("log")

    ax1.grid(True, alpha=0.3)
    plt.show()
    

def show_color_palette(palette: torch.Tensor, show_hex: bool = True, figsize=(8, 2)):
    """
    在 Jupyter 中显示一个色卡。

    Args:
        palette (Tensor): (N, 3) RGB float tensor in [0, 1]
        show_hex (bool): 是否显示颜色值（Hex格式）
        figsize (tuple): 图像尺寸
    """

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    assert palette.ndim == 2 and palette.shape[1] == 3

    n_colors = palette.shape[0]
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.set_xlim(0, n_colors)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for i, color in enumerate(palette):
        rgb = color.cpu().numpy() if isinstance(color, torch.Tensor) else color
        hex_code = "#%02x%02x%02x" % tuple((rgb * 255).astype(int))
        rect = patches.Rectangle((i, 0), 1, 1, color=rgb)
        ax.add_patch(rect)
        if show_hex:
            ax.text(
                i + 0.5,
                0.5,
                hex_code,
                ha="center",
                va="center",
                fontsize=12,
                color="white" if color.mean() < 0.5 else "black",
            )

    plt.show()
