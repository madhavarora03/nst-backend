from pathlib import Path

import numpy as np
import torch
from PIL import Image
from PIL.ImageFile import ImageFile


def load_image(filename: Path,
               size: int = None,
               scale: float = None) -> ImageFile:
    """
    Loads image from given filename, resizes it to given size and scales it to given scale.

    Args:
        filename (Path): Path to image
        size (int, optional): Size of image. Defaults to None.
        scale (int, optional): Scale of image. Defaults to None.

    Returns:
        ImageFile: Loaded image
    """
    img = Image.open(filename).convert('RGB')

    if size is not None:
        size = make_divisible(size)
        img = img.resize((size, size), Image.Resampling.LANCZOS)

    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.Resampling.LANCZOS)

    if size is None:
        orig_w, orig_h = img.size
        new_w, new_h = make_divisible(orig_w), make_divisible(orig_h)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    return img


def save_image(filename: Path,
               data: torch.Tensor) -> None:
    """
    Saves image to given filename

    Args:
        filename (Path): Path to image
        data (torch.Tensor): Image data

    Returns:
        None
    """
    img = data.clone().detach().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)


def make_divisible(value, divisor=32):
    """Ensure height and width are divisible by 32 (required for YOLO)."""
    return int(np.ceil(value / divisor) * divisor)
