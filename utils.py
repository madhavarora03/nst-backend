import torch
import torchvision
from PIL import Image
from pathlib import Path


def load_image(image_path: Path,
               transform: torchvision.transforms,
               device: torch.device = torch.device("cuda")
               ) -> torch.Tensor:
    """
    Loads image from disk and converts it to a tensor
    :param image_path: str
    :param transform: torchvision.transforms
    :param device: torch.device
    :return: torch.Tensor
    """
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)

    return image.to(device)
