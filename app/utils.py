from pathlib import Path
from typing import List

import torch
import torchvision
from PIL import Image


def load_image(image_path: Path,
               transform: torchvision.transforms,
               device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
               ) -> torch.Tensor:
    """
    Loads an image from disk and converts it to a tensor.

    Args:
        image_path (str): Path to the image file.
        transform (torchvision.transforms): Transformations to apply to the image.
        device (torch.device): The device to load the tensor onto (e.g., CPU or GPU).

    Returns:
        torch.Tensor: The image as a tensor after applying the transformation.
    """
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)

    return image.to(device)


def compute_loss(generated_features: List[torch.Tensor],
                 original_img_features: List[torch.Tensor],
                 style_features: List[torch.Tensor],
                 alpha: float,
                 beta: float) -> torch.Tensor:
    """
    Compute the total loss for the current iteration,
    which includes content loss and style loss.

    Args:
        generated_features (List[torch.Tensor]): List of features from the generated image.
        original_img_features (List[torch.Tensor]): List of features from the content image.
        style_features (List[torch.Tensor]): List of features from the style image.
        alpha (float): The weight of the content loss.
        beta (float): The weight of the style loss.

    Returns:
        torch.Tensor: The total loss (content + style).
    """
    original_loss = 0
    style_loss = 0

    for gen_feature, orig_feature, style_feature in zip(
            generated_features, original_img_features, style_features
    ):
        # Content Loss: L2 loss between generated and original image features
        original_loss += torch.mean((gen_feature - orig_feature) ** 2)

        # Style Loss: L2 loss between the Gram matrices of generated and style features
        G = gram_matrix(gen_feature)
        A = gram_matrix(style_feature)
        style_loss += torch.mean((G - A) ** 2)

    total_loss = alpha * original_loss + beta * style_loss
    return total_loss


def gram_matrix(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the Gram matrix of a tensor, used for style loss.

    Args:
        tensor (torch.Tensor): Feature tensor from a layer of the model.

    Returns:
        torch.Tensor: The Gram matrix of the input tensor.
    """
    batch_size, channel, height, width = tensor.shape
    flattened = tensor.view(channel, height * width)
    return flattened.mm(flattened.t())
