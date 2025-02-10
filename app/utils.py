from typing import List

import torch
from PIL import Image


def load_image(filename, size=None, scale=None):
    img = Image.open(filename).convert('RGB')
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img


def save_image(filename, data):
    img = data.clone().detach().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)


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
