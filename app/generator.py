import warnings
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torchvision import models, transforms
from torchvision.utils import save_image

from utils import compute_loss, load_image

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set hyperparameters
EPOCHS = 12000
LR = 1e-3
alpha = 1
beta = 0.01


class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()

        self.chosen_features = ["0", "5", "10", "19", "28"]
        self.model = models.vgg19(pretrained=True).features[:29]

    def forward(self, x):
        features = []

        for layer_num, layer in enumerate(self.model):
            x = layer(x)

            if str(layer_num) in self.chosen_features:
                features.append(x)

        return features


def generate(content_path: Path,
             style_path: Path,
             output_folder: Path) -> None:
    """
    Performs neural style transfer by generating an image that combines
    the content of the `content_image` and the style of the `style_image`
    using a pre-trained VGG19 model.

    Args:
        content_path (Path): Path to the content image. This image will
                              provide the content for the generated image.
        style_path (Path): Path to the style image. This image will provide
                            the style for the generated image.
        output_folder (Path): Path to the folder where the generated images
                               will be saved at specified intervals.

    Returns:
        None: The function saves the generated images to the output folder
              during the training process at regular intervals.
    """
    # Create model instance and set to eval mode
    model = VGG19().to(device).eval()

    ### Uncomment to see model architecture in terminal ###
    # summary(model, (32, 3, 224, 224))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Load content and style images
    content_image = load_image(content_path, transform, device=device)
    style_image = load_image(style_path, transform, device=device)

    # Initialize generated image
    generated_image = content_image.clone().requires_grad_(True)

    # Create Optimizer
    optimizer = Adam(params=[generated_image], lr=LR, weight_decay=beta)

    for epoch in range(EPOCHS):
        # Obtain the convolution features in specifically chosen layers
        generated_features = model(generated_image)
        original_img_features = model(content_image)
        style_features = model(style_image)

        # Calculating total loss
        total_loss = compute_loss(generated_features, original_img_features, style_features, alpha=alpha, beta=beta)

        # Back propagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Save the generated image at specified intervals
        if epoch % 200 == 0 or epoch == EPOCHS - 1:
            print(f"Total loss after {epoch + 1} epochs: {total_loss}")
            name = f"{output_folder}/{str(epoch)}_generated.png"
            save_image(generated_image, name)

# Example Usage
content_image_path = Path("sample/content.jpg")
style_image_path = Path("sample/style.jpg")
output_folder_path = Path("output")

generate(content_image_path, style_image_path, output_folder_path)