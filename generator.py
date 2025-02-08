import warnings

from torch import nn
from torch.optim import Adam
from torchvision import models, transforms
from torchvision.utils import save_image
from pathlib import Path

from utils import *

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


# Create model instance and set to eval mode
model = VGG19().to(device).eval()

### Uncomment to see model architecture in terminal ###
# summary(model, (32, 3, 224, 224))

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load content and style images
content_path = Path("sample/content.jpg")
style_path = Path("sample/style.jpg")

content_image = load_image(content_path, transform)
style_image = load_image(style_path, transform)
generated_image = content_image.clone().requires_grad_(True)

# Set hyperparameters
EPOCHS = 12000
LR = 1e-3
alpha = 1
beta = 0.01

# Create Optimizer
optimizer = Adam(params=[generated_image], lr=LR, weight_decay=beta)

for epoch in range(EPOCHS):
    # Obtain the convolution features in specifically chosen layers
    generated_features = model(generated_image)
    original_img_features = model(content_image)
    style_features = model(style_image)

    # Initializing Loss
    style_loss = original_loss = 0

    # Iterating through all the features for the chosen layers
    for gen_feature, orig_feature, style_feature in zip(
            generated_features, original_img_features, style_features
    ):
        # batch_size = 1
        batch_size, channel, height, width = gen_feature.shape
        original_loss += torch.mean((gen_feature - orig_feature) ** 2)

        # Compute Gram Matrix of generated
        G = gen_feature.view(channel, height * width).mm(
            gen_feature.view(channel, height * width).t()
        )

        # Compute Gram Matrix of Style
        A = style_feature.view(channel, height * width).mm(
            style_feature.view(channel, height * width).t()
        )
        style_loss += torch.mean((G - A) ** 2)

    # Calculate weighted loss
    total_loss = alpha * original_loss + beta * style_loss

    # Back propagation
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if epoch % 200 == 0 or epoch == EPOCHS - 1:
        print(f"Total loss after {epoch + 1} epochs: {total_loss}")
        name = f"output/{str(epoch)}_generated.png"
        save_image(generated_image, name)

print(f"")