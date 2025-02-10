import re
import warnings
from pathlib import Path
from typing import Literal
from uuid import uuid4

import torch
from torchvision import transforms

from TransformerNet import TransformerNet
from utils import load_image, save_image

warnings.filterwarnings("ignore")

StyleTypes = Literal["mosaic", "candy", "rain_princess", "udnie"]


def generate(content_path: Path,
             style: StyleTypes = "mosaic") -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load content image
    content_image = load_image(content_path)

    # Convert image to tensor
    content_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Apply transforms to image
    content_image = content_transform(content_image).unsqueeze(0).to(device)
    _, _, w, h = content_image.shape

    # Load and apply style transfer model
    model = TransformerNet()
    state_dict = torch.load(f"checkpoints/{style}.pth", map_location=device)

    for k in list(state_dict.keys()):
        if re.search(r'in\d+\.running_(mean|var)$', k):
            del state_dict[k]

    model.load_state_dict(state_dict)
    model.to(device).eval()

    output = model(content_image).cpu().detach().squeeze(0)
    output_path = Path(f"output/output_{style}_{uuid4()}.png")

    save_image(output_path, output)
    return output_path

