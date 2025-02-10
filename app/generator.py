import re
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms
from ultralytics import YOLO

from utils import load_image

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_seg = YOLO("checkpoints/yolo/yolov8s-seg.pt")


class TransformerNet(torch.nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
        # Residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
        # Non-linearities
        self.relu = torch.nn.ReLU()

    def forward(self, X):
        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return y


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out


# Create model instance and set to eval mode
model = TransformerNet()
state_dict = torch.load("checkpoints/mosaic.pth")
for k in list(state_dict.keys()):
    if re.search(r'in\d+\.running_(mean|var)$', k):
        del state_dict[k]

model.load_state_dict(state_dict)
model.to(device).eval()

content_image = load_image(Path("sample/content.jpg"))
content_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))
])
content_image = content_transform(content_image)
content_image = content_image.unsqueeze(0).to(device)

output = model(content_image).cpu()
# save_image("output/output.png", output[0])


### Uncomment to see model architecture in terminal ###
# summary(model, (32, 3, 224, 224))
cap = cv2.VideoCapture(0)  # Open webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run YOLOv8-seg for human segmentation
    results = model_seg(frame_rgb)

    # Create an empty mask (assume background initially)
    full_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

    for result in results:
        masks = result.masks.data.cpu().numpy()  # Segmentation masks
        cls = result.boxes.cls.cpu().numpy()  # Class labels

        for i, c in enumerate(cls):
            if int(c) == 0:  # Class 0 = Person
                person_mask = masks[i] * 255  # Convert to uint8
                person_mask = cv2.resize(person_mask, (frame.shape[1], frame.shape[0]))

                # Merge all detected persons into one mask
                full_mask = np.maximum(full_mask, person_mask.astype(np.uint8))

    # **Ensure binary mask**
    _, binary_mask = cv2.threshold(full_mask, 127, 255, cv2.THRESH_BINARY)

    # **Invert the mask to select the background**
    background_mask = cv2.bitwise_not(binary_mask)

    # **Extract the background**
    background = cv2.bitwise_and(frame_rgb, frame_rgb, mask=background_mask)

    # **Apply style transfer to the extracted background**
    background_tensor = content_transform(background).unsqueeze(0).to(device)

    with torch.no_grad():
        stylized_background = model(background_tensor).cpu().squeeze()

    # Convert back to image format
    stylized_background = stylized_background.permute(1, 2, 0).numpy()
    stylized_background = np.clip(stylized_background, 0, 255).astype(np.uint8)
    stylized_background_bgr = cv2.cvtColor(stylized_background, cv2.COLOR_RGB2BGR)

    # **Merge stylized background back into the frame**
    final_frame = np.where(background_mask[..., None] > 0, stylized_background_bgr, frame)

    cv2.imshow("Stylized Background", final_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
