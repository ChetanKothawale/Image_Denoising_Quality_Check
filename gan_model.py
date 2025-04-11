import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the Generator model (U-Net style)
class Generator(nn.Module):
    def __init__(self, img_size=512):
        super(Generator, self).__init__()
        self.enc1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)
        self.enc3 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)
        self.dec1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec2 = nn.ConvTranspose2d(128 + 128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec3 = nn.Conv2d(64 + 64, 3, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        e1 = self.relu(self.enc1(x))
        e2 = self.relu(self.enc2(e1))
        e3 = self.relu(self.enc3(e2))
        d1 = self.relu(self.dec1(e3))
        d1 = torch.cat([d1, e2], dim=1)
        d2 = self.relu(self.dec2(d1))
        d2 = torch.cat([d2, e1], dim=1)
        out = self.sigmoid(self.dec3(d2))
        return out

# Load the trained generator model
def load_gan_model(generator_path):
    generator = Generator(img_size=512).to(device)
    generator.load_state_dict(torch.load(generator_path, map_location=device))
    generator.eval()
    return generator

# Preprocess an uploaded image
def preprocess_image(image):
    img = Image.open(image).convert('RGB')
    img = img.resize((512, 512), Image.Resampling.LANCZOS)
    img = np.array(img) / 255.0
    img = torch.FloatTensor(img).permute(2, 0, 1)
    img = img.unsqueeze(0).to(device)
    return img

# Generate denoised image
def denoise_image(generator, image):
    with torch.no_grad():
        denoised_img = generator(image).cpu().numpy().squeeze(0).transpose(1, 2, 0)
    return (denoised_img * 255).astype(np.uint8)
