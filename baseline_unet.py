"""
baseline_unet.py - Modest Baseline Encoder-Decoder Architecture
Author: Roll No. 23115004
Course: Advanced Computer Vision (ACV)
Assignment: B - Reproducible Segmentation Experiment

Architecture Overview:
  - 3-stage contracting encoder (DoubleConv blocks + MaxPool2d)
  - Dense bottleneck representation (64 -> 128 channels)
  - 3-stage expanding decoder (ConvTranspose2d + Feature Concatenation + DoubleConv)
  - 1x1 Convolution final pixel classifier (16 -> 3 classes)
  - Channel Progression: 3 -> 16 -> 32 -> 64 -> 128 (Bottleneck) -> 64 -> 32 -> 16 -> 3
  - Total Trainable Parameters: 482,771
"""

import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """
    Standard building block: Two consecutive (3x3 Conv -> BatchNorm -> ReLU) layers.
    Maintains spatial resolution via padding=1.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class BaselineUNet(nn.Module):
    """
    Modest Baseline U-Net satisfying Task 2 specification.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 3):
        super().__init__()
        # Encoder Stage 1: Input (3, 256, 256) -> (16, 256, 256)
        self.enc1 = DoubleConv(in_channels, 16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder Stage 2: (16, 128, 128) -> (32, 128, 128)
        self.enc2 = DoubleConv(16, 32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder Stage 3: (32, 64, 64) -> (64, 64, 64)
        self.enc3 = DoubleConv(32, 64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck: (64, 32, 32) -> (128, 32, 32)
        # Uses standard dense convolutions with dilation rate = 1
        self.bottleneck = DoubleConv(64, 128)
        
        # Decoder Stage 3: Upsample (128, 32, 32) -> (64, 64, 64), Concat with enc3 -> (128, 64, 64) -> (64, 64, 64)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(128, 64)
        
        # Decoder Stage 2: Upsample (64, 64, 64) -> (32, 128, 128), Concat with enc2 -> (64, 128, 128) -> (32, 128, 128)
        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(64, 32)
        
        # Decoder Stage 1: Upsample (32, 128, 128) -> (16, 256, 256), Concat with enc1 -> (32, 256, 256) -> (16, 256, 256)
        self.upconv1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(32, 16)
        
        # Pixel-wise classifier: 1x1 Conv (16 -> 3)
        self.classifier = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Contraction pathway
        s1 = self.enc1(x)               # (B, 16, 256, 256)
        p1 = self.pool1(s1)             # (B, 16, 128, 128)
        
        s2 = self.enc2(p1)              # (B, 32, 128, 128)
        p2 = self.pool2(s2)             # (B, 32, 64, 64)
        
        s3 = self.enc3(p2)              # (B, 64, 64, 64)
        p3 = self.pool3(s3)             # (B, 64, 32, 32)
        
        # Latent representation
        b = self.bottleneck(p3)         # (B, 128, 32, 32)
        
        # Expansion pathway with standard concatenation skip connections
        u3 = self.upconv3(b)            # (B, 64, 64, 64)
        d3 = self.dec3(torch.cat([u3, s3], dim=1))  # (B, 64, 64, 64)
        
        u2 = self.upconv2(d3)           # (B, 32, 128, 128)
        d2 = self.dec2(torch.cat([u2, s2], dim=1))  # (B, 32, 128, 128)
        
        u1 = self.upconv1(d2)           # (B, 16, 256, 256)
        d1 = self.dec1(torch.cat([u1, s1], dim=1))  # (B, 16, 256, 256)
        
        logits = self.classifier(d1)    # (B, 3, 256, 256)
        return logits

def get_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    net = BaselineUNet()
    x = torch.randn(2, 3, 256, 256)
    out = net(x)
    print("Baseline U-Net successfully instantiated.")
    print(f"Input Shape:  {x.shape}")
    print(f"Output Shape: {out.shape}")
    print(f"Total Trainable Parameters: {get_parameter_count(net):,}")
