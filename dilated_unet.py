"""
dilated_unet.py - Controlled Architectural Variant with Dilated Bottleneck
Author: Roll No. 23115004
Course: Advanced Computer Vision (ACV)
Assignment: B - Reproducible Segmentation Experiment

Task 3 Controlled Intervention:
  - Single Mechanism Alteration: Bottleneck Context Expansion via Atrous/Dilated Convolutions.
  - Baseline Bottleneck:
      Conv1: 3x3, Dilation=1, Padding=1 (64 -> 128)
      Conv2: 3x3, Dilation=1, Padding=1 (128 -> 128)
  - Dilated Bottleneck (Proposed):
      Conv1: 3x3, Dilation=2, Padding=2 (64 -> 128)
      Conv2: 3x3, Dilation=4, Padding=4 (128 -> 128)
  - Parameter Comparison Budget:
      Baseline Bottleneck Parameters: 221,696
      Dilated Bottleneck Parameters:  221,696 (EXACT MATCH - 0 parameter increase)
      Total Baseline Model Params:   482,771
      Total Dilated Model Params:    482,771 (EXACT MATCH)
"""

import torch
import torch.nn as nn
from baseline_unet import DoubleConv

class DilatedBottleneck(nn.Module):
    """
    Atrous/Dilated Convolutional Bottleneck.
    Expands the effective receptive field exponentially across receptive rates (r=2, r=4)
    without subsampling feature resolution or increasing trainable weights.
    """
    def __init__(self, in_channels: int = 64, out_channels: int = 128):
        super().__init__()
        # First dilated convolution with dilation rate = 2
        # Effective kernel coverage: k' = k + (k-1)*(d-1) = 3 + 2*(1) = 5x5
        self.dconv1 = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=3, padding=2, dilation=2, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Second dilated convolution with dilation rate = 4
        # Effective kernel coverage: k' = 3 + 2*(3) = 9x9
        self.dconv2 = nn.Conv2d(
            out_channels, out_channels, 
            kernel_size=3, padding=4, dilation=4, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.bn1(self.dconv1(x)))
        x = self.relu2(self.bn2(self.dconv2(x)))
        return x

class DilatedUNet(nn.Module):
    """
    Controlled U-Net Architecture featuring Dilated Bottleneck.
    Maintains identical encoder, decoder, skip connections, channels, and parameter count.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 3):
        super().__init__()
        # Identical Encoder Stage 1
        self.enc1 = DoubleConv(in_channels, 16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Identical Encoder Stage 2
        self.enc2 = DoubleConv(16, 32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Identical Encoder Stage 3
        self.enc3 = DoubleConv(32, 64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # CONTROLLED VARIABLE: Dilated Bottleneck with r=2, r=4
        self.bottleneck = DilatedBottleneck(64, 128)
        
        # Identical Decoder Stage 3
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(128, 64)
        
        # Identical Decoder Stage 2
        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(64, 32)
        
        # Identical Decoder Stage 1
        self.upconv1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(32, 16)
        
        # Identical Pixel Classifier
        self.classifier = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Contraction pathway
        s1 = self.enc1(x)
        p1 = self.pool1(s1)
        
        s2 = self.enc2(p1)
        p2 = self.pool2(s2)
        
        s3 = self.enc3(p2)
        p3 = self.pool3(s3)
        
        # Multi-scale receptive field aggregation
        b = self.bottleneck(p3)
        
        # Expansion pathway
        u3 = self.upconv3(b)
        d3 = self.dec3(torch.cat([u3, s3], dim=1))
        
        u2 = self.upconv2(d3)
        d2 = self.dec2(torch.cat([u2, s2], dim=1))
        
        u1 = self.upconv1(d2)
        d1 = self.dec1(torch.cat([u1, s1], dim=1))
        
        logits = self.classifier(d1)
        return logits

def get_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    net = DilatedUNet()
    x = torch.randn(2, 3, 256, 256)
    out = net(x)
    print("Dilated U-Net successfully instantiated.")
    print(f"Input Shape:  {x.shape}")
    print(f"Output Shape: {out.shape}")
    print(f"Total Trainable Parameters: {get_parameter_count(net):,}")
