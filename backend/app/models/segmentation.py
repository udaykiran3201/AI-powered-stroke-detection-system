"""
Stroke Detection System — U-Net Segmentation Model

Segments stroke-affected regions (lesions) in CT brain scans.
Produces a single-channel binary mask indicating lesion vs. background.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from loguru import logger


class DoubleConv(nn.Module):
    """(Conv → BN → ReLU) × 2"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """Max-pool → DoubleConv"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    """Upsample → concat skip → DoubleConv"""

    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Handle size mismatch from odd dimensions
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(
            x1,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
        )
        return self.conv(torch.cat([x2, x1], dim=1))


class UNet(nn.Module):
    """
    Standard U-Net for binary lesion segmentation.

    Input : (B, in_channels, H, W)   — typically 1-ch or 3-ch CT slice
    Output: (B, 1, H, W)             — sigmoid probability mask
    """

    def __init__(self, in_channels: int = 1, bilinear: bool = True):
        super().__init__()
        self.in_channels = in_channels
        factor = 2 if bilinear else 1

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        self.outc = nn.Conv2d(64, 1, kernel_size=1)

        logger.info(
            f"UNet initialised — in_channels={in_channels}, bilinear={bilinear}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits


def load_unet(
    weights_path: Optional[str] = None,
    device: str = "cpu",
    in_channels: int = 1,
) -> UNet:
    """Instantiate U-Net and optionally load saved weights."""
    model = UNet(in_channels=in_channels)
    if weights_path:
        try:
            state_dict = torch.load(weights_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            logger.info(f"Loaded U-Net weights from {weights_path}")
        except FileNotFoundError:
            logger.warning(
                f"Weights not found at {weights_path} — using randomly initialised U-Net"
            )
    model.to(device).eval()
    return model


def segment(
    model: UNet,
    image_tensor: torch.Tensor,
    device: str = "cpu",
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Run segmentation inference.

    Parameters
    ----------
    model : UNet
    image_tensor : torch.Tensor — (1, C, H, W)
    device : str
    threshold : float

    Returns
    -------
    torch.Tensor — binary mask (1, 1, H, W), uint8
    """
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        logits = model(image_tensor)
        probs = torch.sigmoid(logits)
        mask = (probs > threshold).to(torch.uint8)
    return mask.cpu()
