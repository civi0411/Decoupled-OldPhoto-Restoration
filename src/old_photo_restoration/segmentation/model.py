from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class AttentionGate(nn.Module):
    def __init__(self, gating_channels: int, skip_channels: int, inter_channels: int) -> None:
        super().__init__()
        self.gating_projection = nn.Sequential(
            nn.Conv2d(gating_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.skip_projection = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.attention = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Áp attention lên skip tensor theo gating tensor."""
        if g.ndim != 4 or x.ndim != 4:
            raise ValueError("AttentionGate yêu cầu tensor 4 chiều [B, C, H, W]")

        gating = self.gating_projection(g)
        if gating.shape[-2:] != x.shape[-2:]:
            gating = F.interpolate(gating, size=x.shape[-2:], mode="bilinear", align_corners=False)

        skip = self.skip_projection(x)
        attention_map = self.attention(gating + skip)
        return x * attention_map


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.attention = AttentionGate(
            gating_channels=out_channels,
            skip_channels=skip_channels,
            inter_channels=max(out_channels // 2, 1),
        )
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        skip = self.attention(x, skip)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class CrackSegmenter(nn.Module):
    """Attention U-Net tối thiểu cho segmentation vết hỏng ảnh cũ."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.encoder1 = ConvBlock(in_channels, base_channels)
        self.encoder2 = DownBlock(base_channels, base_channels * 2)
        self.encoder3 = DownBlock(base_channels * 2, base_channels * 4)
        self.bottleneck = DownBlock(base_channels * 4, base_channels * 8)

        self.decoder3 = UpBlock(base_channels * 8, base_channels * 4, base_channels * 4)
        self.decoder2 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.decoder1 = UpBlock(base_channels * 2, base_channels, base_channels)
        self.head = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Trả về logits segmentation với cùng tỷ lệ không gian như input."""
        skip1 = self.encoder1(x)
        skip2 = self.encoder2(skip1)
        skip3 = self.encoder3(skip2)
        bottleneck = self.bottleneck(skip3)

        x = self.decoder3(bottleneck, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder1(x, skip1)
        return self.head(x)

    def get_config(self) -> dict[str, int]:
        return {
            "in_channels": int(self.in_channels),
            "out_channels": int(self.out_channels),
            "base_channels": int(self.base_channels),
        }
