from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import utils.device as device_utils


class ConvBlock(nn.Module):
    """Block dùng cho Decoder: 2 lớp Conv liên tiếp kèm BatchNorm và ReLU"""

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


class AttentionGate(nn.Module):
    """Attention Gate giúp lọc nhiễu nền ở các skip connections"""

    def __init__(self, F_g: int, F_l: int, F_int: int) -> None:
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # Tự động resize nếu kích thước khác nhau (do downsample của ResNet)
        if g1.shape[-2:] != x1.shape[-2:]:
            g1 = F.interpolate(g1, size=x1.shape[-2:], mode='bilinear', align_corners=False)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class CrackSegmenter(nn.Module):
    """Attention U-Net với ResNet Encoder và Deep Supervision (Chuẩn Blueprint 2.1)"""

    def __init__(self, pretrained: bool = True, encoder_name: str = "resnet34") -> None:
        super().__init__()

        # 1. ENCODER (Hỗ trợ linh hoạt giữa ResNet-18 và ResNet-34)
        if encoder_name == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet18(weights=weights)
        elif encoder_name == "resnet34":
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet34(weights=weights)
        else:
            raise ValueError(f"Chỉ hỗ trợ encoder_name='resnet18' hoặc 'resnet34', nhận được: {encoder_name}")

        # Trích xuất các tầng của ResNet làm Encoder
        # Khởi đầu: Conv1 -> BN -> ReLU
        self.enc0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool

        # Các layer chính của ResNet (chứa Residual connections xịn)
        self.enc1 = resnet.layer1  # Layer 1: Đầu ra 64 channels
        self.enc2 = resnet.layer2  # Layer 2: Đầu ra 128 channels
        self.enc3 = resnet.layer3  # Layer 3: Đầu ra 256 channels
        self.bottleneck = resnet.layer4  # Layer 4 (Bottleneck của U-Net): Đầu ra 512 channels

        # 2. ATTENTION GATES
        # ag3: g (từ up3) có 512 kênh, x (từ x3) có 256 kênh
        self.ag3 = AttentionGate(F_g=512, F_l=256, F_int=128)

        # ag2: g (từ up2) có 256 kênh, x (từ x2) có 128 kênh
        self.ag2 = AttentionGate(F_g=256, F_l=128, F_int=64)

        # ag1: g (từ up1) có 128 kênh, x (từ x1) có 64 kênh
        self.ag1 = AttentionGate(F_g=128, F_l=64, F_int=32)

        # ag0: g (từ up0) có 64 kênh, x (từ x0) có 64 kênh
        self.ag0 = AttentionGate(F_g=64, F_l=64, F_int=32)

        # 3. DECODER (ĐÃ TÍNH TOÁN LẠI CHÍNH XÁC KÍCH THƯỚC NỐI - CONCAT)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # up3 (512) + ag3 (256) = 768 kênh
        self.dec3 = ConvBlock(768, 256)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # up2 (256) + ag2 (128) = 384 kênh
        self.dec2 = ConvBlock(384, 128)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # up1 (128) + ag1 (64) = 192 kênh
        self.dec1 = ConvBlock(192, 64)

        self.up0 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # up0 (64) + ag0 (64) = 128 kênh
        self.dec0 = ConvBlock(128, 64)

        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # d0 (64) upsample lên rồi vào ConvBlock
        self.final_dec = ConvBlock(64, 32)


        # 4. DEEP SUPERVISION HEADS
        # Nhận output từ decoder blocks và trả về 1 kênh (xác suất vết nứt)
        self.out_4x = nn.Conv2d(64, 1, kernel_size=1)
        self.out_2x = nn.Conv2d(64, 1, kernel_size=1)
        self.out_1x = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x: torch.Tensor):
        # --- ENCODER ---
        x0 = self.enc0(x)  # [B, 64, H/2, W/2]
        x_pool = self.maxpool(x0)  # [B, 64, H/4, W/4]

        x1 = self.enc1(x_pool)  # [B, 64, H/4, W/4]
        x2 = self.enc2(x1)  # [B, 128, H/8, W/8]
        x3 = self.enc3(x2)  # [B, 256, H/16, W/16]
        bottle = self.bottleneck(x3)  # [B, 512, H/32, W/32]

        # --- DECODER + ATTENTION ---
        # Up-sample 3
        up3 = self.up3(bottle)
        concat3 = torch.cat([self.ag3(g=up3, x=x3), up3], dim=1)
        d3 = self.dec3(concat3)

        # Up-sample 2
        up2 = self.up2(d3)
        concat2 = torch.cat([self.ag2(g=up2, x=x2), up2], dim=1)
        d2 = self.dec2(concat2)

        # Up-sample 1
        up1 = self.up1(d2)
        concat1 = torch.cat([self.ag1(g=up1, x=x1), up1], dim=1)
        d1 = self.dec1(concat1)
        out4x = self.out_4x(d1)  # Head 4x

        # Up-sample 0
        up0 = self.up0(d1)
        concat0 = torch.cat([self.ag0(g=up0, x=x0), up0], dim=1)
        d0 = self.dec0(concat0)
        out2x = self.out_2x(d0)  # Head 2x

        # Final Up-sample
        d_final = self.final_dec(self.final_up(d0))
        out1x = self.out_1x(d_final)  # Head 1x

        # --- DEEP SUPERVISION ---
        if self.training:
            # Resize các output trung gian về kích thước ảnh gốc để tính Loss
            out4x_up = F.interpolate(out4x, size=x.shape[2:], mode='bilinear', align_corners=False)
            out2x_up = F.interpolate(out2x, size=x.shape[2:], mode='bilinear', align_corners=False)
            return out1x, out2x_up, out4x_up
        else:
            # Chỉ trả về output chính xác nhất khi chạy inference
            return out1x