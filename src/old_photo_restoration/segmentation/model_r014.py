import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

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

class CrackSegmenterR014ResNet34(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        
        self.pretrained_loaded = False
        resnet = None
        try:
            if pretrained:
                # Try new API
                if hasattr(models, 'ResNet34_Weights'):
                    resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
                else:
                    # Fallback to old API
                    resnet = models.resnet34(pretrained=True)
                self.pretrained_loaded = True
            else:
                resnet = models.resnet34(weights=None)
        except Exception as e:
            print(f"Failed to load pretrained weights: {e}")
            resnet = models.resnet34(weights=None)
            self.pretrained_loaded = False
            
        self.pretrained = self.pretrained_loaded
        
        # Encoder (ResNet-34)
        # Expected shapes for input 512x512
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            # We omit maxpool here to preserve 256x256 resolution for skip connection, or we can use it.
            # standard resnet stem: out is 256x256 before maxpool. 
            # wait, resnet maxpool is applied before layer1.
        )
        self.maxpool = resnet.maxpool # reduces 256x256 to 128x128
        self.layer1 = resnet.layer1   # 64 channels, 128x128
        self.layer2 = resnet.layer2   # 128 channels, 64x64
        self.layer3 = resnet.layer3   # 256 channels, 32x32
        self.layer4 = resnet.layer4   # 512 channels, 16x16
        
        # Decoder
        # decode layer4 + layer3: 16 -> 32
        self.decoder4 = UpBlock(512, 256, 256)
        # decode + layer2: 32 -> 64
        self.decoder3 = UpBlock(256, 128, 128)
        # decode + layer1: 64 -> 128
        self.decoder2 = UpBlock(128, 64, 64)
        # decode + stem: 128 -> 256. stem output is 64 channels.
        self.decoder1 = UpBlock(64, 64, 64)
        
        # final upsample: 256 -> 512
        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = ConvBlock(32, 32)
        
        # final head
        self.head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # stem
        skip_stem = self.stem(x) # [B, 64, 256, 256]
        
        # layer1
        e1 = self.maxpool(skip_stem)
        skip1 = self.layer1(e1) # [B, 64, 128, 128]
        
        # layer2
        skip2 = self.layer2(skip1) # [B, 128, 64, 64]
        
        # layer3
        skip3 = self.layer3(skip2) # [B, 256, 32, 32]
        
        # layer4 (bottleneck)
        bottleneck = self.layer4(skip3) # [B, 512, 16, 16]
        
        # decode
        d4 = self.decoder4(bottleneck, skip3) # [B, 256, 32, 32]
        d3 = self.decoder3(d4, skip2)         # [B, 128, 64, 64]
        d2 = self.decoder2(d3, skip1)         # [B, 64, 128, 128]
        d1 = self.decoder1(d2, skip_stem)     # [B, 64, 256, 256]
        
        # final upsample to 512x512
        f_up = self.final_up(d1)              # [B, 32, 512, 512]
        f_out = self.final_conv(f_up)         # [B, 32, 512, 512]
        
        out = self.head(f_out)                # [B, 1, 512, 512]
        return out
