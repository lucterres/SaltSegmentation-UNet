import torch
from torch import nn
from torch.nn import functional as F

from . import config


class Block(nn.Module):
    """Double Conv: (Conv2d → BN → ReLU) × 2.  padding=1 preserves spatial dims."""
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,  out_channels, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, channels=config.ENCODER_CHANNELS) -> None:
        super().__init__()
        self.blocks   = nn.ModuleList([
            Block(channels[i], channels[i + 1])
            for i in range(len(channels) - 1)
        ])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
            x = self.pool(x)
        return features          # list of feature maps (for skip connections)


class Decoder(nn.Module):
    def __init__(self, channels=config.DECODER_CHANNELS) -> None:
        super().__init__()
        self.channels = channels
        self.up_convs = nn.ModuleList([
            nn.ConvTranspose2d(channels[i], channels[i + 1], 2, 2)
            for i in range(len(channels) - 1)
        ])
        self.blocks = nn.ModuleList([
            Block(channels[i], channels[i + 1])
            for i in range(len(channels) - 1)
        ])

    def forward(self, x, encoder_features):
        for i in range(len(self.channels) - 1):
            x = self.up_convs[i](x)
            skip = encoder_features[i]
            # If spatial dims differ, centre-crop the skip (safety net)
            if x.shape != skip.shape:
                dh = (skip.shape[2] - x.shape[2]) // 2
                dw = (skip.shape[3] - x.shape[3]) // 2
                skip = skip[:, :, dh:dh + x.shape[2], dw:dw + x.shape[3]]
            x = torch.cat([x, skip], dim=1)
            x = self.blocks[i](x)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        encoder_channels=config.ENCODER_CHANNELS,
        decoder_channels=config.DECODER_CHANNELS,
        num_classes: int = config.NUM_CLASSES,
    ) -> None:
        super().__init__()
        self.encoder   = Encoder(encoder_channels)
        self.bottleneck = Block(encoder_channels[-1], encoder_channels[-1] * 2)
        self.decoder   = Decoder((encoder_channels[-1] * 2,) + decoder_channels)
        self.head      = nn.Conv2d(decoder_channels[-1], num_classes, 1)

    def forward(self, x):
        enc_features = self.encoder(x)              # [f1, f2, f3]
        x = enc_features[-1]
        x = self.bottleneck(self.encoder.pool(x))   # bottleneck
        # Reverse encoder features for decoder skip connections
        x = self.decoder(x, enc_features[::-1])
        return self.head(x)                         # logits, shape (B,1,H,W)
