import torch
import torch.nn as nn


class ResidualBlock(nn.Module):

    def __init__(self, channels=64):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)


class ImageRestorationCNN(nn.Module):

    def __init__(self, num_blocks=8):
        super().__init__()

        self.head = nn.Conv2d(1, 64, 3, padding=1)

        self.body = nn.Sequential(
            *[
                ResidualBlock(64)
                for _ in range(num_blocks)
            ]
        )

        self.tail = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x):

        x = self.head(x)

        residual = self.body(x)

        x = x + residual

        x = self.tail(x)

        return x
