import torch
import torch.nn as nn

class ResNetEmbedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.LazyConv2d(3, 1),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.LazyConv2d(3, 1),
            nn.LazyBatchNorm2d(),
        )

    def forward(self, x: torch.Tensor):
        x = self.block(x) + x
        return nn.functional.relu(x)