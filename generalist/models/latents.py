import torch
import torch.nn as nn


class LatentEmbedding(nn.Module):
    """Construct the latent embeddings."""

    def __init__(self, num_latents: int = 784, d_latents: int = 1024):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, d_latents))

    def forward(self, batch_size: int):
        return self.latents.expand(batch_size, -1, -1)
