import torch
from torch import nn
from generalist.generalist_tokenizers.general_embedding import GeneralEmbedding
from generalist.models.latents import LatentEmbedding


class TransformerDecoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        d_model = kwargs.get("d_model", 768)
        nhead = kwargs.get("nhead", 4)
        num_layers = kwargs.get("num_layers", 2)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)

        self.model_max_length = kwargs.get("model_max_length", 2048)

    def forward(self, embedding: GeneralEmbedding | torch.Tensor, latents: torch.Tensor = None):
        if isinstance(embedding, GeneralEmbedding):
            embedding = embedding.embedding

        if latents is None:
            latents = embedding

        return self.transformer_decoder(embedding, latents)


if __name__ == "__main__":
    import torch

    model = TransformerDecoder()
    # latent_embedding = LatentEmbedding(num_latents=50, d_latents=512)
    # latents = latent_embedding(1)
    latents = torch.randn(1, 50, 768)
    # b x t x d
    x = GeneralEmbedding(torch.rand(1, 50, 768))
    out = model(x, latents)
    breakpoint()
