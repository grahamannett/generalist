import torch
from torch import nn
from generalist.generalist_embedding.general_embedding import GeneralizedTensor
from generalist.models.latents import LatentEmbedding


class Transformer(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        d_model = kwargs.get("d_model", 512)
        nhead = kwargs.get("nhead", 8)
        num_encoder_layers = kwargs.get("num_encoder_layers", 4)
        num_decoder_layers = kwargs.get("num_decoder_layers", 4)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
        )

    def forward(self, embedding: GeneralizedTensor, target: GeneralizedTensor):
        out = self.transformer(embedding, target)
        return out


class TransformerDecoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        d_model = kwargs.get("d_model", 512)
        nhead = kwargs.get("nhead", 8)
        num_layers = kwargs.get("num_layers", 4)
        batch_first = kwargs.get("batch_first", True)
        dim_feedforward = kwargs.get("dim_feedforward", 2048)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=batch_first,
            activation="gelu",
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.model_max_length = kwargs.get("model_max_length", 2048)

    def forward(self, embedding: GeneralizedTensor | torch.Tensor, latents: torch.Tensor = None, **kwargs):

        if latents is None:
            latents = embedding

        out = self.transformer_decoder(embedding, latents)
        return out


if __name__ == "__main__":
    import torch

    model = TransformerDecoder()
    # latent_embedding = LatentEmbedding(num_latents=50, d_latents=512)
    # latents = latent_embedding(1)
    latents = torch.randn(1, 50, 768)
    # b x t x d
    x = GeneralizedTensor(torch.rand(1, 50, 768))
    out = model(x, latents)
