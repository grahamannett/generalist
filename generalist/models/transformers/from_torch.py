import torch
from torch import nn
from generalist.data_types.input_types import GeneralizedTensor
from generalist.models.latents import LatentEmbedding

from typing import Optional


class Transformer(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.d_model = kwargs.get("d_model", 768)
        self.nhead = kwargs.get("nhead", 8)
        self.num_encoder_layers = kwargs.get("num_encoder_layers", 3)
        self.num_decoder_layers = kwargs.get("num_decoder_layers", 3)

        self.batch_first = kwargs.get("batch_first", True)
        self.model_max_length = kwargs.get("model_max_length", 1048)

        self.transformer = nn.Transformer(
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            batch_first=self.batch_first,
        )

        self.encoder = self.transformer.encoder
        self.decoder = self.transformer.decoder

    def forward(
        self,
        embedded: torch.Tensor,
        embedded_target: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        src = embedded
        tgt = embedded_target if embedded_target is not None else embedded

        is_batched = src.dim() == 3
        if not self.batch_first and src.size(1) != tgt.size(1) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")
        elif self.batch_first and src.size(0) != tgt.size(0) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")

        if src.size(-1) != self.d_model or tgt.size(-1) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        return output


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
