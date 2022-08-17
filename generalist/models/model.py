from typing import Any, Sequence

import torch
from config import device

from generalist.models.embedding_model import EmbeddingModel
from generalist.models.output_model import GeneralOutput
from generalist.models.transformers.transformerdecoder import TransformerDecoder
from torch import nn


class GeneralistModel(nn.Module):
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        output_model: GeneralOutput,
        embed_dim: int = 768,
        token_idx: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()

        self.embedding_model = embedding_model
        self.output_model = output_model

        self.transformer = TransformerDecoder(
            n_layer=4, embed_dim=embed_dim, num_heads=8, attn_pdrop=0.1, resid_pdrop=0.1, block_size=1024
        )

        self.model_max_length = self.transformer.model_max_length
        self.token_idx = token_idx

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        embedding = self.embedding_model(data)

        # if embedding is list, means we probably got in variable length data for a sample
        # breakpoint()
        if isinstance(embedding, list):
            embedding = torch.cat([torch.cat(emb, dim=1) for emb in embedding], dim=0)
            # embedding = torch.cat(embedding)

        hidden_states = self.transformer(embedding)
        breakpoint()
        out = self.output_model(hidden_states, decoder_query=embedding)
        return out
