from typing import Any, Sequence

import torch
import torch.nn as nn
from config import device

from generalist.models.embedding_model import EmbeddingModel
from generalist.models.output_model import GeneralOutput
from generalist.models.transformers.transformerdecoder import TransformerDecoder


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

        if isinstance(embedding, list):
            if isinstance(embedding[0], list):
                embedding = [torch.cat(emb, dim=1) for emb in embedding]
            embedding = torch.cat(embedding, dim=0)

        hidden_states = self.transformer(embedding)
        out = self.output_model(hidden_states, decoder_query=embedding)
        return out
