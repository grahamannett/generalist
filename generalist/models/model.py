from typing import Any, Sequence

import torch
import torch.nn as nn
from config import device

from generalist.models.embedding_model import EmbeddingModel
from generalist.models.output_model import GeneralOutput
from generalist.models.transformers.transformerdecoder import TransformerDecoder
from generalist.models.transformers.from_torch import Transformer


class GeneralistModel(nn.Module):
    def __init__(
        self,
        output_model: GeneralOutput,
        embed_dim: int = 768,
        token_idx: int = 0,
        embedding_model: EmbeddingModel = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.output_model = output_model
        if embedding_model is not None:
            raise TypeError("Embedding model should be moved outside of Generalist Model")
        self.embedding_model = embedding_model

        self.transformer = Transformer()
        # self.transformer = TransformerDecoder(
        #     n_layer=4, embed_dim=embed_dim, num_heads=8, attn_pdrop=0.1, resid_pdrop=0.1, block_size=1024
        # )

        self.model_max_length = self.transformer.model_max_length
        self.token_idx = token_idx

    def forward(self, embedded: torch.Tensor, embedded_target: torch.Tensor = None, **kwargs) -> torch.Tensor:
        # breakpoint()
        hidden_states = self.transformer(embedded=embedded, embedded_target=embedded_target)
        out = self.output_model(hidden_states, decoder_query=embedded_target)
        return out
