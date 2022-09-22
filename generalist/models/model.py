from typing import Any, Sequence

import torch
import torch.nn as nn

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
        **kwargs,
    ) -> None:
        super().__init__()
        self.token_idx = token_idx
        self.embed_dim = embed_dim

        self.output_model = output_model
        self.transformer = Transformer()

        self.model_max_length = self.transformer.model_max_length

    def forward(self, embedded: torch.Tensor, embedded_target: torch.Tensor = None, **kwargs) -> torch.Tensor:
        hidden_states = self.transformer(embedded=embedded, embedded_target=embedded_target)
        out = self.output_model(hidden_states, decoder_query=embedded_target)
        return out
