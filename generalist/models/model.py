from typing import Any, Sequence

import torch
import torch.nn as nn

from generalist.models.embedding_model import EmbeddingModel
from generalist.models.output_model import GeneralOutput
from generalist.models.transformers.transformerdecoder import TransformerDecoder
from generalist.models.transformers.from_torch import Transformer
from generalist.models.latents import LatentEmbedding


class GeneralistModel(nn.Module):
    def __init__(
        self,
        output_model: GeneralOutput,
        model_dim: int = 768,
        latent_seq_len: int = 32,
        token_idx: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.token_idx = token_idx
        self.model_dim = model_dim

        self.output_model = output_model
        # self.transformer = Transformer()
        self.transformer = nn.Transformer(d_model=self.model_dim, nhead=4, num_encoder_layers=4, num_decoder_layers=4, batch_first=True)
        self.transformer.model_max_length = 512

        self.model_max_length = self.transformer.model_max_length

        self.latent_seq_len = latent_seq_len
        self.latents = LatentEmbedding(self.latent_seq_len, self.model_dim)

    def forward(
        self, embedded: torch.Tensor, embedded_target: torch.Tensor = None, target_attention_mask: torch.Tensor = None, **kwargs
    ) -> torch.Tensor:

        # kind of idea from perceiver
        if embedded_target is None:
            embedded_target = self.latents(embedded.shape[0])

        # hidden_states = self.transformer(embedded=embedded, embedded_target=embedded_target)
        # embedded = torch.rand_like(embedded)
        hidden_states = self.transformer(src=embedded, tgt=embedded_target, tgt_key_padding_mask=target_attention_mask)
        out = self.output_model(hidden_states, decoder_query=embedded_target)
        return out
