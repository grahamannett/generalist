from typing import Any, Sequence

import torch
import torch.nn as nn

from generalist.models.embedding_model import EmbeddingModel
from generalist.models.output_model import GeneralOutput
from generalist.models.transformers.transformerdecoder import TransformerDecoder
from generalist.models.transformers.from_torch import Transformer
from generalist.models.latents import LatentEmbedding


def generate_square_subsequent_mask(sz):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
    Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
    return mask


class GeneralistModel(nn.Module):
    def __init__(
        self,
        output_model: GeneralOutput,
        # BELOW COMES FROM HYDRA CONFIG
        model_dim: int = 768,
        latent_seq_len: int = 32,
        encoder_nhead: int = 4,
        decoder_nhead: int = 4,
        encoder_num_layers: int = 4,
        decoder_num_layers: int = 4,
        enable_nested_tensor: bool = True,
        token_idx: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.token_idx = token_idx
        self.model_dim = model_dim

        self.output_model = output_model
        # self.transformer = Transformer()
        # self.transformer = nn.Transformer(d_model=self.model_dim, nhead=4, num_encoder_layers=4, num_decoder_layers=4, batch_first=True)

        # self.transformer_encoder = self.transformer.encoder
        # self.transformer_decoder = self.transformer.decoder
        self.nhead = 4

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.model_dim, nhead=encoder_nhead, batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.model_dim, nhead=decoder_nhead, batch_first=True)

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=encoder_num_layers, enable_nested_tensor=enable_nested_tensor
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=decoder_num_layers)

        # self.transformer.model_max_length = 512
        # self.model_max_length = self.transformer.model_max_length
        self.model_max_length = 512

        self.latent_seq_len = latent_seq_len
        self.latents = LatentEmbedding(self.latent_seq_len, self.model_dim)

    def forward(
        self, embedded_src: torch.Tensor, embedded_tgt: torch.Tensor = None, tgt_attention_mask: torch.Tensor = None, **kwargs
    ) -> torch.Tensor:

        # kind of idea from perceiver
        if embedded_tgt is None:
            embedded_tgt = self.latents(embedded_src.shape[0])
            # breakpoint()
            # embedded_tgt = embedded_src

        # hidden_states = self.transformer(embedded=embedded, embedded_tgt=embedded_tgt)
        # embedded = torch.rand_like(embedded)

        tgt_mask = (torch.triu(torch.ones(embedded_tgt.shape[1], embedded_tgt.shape[1])) == 1).transpose(0, 1).to(embedded_tgt.device)
        tgt_mask = ~tgt_mask

        # memory = self.transformer_encoder(src=embedded_src, src_key_padding_mask=None, mask=None)
        # breakpoint()
        # hidden_states = self.transformer_decoder(
        #     tgt=embedded_tgt, memory=memory, tgt_mask=tgt_mask, tgt_key_padding_mask=None, memory_key_padding_mask=None
        # )

        # hidden_states = self.transformer(src=embedded, tgt=embedded_tgt, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_attention_mask)
        embedded_src = torch.tensor(embedded_src)

        encoded = self.transformer_encoder(src=embedded_src, src_key_padding_mask=None, mask=None)
        # encoded = embedded_src
        hidden_states = self.transformer_decoder(tgt=embedded_tgt, memory=encoded, tgt_mask=tgt_mask)

        out = self.output_model(hidden_states, decoder_query=embedded_tgt)
        return out

    def embedding(self, data):
        raise NotImplementedError("have to implement embedding")
