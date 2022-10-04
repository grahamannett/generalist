from typing import Any, List, Sequence

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


class Identity(nn.Module):
    def __init__(self, return_key: str = None):
        super().__init__()
        self.return_key = return_key

    def forward(self, *args, **kwargs):
        return kwargs[self.return_key] if self.return_key is not None else args


class GeneralistModel(nn.Module):
    def __init__(
        self,
        output_model: GeneralOutput,
        # BELOW COMES FROM HYDRA CONFIG
        use_encoder: bool = True,
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
        self.use_encoder = use_encoder

        # self.transformer = nn.Transformer(d_model=self.model_dim, nhead=4, num_encoder_layers=4, num_decoder_layers=4, batch_first=True)
        # self.transformer_encoder = self.transformer.encoder
        # self.transformer_decoder = self.transformer.decoder

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.model_dim, nhead=encoder_nhead, batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.model_dim, nhead=decoder_nhead, batch_first=True)

        if self.use_encoder:
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer, num_layers=encoder_num_layers, enable_nested_tensor=enable_nested_tensor
            )
        else:
            self.transformer_encoder = Identity(return_key="src")

        self.transformer_decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=decoder_num_layers)

        # self.transformer.model_max_length = 512
        # self.model_max_length = self.transformer.model_max_length
        self.model_max_length = 512

        # FIXME: from my experiments this doesnt seem to work. but it works in perceiver paper
        self.latent_seq_len = latent_seq_len
        # self.latents = LatentEmbedding(self.latent_seq_len, self.model_dim)

        # self.identiy = nn.Identity()

    def get_memory(self):
        if self.use_encoder:
            return self.transformer_encoder

    def get_tgt_mask(self, embedded_tgt):
        tgt_mask = (torch.triu(torch.ones(embedded_tgt.shape[1], embedded_tgt.shape[1])) == 1).transpose(0, 1).to(embedded_tgt.device)
        tgt_mask = ~tgt_mask
        return tgt_mask

    def forward(
        self,
        embedded_src: torch.Tensor,
        embedded_tgt: torch.Tensor = None,
        tgt_attention_mask: torch.Tensor = None,
        tgt_key_padding_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:

        if tgt_mask is None:
            tgt_mask = self.get_tgt_mask(embedded_tgt)

        encoded = self.transformer_encoder(src=embedded_src, src_key_padding_mask=None, mask=None)
        hidden_states = self.transformer_decoder(
            tgt=embedded_tgt, memory=encoded, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask
        )

        out = self.output_model(hidden_states, decoder_query=embedded_tgt)
        return out

    def embedding(self, data):
        raise NotImplementedError("have to implement embedding")
