from typing import Any, List

import torch
from torch import nn

from config import DEVICE
from generalist.generalist_tokenizers.general_embedding import GeneralEmbedding, GeneralizedTokens
from generalist.generalist_tokenizers.image_path import ImageEmbeddingPath, ImagePath
from generalist.generalist_tokenizers.text_path import TextEmbeddingPath
from generalist.models.gpt_fix import TransformerDecoder


class EmbeddingModel(nn.Module):
    def __init__(self, model_dim: int = 1024, device: str = DEVICE) -> None:
        super().__init__()

        self.device = device

        self.text_path = TextEmbeddingPath()
        self.image_path = ImageEmbeddingPath()

        self.data_type = nn.ModuleDict(
            {
                self.text_path.data_type: self.text_path,
                self.image_path.data_type: self.image_path,
            }
        )

        self.model_dim = model_dim

    def make_target(self, target: str):
        with torch.no_grad():
            return self.data_type["text"].make_target(target)

    def forward(self, data: List[GeneralizedTokens]) -> GeneralEmbedding:

        embedded = [self.data_type[d.data_type](d) for d in data]

        token_size = sum([_.embedding.shape[1] for _ in embedded])

        max_dims = [self.model_dim - token_size + _.embedding.shape[1] for _ in embedded]

        hidden_states = []

        for idx, _emb in enumerate(embedded):

            if max_dims[idx] > 0:
                hidden_states.append(_emb.embedding[:, : max_dims[idx]])
            else:
                hidden_states.append(_emb.embedding)

        hidden_states = torch.cat(hidden_states, dim=1)

        embeded = GeneralEmbedding(embedding=hidden_states)
        return embeded


class GeneralistModel(nn.Module):
    def __init__(self, output_dim: int = 33024, device: str = DEVICE) -> None:
        super().__init__()
        self.device = device

        self.transformer = TransformerDecoder.from_pretrained("gpt2")
        self.output = nn.LazyLinear(output_dim)

    def forward(self, x: GeneralEmbedding) -> torch.Tensor:
        x = self.transformer(x)
        x = self.output(x)
        return x
