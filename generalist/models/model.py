from typing import Any, Sequence

import torch
from torch import nn

from config import device
from generalist.generalist_tokenizers.general_embedding import GeneralEmbedding, GeneralizedTokens
from generalist.generalist_tokenizers.image_path import ImageEmbeddingPath, ImagePath
from generalist.generalist_tokenizers.text_path import TextEmbeddingPath
from generalist.models.pretrained.gpt import TransformerDecoder as TransformerDecoderGPT
from generalist.models.pretrained.perceiver import TransformerDecoder as TransformerDecoderPerceiver
from generalist.models.transformer import TransformerDecoder as TransformerDecoder


class EmbeddingModel(nn.Module):
    def __init__(self, model_dim: int = 1024, **kwargs) -> None:
        super().__init__()

        self.text_path = TextEmbeddingPath()
        self.image_path = ImageEmbeddingPath()

        self.data_type = nn.ModuleDict(
            {
                self.text_path.data_type: self.text_path,
                self.image_path.data_type: self.image_path,
            }
        )

        self.model_dim = model_dim

    def forward(self, data: Sequence[GeneralizedTokens]) -> GeneralEmbedding:
        return [self.handle_sample(d) for d in data]

    def handle_sample(self, data: GeneralizedTokens | Sequence[GeneralizedTokens]) -> GeneralEmbedding:
        if isinstance(data, list):
            embedding = self.combine_embeddings([self.data_type[d.data_type](d) for d in data])
        else:
            embedding = self.data_type[data.data_type](data)

        embedded = GeneralEmbedding(embedding=embedding.embedding)
        return embedded

    def combine_embeddings(self, embeddings: Sequence[GeneralEmbedding]) -> GeneralEmbedding:

        token_size = sum([e.embedding.shape[1] for e in embeddings])
        max_dims = [self.model_dim - (token_size - e.embedding.shape[1]) for e in embeddings]
        hidden_states = []

        for idx, _emb in enumerate(embeddings):

            if max_dims[idx] > 0:
                hidden_states.append(_emb.embedding[:, : max_dims[idx]])
            else:
                hidden_states.append(_emb.embedding)

        embedded = torch.cat(hidden_states, dim=1)

        return GeneralEmbedding(embedding=embedded)


class GeneralistModel(nn.Module):
    def __init__(self, output_dim: int = 33024, pretrained_name: str = "gpt2", **kwargs) -> None:
        super().__init__()

        self.output_dim = output_dim
        self.pretrained_name = pretrained_name

        # self.transformer = TransformerDecoder.from_pretrained(self.pretrained_name)
        # self.transformer = TransformerDecoderPerceiver()
        self.transformer = TransformerDecoder()

        self.output = nn.LazyLinear(output_dim, bias=False)

        self.model_max_length = self.transformer.model_max_length

    def forward_sample(self, x: GeneralEmbedding) -> torch.Tensor:
        x = self.transformer(x)
        x = self.output(x)
        return x

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        out = self.transformer(data)
        out = self.output(out[:, 0])
        return out

    # def forward(self, data: Sequence[GeneralEmbedding]) -> Sequence[torch.Tensor]:
    #     return [self.forward_sample(x) for x in data]
