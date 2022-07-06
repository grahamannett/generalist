from typing import Any, List

import torch
from torch import nn

from config import device
from generalist.generalist_tokenizers.general_embedding import GeneralEmbedding, GeneralizedTokens
from generalist.generalist_tokenizers.image_path import ImageEmbeddingPath, ImagePath
from generalist.generalist_tokenizers.text_path import TextEmbeddingPath
from generalist.models.gpt_fix import TransformerDecoder


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

    def forward(self, data: List[GeneralizedTokens]) -> GeneralEmbedding:
        return [self.handle_sample(d) for d in data]

    def make_target(self, target: str):
        with torch.no_grad():
            return self.data_type["text"].make_target(target)

    def handle_sample(self, data: List[GeneralizedTokens]) -> Any:
        embeddings = [self.data_type[d.data_type](d) for d in data]

        embedding = self.combine_embeddings(embeddings)
        embedded = GeneralEmbedding(embedding=embedding)
        return embedded

    def handle_batch(self, data: List[List[GeneralizedTokens]]) -> Any:
        return [self.handle_sample(d) for d in data]

    def combine_embeddings(self, embeddings: List[GeneralEmbedding]) -> GeneralEmbedding:
        token_size = sum([e.embedding.shape[1] for e in embeddings])
        max_dims = [self.model_dim - (token_size - e.embedding.shape[1]) for e in embeddings]
        hidden_states = []

        for idx, _emb in enumerate(embeddings):

            if max_dims[idx] > 0:
                hidden_states.append(_emb.embedding[:, : max_dims[idx]])
            else:
                hidden_states.append(_emb.embedding)

        embedded = torch.cat(hidden_states, dim=1)

        # if token_size != embedded.shape[1]:
        #     breakpoint()

        return embedded


class GeneralistModel(nn.Module):
    def __init__(self, output_dim: int = 33024, pretrained_name: str = "gpt2", **kwargs) -> None:
        super().__init__()

        self.output_dim = output_dim
        self.pretrained_name = pretrained_name

        self.transformer = TransformerDecoder.from_pretrained(self.pretrained_name)
        self.output = nn.LazyLinear(output_dim)

        self.model_max_length = self.transformer.config.n_ctx

    def forward_sample(self, x: GeneralEmbedding) -> torch.Tensor:
        x = self.transformer(x)
        x = self.output(x)
        return x

    def forward(self, data: List[GeneralEmbedding]) -> List[torch.Tensor]:
        return [self.forward_sample(x) for x in data]
