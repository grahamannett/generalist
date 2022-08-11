from dataclasses import dataclass
from typing import List, Sequence

import torch
import torch.nn as nn
from generalist.generalist_embedding.image_embedding import ImageEmbeddingPath, ImagePath
from generalist.generalist_embedding.general_embedding import GenearlizedTensor
from generalist.generalist_tokenizers.text_path import TextEmbeddingPath


@dataclass
class EmbeddingPath:
    module: nn.Module
    name: str = None
    data_type: str = None


def default_embedding_paths() -> List[EmbeddingPath]:
    return [
        EmbeddingPath(module=ImagePath(), name="image_path", data_type=ImagePath.data_type),
        # EmbeddingPath(module=TextEmbeddingPath(), name="text_path", data_type=TextEmbeddingPath.data_type),
    ]


class EmbeddingModel(nn.Module):
    def __init__(
        self,
        embedding_paths: List[EmbeddingPath] = None,
        model_dim: int = 1024,
        **kwargs,
    ) -> None:
        super().__init__()

        self.data_type = nn.ModuleDict({})

        if embedding_paths is None:
            embedding_paths = default_embedding_paths()
        for embedd_path in embedding_paths:
            self._setup_path(embedd_path, **kwargs)

        self.model_dim = model_dim

    # def _setup_path(self, module: nn.Module, name: str = None, data_type: str = None, **kwargs) -> None:
    def _setup_path(self, embedding_path: EmbeddingPath, **kwargs) -> None:
        name = getattr(embedding_path, "name", embedding_path.module.__class__.__name__)
        data_type = getattr(embedding_path, "data_type", embedding_path.module.data_type)

        setattr(self, name, embedding_path.module)
        self.data_type[data_type] = embedding_path.module

    def swap_data_type(self, module: nn.Module, data_type: str = None) -> "EmbeddingModel":
        if (data_type := getattr(module, "data_type", data_type)) is None:
            raise ValueError("data_type must be arg or property of module")

        self.data_type[data_type] = module
        return self

    def forward(self, data: Sequence[GenearlizedTensor]) -> GenearlizedTensor:
        # this handles a list of tokenized data.
        # it is a list since the data can be of differening tokenized shapes/sizes.
        # for instance if the input is text and image versus input of just text
        return [self.handle_sample(d) for d in data]

    def handle_sample(self, data: GenearlizedTensor | Sequence[GenearlizedTensor]) -> GenearlizedTensor:

        if isinstance(data, list):
            embedding = [self.data_type[d.data_type](d) for d in data]
            embedding = torch.cat(embedding, dim=1)
        else:
            embedding = self.data_type[data.data_type](data)

        return embedding

    def _combine_embeddings(self, embeddings: Sequence[GenearlizedTensor]) -> GenearlizedTensor:

        token_size = sum([e.embedding.shape[1] for e in embeddings])
        max_dims = [self.model_dim - (token_size - e.embedding.shape[1]) for e in embeddings]
        hidden_states = []

        for idx, _emb in enumerate(embeddings):

            if max_dims[idx] > 0:
                hidden_states.append(_emb.embedding[:, : max_dims[idx]])
            else:
                hidden_states.append(_emb.embedding)

        embedded = torch.cat(hidden_states, dim=1)

        return GenearlizedTensor(embedded)
