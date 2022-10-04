from dataclasses import dataclass
from typing import List, Sequence, Dict

import torch
import torch.nn as nn
from generalist.generalist_embedding import image_embedding

# from generalist.generalist_embedding.image_embedding import ImageEmbeddingPath, TorchvisionPretrained, ImagePath, ImagePathConv
from generalist.generalist_embedding.text_embedding import TextEmbeddingPath
from generalist.data_types.input_types import GeneralizedTensor
from generalist.data_types.helper_types import DataHandlerPath


def default_embedding_paths(model_dim: int = 768) -> List[DataHandlerPath]:
    return [
        DataHandlerPath(
            module=image_embedding.ImagePathConv(model_dim=model_dim),
            name="image_path",
            data_type=image_embedding.data_type,
        ),
        DataHandlerPath(
            module=TextEmbeddingPath(model_dim=model_dim),
            name="text_path",
            data_type=TextEmbeddingPath.data_type,
        ),
    ]


class EmbeddingModel(nn.Module):
    def __init__(
        self,
        embedding_paths: List[DataHandlerPath] = None,
        model_dim: int = 768,
        **kwargs,
    ) -> None:
        super().__init__()

        self.data_type = nn.ModuleDict({})

        if embedding_paths is None:
            embedding_paths = default_embedding_paths(model_dim=model_dim)
        for embedd_path in embedding_paths:
            self.setup_path(embedd_path, **kwargs)

        self.model_dim = model_dim

    def embed_data(self, data: torch.Tensor, data_type: str) -> torch.Tensor:
        return self.data_type[data_type](data)

    def forward(self, data: Dict[str, GeneralizedTensor]) -> Dict[str, GeneralizedTensor]:
        return {k: self.data_type[k](v) for k, v in data.items()}

    def forward(self, data: GeneralizedTensor | List[GeneralizedTensor]) -> GeneralizedTensor:
        if isinstance(data, GeneralizedTensor):
            return self.data_type[data.data_type](data)
        elif isinstance(data, Sequence):
            return self.forward_sequence(data)

    def forward_sequence(self, data: Sequence[GeneralizedTensor]) -> GeneralizedTensor:
        # this handles a list of tokenized data.
        # it is a list since the data can be of differening tokenized shapes/sizes.
        # for instance if the input is text and image versus input of just text
        embeddings = []
        for sample in data:
            if isinstance(sample, list):
                # this means that this sample has multiple data types

                emb = torch.cat([self.data_type[d.data_type](d) for d in sample], dim=1)
            else:
                # each item in list is a data type
                emb = self.data_type[sample.data_type](sample)
            embeddings.append(emb)

        embeddings = self.combine_embeddings(embeddings)
        return embeddings

    def combine_embeddings(self, embeddings: List[GeneralizedTensor]) -> torch.Tensor:
        # NOTE: simple now but assumes all embeddings have same length when combined...
        # might not be true later and needs to be fixed

        return torch.cat(embeddings, dim=0)

    def setup_path(self, embedding_path: DataHandlerPath, **kwargs) -> None:
        setattr(self, embedding_path.name, embedding_path.module)
        self.data_type[embedding_path.data_type] = embedding_path.module

    def swap_data_type(self, module: nn.Module, data_type: str = None) -> "EmbeddingModel":
        if (data_type := getattr(module, "data_type", data_type)) is None:
            raise ValueError("data_type must be arg or property of module")

        self.data_type[data_type] = module
        return self
