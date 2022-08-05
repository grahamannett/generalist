from typing import Any, Sequence

import torch
from torch import nn

from config import device
from generalist.generalist_embedding.image_embedding import ImageEmbeddingPath
from generalist.generalist_tokenizers.general_embedding import GenearlizedTensor
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


class GeneralistModel(nn.Module):
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        output_dim: int = 33024,
        # pretrained_name: str = "gpt2",
        token_idx: int = 0,
        **kwargs
    ) -> None:
        super().__init__()

        self.embedding_model = embedding_model

        self.output_dim = output_dim

        # self.pretrained_name = pretrained_name
        # self.transformer = TransformerDecoder.from_pretrained(self.pretrained_name)
        # self.transformer = TransformerDecoderPerceiver()
        self.transformer = TransformerDecoder()

        self.output = nn.LazyLinear(output_dim, bias=False)
        self.model_max_length = self.transformer.model_max_length
        self.token_idx = token_idx

    def forward_sample(self, x: GenearlizedTensor) -> torch.Tensor:
        x = self.transformer(x)
        x = self.output(x)
        return x

    def forward(self, data: torch.Tensor) -> torch.Tensor:

        out = self.embedding_model(data)
        out = torch.cat(out)
        out = self.transformer(out)

        # if self.token_idx is not None:
        #     out = out[:, self.token_idx]

        out = out.mean(dim=1)
        out = self.output(out)
        return out

    # def forward(self, data: Sequence[GenearlizedTensor]) -> Sequence[torch.Tensor]:
    #     return [self.forward_sample(x) for x in data]
