from typing import Any, Sequence

import torch
from torch import nn

from config import device

from generalist.generalist_tokenizers.general_embedding import GenearlizedTensor
from generalist.models.pretrained.gpt import TransformerDecoder as TransformerDecoderGPT
from generalist.models.pretrained.perceiver import TransformerDecoder as TransformerDecoderPerceiver
from generalist.models.transformer import TransformerDecoder as TransformerDecoder

from generalist.models.embedding_model import EmbeddingModel


class GeneralOutput(nn.Module):
    def __init__(self, output_dim: int = 33024, bias: bool = False) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.output = nn.LazyLinear(self.output_dim, bias=False)

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.output(hidden_states)


class GeneralClassificationOutput(GeneralOutput):
    def __init__(self, num_classes: int = 10, reduce_type: str = "mean", bias: bool = False) -> None:
        super().__init__(output_dim=num_classes, bias=bias)

        self.num_classes = num_classes
        self.reduce_type = reduce_type

        self.reduce_dict = {
            "mean": self.reduce_mean,
            "token_idx": self.reduce_token_idx,
        }

    def reduce_token_idx(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, 0]

    def reduce_mean(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mean(x, dim=1)

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.reduce_type is not None:
            hidden_states = self.reduce_dict[self.reduce_type](hidden_states)

        return self.output(hidden_states)


class GeneralistModel(nn.Module):
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        output_model: GeneralOutput,
        # output_dim: int = 33024,
        # pretrained_name: str = "gpt2",
        token_idx: int = 0,
        **kwargs
    ) -> None:
        super().__init__()

        self.embedding_model = embedding_model
        self.transformer = TransformerDecoder(**kwargs)

        self.output = output_model

        self.model_max_length = self.transformer.model_max_length
        self.token_idx = token_idx

    def forward_sample(self, x: GenearlizedTensor) -> torch.Tensor:
        x = self.transformer(x)
        x = self.output(x)
        return x

    def forward(self, data: torch.Tensor) -> torch.Tensor:

        embedding = self.embedding_model(data)
        if isinstance(embedding, list):
            embedding = torch.cat(embedding)
        hidden_states = self.transformer(embedding)

        out = self.output(hidden_states, decoder_query=embedding)
        return out

    # def forward(self, data: Sequence[GenearlizedTensor]) -> Sequence[torch.Tensor]:
    #     return [self.forward_sample(x) for x in data]
