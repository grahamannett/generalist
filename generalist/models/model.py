from enum import Enum
from turtle import forward
from typing import Any
from generalist.generalist_tokenizers.patch_embedding import ImageEmbedding
from generalist.generalist_tokenizers.tokenizer_helpers import TextTokenizer

import torch
from torch import nn

from generalist.models.gpt_fix import GPT2Model_Fix, TransformerDecoder

from generalist.generalist_tokenizers.text_embedding import TextEmbedding


class TextPath(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.tokenizer = TextTokenizer()
        self.embedder = TextEmbedding()

    def forward(self, data: Any) -> torch.Tensor:
        data_ = self.tokenizer(data)
        data_ = self.embedder(**data_)
        return data_
        # return self.embedder(self.tokenizer(data))


class GeneralistModel(nn.Module):
    def __init__(self, output_dim: int = 33024) -> None:
        super().__init__()

        self.transformer = TransformerDecoder.from_pretrained("gpt2")
        self.output = nn.LazyLinear(output_dim)

        # self.text_embedding = TextEmbedding()
        # self.text_tokenizer = TextTokenizer()

        # self.image_embedding = ImageEmbedding()

        # self.image_path = ImageEmbedding()
        # self.text_path = TextEmbedding()

        self.task_path = nn.ModuleDict(
            {
                "text": TextPath(),
                "image": ImageEmbedding(),
            }
        )

    def forward(self, data) -> torch.Tensor:

        enc = self.task_path[data["type"]](data["data"])

        x = self.transformer(enc)
        breakpoint()
        x = self.output(x)
        return x
