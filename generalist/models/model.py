from re import X
import torch
from torch import nn

from generalist.generalist_tokenizers.image_path import ImagePath
from generalist.generalist_tokenizers.text_path import TextPath
from generalist.models.gpt_fix import TransformerDecoder


class EmbeddingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.data_type = nn.ModuleDict(
            {
                "text": TextPath(),
                "image": ImagePath(),
            }
        )

    def forward(self, data):
        x = self.task_path[data["type"]](data["data"])
        return x

    def make_target(self, target: str):
        pass


class GeneralistModel(nn.Module):
    def __init__(self, output_dim: int = 33024) -> None:
        super().__init__()

        self.transformer = TransformerDecoder.from_pretrained("gpt2")
        self.output = nn.LazyLinear(output_dim)

    def forward(self, x) -> torch.Tensor:
        x = self.transformer(x)
        x = self.output(x)
        return x
