import torch
from torch import nn

from generalist.generalist_tokenizers.patch_embedding import ImageEmbedding
from generalist.generalist_tokenizers.text_path import TextPath
from generalist.models.gpt_fix import TransformerDecoder


class GeneralistModel(nn.Module):
    def __init__(self, output_dim: int = 33024) -> None:
        super().__init__()

        self.task_path = nn.ModuleDict(
            {
                "text": TextPath(),
                "image": ImageEmbedding(),
            }
        )


        self.transformer = TransformerDecoder.from_pretrained("gpt2")
        self.output = nn.LazyLinear(output_dim)

    def forward(self, data) -> torch.Tensor:

        enc = self.task_path[data["type"]](data["data"])

        x = self.transformer(enc)
        x = self.output(x)
        return x
