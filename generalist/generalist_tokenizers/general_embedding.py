from dataclasses import dataclass
from enum import Enum
from typing import Any, List

import torch
from torchvision.io import read_image
from collections import namedtuple


# dataloader requires namedtuple/list/tensor/dict
# @dataclass isnt allowed
# ImageType = namedtuple("ImageType", "data")
# TextType = namedtuple("TextType", "data")


@dataclass
class GeneralEmbedding:
    embedding: torch.Tensor
    attention_mask: torch.Tensor = None
    output_shape: torch.Tensor = None
    encoder_hidden_states: torch.Tensor = None
    encoder_attention_mask: torch.Tensor = None
    use_cache: bool = True
    output_attentions: bool = False

    input_shape: torch.Tensor = None

    def asdict(self):
        return self.__dict__

    @classmethod
    def combine(cls, *args):
        new_encoded = cls(
            hidden_states=torch.cat([arg.hidden_states for arg in args], dim=1),
        )
        return new_encoded
