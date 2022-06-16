from dataclasses import dataclass
from enum import Enum
from typing import Any, List

import torch
from torchvision.io import read_image
from collections import namedtuple

from config import device


def _device_to(self, name: str, device_: str = device):
    if (_tensor := getattr(self, name, None)) is not None:
        setattr(self, name, _tensor.to(device_))


class GenearlizedInput:
    def set_data_type(self, data_type: str):
        self.data_type = data_type
        return self


@dataclass
class GeneralizedTokens(GenearlizedInput):
    tokens: torch.Tensor

    data_type: str = None
    attention_mask: torch.Tensor = None
    token_type_ids: torch.Tensor = None

    def __post_init__(self):
        self.tokens = self.tokens.to(device)

        _device_to(self, "attention_mask")
        _device_to(self, "token_type_ids")


@dataclass
class GeneralEmbedding(GenearlizedInput):
    embedding: torch.Tensor = None
    attention_mask: torch.Tensor = None
    output_shape: torch.Tensor = None
    encoder_hidden_states: torch.Tensor = None
    encoder_attention_mask: torch.Tensor = None
    use_cache: bool = True
    output_attentions: bool = False

    input_shape: torch.Tensor = None
    data_type: str = None

    def asdict(self):
        return self.__dict__

    @classmethod
    def combine(cls, *args):
        new_encoded = cls(
            hidden_states=torch.cat([arg.hidden_states for arg in args], dim=1),
        )
        return new_encoded
