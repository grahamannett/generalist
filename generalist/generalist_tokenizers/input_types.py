from dataclasses import KW_ONLY, dataclass
from typing import Any, List, NamedTuple, Optional, Union
from enum import Enum

from generalist.generalist_tokenizers.tokenizer_utils import GeneralTokenizer

import torch
from torch import nn
from torchvision.transforms import functional as F


@dataclass
class DataHandlerPath:
    module: nn.Module | GeneralTokenizer
    name: str = None  # the name to bind to the handler object
    data_type: str = None  # the data type it handles

    def __post_init__(self):
        if self.name is None:
            self.name = self.module.__class__.__name__
        if self.data_type is None:
            self.data_type = self.module.data_type


class InputTypes(str, Enum):
    generic = "generic"
    image = "image"  # PIL or similar
    text = "text"  # str/text
    rl = "rl"  # actions/states/rewards


class InputType:
    data: Any
    _: KW_ONLY
    data_type = InputTypes.generic.name


@dataclass
class SampleMetaData:
    idx: Any = None
    dataset_name: Any = None


@dataclass
class Sample:
    data: List[InputType] = None
    target: Any = None

    metadata: Optional[SampleMetaData] = None

    def __iter__(self):
        yield self.data, self.target


class TextType(torch.Tensor, InputType):
    data_type = InputTypes.text.name


class ImageType(torch.Tensor, InputType):
    data_type = InputTypes.image.name

    def resize(self, size: int) -> "ImageType":
        self.data = F.resize(self.data, size)
        return self


@dataclass
class RLType(InputType):
    observation: Any
    action: Any
    reward: Any
