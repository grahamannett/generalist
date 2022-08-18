from dataclasses import KW_ONLY, dataclass
from enum import Enum
from typing import Any, List, TypeVar

import torch

# from generalist.generalist_tokenizers.general_tokenizer import GeneralTokenizer
from torch import nn
from torchvision.transforms import functional as F

GeneralTokenizer = TypeVar("GeneralTokenizer")


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

    tokenizer: GeneralTokenizer = None

    def tokenize(self, tokenizer: GeneralTokenizer = None):
        tokenizer = self.get_tokenizer(tokenizer)
        return tokenizer(self.data)

    def get_tokenizer(self, tokenizer: GeneralTokenizer = None):
        if tokenizer is None:
            if self.tokenizer is None:
                raise Exception("tokenizer or InputTypes tokenizer must be set")
            tokenizer = self.tokenizer

        return tokenizer


class GenearlizedTensor(torch.Tensor):
    data_type: str = None
    _custom_prop: List[str] = None

    def set_data_type(self, data_type: str):
        self.data_type = data_type
        return self


@dataclass
class TextTypeRaw(InputType):
    data: Any
    data_type = InputTypes.text.name

    def tokenize(self, tokenizer: GeneralTokenizer = None):
        tokenizer = super().get_tokenizer(tokenizer)
        return tokenizer(self.data)

    # def convert(self, tokenizer: GeneralTokenizer):
    #     return TextType(tokenizer(self.data))


class TextType(InputType, GenearlizedTensor):
    data: torch.Tensor
    data_type = InputTypes.text.name


class ImageType(InputType, GenearlizedTensor):
    data: torch.Tensor
    data_type = InputTypes.image.name

    def resize_image(self, size: List[int], **kwargs) -> "ImageType":
        self.data = F.resize(self.data, size=size, **kwargs)
        return self


@dataclass
class RLType(InputType):
    observation: Any
    action: Any
    reward: Any


if __name__ == "__main__":
    x = TextType("this is a string")
    x2 = TextType(torch.Tensor([1, 2, 3]))
    x3 = ImageType(torch.rand(3, 224, 224))
    assert type(x) == TextType
    assert type(x2) == TextType
    breakpoint()