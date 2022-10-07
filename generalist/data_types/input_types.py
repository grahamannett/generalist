from dataclasses import KW_ONLY, dataclass
from enum import Enum
from typing import Any, List, TypeVar
from typing_extensions import Self

import torch
from torch import nn
from torchvision.transforms import functional as F

# from generalist.generalist_tokenizers.general_tokenizer import GeneralTokenizer


# class InputTypes(str, Enum):
class InputTypes:
    generic = "generic"
    image = "image"  # PIL or similar
    text = "text"  # str/text
    rl = "rl"  # actions/states/rewards


class InputType:
    data: Any
    _: KW_ONLY
    data_type = InputTypes.generic

    # tokenizer: "GeneralTokenizer" = None

    # def tokenize(self, tokenizer: "GeneralTokenizer" = None):
    #     tokenizer = self.get_tokenizer(tokenizer)
    #     return tokenizer(self.data)

    # def get_tokenizer(self, tokenizer: "GeneralTokenizer" = None):
    #     if tokenizer is None:
    #         if self.tokenizer is None:
    #             raise Exception("tokenizer or InputTypes tokenizer must be set")
    #         tokenizer = self.tokenizer

    #     return tokenizer


class GeneralizedTensor(torch.Tensor):
    data_type: str = None
    _custom_prop: List[str] = None

    def set_data_type(self, data_type: str):
        self.data_type = data_type
        return self


class TextType(InputType):
    data: Any
    data_type = InputTypes.text

    def __init__(self, data: Any, *args, **kwargs):
        self.data = data

    # TODO: make this a __new__ method
    @classmethod
    def from_data(cls, *args, **kwargs):
        cls = TextTypeTensor if isinstance(args[0], torch.Tensor) else cls
        return cls(*args, **kwargs)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.data})"


class TextTypeTensor(InputType, GeneralizedTensor):
    data: torch.Tensor
    data_type = InputTypes.text


class ImageType(InputType):
    data: Any
    data_type = InputTypes.image

    @classmethod
    def from_data(cls, *args, **kwargs):
        cls = ImageTypeTensor if isinstance(args[0], torch.Tensor) else cls
        return cls(*args, **kwargs)


class ImageTypeTensor(InputType, GeneralizedTensor):
    data: torch.Tensor
    data_type = InputTypes.image

    def resize_image(self, size: List[int], **kwargs) -> "ImageType":
        self.data = F.resize(self.data, size=size, **kwargs)
        return self
