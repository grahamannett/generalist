from dataclasses import KW_ONLY, dataclass
from enum import Enum
from typing import Any, Callable, List, TypeVar

import torch
from torch import nn
from torchvision.transforms import functional as F
from typing_extensions import Self


class InputTypes:
    generic = "generic"
    image = "image"  # PIL, tensor or similar
    text = "text"  # str/text or text tokens
    rl = "rl"  # actions/states/rewards


class InputType:
    data: Any
    _: KW_ONLY
    data_type = InputTypes.generic

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.data})"


class GeneralizedTensor(torch.Tensor):
    data_type: str = None
    _custom_prop: List[str] = None

    def set_data_type(self, data_type: str):
        # this is so its chainable
        self.data_type = data_type
        return self


class TextType(InputType):
    data: Any
    data_type = InputTypes.text

    def __new__(cls: type[Self], *args, **kwargs) -> Self:
        if isinstance(args[0], torch.Tensor):
            return TextTypeTensor(*args, **kwargs)
        return super().__new__(cls)

    def __init__(self, data: Any, **kwargs):
        self.data = data

    @classmethod
    def transform(cls, *args, **kwargs):
        return


class TextTypeTensor(GeneralizedTensor):
    data: torch.Tensor
    data_type = InputTypes.text


class ImageType(InputType):
    data: Any
    data_type = InputTypes.image

    def __new__(cls: type[Self], *args, **kwargs) -> Self:
        if isinstance(args[0], torch.Tensor):
            return ImageTypeTensor(*args, **kwargs)
        return super().__new__(cls)

    def __init__(self, data: Any, **kwargs):
        self.data = data

    @classmethod
    def transform(cls, *args, **kwargs):
        return cls(*args).unsqueeze(0)


class ImageTypeTensor(GeneralizedTensor):
    data: torch.Tensor
    data_type = InputTypes.image

    def resize_image(self, size: List[int], **kwargs) -> "ImageType":
        self.data = F.resize(self.data, size=size, **kwargs)
        return self


@dataclass
class RLType(InputType):
    observation: Any
    action: Any
    reward: Any


class DataWithMask:
    def __init__(self, data: Any, mask: Any, data_cls: Callable = TextTypeTensor):
        self.data = data
        self.mask = mask
        self.data_cls = data_cls

    def get(self):
        return self.data_cls(self.data), self.mask


if __name__ == "__main__":
    x = TextType("this is a string")
    x2 = TextType(torch.Tensor([1, 2, 3]))
    x3 = ImageType(torch.rand(3, 224, 224))

    assert type(x) == TextType
    assert type(x2) == TextTypeTensor
