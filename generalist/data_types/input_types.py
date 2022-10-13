from dataclasses import KW_ONLY, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, List

import torch
from torch import nn
from torchvision.io import read_image
from torchvision.transforms import functional as F


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

    def __new__(cls: type["TextType"], *args, **kwargs):
        if isinstance(args[0], torch.Tensor):
            return TextTypeTensor(*args, **kwargs)
        return super().__new__(cls)

    def __init__(self, data: Any, **kwargs):
        self.data = data

    def __len__(self):
        return len(self.data)


class TextTypeTensor(GeneralizedTensor):
    data: torch.Tensor
    data_type = InputTypes.text


class ImageType(InputType):
    data: Any
    data_type = InputTypes.image

    def __new__(cls: type["ImageType"], *args, **kwargs):
        if isinstance(args[0], torch.Tensor):
            return ImageTypeTensor(*args, **kwargs)
        return super().__new__(cls)

    def __init__(self, data: Any, **kwargs):
        self.data = data

    @classmethod
    def from_file(cls, filepath: str | Path):
        image = read_image(filepath)
        return cls(image)


class ImageTypeTensor(GeneralizedTensor):
    data: torch.Tensor
    data_type = InputTypes.image

    def resize_image(self, size: List[int], **kwargs) -> "ImageTypeTensor":
        self.data = F.resize(self.data, size=size, **kwargs)
        return self


# @dataclass
class OfflineRLType(InputType):
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor

    data_type = InputTypes.rl

    def __init__(self, observations: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor) -> None:
        super().__init__()
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.dones = dones

    def __repr__(self):
        return f"{self.__class__.__name__}(observations={self.observations.shape}, actions={self.actions.shape}, rewards={self.rewards.shape}, dones={self.dones.shape})"

    def hook_attributes(self, sample: "Sample"):
        sample.observations = self.observations
        sample.actions = self.actions
        sample.rewards = self.rewards
        sample.dones = self.dones


class DataWithMask:
    def __init__(self, data: Any, mask: Any, data_cls: Callable = TextTypeTensor):
        self.data = data
        self.mask = mask
        self.data_cls = data_cls

    def get(self):
        return self.data_cls(self.data), self.mask
