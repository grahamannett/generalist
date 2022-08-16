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


# this is basically just a torch tensor but make it easier to add data_type
class GenearlizedTensor(torch.Tensor):
    data_type: str = None
    _custom_prop: List[str] = None
    # _custom_prop = []

    # def set_prop(self, prop: str, value: Any):
    #     if hasattr(self, prop) and prop not in self._custom_prop:
    #         raise KeyError(f"only use set_prop for {prop}")
    #     setattr(self, prop, value)
    #     return self

    def set_data_type(self, data_type: str):
        self.data_type = data_type
        return self
