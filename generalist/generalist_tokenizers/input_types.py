from dataclasses import dataclass
from typing import Any, List, NamedTuple, Union
from enum import Enum

from torch import Tensor


class InputTypes(str, Enum):
    generic = "generic"
    image = "image"
    text = "text"


@dataclass
class InputType:
    data: Any
    data_type = InputTypes.generic.name


@dataclass
class Sample:
    data: List[InputType]
    target: Any = None


@dataclass
class TextType(InputType):
    data: str
    data_type = InputTypes.text.name


@dataclass
class ImageType(InputType):
    data: Tensor
    data_type = InputTypes.image.name


@dataclass
class RLType(InputType):
    observation: Any
    action: Any
    reward: Any
