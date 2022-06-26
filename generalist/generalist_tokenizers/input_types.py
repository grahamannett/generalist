from dataclasses import dataclass
from typing import Any, List, NamedTuple
from enum import Enum

from torch import Tensor


class InputTypes(str, Enum):
    image = "image"
    text = "text"

class InputType:
    data: Any

@dataclass
class Sample:
    data: List[InputType]
    label: Any = None

@dataclass
class TextType(InputType):
    data: str
    data_type = InputTypes.text.name

@dataclass
class ImageType(InputType):
    data: Tensor
    data_type = InputTypes.image.name

