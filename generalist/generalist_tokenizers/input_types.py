from typing import NamedTuple
from enum import Enum

from torch import Tensor


class InputTypes(str, Enum):
    image = "image"
    text = "text"


class InputTypes_(NamedTuple):
    pass


class TextType(NamedTuple):
    data: str
    data_type = InputTypes.text.name


class ImageType(NamedTuple):
    data: Tensor
    data_type = InputTypes.image.name
