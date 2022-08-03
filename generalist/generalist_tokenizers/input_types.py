from dataclasses import KW_ONLY, dataclass
from typing import Any, List, NamedTuple, Optional, Union
from enum import Enum

from torch import Tensor
from torchvision.transforms import functional as F


class InputTypes(str, Enum):
    generic = "generic"
    image = "image"
    text = "text"


@dataclass
class InputType:
    data: Any

    # _: KW_ONLY
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


@dataclass
class TextType(InputType):
    data: str
    # _: KW_ONLY
    data_type = InputTypes.text.name


@dataclass
class ImageType(InputType):
    data: Tensor
    data_type = InputTypes.image.name

    def resize(self, size: int) -> "ImageType":
        self.data = F.resize(self.data, size)
        return self


@dataclass
class RLType(InputType):
    observation: Any
    action: Any
    reward: Any
