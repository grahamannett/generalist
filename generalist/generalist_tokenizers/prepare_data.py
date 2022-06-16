from collections import namedtuple
from typing import Any, List, NamedTuple

from generalist_tokenizers.text_path import TextTokenizer
from generalist_tokenizers.image_path import ImageTokenizer

from generalist.generalist_tokenizers.input_types import ImageType, TextType


class PrepareData:
    def __init__(self):

        self.image = ImageTokenizer()
        self.text = TextTokenizer()
        self.tokenizer = self.text.tokenizer

        self.path = {
            self.image.data_type: self.image,
            self.text.data_type: self.text,
        }

    def __call__(self, data: List[NamedTuple]) -> Any:
        # data_ = [self.path[d.dtype[0]](d.data) for d in data]

        breakpoint()
        out = [self.path[type(d)](d.data) for d in data]
        breakpoint()
        return out
