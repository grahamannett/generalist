from typing import Any, List, NamedTuple, Tuple
from generalist.generalist_tokenizers.general_embedding import GeneralizedTokens

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
        out = [self.prepare_data(d) for d in data]
        return out

    def prepare_data(self, data: NamedTuple) -> GeneralizedTokens:
        return self.path[data.data_type](data.data)

    # def make_label(self, label)
