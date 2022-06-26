from functools import singledispatchmethod
from typing import Any, List, Sequence, NamedTuple, Tuple
from generalist.generalist_tokenizers.general_embedding import GeneralizedTokens

from generalist_tokenizers.text_path import TextTokenizer
from generalist_tokenizers.image_path import ImageTokenizer

from generalist.generalist_tokenizers.input_types import ImageType, TextType

from typing import TypeVar
from collections import abc

T = TypeVar("T")

class PrepareData:
    def __init__(self):

        self.image = ImageTokenizer()
        self.text = TextTokenizer()
        self.tokenizer = self.text.tokenizer

        self.path = {
            self.image.data_type: self.image,
            self.text.data_type: self.text,
        }

    def __call__(self, batch: List[Any]) -> Any:
        out = [self.prepare_data(data) for data in batch]
        return out


    @singledispatchmethod
    def prepare_data(self, data: T) -> None:
        raise NotImplementedError("Method has not been implemented for this type.")


    @prepare_data.register
    def prepare_data_list(self, data: abc.Sequence) -> List[Any]:
        out =  [self.path[d.data_type](d) for d in data]
        return out


    # @prepare_data.register
    # def prepare_data_namedtuple(self, data: NamedTuple) -> GeneralizedTokens:
    #     return self.path[data.data_type](data.data)

