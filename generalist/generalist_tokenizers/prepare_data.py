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
        self.max_length = self.tokenizer.model_max_length

        self.path = {
            self.image.data_type: self.image,
            self.text.data_type: self.text,
        }

    def __call__(self, batch: List[Any]) -> Any:
        out = [self.prepare_data(data) for data in batch]
        return out

    def prepare_label(self, labels, out):
        max_lens = [min(self.tokenizer.model_max_length, o.shape[1]) for o in out]
        labels_out = [
            self.tokenizer(
                l.data,
                padding="max_length",
                truncation=True,
                max_length=max_lens[l_i],
                return_tensors="pt",
            )
            for l_i, l in enumerate(labels)
        ]
        return labels_out

    def encode(self, *args, **kwargs):
        return self.tokenizer.encode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @singledispatchmethod
    def prepare_data(self, data: T) -> None:
        raise NotImplementedError("Method has not been implemented for this type.")

    @prepare_data.register
    def prepare_data_list(self, data: abc.Sequence) -> List[Any]:
        out = [self.path[d.data_type](d) for d in data]
        return out
