from typing import Any, TypeVar

import torch.nn as nn
from generalist.generalist_tokenizers.image_path import ImageTokenizer
from generalist.generalist_tokenizers.input_types import InputType
from generalist.generalist_tokenizers.text_path import TextTokenizer
from generalist.models.model import EmbeddingModel, GeneralistModel


class PrepareData:
    def __init__(self, embedding_model: EmbeddingModel, generalist_model: GeneralistModel):
        self.embedding_model = embedding_model
        self.generalist_model = generalist_model
        self.model_max_length = self.generalist_model.model_max_length

        self.image = ImageTokenizer()
        self.text = TextTokenizer(max_length=self.generalist_model.model_max_length)
        self.tokenizer = self.text.tokenizer

        self.path = {
            self.image.data_type: self.image,
            self.text.data_type: self.text,
        }

    def __call__(self, batch: Any) -> Any:

        out = []
        for data in batch:
            if isinstance(data, InputType):
                out.append(self._path_data(data))
            elif isinstance(data, list):
                out.append([self._path_data(d) for d in data])
        return out

    def _path_data(self, data: Any) -> Any:
        return self.path[data.data_type](data)

    def prepare_targets(self, targets, out, padding="max_length", truncation=True):
        encoded_targets = [
            self.tokenizer.encode(
                target.data,
                max_length=out[target_idx].shape[1],
                padding=padding,
                truncation=truncation,
                return_tensors="pt",
            )
            for target_idx, target in enumerate(targets)
        ]
        return encoded_targets

    def encode(self, *args, **kwargs):
        return self.tokenizer.encode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)
