from typing import Any, TypeVar

import torch.nn as nn
from generalist.generalist_tokenizers.image_path import ImageTokenizer
from generalist.generalist_tokenizers.input_types import InputType
from generalist.generalist_tokenizers.text_path import TextTokenizer
from generalist.models.model import EmbeddingModel, GeneralistModel


class PrepareData:
    def __init__(self, embedding_model: EmbeddingModel, generalist_model: GeneralistModel, device: str):
        self.embedding_model = embedding_model
        self.generalist_model = generalist_model
        self.model_max_length = self.generalist_model.model_max_length

        self.image = ImageTokenizer(device=device)
        self.text = TextTokenizer(max_length=self.generalist_model.model_max_length, device=device)
        self.tokenizer = self.text.tokenizer

        self.path = {
            self.image.data_type: self.image,
            self.text.data_type: self.text,
        }

        self.device = device

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
        data = [t.data for t in targets]
        out_max_length = max((o.shape[1] for o in out))
        encoded_targets = self.tokenizer(
            data,
            max_length=out_max_length,
            padding=padding,
            truncation=truncation,
            return_tensors="pt",
        )
        encoded_targets = encoded_targets["input_ids"]
        return encoded_targets

    def encode(self, *args, **kwargs):
        return self.tokenizer.encode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)
