from typing import Any, Callable, TypeVar

import torch
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
        self.text = TextTokenizer(max_length=self.model_max_length, device=device)
        self.tokenizer = self.text.tokenizer

        self.path = {
            self.image.data_type: self.image,
            self.text.data_type: self.text,
        }

        self.device = device

    def __call__(self, data: Any) -> Any:
        out = []
        for d in data:
            if isinstance(d, InputType):
                out.append(self._path_data(d))
            elif isinstance(d, list):
                out.append([self._path_data(dd) for dd in d])

        return out

    def _path_data(self, data: Any) -> Any:
        return self.path[data.data_type](data)

    def prepare_targets(self, targets, logits_max_length=None, padding="max_length", truncation=True):
        # if logits:
        #     logits_max_length = max((l.shape[1] for l in logits))

        data = [t.data for t in targets]
        encoded_targets = self.tokenizer(
            data,
            max_length=logits_max_length,
            padding=padding,
            truncation=truncation,
            return_tensors="pt",
        )
        encoded_targets = encoded_targets["input_ids"]
        return encoded_targets

    def handle_targets(
        self, logits: torch.Tensor, labels: torch.Tensor, loss_fn: Callable, shift_labels: bool = False
    ):
        # Shift so that tokens < n predict n'th token
        if shift_labels:
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()

        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        return loss

    def encode(self, *args, **kwargs):
        return self.tokenizer.encode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)
