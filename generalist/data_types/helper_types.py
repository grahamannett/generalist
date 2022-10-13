from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import torch
import torch.nn as nn

# commented until i fix circular import
from generalist.data_types.input_types import InputType
from generalist.generalist_tokenizers.general_tokenizer import GeneralTokenizer


# @dataclass
class SampleMetaData:
    idx: Any = None
    dataset_name: Any = None

    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.__dict__})"

    @classmethod
    def make(cls, idx: int | slice, **kwargs):
        if isinstance(idx, slice):
            raise NotImplementedError("metadata for slices not implemented")
        metadata = cls(idx=idx, **kwargs)
        return metadata


class Sample:
    def __init__(self, data: List[InputType] = None, target: Any = None, masks: Dict[str, Any] = {}, metadata: SampleMetaData = None):
        self.data = data
        self.target = target
        self.metadata = metadata
        self.masks = masks

    def __call__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

    def __iter__(self):
        yield self._data, self._target

    def __repr__(self) -> str:
        string = f"Sample(data={self.data}, target={self.target}"
        if self.masks:
            string += f", masks={self.masks}"

        if self.metadata is not None:
            string += f", metadata={self.metadata}"
        string += ")"
        return string

    def _input_type_hook(self, hook_fn: Callable):
        hook_fn(self)

    @classmethod
    def new(cls, **kwargs):
        return cls(**kwargs)


class SampleBuilder:
    metadata: SampleMetaData = SampleMetaData
    preprocessing: List[Callable] = []

    def __call__(self, *args, **kwargs):
        key: str
        val: Dict[str, Any]

        sample = Sample(*args, **kwargs)
        for func in self.preprocessing:
            if isinstance(func, str):
                func = getattr(sample.data, func)(sample)

            else:
                func(sample)

        return sample

    def use_preprocessing(self, func: Callable):
        self.preprocessing.append(func)


class SampleBuilderMixin:
    sample_builder = SampleBuilder()


class Batch:
    def __init__(self, samples: List[Sample] = None, return_tensors: str = None, device: str = None, **kwargs):
        self.samples = samples
        self.return_tensors = return_tensors
        self.device = device

    def attr_get(self, key: str):
        out = [getattr(s, key) for s in self.samples]
        if self.return_tensors == "pt":
            out = torch.cat(out).to(self.device)
        return out

    @property
    def data(self):
        out = defaultdict(list)
        sample: Sample
        for sample in self.samples:
            out[sample.data.data_type].append(sample.data)

        if self.return_tensors == "pt":
            for key, val in out.items():
                out[key] = torch.cat(val).to(self.device)

        return out

    @property
    def target(self):
        return self.attr_get("target")

    # TODO: refactor this so it is similar to attr_get
    def get_masks(self, key: str):
        out = [s.masks[key] for s in self.samples]
        if self.return_tensors == "pt":
            out = torch.cat(out).to(self.device)

        return out

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, key: int):
        return self.samples[key]

    def __iter__(self):
        breakpoint()
        return iter(self.samples)

    @classmethod
    def collate_fn(cls, samples: List[Sample]) -> "Batch":
        """collate_fn for torch.utils.data.DataLoader"""
        batch = cls(samples)
        return batch


@dataclass
class DataHandlerPath:
    module: nn.Module | GeneralTokenizer
    name: str = None  # the name to bind to the handler object
    data_type: str = None  # the data type it handles

    def __post_init__(self):
        if self.name is None:
            self.name = self.module.__class__.__name__
        if self.data_type is None:
            self.data_type = self.module.data_type
