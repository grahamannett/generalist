from numpy import isin, mat
import torch.nn as nn

from functools import wraps
from typing import Any


from torch.utils.data import Dataset

from generalist.generalist_tokenizers.image_path import ImageTokenizer
from generalist.generalist_tokenizers.input_types import InputType, Sample
from generalist.generalist_tokenizers.prepare_data import PrepareData
from generalist.generalist_tokenizers.text_path import TextTokenizer
from generalist.generalist_tokenizers.tokenizer_utils import GeneralTokenizer


class DataPaths:
    # image_path = ImageTokenizer()
    # text_path = TextTokenizer()
    image_path = None
    text_path = None

    @classmethod
    def setup(cls, **kwargs) -> None:
        def _helper(base, prop, init):
            if getattr(base, prop) is None:
                setattr(base, prop, init(**kwargs))

        _helper(cls, "image_path", ImageTokenizer)
        _helper(cls, "text_path", TextTokenizer)


class GeneralistDataset(Dataset):
    tokenizers = {}

    def __init__(self, return_raw: bool = True, **kwargs) -> None:
        self._return_raw = return_raw
        self._use_prepare_data = kwargs.get("use_prepare_data", False)

    @property
    def return_raw(self) -> bool:
        return self._return_raw

    @return_raw.setter
    def return_raw(self, value: bool) -> None:
        self._return_raw = value

    def path(self, data_type: str) -> Any:
        match data_type:
            case self.paths.image_path.data_type:
                return data_type
            case self.paths.text_path.data_type:
                return data_type

    def use_prepare_data(self, prepare_data: PrepareData) -> None:
        self._use_prepare_data = True
        self.tokenizers = prepare_data.path

    def process_sample(self, sample: Sample, return_raw: bool = None) -> Sample:
        if self.return_raw or return_raw:
            return sample

        self._process_sample_property(sample, "data")
        self._process_sample_property(sample, "target")

        return sample

    def _process_sample_property(self, sample: Sample, prop: str) -> Sample:
        prop_value = getattr(sample, prop)

        if isinstance(prop_value, list):
            setattr(sample, prop, [self.process_input_type(data) for data in prop_value])
        else:
            setattr(sample, prop, self.process_input_type(prop_value))

    def process_input_type(self, input_type: InputType):
        return self.tokenizers[input_type.data_type](input_type)


class DatasetRegistry:
    registry = {}

    def __init__(self) -> None:
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass

    def __class_getitem__(cls, dataset: str):
        return cls.registry.get(dataset, None)

    @staticmethod
    def get(dataset: str, *args, **kwargs):
        if dataset not in GeneralistDataset.registry:
            raise KeyError(f"No dataset registered for {dataset}")
        return DatasetRegistry.registry.get(dataset)(*args, **kwargs)

    @staticmethod
    def register(dataset_class, *args, **kwargs):
        DatasetRegistry.registry[dataset_class.shortname] = dataset_class
        return dataset_class
