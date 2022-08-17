from numpy import isin, mat
import torch
import torch.nn as nn

from typing import Any, Callable, Sequence
from generalist.generalist_datasets.base import GeneralistDataset

from generalist.utils.utils import get_device
from generalist.generalist_tokenizers.general_tokenizer import GeneralTokenizer

from generalist.generalist_tokenizers.image_tokenizer import ImageTokenizer
from generalist.generalist_tokenizers.input_types import InputType, Sample, SampleMetaData
from generalist.generalist_tokenizers.prepare_data import PrepareData
from generalist.generalist_tokenizers.text_path import TextTokenizer


class DataPaths:
    @classmethod
    def setup(cls, **kwargs) -> None:
        def _helper(base, prop, init):
            if getattr(base, prop) is None:
                setattr(base, prop, init(**kwargs))

        _helper(cls, "image_path", ImageTokenizer)
        _helper(cls, "text_path", TextTokenizer)


class DatasetRegistry:
    registry = {}

    def __init__(self) -> None:
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass

    def __class_getitem__(cls, dataset: str):
        return cls.registry.get(dataset, None)

    @classmethod
    def list(cls):
        return list(cls.registry.keys())

    @classmethod
    def add_dataset(self, dataset_name, dataset):
        pass

    @staticmethod
    def get(dataset: str, *args, **kwargs):
        if dataset not in GeneralistDataset.registry:
            raise KeyError(f"No dataset registered for {dataset}")
        return DatasetRegistry.registry.get(dataset)(*args, **kwargs)

    @staticmethod
    def register(dataset_class: Any = None, *args, **kwargs):
        DatasetRegistry.registry[dataset_class.shortname] = dataset_class
        return dataset_class

    @staticmethod
    def register_(shortname: str, *args, **kwargs) -> Callable:
        DatasetRegistry.add_dataset(shortname)


class CombinedDataset(GeneralistDataset):
    def __init__(self, datasets: Sequence[GeneralistDataset], batch_same: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self._datasets = datasets

        self.batch_same = batch_same

    def __len__(self) -> int:
        return sum([len(dataset) for dataset in self._datasets])

    def __getitem__(self, index) -> Sample:
        dataset_idx = [_ for _ in self._lengths_idx if _ <= index].pop()
        return self._datasets[dataset_idx].__getitem__(index - self._lengths_idx[dataset_idx])
