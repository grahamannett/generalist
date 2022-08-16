from numpy import isin, mat
import torch
import torch.nn as nn

from typing import Any, Callable, Sequence


from torch.utils.data import Dataset

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


class GeneralistDataset(Dataset):
    shortname = None
    tokenizers = {}

    def __init__(self, return_raw: bool = False, **kwargs) -> None:
        self._sample_metadata = kwargs.get("sample_metadata", True)

        self.return_raw = return_raw
        self.process_sample_data = kwargs.get("process_sample_data", True)
        self.process_sample_target = kwargs.get("process_sample_target", True)

        self.device = kwargs.get("device", get_device())

    @classmethod
    def use_tokenizers(cls, tokenizers: Sequence[GeneralTokenizer], *args, **kwargs) -> None:
        if isinstance(tokenizers, GeneralTokenizer):
            tokenizers = [GeneralTokenizer]
        if args:
            tokenizers.extend(args)
        cls.tokenizers = {tok.data_type: tok for tok in tokenizers}

    def __getitem__(self, idx: int, **kwargs) -> Sample:
        sample = Sample()
        if self._sample_metadata:
            sample.metadata = SampleMetaData(idx=idx, dataset_name=self.shortname)
        else:
            delattr(sample, "metadata")

        return sample

    def _return_tuple(self):
        pass

    def path(self, data_type: str) -> Any:
        match data_type:
            case self.paths.image_path.data_type:
                return data_type
            case self.paths.text_path.data_type:
                return data_type

    def process_sample(self, sample: Sample, return_raw: bool = None) -> Sample:
        if self.return_raw or return_raw:
            return sample

        if self.process_sample_data:
            self.process_sample_property(sample, "data")
        if self.process_sample_target:
            self.process_sample_property(sample, "target")

        return sample

    def process_sample_property(self, sample: Sample, prop: str) -> Sample:
        prop_attr = getattr(sample, prop)

        match prop_attr:
            case list():
                new_prop = [
                    self.tokenizers[data.data_type](data).to(self.device).set_data_type(data.data_type)
                    for data in prop_attr
                ]
                setattr(sample, prop, new_prop)
            case InputType():
                new_val = (
                    self.tokenizers[prop_attr.data_type](prop_attr)
                    .to(self.device)
                    .set_data_type(prop_attr.data_type)
                )
                setattr(sample, prop, new_val)
            case _:
                pass
                # raise ValueError(f"{prop} is not a list or InputType")


class ChainedGenearlistDataset(Dataset):
    def __init__(
        self, datasets: Sequence[GeneralistDataset], sample_weights: Sequence[float], **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self._datasets = datasets
        self.sample_weights = sample_weights

        self._lengths = [len(dataset) for dataset in self._datasets]
        self._lengths_idx = [sum(self._lengths[:i]) for i in range(len(self._lengths))]

    def __len__(self) -> int:
        return sum(self._lengths)

    def __getitem__(self, index: int) -> Sample:
        dataset_idx = [_ for _ in self._lengths_idx if _ <= index].pop()
        return self._datasets[dataset_idx].__getitem__(index - self._lengths_idx[dataset_idx])


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
