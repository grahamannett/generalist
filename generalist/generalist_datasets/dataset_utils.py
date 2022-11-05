from typing import Any, Callable
from generalist.generalist_datasets.base import GeneralistDataset
from torch.utils.data import Dataset
import functools

from generalist.generalist_tokenizers.image_tokenizers import ImageTokenizer
from generalist.generalist_tokenizers.text_tokenizers import TextTokenizer


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

    # def __init__(self, dataset_class, **kwargs) -> None:
    #     # breakpoint()
    #     functools.update_wrapper(self, dataset_class)
    #     self.register_dataset(dataset_class)

    # def __call__(self, *args: Any, **kwds: Any) -> Any:
    #     print("in call...==>")
    #     return kwds

    #     # pass
    #     pass

    @classmethod
    def _register_dataset(cls, dataset_class: Dataset):
        if (shortname := getattr(dataset_class, "shortname", None)) is None:
            shortname = dataset_class.__name__.lower()
        cls.registry[shortname] = dataset_class

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

    @classmethod
    def register(cls, dataset_class: Dataset, *args, **kwargs) -> Dataset:
        # functools.update_wrapper(cls, dataset_class)
        cls._register_dataset(dataset_class)
        return dataset_class
