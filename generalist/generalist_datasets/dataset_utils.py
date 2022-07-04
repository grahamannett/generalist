import torch.nn as nn

from functools import wraps


class GeneralistDataset:
    registry = {}

    @staticmethod
    def register(model_name: str):
        def outer_wrapper(dataset_class, *args, **kwargs):
            GeneralistDataset.registry[model_name] = dataset_class
            return dataset_class(*args, **kwargs)

        return outer_wrapper

    def __class_getitem__(cls, key):
        return GeneralistDataset.registry.get(key, None)

    @staticmethod
    def get(key: str):
        if key not in GeneralistDataset.registry:
            raise KeyError(f"No dataset registered for {key}")
        return GeneralistDataset.registry.get(key)()
