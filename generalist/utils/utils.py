from dataclasses import dataclass
from typing import Any, Callable, List

import torch
from generalist.generalist_tokenizers.input_types import Sample



def get_hostname():
    import platform

    return platform.node()


def matplotlib_system_setup():
    import platform

    import matplotlib

    match platform.system().lower():
        case "darwin":
            matplotlib.use("MacOSX")
        case "linux":
            # not sure if i even need this but just doing for uniformity, might need to pass
            matplotlib.use("agg")


def _all_keys_match(batch):
    all_match = True
    _keys = list(batch[0].__annotations__.keys())
    for _batch in batch:
        if _keys != list(_batch.__annotations__.keys()):
            all_match = False
    return all_match, _keys


@dataclass
class BatchOld:
    """genearic batch class

    usage (one of):
        data, target = batch
        data, target = batch.data, batch.target
        data, target = batch['data'], batch['target']
        sample = batch[0]


    Returns:
        _type_: _description_
    """

    data: Any = None
    target: Any = None

    @classmethod
    def collate_fn(cls, samples: List[Sample]) -> "Batch":
        """collate_fn for torch.utils.data.DataLoader"""
        batch = cls(*zip(*((s.data, s.target) for s in samples)))
        return batch

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key: str | int):
        match key:
            case int():
                return Sample(self.data[key], self.target[key])
            case str():
                return getattr(self, key, None)

    def __iter__(self):
        return iter((self.data, self.target))


@dataclass
class Batch:
    samples: List[Sample] = None

    def __post_init__(self):
        # not sure which is quicker:
        # list(zip(*((s.data, s.target) for s in self.samples)))
        # or below
        self.data = [s.data for s in self.samples]
        self.target = [s.target for s in self.samples]

    @classmethod
    def collate_fn(cls, samples: List[Sample]) -> "Batch":
        """collate_fn for torch.utils.data.DataLoader"""
        batch = cls(samples)
        return batch

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, key: int):
        return self.samples[key]

    def __iter__(self):
        return iter(self.samples)


class BatchAdvanced_:
    def __init__(self, samples: List[Sample] = None):
        self.samples = samples

        self.data = [s.data for s in self.samples]
        self.target = [s.target for s in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, key: int):
        return self.samples[key]

    def __iter__(self):
        return iter(self.samples)


@dataclass
class collate_func_transform:
    transform: Callable[[Any], Any] = None
    dtype: type | torch.dtype = None


class collate_func:
    def __init__(
        self,
        device: str = None,
        return_data: collate_func_transform = None,
        return_target: collate_func_transform = None,
    ):
        self.device = device
        self.return_data = return_data
        self.return_target = return_target

    def __call__(self, samples: List[Sample]) -> Batch:
        batch = BatchAdvanced_(samples)

        self._return_tensor(self.return_data, batch, "data")
        self._return_tensor(self.return_target, batch, "target")

        for i, sample in enumerate(batch.samples):
            if isinstance(sample.data.data, torch.Tensor):
                sample.data.data = sample.data.data.to(self.device)
        return batch

    def _return_tensor(self, flag: bool, obj: BatchAdvanced_, prop: str):
        match flag:
            case None:
                pass
            case "pt":
                setattr(obj, prop, torch.tensor(getattr(obj, prop)))
            case _:
                breakpoint()
                # raise ValueError(f"{flag} is not a valid return_data flag")


def sample_collate_fn(samples: List[Sample]) -> Batch:
    batch = Batch.collate_fn(samples)
    return batch
