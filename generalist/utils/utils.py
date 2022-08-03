import argparse
from dataclasses import astuple, dataclass
from typing import Any, List

from generalist.generalist_tokenizers.input_types import Sample


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", type=str, default=["xsum"])
    parser.add_argument("-bs", "--batch_size", type=int, default=8, dest="batch_size")
    parser.add_argument("--n_epochs", type=int, default=1, dest="n_epochs")
    parser.add_argument("--display", action=argparse.BooleanOptionalAction)
    return parser.parse_args()


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


def sample_collate_fn(samples: List[Sample]) -> Batch:
    batch = Batch.collate_fn(samples)
    return batch
