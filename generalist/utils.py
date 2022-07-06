import argparse
from collections import UserDict
from dataclasses import astuple, dataclass
from sys import breakpointhook
from typing import Any, List

from generalist.generalist_tokenizers.input_types import Sample


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", type=str, default=["xsum"])
    return parser.parse_args()


def matplotlib_system_setup():
    import platform
    import matplotlib

    match platform.system().lower():
        case "darwin":
            matplotlib.use("MacOSX")


def _all_keys_match(batch):
    all_match = True
    _keys = list(batch[0].__annotations__.keys())
    for _batch in batch:
        if _keys != list(_batch.__annotations__.keys()):
            all_match = False
    return all_match, _keys


@dataclass
class Batch:
    """genearic batch class

    usage (one of):
        data, target = batch
        data, target = batch.data, batch.target
        data, target = batch['data'], batch['target']


    Returns:
        _type_: _description_
    """

    data: Any = None
    target: Any = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        return getattr(self, key, None)

    def __iter__(self):
        return iter((self.data, self.target))


def collate_fn(samples: List[Sample]):
    return Batch(data=[s.data for s in samples], target=[s.target for s in samples])
