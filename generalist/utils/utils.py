from dataclasses import dataclass
from typing import Any, Callable, List

import torch
from generalist.data_types.helper_types import Sample, Batch


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
class collate_func_transform:
    transform: Callable[[Any], Any] = None
    dtype: type | torch.dtype = None


class collate_func:
    def __init__(
        self,
        device: str = None,
        return_data: collate_func_transform = None,
        return_target: collate_func_transform = None,
        **kwargs,
    ):
        self.device = device
        self.return_data = return_data
        self.return_target = return_target

    def __call__(self, samples: List[Sample]) -> Batch:
        batch = Batch(samples)
        for i, sample in enumerate(batch.samples):
            sample.data = self.fix_prop(sample.data)
            sample.target = self.fix_prop(sample.target)

        return batch

    def fix_prop(self, prop):
        if isinstance(prop, list):
            new_val = [d.to(self.device) if isinstance(d, torch.Tensor) else d for d in prop]
        else:
            new_val = prop.to(self.device) if isinstance(prop, torch.Tensor) else prop

        return new_val

    def _return_tensor(self, flag: bool, obj: Batch, prop: str):
        match flag:
            case None:
                pass
            case "pt":
                # setattr(obj, prop, torch.tensor(getattr(obj, prop)))
                pass
            case _:
                breakpoint()
                raise ValueError(f"{flag} is not a valid return_data flag")


def sample_collate_fn(samples: List[Sample]) -> Batch:
    batch = Batch.collate_fn(samples)
    return batch
