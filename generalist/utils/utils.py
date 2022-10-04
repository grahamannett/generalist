from dataclasses import dataclass
from typing import Any, Callable, List

from pathlib import Path
import torch
from generalist.data_types.helper_types import Sample, Batch
import datetime


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


def save_checkpoint(
    model_save_dir: Path,
    obj: Any,
    filename: str = None,
):
    """helper function to checkpoint an object.  inteded for use with saving embedding_model/model

    Args:
        model_save_dir (Path): directory to save checkpoint
        obj (Any): object containing
        filename (str, optional): _description_. Defaults to None.
    """

    if filename is None:
        filename = datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S") + ".pt"

    torch.save(
        obj,
        model_save_dir / f"{filename}",
    )


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
        return_tensors: str = None,
        **kwargs,
    ):
        self.device = device
        self.return_data = return_data
        self.return_target = return_target
        self.return_tensors = return_tensors
        self.batch_kwargs = {}

        if self.return_tensors:
            self.batch_kwargs["return_tensors"] = self.return_tensors

    # def __call__(self, samples: List[Sample]):
    #     out = {}
    #     data_idxs = {}
    #     for i, sample in enumerate(samples):

    def __call__(self, samples: List[Sample]) -> Batch:

        batch = Batch(samples, **self.batch_kwargs)
        for i, sample in enumerate(batch.samples):
            sample.data = self.fix_prop(sample.data)
            sample.target = self.fix_prop(sample.target)

            # if hasattr(sample, "tgt_attention_mask"):
            #     sample.tgt_attention_mask = self.fix_prop(sample.tgt_attention_mask)

        return batch

    def fix_prop(self, prop):
        if isinstance(prop, list):
            new_val = [d.to(self.device) if isinstance(d, torch.Tensor) else d for d in prop]
        else:
            new_val = prop.to(self.device) if isinstance(prop, torch.Tensor) else prop
        return new_val

    def _fix_prop_list(self, prop):
        return [d.to(self.device) if isinstance(d, torch.Tensor) else d for d in prop]

    def _fix_prop_single(self, prop):
        return prop.to(self.device) if isinstance(prop, torch.Tensor) else prop

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
