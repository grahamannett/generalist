from typing import Any, Callable, List
from dataclasses import dataclass

import torch
from typing import Callable
from generalist.generalist_datasets.base import TokenizersHandler


from generalist.data_types.helper_types import Sample, Batch


@dataclass
class collate_func_transform:
    transform: Callable[[Any], Any] = None
    dtype: type | torch.dtype = None


from torch.nn.utils.rnn import pad_sequence


class collate_func_modalities:
    def __init__(self, device: str):
        self.device = device

    def __call__(self, samples: List[Any]):
        batch = {}
        for sample in samples:
            for key, val in sample.items():
                if key not in batch:
                    batch[key] = []
                batch[key].append(val)

        for key, val in batch.items():
            batch[key] = pad_sequence(val, batch_first=True)

        return batch


class collate_func_helper:
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

        self.tokenizers = TokenizersHandler()

        if self.return_tensors:
            self.batch_kwargs["return_tensors"] = self.return_tensors

    def __call__(self, samples: List[Sample]) -> Batch:

        batch = Batch(samples, **self.batch_kwargs)
        return batch

        # TODO: this might need to be a part of the dataset
        # batch = Batch(samples, **self.batch_kwargs)
        # for i, sample in enumerate(batch.samples):
        #     sample.data = self.fix_prop(sample.data)
        #     sample.target = self.fix_prop(sample.target)
        #     sample.masks = {k: self.fix_prop(v) for k, v in sample.masks.items()}

        # return batch

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

    def use_tokenizers(self):
        pass
