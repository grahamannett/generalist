from dataclasses import dataclass
from typing import Any, List
from generalist.data_types.input_types import InputType

import torch


@dataclass
class SampleMetaData:
    idx: Any = None
    dataset_name: Any = None


class Sample:
    def __init__(self, data: List[InputType] = None, target: Any = None, metadata: SampleMetaData = None):
        self.data = data
        self.target = target
        self.metadata = metadata

    def __iter__(self):
        yield self.data, self.target

    def __repr__(self) -> str:
        string = f"Sample(data={self.data}, target={self.target}"
        if self.metadata is not None:
            string += f", metadata={self.metadata}"
        string += ")"
        return string


def _new_tensor_helper(tensor_subclass):
    def __new__(cls, *args):
        if isinstance(args[0], torch.Tensor):
            return tensor_subclass(args[0])
        return super(cls).__new__(cls)

    return __new__


class Batch:
    def __init__(self, samples: List[Sample] = None, device: str = None):
        self.samples = samples

        self.data = [s.data for s in self.samples]
        self.target = [s.target for s in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, key: int):
        return self.samples[key]

    def __iter__(self):
        return iter(self.samples)

    @classmethod
    def collate_fn(cls, samples: List[Sample]) -> "Batch":
        """collate_fn for torch.utils.data.DataLoader"""
        batch = cls(samples)
        return batch

    # def fix(self):
    #     self.data = [[d_.to(self.device) for d_ in d] for d in self.data]
    #     self.target = [d.to(self.device) for d in self.target]


# @dataclass
# class Batch:
#     samples: List[Sample] = None

#     def __post_init__(self):
#         # not sure which is quicker:
#         self.data = [s.data for s in self.samples]
#         self.target = [s.target for s in self.samples]

#     @classmethod
#     def collate_fn(cls, samples: List[Sample]) -> "Batch":
#         """collate_fn for torch.utils.data.DataLoader"""
#         batch = cls(samples)
#         return batch

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, key: int):
#         return self.samples[key]

#     def __iter__(self):
#         return iter(self.samples)