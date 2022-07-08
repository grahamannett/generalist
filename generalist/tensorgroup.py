import torch
from typing import Sequence

class TensorGroup:
    def __init__(self, tensors: Sequence[torch.Tensor]) -> None:
        self.tensors = tensors