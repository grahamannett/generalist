import torch
from typing import Sequence

# https://pytorch.org/docs/stable/notes/extending.html#extending-torch-with-a-tensor-like-type
# think i want to rename this but not sure what exactly
# also not sure if i should be using vmap from functorch
# this should be similar to a ragged tensor in tf or nested tensor in pytorch
class TensorGroup:
    def __init__(self, tensors: Sequence[torch.Tensor]) -> None:
        self.tensors = tensors
