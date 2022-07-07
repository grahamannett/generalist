import torch


class GeneralTokenizer:
    def __init__(self, device: str = "cpu", **kwargs):
        self.device = device

    def fix_device(self, prop: str) -> None:
        tensor = getattr(self, prop, None)
        if isinstance(tensor, torch.Tensor):
            setattr(self, prop, tensor.to(self.device))
