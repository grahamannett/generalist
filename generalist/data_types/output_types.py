from typing import Callable
import torch


def get_output_types(targets):
    return set([type(target) for target in targets])


class LossFuncMixin:
    loss_func: Callable

    def get_loss(self, logits: torch.Tensor, targets: torch.Tensor):
        return self.loss_func(logits, targets)


class TargetTypeMixin:
    data: torch.Tensor


class SequenceOutput(LossFuncMixin, torch.Tensor):
    data: torch.Tensor


class ClassificationOutput(LossFuncMixin, torch.Tensor):
    data: torch.Tensor
