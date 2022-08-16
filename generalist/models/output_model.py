import torch
import torch.nn as nn


def reduce_dummy_(x: torch.Tensor, **kwargs) -> torch.Tensor:
    return x


def reduce_cls_(x: torch.Tensor, dim: int = 0, **kwargs) -> torch.Tensor:
    return x[:, dim]


def reduce_mean_(x: torch.Tensor, dim: int = 1, **kwargs) -> torch.Tensor:
    return x.mean(dim=dim)


class GeneralOutput(nn.Module):
    def __init__(self, model_dim: int = 768, output_dim: int = 33024, bias: bool = False) -> None:
        super().__init__()
        # self.output_dim = output_dim

        self.output = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, output_dim, bias=False))

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.output(hidden_states)


class GeneralClassificationOutput(GeneralOutput):
    reduce_dict = {
        "mean": reduce_mean_,
        "cls": reduce_cls_,
        None: reduce_dummy_,
    }

    def __init__(
        self, model_dim: int, num_classes: int = 10, reduce_type: str = "mean", bias: bool = False
    ) -> None:
        super().__init__(model_dim=model_dim, output_dim=num_classes, bias=bias)

        self.num_classes = num_classes
        self.reduce_type = reduce_type

        if reduce_type not in self.reduce_dict:
            raise ValueError(f"reduce_type {reduce_type} not in {self.reduce_dict}")

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = self.reduce_dict[self.reduce_type](hidden_states)
        return self.output(hidden_states)
