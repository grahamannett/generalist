import torch
import torch.nn as nn


def identity(x: torch.Tensor, **kwargs) -> torch.Tensor:
    return x


def reduce_cls(x: torch.Tensor, dim: int = 0, **kwargs) -> torch.Tensor:
    return x[:, dim]


def reduce_mean(x: torch.Tensor, dim: int = 1, **kwargs) -> torch.Tensor:
    return x.mean(dim=dim)


class GeneralOutput(nn.Module):
    # output dim of gato is 33024 but depends on tokenizer
    def __init__(self, model_dim: int = 768, output_dim: int = 33024, bias: bool = False) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(model_dim)
        self.linear_out = nn.Linear(in_features=model_dim, out_features=output_dim, bias=bias)

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.linear_out(self.layer_norm(hidden_states))


class GeneralClassificationOutput(GeneralOutput):
    reduce_dict = {
        "mean": reduce_mean,
        "cls": reduce_cls,
        None: identity,
    }

    def __init__(self, model_dim: int, num_classes: int = 10, reduce_type: str = "mean", bias: bool = False) -> None:
        super().__init__(model_dim=model_dim, output_dim=num_classes, bias=bias)

        self.num_classes = num_classes
        self.reduce_type = reduce_type

        if reduce_type not in self.reduce_dict:
            raise ValueError(f"reduce_type {reduce_type} not in {self.reduce_dict}")

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = self.reduce_dict[self.reduce_type](hidden_states)
        return self.output(hidden_states)


class GeneralImageOutput(GeneralOutput):
    def __init__(self, model_dim: int, output_dim: int = 768, bias: bool = False) -> None:
        super().__init__(model_dim=model_dim, output_dim=output_dim, bias=bias)

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.output(hidden_states)
