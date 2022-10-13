
import torch
import torch.nn as nn


# similar to https://github.com/kzl/decision-transformer/blob/master/gym/decision_transformer/models/decision_transformer.py


class DecisionEmbedding(nn.Module):
    def __init__(self, max_ep_len: int, model_dim: int = 768, **kwargs) -> None:
        super().__init__()
        self.model_dim = model_dim

        self.embed_timestep = nn.Embedding(max_ep_len, model_dim)
        self.embed_return = torch.nn.Linear(1, model_dim)
        self.embed_state = torch.nn.Linear(self.state_dim, model_dim)
        self.embed_action = torch.nn.Linear(self.act_dim, model_dim)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return data


class DecisionOutput(nn.Module):
    def __init__(self, model_dim: int = 768, **kwargs) -> None:
        super().__init__()
        self.model_dim = model_dim

        self.embed_ln = nn.LayerNorm(model_dim)

        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(model_dim, self.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(model_dim, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        self.predict_return = torch.nn.Linear(model_dim, 1)
