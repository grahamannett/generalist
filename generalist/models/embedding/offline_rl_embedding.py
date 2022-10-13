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

        self.embed_ln = nn.LayernNorm(model_dim)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        state_embeddings =self.embed_state(data.state)
        action_embeddings = self.embed_action(data.action)
        returns_embeddings= self.embed_return(data.returns)
        timestep_embeddings = self.embed_timestep(data.timestep)

        state_embeddings = state_embeddings + timestep_embeddings
        action_embeddings = action_embeddings + timestep_embeddings
        returns_embeddings = returns_embeddings + timestep_embeddings

        stacked_embeddings = torch.stack([returns_embeddings, state_embeddings, action_embeddings], dim=1)
        stacked_embeddings = self.embed_ln(stacked_embeddings)


class DecisionOutput(nn.Module):
    # https://github.com/kzl/decision-transformer/blob/master/gym/decision_transformer/models/decision_transformer.py
    def __init__(self, model_dim: int = 768, action_tanh: bool = True, **kwargs) -> None:
        super().__init__()
        self.model_dim = model_dim

        self.embed_ln = nn.LayerNorm(model_dim)

        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(model_dim, self.state_dim)
        self.predict_action = nn.Sequential(*([nn.Linear(model_dim, self.act_dim)] + ([nn.Tanh()] if action_tanh else [])))
        self.predict_return = torch.nn.Linear(model_dim, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pass