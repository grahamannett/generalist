import torch
from dataclasses import dataclass


@dataclass
class GeneralEncoded:
    hidden_states: torch.Tensor
    attention_mask: torch.Tensor = None
    output_shape: torch.Tensor = None
    encoder_hidden_states: torch.Tensor = None
    encoder_attention_mask: torch.Tensor = None
    use_cache: bool = True
    output_attentions: bool = False

    input_shape: torch.Tensor = None

    def asdict(self):
        return self.__dict__
