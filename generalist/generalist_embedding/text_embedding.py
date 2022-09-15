from typing import Any

import torch

from generalist.data_types.input_types import GeneralizedTensor, TextTypeRaw, TextType
from transformers import GPT2PreTrainedModel

import torch.nn as nn

# from generalist.utils.device import get_device


class TextEmbeddingPath(nn.Module):
    data_type = TextType.data_type

    def __init__(self, device: str = None, **kwargs) -> None:
        super().__init__()

        self.embedder = TextEmbedding.from_pretrained("gpt2")
        self.device = device
        # self.device = get_device() if device is None else device

    def forward(self, data: Any) -> torch.Tensor:
        embedding = self.embedder(data)
        return embedding.set_data_type(self.data_type)


class TextEmbedding(GPT2PreTrainedModel):
    # unclear if i should use GPT2Model or GPT2PreTrainedModel for weights
    _keys_to_ignore_on_load_unexpected = ["h", "ln_f"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.max_position_embeddings, config.n_embd)

        self.drop = nn.Dropout(config.embd_pdrop)

        self.post_init()

    def forward(
        self,
        data: TextType,
    ):

        tokens = data
        input_shape = tokens.shape
        attention_mask = getattr(data, "attention_mask", None)
        token_type_ids = getattr(data, "token_type_ids", None)

        tokens = tokens.view(-1, input_shape[-1])
        batch_size = tokens.shape[0]

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        past_length = 0
        position_ids = torch.arange(
            past_length, input_shape[-1] + past_length, dtype=torch.long, device=tokens.device
        )

        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # UNSURE IF THIS SHOULD BE IN EMBEDDING OR IN TRANSFORMER
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch size should be positive")

            attention_mask = attention_mask.view(batch_size, -1)

            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        input_embeds = self.wte(tokens)
        position_embeds = self.wpe(position_ids)
        hidden_states = input_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)

        out = hidden_states
        return out

        # return GeneralEmbedding(
        #     embedding=hidden_states,
        #     attention_mask=attention_mask,
        #     output_shape=output_shape,
        #     input_shape=input_shape,
        # )
