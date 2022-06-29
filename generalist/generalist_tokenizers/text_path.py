from typing import Any

import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2PreTrainedModel, XLNetTokenizer

from config import device

from generalist.generalist_tokenizers.general_embedding import (
    GeneralEmbedding,
    GeneralizedTokens,
)
from generalist.generalist_tokenizers.input_types import TextType


class TextTokenizer:
    """
    Text is encoded via SentencePiece (Kudo and Richardson, 2018) with 32000 subwords into the integer range [0, 32000).
    """

    data_type = TextType.data_type

    def __init__(
        self,
        padding: bool = True,
        tokenizer_class=XLNetTokenizer,
        pretrained_model_or_path: str = "xlnet-base-cased",
        model_input_length: int = 1024,
        **kwargs
    ) -> None:

        self.tokenizer = tokenizer_class.from_pretrained(
            pretrained_model_or_path, padding=padding, model_max_length=model_input_length
        )
        self.return_tensors = "pt"
        self.model_input_length = model_input_length

    def __call__(self, sample: str, **kwargs) -> torch.Tensor:
        x = sample.data if isinstance(sample, TextType) else sample

        encoded = self.encode(x, **kwargs)
        out = GeneralizedTokens(
            tokens=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            token_type_ids=encoded["token_type_ids"],
            data_type=self.data_type,
        )
        return out

    def encode(self, x: str, **kwargs) -> torch.Tensor:
        encoded = self.tokenizer(x, return_tensors=self.return_tensors, **kwargs)
        return encoded


class TextEmbeddingPath(nn.Module):
    data_type = TextType.data_type

    def __init__(self, device: str = device, **kwargs) -> None:
        super().__init__()

        self.embedder = TextEmbedding.from_pretrained("gpt2")

        self.device = device

    def forward(self, data: Any) -> torch.Tensor:
        data = self.embedder(data)
        return data


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
        data: GeneralizedTokens,
    ):

        tokens = data.tokens
        input_shape = tokens.shape
        attention_mask = getattr(data, "attention_mask", None)
        token_type_ids = getattr(data, "token_type_ids", None)
        # tokens: torch.Tensor,
        # token_type_ids: torch.Tensor = None,
        # attention_mask: torch.Tensor = None,
        # input_shape = tokens.size()
        tokens = tokens.view(-1, input_shape[-1])
        batch_size = tokens.shape[0]

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        past_length = 0
        position_ids = torch.arange(
            past_length, input_shape[-1] + past_length, dtype=torch.long, device=device
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

        return GeneralEmbedding(
            embedding=hidden_states,
            attention_mask=attention_mask,
            output_shape=output_shape,
            input_shape=input_shape,
        )
