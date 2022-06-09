import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2PreTrainedModel

from config import device

from generalist.generalist_tokenizers.general_encoded import GeneralEncoded


class TextEmbedding2(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"h", r"ln_f"]
    _keys_to_ignore_on_load_unexpected = [r"h"]

    def __init__(self, config) -> None:
        super().__init__(config)

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.max_position_embeddings, config.n_embd)

        self.drop = nn.Dropout(config.embd_pdrop)

        # breakpoint()
        # honestly cant tell if post_init is necessary
        # https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L971
        # self.post_init()

    def forward(self, input_ids, position_ids=None, token_type_ids=None, past=None, head_mask=None):
        return None


class TextEmbedding(nn.Module):
    _type = "text"

    def __init__(
        self, embed_dim: int = 768, vocab_size: int = 3200, max_position_embeddings: int = 1024
    ) -> None:
        super().__init__()

        self.wte = nn.Embedding(vocab_size, embed_dim)
        self.wtp = nn.Embedding(max_position_embeddings, embed_dim)
        self.drop = nn.Dropout(0.1)

        self.use_pretrained()

    def use_pretrained(self, pretrained_model_name_or_path: str = "gpt2") -> None:
        model = GPT2Model.from_pretrained(pretrained_model_name_or_path)

        self.config = model.config

        # layers and functions we need to copy from the pretrained model
        self.wte = model.wte
        self.wtp = model.wpe
        self.drop = model.drop
        self.dtype = model.dtype
        self.get_head_mask = model.get_head_mask
        print("using pretrained vals")

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
    ):
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        batch_size = input_ids.shape[0]

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

        input_embeds = self.wte(input_ids)
        position_embeds = self.wtp(position_ids)
        hidden_states = input_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)

        output = GeneralEncoded(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_shape=output_shape,
        )

        return output
