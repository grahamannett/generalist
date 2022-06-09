from generalist.generalist_tokenizers.general_encoded import GeneralEncoded
import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2PreTrainedModel,
    GPT2Model,
    GPT2Block,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)


class TransformerDecoder(GPT2Model):
    # _keys_to_ignore_on_load_unexpected = ["wte", r"wpe"]

    def __init__(self, config) -> None:
        super().__init__(config)

        self.embed_dim = config.hidden_size

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def _get_output_shape(self, input_shape: torch.Tensor, hidden_states):
        return input_shape + (hidden_states.size(-1),)

    def forward(self, general_encoded: GeneralEncoded, **kwargs):
        output_attentions = kwargs.get("output_attentions", self.config.output_attentions)
        output_hidden_states = kwargs.get("output_hidden_states", self.config.output_hidden_states)
        use_cache = kwargs.get("use_cache", self.config.use_cache)
        head_mask = self.get_head_mask(kwargs.get("head_mask", None), self.config.n_layer)

        past_length = 0
        past_key_values = tuple([None] * len(self.h))

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None

        encoder_hidden_states = None
        encoder_attention_mask = None

        hidden_states = general_encoded.hidden_states
        attention_mask = general_encoded.attention_mask

        output_shape = getattr(
            general_encoded,
            "output_shape",
            self._get_output_shape(general_encoded.input_shape, general_encoded.hidden_states),
        )

        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return hidden_states


def return_previous_output(
    hidden_states,
    presents,
    all_hidden_states,
    all_self_attentions,
    all_cross_attentions,
    return_dict: bool = False,
):
    if not return_dict:
        return tuple(
            v
            for v in [
                hidden_states,
                presents,
                all_hidden_states,
                all_self_attentions,
                all_cross_attentions,
            ]
            if v is not None
        )
    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,
        past_key_values=presents,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
        cross_attentions=all_cross_attentions,
    )
