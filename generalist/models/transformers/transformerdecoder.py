import torch
import torch.nn as nn

from .layers import Block


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        n_layer: int,
        embed_dim: int,
        num_heads: int,
        attn_pdrop: float,
        resid_pdrop: float,
        block_size: int,
    ) -> None:
        super().__init__()

        self.transformer = nn.ModuleDict(
            dict(
                h=nn.ModuleList(
                    [
                        Block(
                            embed_dim=embed_dim,
                            num_heads=num_heads,
                            attn_pdrop=attn_pdrop,
                            resid_pdrop=resid_pdrop,
                            block_size=block_size,
                        )
                        for _ in range(n_layer)
                    ]
                ),
                ln_f=nn.LayerNorm(embed_dim),
            )
        )

        self.model_max_length = block_size

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:

        for block in self.transformer.h:
            hidden_states = block(hidden_states)

        hidden_states = self.transformer.ln_f(hidden_states)
        return hidden_states


if __name__ == "__main__":
    # sa = CausalSelfAttention(512, 4, 0.1)
    # msa = nn.MultiheadAttention(512, 4, dropout=0.1)
    model = TransformerDecoder(
        n_layer=2, embed_dim=512, num_heads=4, attn_pdrop=0.1, resid_pdrop=0.1, block_size=1024
    )
    x = torch.randn(1, 10, 512)
    out = model(x)
    breakpoint()
