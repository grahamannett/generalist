import math
import torch
import torch.nn as nn


class LearnedPositionalEmbeddings(nn.Module):
    """
    This adds learned positional embeddings to patch embeddings.
    """

    def __init__(self, model_dim: int, max_len: int = 5_000):
        """
        * `model_dim` is the transformer embeddings size
        * `max_len` is the maximum number of patches
        """
        super().__init__()
        # Positional embeddings for each location
        self.positional_encodings = nn.Parameter(torch.zeros(max_len, 1, model_dim), requires_grad=True)

    def forward(self, x: torch.Tensor):
        """
        * `x` is the patch embeddings of shape `[patches, batch_size, model_dim]`
        """
        # Get the positional embeddings for the given patches

        pe = self.positional_encodings[x.shape[0]]

        if (scale_factor := (x.shape[-1] / pe.shape[-1])) != 1.0:
            pe = torch.nn.functional.interpolate(pe.unsqueeze(0), scale_factor=scale_factor).squeeze(0)

        return pe


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.

    https://github.com/facebookresearch/detr/blob/main/models/position_encoding.py
    https://github.com/saahiluppal/catr/blob/master/models/position_encoding.py
    """

    def __init__(self, num_pos_feats: int = 64, temperature: int = 10000, normalize: bool = False, scale: float = None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensors: torch.Tensor, mask: torch.Tensor):
        # x = tensor_list.tensors
        # mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class HeightWidthPositionEmbedding(nn.Module):
    """
    Absolute pos embedding, learned.  From https://github.com/saahiluppal/catr
    """

    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        h, w = images.shape[-2:]
        i = torch.arange(w, device=images.device)
        j = torch.arange(h, device=images.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = (
            torch.cat(
                [
                    x_emb.unsqueeze(0).repeat(h, 1, 1),
                    y_emb.unsqueeze(1).repeat(1, w, 1),
                ],
                dim=-1,
            )
            .permute(2, 0, 1)
            .unsqueeze(0)
            .repeat(images.shape[0], 1, 1, 1)
        )
        return pos
