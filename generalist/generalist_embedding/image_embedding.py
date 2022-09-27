import math
from typing import Tuple

import torch
import torch.nn as nn
import torchvision

from generalist.data_types.input_types import ImageType, GeneralizedTensor
from generalist.generalist_tokenizers.image_tokenizers import normalize_image

from einops import repeat


def calculate_dims(img_size: torch.Size | torch.Tensor, patch_size: int) -> Tuple[int, int]:
    if isinstance(img_size, torch.Size):
        img_size = torch.as_tensor(img_size)

    d_model = (patch_size**2) * img_size[0]
    seq_length = img_size.prod() / d_model
    return seq_length, d_model


class PatchEmbeddings(nn.Module):
    """
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/vit/__init__.py
    """

    def __init__(self, d_model: int, patch_size: int, in_channels: int, batch_first: bool = True):
        """
        * `d_model` is the transformer embeddings size
        * `patch_size` is the size of the patch
        * `in_channels` is the number of channels in the input image (3 for rgb)
        """
        super().__init__()

        # We create a convolution layer with a kernel size and and stride length equal to patch size.
        # This is equivalent to splitting the image into patches and doing a linear
        # transformation on each patch.
        self.conv = nn.Conv2d(in_channels, d_model, patch_size, stride=patch_size)
        self.batch_first = batch_first

    def forward(self, x: torch.Tensor):
        """
        * `x` is the input image of shape `[batch_size, channels, height, width]`
        """

        if x.ndim == 3:
            x = x.unsqueeze(0)

        x = self.conv(x)

        bs, c, h, w = x.shape

        if self.batch_first:
            return x.view(bs, h * w, c)
        else:
            # Rearrange to shape `[patches, batch_size, d_model]`
            x = x.permute(2, 3, 0, 1)
            x = x.view(h * w, bs, c)

            return x


class LearnedPositionalEmbeddings(nn.Module):
    """
    This adds learned positional embeddings to patch embeddings.
    """

    def __init__(self, d_model: int, max_len: int = 5_000):
        """
        * `d_model` is the transformer embeddings size
        * `max_len` is the maximum number of patches
        """
        super().__init__()
        # Positional embeddings for each location
        self.positional_encodings = nn.Parameter(torch.zeros(max_len, 1, d_model), requires_grad=True)

    def forward(self, x: torch.Tensor):
        """
        * `x` is the patch embeddings of shape `[patches, batch_size, d_model]`
        """
        # Get the positional embeddings for the given patches

        pe = self.positional_encodings[x.shape[0]]

        if (scale_factor := (x.shape[-1] / pe.shape[-1])) != 1.0:
            pe = torch.nn.functional.interpolate(pe.unsqueeze(0), scale_factor=scale_factor).squeeze(0)

        return x + pe


class ImageEmbeddingPath(nn.Module):
    data_type = ImageType.data_type

    def __init__(self, d_model: int = 768, patch_size: int = 16, in_channels: int = 3):
        super().__init__()

        self.patch_size = patch_size
        self.d_model = d_model

        self.positional_embeddings = LearnedPositionalEmbeddings(d_model=d_model)
        self.cls_token_emb = nn.Parameter(torch.randn(1, 1, d_model), requires_grad=True)

    def forward(self, data: GeneralizedTensor):
        embeddings = self.positional_embeddings(data)

        cls_tokens = repeat(self.cls_token_emb, "1 1 d -> b 1 d", b=len(embeddings))
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        embeddings = GeneralizedTensor(embeddings)

        embeddings.set_data_type(self.data_type)

        return embeddings


class ImagePath(nn.Module):
    data_type = ImageType.data_type

    def __init__(self, d_model: int = 768, patch_size: int = 16, in_channels: int = 3):
        super().__init__()

        self.patch_size = 16
        self.patch_embeddings = PatchEmbeddings(d_model=d_model, patch_size=patch_size, in_channels=in_channels)
        self.positional_embeddings = LearnedPositionalEmbeddings(d_model=d_model)
        # self.cls_token_emb = nn.Parameter(torch.randn(1, 1, d_model), requires_grad=True)

    def forward(self, x: torch.Tensor):
        x = self.patch_embeddings(x)
        x = normalize_image(x)
        x = self.positional_embeddings(x)

        # i dont know if we need this?  and it makes the dims wrong
        # cls_token_emb = self.cls_token_emb.expand(-1, x.shape[1], -1)
        # x = torch.cat([cls_token_emb, x])
        return GeneralizedTensor(x).set_data_type(self.data_type)


class ResNetEmbedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.LazyConv2d(3, 1),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.LazyConv2d(3, 1),
            nn.LazyBatchNorm2d(),
        )

    def forward(self, x: torch.Tensor):
        x = self.block(x) + x
        return nn.functional.relu(x)


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


class TorchvisionPretrained(nn.Module):
    data_type = ImageType.data_type

    def __init__(self, name: str = "resnet101") -> None:
        super().__init__()
        self.dilation = True
        self.layer = torchvision.models.resnet101(replace_stride_with_dilation=[False, False, self.dilation], pretrained=True)
        self.pos_enc = PositionEmbeddingSine()

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        if mask is None:
            b, c, h, w = x.shape
            mask = torch.ones((c, h, w), dtype=torch.bool, device=x.device)
        pos = self.pos_enc(mask)
        x = self.layer(x)
        return x


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.

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

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
