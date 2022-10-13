import math
from typing import Tuple

import torch
import torch.nn as nn
import torchvision
from torchvision.models._utils import IntermediateLayerGetter

from generalist.data_types.input_types import ImageType, GeneralizedTensor
from generalist.models.embedding.resnet_embedding import build_backbone
from generalist.models.embedding.image_positional_embeddings import LearnedPositionalEmbeddings, PositionEmbeddingSine
from generalist.generalist_tokenizers.image_tokenizers import normalize_image

from einops import repeat

data_type = ImageType.data_type


def calculate_dims(img_size: torch.Size | torch.Tensor, patch_size: int) -> Tuple[int, int]:
    if isinstance(img_size, torch.Size):
        img_size = torch.as_tensor(img_size)

    model_dim = (patch_size**2) * img_size[0]
    seq_length = img_size.prod() / model_dim
    return seq_length, model_dim


class PatchEmbeddings(nn.Module):
    """
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/vit/__init__.py
    """

    def __init__(self, model_dim: int, patch_size: int, in_channels: int, batch_first: bool = True):
        """
        * `model_dim` is the transformer embeddings size
        * `patch_size` is the size of the patch
        * `in_channels` is the number of channels in the input image (3 for rgb)
        """
        super().__init__()
        self.model_dim = model_dim
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.batch_first = batch_first

        # We create a convolution layer with a kernel size and and stride length equal to patch size.
        # This is equivalent to splitting the image into patches and doing a linear
        # transformation on each patch.
        # self.conv = nn.Conv2d(in_channels, model_dim, patch_size, stride=patch_size)
        # self.conv = nn.Conv2d(400, model_dim, patch_size, stride=patch_size)
        self.conv = nn.LazyConv2d(out_channels=model_dim, kernel_size=patch_size, stride=patch_size)

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
            # Rearrange to shape `[patches, batch_size, model_dim]`
            x = x.permute(2, 3, 0, 1)
            x = x.view(h * w, bs, c)

            return x


class ImageEmbeddingPath(nn.Module):
    data_type = ImageType.data_type

    def __init__(self, model_dim: int = 768, patch_size: int = 16, in_channels: int = 3):
        super().__init__()

        self.patch_size = patch_size
        self.model_dim = model_dim

        self.positional_embeddings = LearnedPositionalEmbeddings(model_dim=model_dim)
        self.cls_token_emb = nn.Parameter(torch.randn(1, 1, model_dim), requires_grad=True)

    def forward(self, data: GeneralizedTensor):
        embeddings = data + self.positional_embeddings(data)

        cls_tokens = repeat(self.cls_token_emb, "1 1 d -> b 1 d", b=len(embeddings))
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        embeddings = GeneralizedTensor(embeddings)

        embeddings.set_data_type(self.data_type)

        return embeddings


class ImageBackbone(nn.Module):
    data_type = ImageType.data_type

    def __init__(self, model_dim: int = 768, patch_size: int = 16, in_channels: int = 3) -> None:
        super().__init__()
        self.backbone = build_backbone(hidden_dim=model_dim)

    def forward(self, image: torch.Tensor, mask: torch.Tensor = None):
        if mask is None:
            mask = torch.ones_like(image)

        out, pos = self.backbone(image, mask)
        return out


class ImagePathConv(nn.Module):
    data_type = ImageType.data_type

    def __init__(self, model_dim: int = 768, patch_size: int = 16, in_channels: int = 3):
        super().__init__()

        self.patch_size = patch_size
        self.model_dim = model_dim

        self.cls_token_emb = nn.Parameter(torch.randn(1, 1, model_dim), requires_grad=True)
        self.proj = nn.Conv2d(in_channels=in_channels, out_channels=model_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(model_dim)
        self.positional_embeddings = LearnedPositionalEmbeddings(model_dim=model_dim)

    def forward(self, x: torch.Tensor):
        if x.ndim == 3:
            x = x.unsqueeze(0)

        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x += self.positional_embeddings(x)
        return GeneralizedTensor(x).set_data_type(self.data_type)


class ImagePath(nn.Module):
    data_type = ImageType.data_type

    def __init__(self, model_dim: int = 768, patch_size: int = 16, in_channels: int = 3):
        super().__init__()

        self.patch_size = 16
        self.patch_embeddings = PatchEmbeddings(model_dim=model_dim, patch_size=patch_size, in_channels=in_channels)
        self.positional_embeddings = LearnedPositionalEmbeddings(model_dim=model_dim)
        # self.cls_token_emb = nn.Parameter(torch.randn(1, 1, model_dim), requires_grad=True)

    def forward(self, x: torch.Tensor):
        x = self.patch_embeddings(x)
        x = normalize_image(x)
        x += self.positional_embeddings(x)
        # return x

        # i dont know if we need this?  and it makes the dims wrong
        # cls_token_emb = self.cls_token_emb.expand(-1, x.shape[1], -1)
        # x = torch.cat([cls_token_emb, x])
        return GeneralizedTensor(x).set_data_type(self.data_type)


class TorchvisionPretrained(nn.Module):
    data_type = ImageType.data_type

    def __init__(self, name: str = "resnet101") -> None:
        super().__init__()
        self.dilation = True
        model_ = torchvision.models.resnet101(replace_stride_with_dilation=[False, False, self.dilation], pretrained=True)
        return_layers = {"layer4": "0"}
        self.body = IntermediateLayerGetter(model_, return_layers=return_layers)
        self.pos_enc = PositionEmbeddingSine()

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        if mask is None:
            c, h, w = x.shape
            mask = torch.ones((c, h, w), dtype=torch.bool, device=x.device)
        pos = self.pos_enc(x, mask)
        breakpoint()
        if x.ndim == 3:
            x = x.unsqueeze(0)
        x = self.body(x)["0"]
        x = torch.cat([x, pos], dim=1)
        return x
