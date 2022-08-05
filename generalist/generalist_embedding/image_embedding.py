import torch
import torch.nn as nn
from generalist.generalist_tokenizers.general_embedding import GenearlizedTensor
from generalist.generalist_tokenizers.input_types import ImageType


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

    def __init__(self, d_model: int = 704, patch_size: int = 16, in_channels: int = 3):
        super().__init__()

        self.patch_size = patch_size
        self.d_model = d_model

        self.positional_embeddings = LearnedPositionalEmbeddings(d_model=d_model)
        self.cls_token_emb = nn.Parameter(torch.randn(1, 1, d_model), requires_grad=True)

    def forward(self, data: GenearlizedTensor):
        # embeddings = self.positional_embeddings(data.tokens)
        embeddings = self.positional_embeddings(data)
        # return GenearlizedTensor(embedding=embeddings, data_type=self.data_type)
        return GenearlizedTensor(embeddings).set_data_type(self.data_type)


class ImagePath(nn.Module):
    def __init__(self, d_model: int = 704, patch_size: int = 16, in_channels: int = 3):
        super().__init__()

        self.patch_size = 16
        self.patch_embeddings = PatchEmbeddings(
            d_model=d_model, patch_size=patch_size, in_channels=in_channels
        )
        self.positional_embeddings = LearnedPositionalEmbeddings(d_model=d_model)

        self.cls_token_emb = nn.Parameter(torch.randn(1, 1, d_model), requires_grad=True)

    def forward(self, x: torch.Tensor):
        # input_shape = x.shape
        x = self.patch_embeddings(x)
        x = self.normalize(x)
        x = self.positional_embeddings(x)
        return GenearlizedTensor(x)
        # i dont know if we need this?  and it makes the dims wrong
        # cls_token_emb = self.cls_token_emb.expand(-1, x.shape[1], -1)
        # x = torch.cat([cls_token_emb, x])
        # return GenearlizedTensor(hidden_states=x, input_shape=input_shape, output_shape=x.shape)


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
