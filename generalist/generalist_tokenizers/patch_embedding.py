import torch
import torch.nn as nn


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
        # Apply convolution layer
        x = self.conv(x)

        # Get the shape.
        bs, c, h, w = x.shape
        # Rearrange to shape `[patches, batch_size, d_model]`

        if self.batch_first:
            return x.view(bs, h * w, c)
        else:
            x = x.permute(2, 3, 0, 1)
            x = x.view(h * w, bs, c)

            # Return the patch embeddings
            return x


class LearnedPositionalEmbeddings(nn.Module):
    """
    <a id="LearnedPositionalEmbeddings"></a>
    ## Add parameterized positional encodings
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
        # Add to patch embeddings and return
        return x + pe


class ImageEmbedding(nn.Module):
    def __init__(self, d_model: int = 768, patch_size: int = 16, in_channels: int = 3):
        super().__init__()
        self.patch_embeddings = PatchEmbeddings(
            d_model=d_model, patch_size=patch_size, in_channels=in_channels
        )
        self.positional_embeddings = LearnedPositionalEmbeddings(d_model=d_model)

        self.cls_token_emb = nn.Parameter(torch.randn(1, 1, d_model), requires_grad=True)

    def forward(self, x: torch.Tensor):
        x = self.patch_embeddings(x)
        x = self.positional_embeddings(x)

        cls_token_emb = self.cls_token_emb.expand(-1, x.shape[1], -1)
        x = torch.cat([cls_token_emb, x])
        return x


# class ImageEmbedding(nn.Module):
#     """this one was meant to be most similar to gato but not sure what groups and various params"""

#     _type = "image"

#     def __init__(self, embed_dim: int = 196, kernel_size: int = 1, num_groups: int = 2) -> None:
#         super().__init__()
#         self.embed_dim = embed_dim
#         # self.num_channels =

#         # self.pte = nn.Embedding(patch_size, embed_dim)

#         # self.gn1 = nn.GroupNorm(32, 32)

#         self.group_norm_1 = nn.GroupNorm(num_groups=num_groups, num_channels=embed_dim)
#         self.gelu_1 = nn.GELU()
#         self.conv_1 = nn.LazyConv2d(1, kernel_size=kernel_size)

#         self.group_norm_2 = nn.GroupNorm(num_groups=num_groups, num_channels=embed_dim)
#         self.gelu_2 = nn.GELU()
#         self.conv_2 = nn.LazyConv2d(1, kernel_size=kernel_size)

#     def forward(self, x: torch.Tensor):
#         # x = x.transpose(-1, -2)

#         # breakpoint()
#         x_ = self.group_norm_1(x)
#         x_ = self.gelu_1(x_)
#         x_ = self.conv_1(x_)
#         # breakpoint()

#         x_ = self.group_norm_2(x_)
#         x_ = self.gelu_2(x_)
#         x_ = self.conv_2(x_)

#         x = x + x_
#         # breakpoint()
#         return x


# class PatchEmbedResNet:
#     def __init__(self, pretrained_model_name_or_path: str = "microsoft/resnet-50") -> None:
#         self.feature_extractor = AutoFeatureExtractor.from_pretrained(
#             pretrained_model_name_or_path=pretrained_model_name_or_path
#         )

#     def __call__(self, x):
#         x = self.feature_extractor(x)
#         return x


# # Adapted from https://amaarora.github.io/2021/01/18/ViT.html
# class PatchEmbed(nn.Module):
#     def __init__(
#         self,
#         img_size: int = 224,
#         patch_size: int = 16,
#         in_channels: int = 3,
#         embed_dim: int = 768,
#         dilation: int = 1,
#     ):
#         super().__init__()
#         num_patches = (img_size // patch_size) * (img_size // patch_size)
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.num_patches = num_patches
#         self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

#     def forward(self, x):
#         x = self.proj(x).flatten(2).transpose(1, 2)
#         return x
