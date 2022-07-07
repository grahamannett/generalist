import torch
import torch.nn as nn
from config import device
from generalist.generalist_tokenizers.general_embedding import GeneralEmbedding, GeneralizedTokens
from generalist.generalist_tokenizers.input_types import ImageType
from generalist.generalist_tokenizers.tokenizer_utils import GeneralTokenizer

from einops import rearrange


def normalize_image(
    x: torch.Tensor, patch_size: int = 16, lower_bound: float = -1, upper_bound: float = 1
) -> torch.Tensor:
    x = ((x - x.max()) / (x.max() - x.min())) * (upper_bound - lower_bound) + lower_bound
    x /= patch_size ** (1 / 2)
    return x


class ImageTokenizer(GeneralTokenizer):
    data_type = ImageType.data_type

    def __init__(
        self, p1: int = 16, p2: int = 16, upper_bound: int = 1, lower_bound: int = -1, **kwargs
    ) -> None:

        super().__init__(**kwargs)
        self.p1 = p1  # h
        self.p2 = p2  # w
        # self.patch_size = self.p1 * self.p2
        self.patch_size = self.p1

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __call__(self, img: torch.Tensor):
        img = img.data if isinstance(img, ImageType) else img

        img = self.to_patches(img)
        img = normalize_image(img, self.patch_size, self.lower_bound, self.upper_bound)

        out = GeneralizedTokens(tokens=img, data_type=self.data_type)
        out.tokens = out.tokens.to(self.device)
        return out

    def to_patches(self, img: torch.Tensor):
        if img.ndim == 3:
            img = img.unsqueeze(0)
        img = rearrange(img, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=self.p1, p2=self.p2)
        return img


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

        return x + pe


class ImageEmbeddingPath(nn.Module):
    data_type = ImageType.data_type

    def __init__(self, d_model: int = 768, patch_size: int = 16, in_channels: int = 3, device: str = device):
        super().__init__()

        self.patch_size = patch_size
        self.d_model = d_model

        self.positional_embeddings = LearnedPositionalEmbeddings(d_model=d_model)
        self.cls_token_emb = nn.Parameter(torch.randn(1, 1, d_model), requires_grad=True)

    def forward(self, data: GeneralizedTokens):
        embeddings = self.positional_embeddings(data.tokens)
        return GeneralEmbedding(embedding=embeddings, data_type=self.data_type)


class ImagePath(nn.Module):
    def __init__(self, d_model: int = 768, patch_size: int = 16, in_channels: int = 3, device: str = device):
        super().__init__()
        self.device = device

        self.patch_size = 16
        self.patch_embeddings = PatchEmbeddings(
            d_model=d_model, patch_size=patch_size, in_channels=in_channels
        )
        self.positional_embeddings = LearnedPositionalEmbeddings(d_model=d_model)

        self.cls_token_emb = nn.Parameter(torch.randn(1, 1, d_model), requires_grad=True)

    def forward(self, x: torch.Tensor):
        x = x.to(self.device)
        input_shape = x.shape
        x = self.patch_embeddings(x)
        x = self.normalize(x)
        x = self.positional_embeddings(x)

        # i dont know if we need this?  and it makes the dims wrong
        # cls_token_emb = self.cls_token_emb.expand(-1, x.shape[1], -1)
        # x = torch.cat([cls_token_emb, x])
        return GeneralEmbedding(hidden_states=x, input_shape=input_shape, output_shape=x.shape)


def img_to_patch(x, patch_size: int = 16, flatten_channels: bool = True):
    """
    https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial15/Vision_Transformer.html
    Inputs:
        x - torch.Tensor representing the image of shape [B, C, H, W]
        patch_size - Number of pixels per dimension of the patches (integer)
        flatten_channels - If True, the patches will be returned in a flattened format
                        as a feature vector instead of a image grid.
    """
    if x.ndim == 3:
        x = x.unsqueeze(0)
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5)  # [B, H', W', C, p_H, p_W]
    x = x.flatten(1, 2)  # [B, H'*W', C, p_H, p_W]
    if flatten_channels:
        x = x.flatten(2, 4)  # [B, H'*W', C*p_H*p_W]
    return x
