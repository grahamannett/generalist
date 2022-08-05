import torch
import torch.nn as nn
from generalist.generalist_tokenizers.general_embedding import GenearlizedTensor
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

    def __call__(self, img: torch.Tensor | ImageType):
        img = img.data if isinstance(img, ImageType) else img

        img = self.to_patches(img)
        img = normalize_image(img, self.patch_size, self.lower_bound, self.upper_bound)

        out = GenearlizedTensor(img).set_data_type(self.data_type)
        return out

    def to_patches(self, img: torch.Tensor):
        if img.ndim == 3:
            img = img.unsqueeze(0)

        if img.shape[-1] % self.p1 != 0:
            raise ValueError(
                "Image height and width must be divisible by patch size. "
                f"Got {img.shape[-1]} and {self.p1}"
            )

        img = rearrange(img, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=self.p1, p2=self.p2)
        return img


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
