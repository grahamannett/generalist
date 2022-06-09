from typing import Text
import torch
from torch import nn

from torchvision.io import read_image
from transformers import XLNetTokenizer


from einops import rearrange
import math


class TextTokenizer:
    """

    Text is encoded via SentencePiece (Kudo and Richardson, 2018) with 32000 subwords into the integer range [0, 32000).
    """

    def __init__(self) -> None:
        self.tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
        self.return_tensors = "pt"

    def __call__(self, x: str) -> torch.Tensor:
        return self.tokenizer(x, return_tensors=self.return_tensors)


class ImageTokenizer:
    def __init__(self, p1: int = 16, p2: int = 16, upper_bound: int = 1, lower_bound: int = -1) -> None:
        self.p1 = p1  # h
        self.p2 = p2  # w
        # self.patch_size = self.p1 * self.p2
        self.patch_size = self.p1

        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

    def __call__(self, img: torch.Tensor):
        if img.ndim == 3:
            img = img.unsqueeze(0)

        # i think this is the p1xp2 raster order as described in vit
        img = rearrange(img, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=self.p1, p2=self.p2)

        img = ((img - img.max()) / (img.max() - img.min())) * (
            self.upper_bound - self.lower_bound
        ) + self.lower_bound

        # square root of patch size?
        img /= math.sqrt(self.patch_size)
        return img

    def img_to_patch(self, x, patch_size, flatten_channels=True):
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


# if __name__ == "__main__":

#     train_data = load_aokvqa(AOKVQA_DIR, "train")
#     dataset_example = train_data[0]

#     image_path = get_coco_path("train", dataset_example["image_id"], COCO_DIR)

#     img = read_image(image_path)

#     text_tokenizer = TextTokenizer()
#     img_tokenizer = ImageTokenizer()

#     patches1 = img_tokenizer.img_to_patch(img, 16)
#     patches2 = img_tokenizer.img_to_patch(img, 16, False)
#     patches3 = img_tokenizer(img)

#     breakpoint()
