from torchvision import transforms
from torchvision.io import read_image as _read_image

import torch


class ImageDatasetMixin:
    def default_image_transform(self):
        transform = transforms.Compose(
            [
                transforms.Resize(320),
                transforms.ToTensor(),
            ]
        )
        return transform

    def read_image(self, *args, **kwargs):
        return _read_image(*args, **kwargs)

    def generate_mask(self, image: torch.Tensor):
        mask = torch.ones_like(image)
        return mask
