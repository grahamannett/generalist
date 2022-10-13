import torch


def fix_channels(image: torch.Tensor) -> torch.Tensor:
    # if greyscale, repeat channels
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    return image


def normalize_image(image: torch.Tesnsor) -> torch.Tensor:
    image = image / 255.0
    return image
