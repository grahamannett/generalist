from dataclasses import dataclass

from torchvision.io import read_image


@dataclass
class ImageType:
    image_path: str

    def __post_init__(self):
        self.image = read_image(self.image_path)
        self.image_size = self.image.size()


@dataclass
class TextType:
    text: str
