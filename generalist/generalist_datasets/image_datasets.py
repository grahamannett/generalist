import torch
from generalist.generalist_datasets.base import GeneralistDataset
from generalist.data_types.input_types import ImageType, TextType, TextTypeRaw
from generalist.data_types.helper_types import Sample
from torchvision import datasets, transforms
from torchvision.io import read_image as _read_image


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


class MNISTDataset(ImageDatasetMixin, GeneralistDataset):
    shortname = "mnist"

    def __init__(self, train: bool = True, out_channels: int = 1, **kwargs):
        super().__init__(**kwargs)

        self.out_channels = out_channels

        self.feature_extractor = kwargs.get("feature_extractor", None)
        transform = kwargs.get("transform", self.transform_helper())

        self.dataset = datasets.MNIST("../data", train=train, download=True, transform=transform)

    def transform_helper(self):
        transform = self.default_image_transform()
        transform.transforms.append(transforms.Normalize((0.1307,), (0.3081,)))
        return transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int, **kwargs) -> Sample:
        sample = super().__getitem__(idx, **kwargs)
        image, label = self.dataset[idx]

        if self.out_channels > 1 and isinstance(image, torch.Tensor):
            image = image.repeat(self.out_channels, 1, 1)

        if self.feature_extractor:
            image = self.feature_extractor(image, return_tensors="pt")
            image = image["input_ids"].squeeze(0)

        # return image, label
        image_ = ImageType(image)

        image_.resize_image(320)
        image_ = image_.tokenize()

        target = TextTypeRaw(str(label))
        target_encoded = target.tokenize()
        breakpoint()

        sample.data, sample.target = image_, target_encoded
        return sample
