from generalist.data_types.helper_types import Sample
from generalist.data_types.input_types import ImageType, TextType, TextTypeRaw
from generalist.generalist_datasets.image_datasets import ImageDatasetMixin
from generalist.generalist_datasets.base import GeneralistDataset

import torch
from torchvision import datasets, transforms


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
        # breakpoint()

        sample.data, sample.target = image_, target_encoded
        return sample
