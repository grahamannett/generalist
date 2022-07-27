from generalist.generalist_datasets.dataset_utils import GeneralistDataset
from torchvision import datasets, transforms

from generalist.generalist_tokenizers.input_types import ImageType, Sample, TextType


class ImageDataset(GeneralistDataset):
    def __init__(self):
        pass

    def __getitem__(self, index):
        pass


# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# dataset1 = datasets.MNIST("../data", train=True, download=True, transform=transform)

# dataset2 = datasets.MNIST("../data", train=False, transform=transform)


class MNISTDataset(GeneralistDataset):
    shortname = "mnist"

    def __init__(self, train: bool = True, out_channels: int = 1, **kwargs):
        super().__init__()
        self.out_channels = out_channels

        transform = kwargs.get("transform", self._default_transform())
        self.dataset = datasets.MNIST("../data", train=train, download=True, transform=transform)

    def _default_transform(self):
        transform = transforms.Compose(
            [
                # transforms.Resize(320),
                transforms.ToTensor(),
                # transforms.Normalize((0.1307,), (0.3081,))
            ]
        )
        return transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int, **kwargs) -> Sample:
        sample = super().__getitem__(idx, **kwargs)
        image, label = self.dataset[idx]

        if self.out_channels > 1:
            image = image.repeat(self.out_channels, 1, 1)

        return image, label

        # sample.data = [ImageType(image), TextType("what number is this?")]
        # sample.target = TextType(str(label))
        # return sample
