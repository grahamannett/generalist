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

    def __init__(self, train: bool = True, **kwargs):
        super().__init__()
        self.transform = transforms.Compose(
            [transforms.Resize(320), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        self.dataset = datasets.MNIST("../data", train=train, download=True, transform=self.transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int, **kwargs) -> Sample:
        sample = super().__getitem__(idx, **kwargs)
        image, label = self.dataset[idx]
        image = image.repeat(3, 1, 1)

        sample.data = [ImageType(image), TextType("what number is this?")]
        sample.target = TextType(str(label))
        return sample
