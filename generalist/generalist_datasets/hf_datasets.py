from datasets import load_dataset
from generalist.generalist_datasets.dataset_utils import GeneralistDataset
from torch.utils.data import Dataset
from typing import Any


class BaseHFDataset(Dataset):
    def __init__(self, dataset_name: str, *args, **kwargs):
        self.dataset_name = dataset_name


class QuestionAnswering(Dataset):
    def __init__(self, name: str, split: str = "train"):
        self.dataset_name = name
        self.split = split
        self._dataset = load_dataset(self.dataset_name)[split]

    def load_dataset(self):
        raise NotImplementedError

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        return self._dataset[idx]


@GeneralistDataset.register("hf_summarization")
class SummarizationDataset(Dataset):
    def __init__(self, dataset_name: str = "xsum", split: str = "train") -> None:
        super().__init__()

        self.dataset_name = dataset_name
        self.split = split
        self._dataset = load_dataset(self.dataset_name)[split]

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Any:
        return self._dataset[idx]
