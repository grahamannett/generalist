from datasets import load_dataset
from generalist.generalist_datasets.dataset_utils import GeneralistDataset, DatasetRegistry
from torch.utils.data import Dataset
from typing import Any

from generalist.generalist_tokenizers.input_types import Sample, TextType


class BaseHFDataset(Dataset):
    def __init__(self, dataset_name: str, *args, **kwargs):
        self.dataset_name = dataset_name


class QuestionAnswering(Dataset):
    shortname = "hf_qa"

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


@DatasetRegistry.register
class SummarizationDataset(GeneralistDataset):
    shortname = "hf_summarization"

    def __init__(self, dataset_name: str = "xsum", split: str = "train") -> None:
        super().__init__()

        self.dataset_name = dataset_name
        self.split = split
        self._dataset = load_dataset(self.dataset_name)[split]

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Any:
        item = self._dataset[idx]
        sample = Sample(data=TextType(item["document"]), target=TextType(item["summary"]))

        return self.process_sample(sample)
