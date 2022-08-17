from datasets import load_dataset
from generalist.generalist_datasets.base import GeneralistDataset
from generalist.generalist_datasets.dataset_utils import DatasetRegistry
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

    def __init__(self, dataset_name: str = "xsum", split: str = "train", **kwargs) -> None:
        super().__init__(**kwargs)

        self.dataset_name = dataset_name
        self.split = split
        self._dataset = load_dataset(self.dataset_name)[split]

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int, **kwargs) -> Sample:
        sample = super().__getitem__(idx, **kwargs)

        item = self._dataset[idx]
        sample.data = TextType(item["document"])
        sample.target = TextType(item["summary"])
        return sample


@DatasetRegistry.register
class LanguageModelingDataset(GeneralistDataset):
    shortname = "hf_language_modeling"

    def __init__(
        self,
        dataset_path: str = "wikitext",
        dataset_name: str = "wikitext-103-raw-v1",
        split: str = "train",
        return_raw: bool = True,
        **kwargs
    ) -> None:
        super().__init__(return_raw, **kwargs)

        self._dataset_path = dataset_path
        self._dataset_name = dataset_name
        self.split = split
        self._dataset = load_dataset(dataset_path, dataset_name)[split]

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int, **kwargs) -> Sample:
        sample = super().__getitem__(idx, **kwargs)
        item = self._dataset[idx]
        sample.data = TextType(item["text"])
        sample.target = TextType(item["text"])
        return sample
