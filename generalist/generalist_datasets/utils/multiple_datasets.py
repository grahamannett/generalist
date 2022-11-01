from torch.utils.data import Dataset, IterableDataset

from typing import Any, Sequence
from generalist.generalist_tokenizers.general_tokenizer import GeneralTokenizer
from generalist.data_types.helper_types import Sample, SampleMetaData

import random


class CombinedDataset(Dataset):
    def __init__(self, datasets: Sequence[Dataset], sample_weights: Sequence[float] = None, **kwargs) -> None:
        if sample_weights is not None:
            assert sum(sample_weights) == 1.0, "sample_weights must sum to 1.0"

        super().__init__(**kwargs)
        self.datasets = datasets

        self.sample_weights = sample_weights

        # self._lengths = [len(dataset) for dataset in self._datasets]
        # self._lengths_idx = [sum(self._lengths[:i]) for i in range(len(self._lengths))]

        # #TODO fix this since its bad and slow, but couldnt figure out indexing good way simply
        self._idx_dataset = []

        for d_i, dataset in enumerate(self.datasets):
            for s_i in range(len(dataset)):
                self._idx_dataset.append((d_i, s_i))

    def __len__(self) -> int:
        # return
        return len(self._idx_dataset)

    def __getitem__(self, index: int) -> Sample:
        d_i, s_i = self._idx_dataset[index]
        return self.datasets[d_i][s_i]


class BatchUniformDatasetSampler:
    def __init__(self, datasets: CombinedDataset, batch_size: int, drop_last: bool = True, shuffle: bool = True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.drop_last = drop_last

        new_idxs = []

        curr_length = 0
        for dataset_idx in range(len(datasets.datasets)):
            filtered_items = [(item[0], item[1] + curr_length) for item in datasets._idx_dataset if item[0] == dataset_idx]
            random.shuffle(filtered_items)
            for group_end in range(self.batch_size, len(filtered_items), self.batch_size):
                new_idxs.append(filtered_items[group_end : group_end + self.batch_size])
            if len(new_idxs[-1]) < self.batch_size:
                popped = new_idxs.pop()
            curr_length += len(datasets.datasets[dataset_idx])

        random.shuffle(new_idxs)

        self.new_idxs = new_idxs

    def __len__(self, *args, **kwargs):
        return len(self.new_idxs)

    def __iter__(self, *args, **kwargs):
        for batch in self.new_idxs:
            dataset_idx = [item[1] for item in batch]
            yield dataset_idx
