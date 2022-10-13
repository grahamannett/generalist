from torch.utils.data import Dataset, IterableDataset

from typing import Any, Sequence
from generalist.generalist_tokenizers.general_tokenizer import GeneralTokenizer
from generalist.data_types.helper_types import Sample, SampleMetaData


class ChainedDataset(Dataset):
    def __init__(self, datasets: Sequence[Dataset], sample_weights: Sequence[float] = None, **kwargs) -> None:
        if sample_weights is not None:
            assert sum(sample_weights) == 1.0, "sample_weights must sum to 1.0"

        super().__init__(**kwargs)
        self._datasets = datasets

        self.sample_weights = sample_weights

        # self._lengths = [len(dataset) for dataset in self._datasets]
        # self._lengths_idx = [sum(self._lengths[:i]) for i in range(len(self._lengths))]

        # #TODO fix this since its bad and slow, but couldnt figure out indexing good way simply
        self._idx_dataset = []

        for d_i, dataset in enumerate(self._datasets):
            for s_i in range(len(dataset)):
                self._idx_dataset.append((d_i, s_i))


    def __len__(self) -> int:
        # return
        return len(self._idx_dataset)

    def __getitem__(self, index: int) -> Sample:
        d_i, s_i = self._idx_dataset[index]
        return self._datasets[d_i][s_i]

        # _dataset_idx = 0
        # for i, length in enumerate(self._lengths):
        #     if index < self._lengths_idx[i]:
        #         break
        #     _dataset_idx += i

        # relative_idx = self._lengths_idx[index - self._lengths_idx[i]]
        # return self._datasets[i].__getitem__(relative_idx)
