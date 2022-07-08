
from generalist.generalist_datasets.dataset_utils import GeneralistDataset
from generalist.generalist_tokenizers.input_types import Sample


class OfflineRLDataset(GeneralistDataset):
    def __init__(self, return_raw: bool = True, **kwargs) -> None:
        super().__init__(return_raw, **kwargs)

    def __getitem__(self, idx: int, **kwargs) -> Sample:
        sample = super().__getitem__(idx, **kwargs)
