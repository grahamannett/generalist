from typing import Any
from torch.utils.data import Dataset


class GeneralistDataset(Dataset):
    def __init__(self, *args, **kwargs):
        pass

    def data_prepare(self, **kwargs) -> Any:
        return kwargs

    def __getitem__(self, index: int, **kwargs) -> Any:
        return self.data_prepare(**self.data[index])
