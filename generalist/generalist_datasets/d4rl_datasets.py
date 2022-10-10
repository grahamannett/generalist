from generalist.generalist_datasets.base import GeneralistDataset
from generalist.data_types.input_types import Sample

import torch


class OfflineRLDataset(GeneralistDataset):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.actions = []
        self.rewards = []
        self.states = []

        # to get it flat will be something like
        self.sequence = torch.vstack([self.actions, self.rewards, self.states]).T.reshape(-1)

    def __getitem__(self, idx: int, **kwargs) -> Sample:
        sample = super().__getitem__(idx, **kwargs)
