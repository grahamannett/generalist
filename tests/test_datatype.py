from collections import UserDict
from typing import NamedTuple
import unittest

from generalist.generalist_tokenizers import text_path
from config import device

# class TestText(unitttest.TestCase):
#     device = device

#     def setUp(self):
#         self.path = text_path.TextEmbeddingPath(device=self.device)
#         self.embedding = text_path.TextEmbedding(device=self.device)
from dataclasses import dataclass

from torch.utils.data import Dataset, DataLoader


@dataclass
class DataInstance:
    val: int = 0

class ExDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

        self.vals = [DataInstance(val=i) for i in range(10)]

    def __getitem__(self, idx: int):
        return self.vals[idx]

    def __len__(self):
        return len(self.vals)

dataset = ExDataset()

val1 = dataset[0]
print(val1)

def _collate_fn(batch):
    return batch

dataloader = DataLoader(dataset, batch_size=3, shuffle=True, collate_fn=_collate_fn)

for batch_idx, batch in enumerate(dataloader):
    print(batch_idx, batch)
