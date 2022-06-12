import argparse
from typing import Any, Dict, List
from torch.utils.data import Dataset

import os
import json

from config import AOKVQA_DIR, COCO_DIR

from dataclasses import dataclass


@dataclass
class AokvqaInstance:
    split: str
    image_id: int
    question_id: str
    question: str
    choices: List[str]
    correct_choice_idx: int
    direct_answers: List[str]
    difficult_direct_answer: bool
    rationales: List[str]

    # OTHER
    aokvka_dir: str = AOKVQA_DIR
    coco_dir: str = COCO_DIR

    @property
    def image_path(self):
        filepath = f"{self.coco_dir}/{self.split}2017/{self.image_id:012}.jpg"
        return filepath

    def inputs(self):
        return [self.image_path]

    def label(self):
        return self.direct_answers[self.correct_choice_idx]


def load_aokvqa(aokvqa_dir: str = AOKVQA_DIR, split: str = "train", version="v1p0"):
    assert split in ["train", "val", "test", "test_w_ans"]
    dataset = json.load(open(os.path.join(aokvqa_dir, f"aokvqa_{version}_{split}.json")))
    return dataset


# OLD
def _get_coco_path(split: str, image_id: str, coco_dir: str = COCO_DIR):
    return os.path.join(coco_dir, f"{split}2017", f"{image_id:012}.jpg")


class AokvqaDataset(Dataset):
    def __init__(self, aokvqa_dir: str = AOKVQA_DIR, split: str = "train", version="v1p0"):
        self.aokvqa_dir = aokvqa_dir
        self.split = split
        self.version = version
        self.dataset = load_aokvqa(aokvqa_dir, split, version)

        self.dataset = [AokvqaInstance(**instance) for instance in self.dataset]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
