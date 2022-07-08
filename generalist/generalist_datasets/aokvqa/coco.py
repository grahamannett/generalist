from pathlib import Path
import pickle
import sys
from typing import Tuple

import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer


class CocoDataset(Dataset):
    def __init__(self, coco_dir: str, split: str = "train") -> None:
        coco_dir = Path(coco_dir)
        super().__init__()

        self.coco_dir = coco_dir
        self.split = split

        self.img_dir = coco_dir / f"{split}2017"
        self.captions = coco_dir / f"captions_{split}2017.json"
        self.instances = coco_dir / f"instances_{split}2017.json"
        self.person_keypoints = coco_dir / f"person_keypoints_{split}2017.json"
