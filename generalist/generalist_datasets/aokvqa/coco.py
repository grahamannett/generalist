from pathlib import Path
import pickle
import sys
from typing import Tuple

import torch
from torch.utils.data import Dataset


import json

from generalist.generalist_tokenizers.input_types import ImageType, Sample


class CocoDataset(Dataset):
    def __init__(self, coco_dir: str, split: str = "train") -> None:
        coco_dir = Path(coco_dir)
        super().__init__()

        self.coco_dir = coco_dir
        self.split = split

        self.img_dir = coco_dir / f"{split}2017"
        self.captions = coco_dir / "annotations" / f"captions_{split}2017.json"
        self.instances = coco_dir / "annotations" / f"instances_{split}2017.json"
        self.person_keypoints = coco_dir / "annotations" / f"person_keypoints_{split}2017.json"

        self.captions_data = json.load(open(self.captions))

        # these other ones only have segmentation maps
        # self.instances_data = json.load(open(self.instances))
        # self.person_keypoints_data = json.load(open(self.person_keypoints))
        self.process()

    def process(self) -> None:
        self.image_annotation = {obj["image_id"]: obj for obj in self.captions_data["annotations"]}

        self._dataset = []
        for image_info in self.captions_data["images"]:
            image_id = image_info["id"]
            image_path = self.img_dir / image_info["file_name"]
            caption = self.image_annotations.get(image_id, None)

            self._dataset.append({
                "image_id": image_id,
                "image_path": str(image_path),
                "caption": caption,
            })

    def __getitem__(self, idx: int, **kwargs) -> Sample:
        sample = super().__getitem__(idx, **kwargs)

        item = self._dataset[idx]
        image = item["image_path"]
        sample.data = ImageType()


dataset = CocoDataset(coco_dir="/Users/graham/code/boise_research_group/generalist/data/coco", split="train")
