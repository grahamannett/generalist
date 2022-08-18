import json

from pathlib import Path


import torch
import numpy as np

from generalist.generalist_datasets.base import GeneralistDataset
from generalist.generalist_datasets.image_datasets import ImageDatasetMixin
from generalist.data_types.input_types import ImageType, TextTypeRaw
from generalist.data_types.helper_types import Sample
from torchvision import transforms


_train_transform = transforms.Compose(
    [
        transforms.ColorJitter(brightness=[0.5, 1.3], contrast=[0.8, 1.5], saturation=[0.2, 1.5]),
        transforms.RandomHorizontalFlip(),
        # transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

_val_transform = transforms.Compose(
    [
        # transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

_transforms = {
    "train": _train_transform,
    "val": _val_transform,
}


class CocoDataset(ImageDatasetMixin, GeneralistDataset):
    def __init__(self, coco_dir: str = None, split: str = "train", **kwargs) -> None:
        assert split in ["train", "test", "val"]

        super().__init__()
        coco_dir = Path(coco_dir)
        self.coco_dir = coco_dir
        self.split = split

        self.img_dir = coco_dir / f"{split}2017"
        self.captions = coco_dir / "annotations" / f"captions_{split}2017.json"
        self.instances = coco_dir / "annotations" / f"instances_{split}2017.json"
        self.person_keypoints = coco_dir / "annotations" / f"person_keypoints_{split}2017.json"

        self.captions_data = json.load(open(self.captions))

        self.image_transform = _transforms[self.split]

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
            caption = self.image_annotation.get(image_id, None)

            self._dataset.append(
                {
                    "image_id": image_id,
                    "image_path": str(image_path),
                    "caption": caption,
                }
            )

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx: int, **kwargs) -> Sample:
        sample = super().__getitem__(idx, **kwargs)
        item = self._dataset[idx]

        image = ImageType(self.read_image(item["image_path"]))
        image.resize_image((320, 320))
        image = image / 255.0
        data = self.image_transform(image)
        target = TextTypeRaw(item["caption"]["caption"])

        target = target.tokenize()
        data = data.tokenize()
        # target = target.tokenize(self.tokenizers["text"])

        sample.data, sample.target = data, target

        # self.process_sample(sample)
        return sample
