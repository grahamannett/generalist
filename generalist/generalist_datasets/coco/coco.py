import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from generalist.data_types.helper_types import Sample
from generalist.data_types.input_types import ImageType, TextTypeRaw, TextType
from generalist.generalist_datasets.base import GeneralistDataset
from generalist.generalist_datasets.image_datasets import ImageDatasetMixin
from torchvision import transforms

from generalist.generalist_datasets.tasks_utils import TaskInterface


def fix_channels(image):
    # if greyscale, repeat channels
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    return image


# def normalize_image(image):
#     image = image / 255.0
#     return image

_splits_available = ["train", "val", "test"]


_train_transform = transforms.Compose(
    [
        transforms.Lambda(fix_channels),
        # transforms.Lambda(normalize_image),
        transforms.Resize((320, 320)),
        # transforms.ColorJitter(brightness=[0.5, 1.3], contrast=[0.8, 1.5], saturation=[0.2, 1.5]),
        transforms.RandomHorizontalFlip(),
        # transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

_val_transform = transforms.Compose(
    [
        # transforms.Lambda(normalize_image),
        transforms.Lambda(fix_channels),
        transforms.Resize((320, 320)),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

_transforms = {
    "train": _train_transform,
    "val": _val_transform,
}





class CocoDataset(ImageDatasetMixin, GeneralistDataset):
    shortname = "coco"

    def __init__(self, coco_dir: str = None, split: str = "train", **kwargs) -> None:
        assert split in ["train", "test", "val"]
        assert split != "test", "COCO test split is not available since it doesnt have captions/keypoints"

        super().__init__(**kwargs)
        self.split = split
        self.image_transform = _transforms[self.split]

        self.coco_dir = Path(coco_dir)
        self.img_dir = self.coco_dir / f"{split}2017"
        self.captions_path = self.coco_dir / "annotations" / f"captions_{split}2017.json"
        self.instances_path = self.coco_dir / "annotations" / f"instances_{split}2017.json"
        self.person_keypoints_path = self.coco_dir / "annotations" / f"person_keypoints_{split}2017.json"

        self.captions_data = json.load(open(self.captions_path))
        self._process_captions()
        # these other ones only have segmentation maps
        # self.instances_data = json.load(open(self.instances))
        # self.person_keypoints_data = json.load(open(self.person_keypoints))
        self.text_tokenizer_kwargs = {
            "return_tensors": "pt",
            "truncation": True,
            "padding": "max_length",
            "max_length": 32,
            "return_attention_mask": True,
        }

        # tasks available
        self.tasks = TaskInterface()

    def _process_captions(self) -> None:
        self._image_info = {}
        self.image_annotation = {}

        # there are multiple captions per image
        for annotation in self.captions_data["annotations"]:
            if annotation["image_id"] not in self.image_annotation:
                self.image_annotation[annotation["image_id"]] = []
            self.image_annotation[annotation["image_id"]].append(annotation)

        self._dataset = []
        for image_info in self.captions_data["images"]:
            if image_info["id"] not in self._image_info:
                self._image_info[image_info["id"]] = image_info
            else:
                raise ValueError("Duplicate image id! This should not happen.")

            image_id = image_info["id"]
            image_path = self.img_dir / image_info["file_name"]
            captions = self.image_annotation.get(image_id, None)

            self._dataset.append(
                {
                    "image_id": image_id,
                    "image_path": str(image_path),
                    "caption": captions,
                }
            )

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx: int, **kwargs) -> Sample:
        sample = super().__getitem__(idx, **kwargs)

        item = self._dataset[idx]
        self.extra_metadata(sample, item)

        image = self.read_image(item["image_path"])
        image = self.image_transform(ImageType(image))
        # pick from one of the captions
        _caption = random.choice(item["caption"])
        caption = TextTypeRaw(_caption["caption"])
        # instruction = TextTypeRaw(f"Describe this image.")

        # image = image.tokenize(tokenizer=self.tokenizers["image"])
        # caption_out = caption.tokenize(tokenizer=self.tokenizers["text"], **self.text_tokenizer_kwargs)

        sample.data = image
        sample.target = caption

        text_tokenizer_kwargs = {**self.text_tokenizer_kwargs, **kwargs.get("text_tokenizer_kwargs", {})}

        if not kwargs.get("raw_data", False):
            data = self.tokenizers.image(sample.data)
            sample.data = data

        if not kwargs.get("raw_target", False):
            _caption = f"{self.tasks.caption(caption.data)}"

            target = self.tokenizers.text.encode_plus(_caption, **text_tokenizer_kwargs)
            sample.target = TextType(target["input_ids"])
            sample.masks["target"] = target["attention_mask"]

        return sample

    def extra_metadata(self, sample: Sample, item: Dict[Any, Any]) -> dict:
        sample.metadata.item_path = item["image_path"]
        sample.metadata.image_id = item["image_id"]
