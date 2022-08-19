import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from config import config
from generalist.data_types.input_types import ImageType, TextType
from generalist.data_types.helper_types import Sample
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms.functional import resize

from generalist.generalist_datasets.base import GeneralistDataset


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
    aokvqa_dir: str = config.aokvqa_dir
    coco_dir: str = config.coco_dir

    @property
    def image_path(self):
        filepath = f"{self.coco_dir}/{self.split}2017/{self.image_id:012}.jpg"
        return filepath

    def inputs(self):
        return [self.image_path]

    def label(self):
        return self.direct_answers[self.correct_choice_idx]


class AokvqaDataset(GeneralistDataset):
    shortname = "aokvqa"

    def __init__(
        self, aokvqa_dir: str = AokvqaInstance.aokvqa_dir, split: str = "train", version="v1p0", **kwargs
    ):
        super().__init__(**kwargs)
        self.aokvqa_dir = aokvqa_dir
        self.split = split
        self.version = version
        self.dataset = self.load_aokvqa(aokvqa_dir, split, version)

        self.dataset = [AokvqaInstance(**instance) for instance in self.dataset]

    @staticmethod
    def load_aokvqa(aokvqa_dir: str = AokvqaInstance.aokvqa_dir, split: str = "train", version="v1p0"):
        assert split in ["train", "val", "test", "test_w_ans"]
        dataset = json.load(open(os.path.join(aokvqa_dir, f"aokvqa_{version}_{split}.json")))
        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int, **kwargs) -> Sample:
        sample = super().__getitem__(idx, **kwargs)
        item = self.dataset[idx]

        question = f"question: {item.question} choices: {', '.join(item.choices)}"
        answer = f"{item.choices[item.correct_choice_idx]}. {item.rationales[0]}"

        # probably want to put this into a different transform type
        image = self.image_transform(read_image(item.image_path))

        sample.data = [ImageType(image), TextType(question)]
        sample.target = TextType(answer)
        return sample

    def image_transform(self, image: torch.Tensor):
        # some images are greyscale and should probably convert ot rgb but this suffices
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        image = resize(image, (320, 320))
        return image / 255.0

    def find_question(self, question: str):
        res = [sample for sample in self.dataset if question in sample.question]
        return res
