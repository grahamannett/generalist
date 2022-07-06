import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from config import config
from generalist.generalist_tokenizers.input_types import ImageType, Sample, TextType
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms.functional import resize

from generalist.generalist_datasets.dataset_utils import GeneralistDataset


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
    aokvka_dir: str = config.AOKVQA_DIR
    coco_dir: str = config.COCO_DIR

    @property
    def image_path(self):
        filepath = f"{self.coco_dir}/{self.split}2017/{self.image_id:012}.jpg"
        return filepath

    def inputs(self):
        return [self.image_path]

    def label(self):
        return self.direct_answers[self.correct_choice_idx]


class AokvqaDataset(GeneralistDataset):
    def __init__(self, aokvqa_dir: str = config.AOKVQA_DIR, split: str = "train", version="v1p0"):
        super().__init__()
        self.aokvqa_dir = aokvqa_dir
        self.split = split
        self.version = version
        self.dataset = self.load_aokvqa(aokvqa_dir, split, version)

        self.dataset = [AokvqaInstance(**instance) for instance in self.dataset]

    @staticmethod
    def load_aokvqa(aokvqa_dir: str = config.AOKVQA_DIR, split: str = "train", version="v1p0"):
        assert split in ["train", "val", "test", "test_w_ans"]
        dataset = json.load(open(os.path.join(aokvqa_dir, f"aokvqa_{version}_{split}.json")))
        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        question = f"question: {sample.question} choices: {', '.join(sample.choices)}"
        answer = f"{sample.choices[sample.correct_choice_idx]}. {sample.rationales[0]}"

        # probably want to put this into a different transform type
        image = read_image(sample.image_path)
        image = self.image_transform(image)

        inputs = {
            "data": [ImageType(image), TextType(question)],
            "target": TextType(answer),
        }
        # return inputs

        sample = Sample(**inputs)
        return sample

    def image_transform(self, image: torch.Tensor):
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        image = resize(image, (320, 320))
        return image / 255.0

    def find_question(self, question: str):
        res = [sample for sample in self.dataset if question in sample.question]
        return res
