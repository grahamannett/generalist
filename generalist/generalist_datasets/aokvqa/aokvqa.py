import argparse
from typing import Any, Dict, List
from torch.utils.data import Dataset

import os
import json

from config import AOKVQA_DIR, COCO_DIR

from dataclasses import dataclass


@dataclass
class AokvqaInstance(Dataset):
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
    def coco_path(self):
        filepath = f"{self.coco_dir}/{self.split}2017/{self.image_id:012}.jpg"
        return filepath


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aokvqa_dir", type=str, default=AOKVQA_DIR)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--version", type=str, default="v1p0")
    args = parser.parse_args()

    aokvqa_dataset = AokvqaDataset(args.aokvqa_dir, args.split, args.version)
    aokvqa_dataset_val = AokvqaDataset(args.aokvqa_dir, "val", args.version)
    # print(len(aokvqa_dataset))

    data1 = aokvqa_dataset[0]
    val1 = aokvqa_dataset_val[0]
    filepath1 = data1.coco_path
    filepath2 = val1.coco_path

    print(os.path.exists(filepath1))
    print(os.path.exists(filepath2))
    breakpoint()

# def load_data_super(cfg: argparse.Namespace, split: str, eval: bool = False):
#     features = vars(cfg).get(f"{split}_features", None)

#     return AokvqaDatasetSuper(
#         aokvqa_dir=cfg.aokvqa_dir,
#         split=split,
#         features=features,
#         prompt_with_choices=cfg.prompt_with_choices,
#         generation_target=cfg.generation_target,
#         prefix_length=cfg.prefix_length,
#         normalize_prefix=cfg.normalize_prefix,
#         gpt2_type="gpt2",
#         eval=eval,
#     )


# class AokvqaDatasetSuper(AokvqaDataset):
#     def __init__(self, **kwargs) -> None:
#         super().__init__(**kwargs)

#         self._aokvqa_set_kwargs = {
#             "aokvqa_dir": kwargs["aokvqa_dir"],
#             "split": kwargs["split"],
#         }

#     def __getitem__(self, i: int):
#         prefix, input_tokens, prompt_len, target_len = super().__getitem__(i)
#         out = {
#             "prefix": prefix,
#             "input_tokens": input_tokens,
#             "prompt_len": prompt_len,
#             "target_len": target_len,
#         }
#         return out

#     def load_data_extra(self):
#         aokvqa_set = load_aokvqa(**self._aokvqa_set_kwargs)
#         return aokvqa_set

#     def load_func_extra(self):
#         self.target_texts = target_texts
#         self.prompt_text = prompt_text
