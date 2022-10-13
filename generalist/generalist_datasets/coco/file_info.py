from dataclasses import dataclass


def _setup_paths(self):
    coco_dir = self.coco_dir
    split = self.split
    images_root: str = f"{coco_dir}/{split}2017"
    captions_filepath: str = f"{coco_dir}/annotations/captions_{split}2017.json"
    instances_filepath: str = f"{coco_dir}/annotations/instances_{split}2017.json"
    person_keypoints_filepath: str = f"{coco_dir}/annotations/person_keypoints_{split}2017.json"


@dataclass
class CocoFilepathsBase:
    base_dir: str
    split: str
