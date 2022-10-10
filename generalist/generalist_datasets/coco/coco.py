from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch

import torchvision
from generalist.data_types.helper_types import Sample, SampleBuilder
from generalist.data_types.input_types import ImageType, TextType
from generalist.generalist_datasets.base import GeneralistDataset, SampleBuilderMixin
from generalist.generalist_datasets.image_datasets import ImageDatasetMixin
from torchvision import transforms

from generalist.generalist_datasets.utils.tasks_utils import TaskInterface
from generalist.generalist_tokenizers import text_tokenizers

from pycocotools import mask as coco_mask


def fix_channels(image):
    # if greyscale, repeat channels
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    return image


# def normalize_image(image):
#     image = image / 255.0
#     return image


# _train_transform = transforms.Compose(
#     [
#         transforms.Lambda(fix_channels),
#         # transforms.Lambda(normalize_image),
#         transforms.Resize((320, 320)),
#         # transforms.ColorJitter(brightness=[0.5, 1.3], contrast=[0.8, 1.5], saturation=[0.2, 1.5]),
#         transforms.RandomHorizontalFlip(),
#         # transforms.ToTensor(),
#         # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#     ]
# )

# _val_transform = transforms.Compose(
#     [
#         # transforms.Lambda(normalize_image),
#         transforms.Lambda(fix_channels),
#         transforms.Resize((320, 320)),
#         # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#     ]
# )

# _transforms = {
#     "train": _train_transform,
#     "val": _val_transform,
# }

# _coco_train_transform =

coco_transforms = {}


def coco_get_filepaths(coco_dir: str, split: str):
    assert split in ["train", "val", "test"]

    @dataclass
    class CocoFilepaths:
        base_dir: str = coco_dir
        images_root: str = f"{coco_dir}/{split}2017"
        captions_filepath: str = f"{coco_dir}/annotations/captions_{split}2017.json"
        instances_filepath: str = f"{coco_dir}/annotations/instances_{split}2017.json"
        person_keypoints_filepath: str = f"{coco_dir}/annotations/person_keypoints_{split}2017.json"

    return CocoFilepaths()


class CocoCaption(SampleBuilderMixin, torchvision.datasets.CocoCaptions):
    # sample_builder = SampleBuilder()

    # https://github.com/pytorch/vision/blob/main/torchvision/datasets/coco.py
    def __init__(
        self,
        root: str,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        **kwargs,
    ) -> None:
        super().__init__(root=root, annFile=annFile, transform=transform, target_transform=target_transform, transforms=transforms)

    def __getitem__(self, *args, **kwargs) -> Tuple[Any, Any]:
        image, (caption, caption_mask) = super().__getitem__(*args, **kwargs)

        masks = {"caption": caption_mask}

        sample_metadata = self.sample_builder.metadata.make(*args, dataset_name=self.__class__.__name__)
        sample = self.sample_builder(data=image, target=caption, masks=masks, metadata=sample_metadata)
        return sample


# class CocoDetection(SampleBuilderMixin, torchvision.datasets.CocoDetection):
class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, return_masks: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, index: int, *args, **kwargs) -> Tuple[Any, Any]:
        # img, target = super().__getitem__(idx, *args, **kwargs)
        img, target = super(CocoDetection, self).__getitem__(index=index)
        image_id = self.ids[index]
        target2 = {"image_id": image_id, "annotations": target}
        img3, target3 = self.prepare(img, target2)
        return img3, target3


def convert_coco_poly_to_mask(segmentations, height, width):
    """https://github.com/facebookresearch/detr/blob/main/datasets/coco.py"""
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    """https://github.com/facebookresearch/detr/blob/main/datasets/coco.py"""

    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


# --- ---- --- --- old below
# class CocoDatasetMultipleTask(ImageDatasetMixin, GeneralistDataset):
#     """
#     this was the dataset before i looked at using the torchvision coco dataset.  slower but more flexible
#     """

#     shortname = "coco"

#     def __init__(self, coco_dir: str = None, split: str = "train", **kwargs) -> None:
#         assert split in ["train", "test", "val"]
#         assert split != "test", "COCO test split is not available since it doesnt have captions/keypoints"

#         super().__init__(**kwargs)
#         self.split = split
#         self.image_transform = _transforms[self.split]

#         self.coco_dir = Path(coco_dir)
#         self.img_dir = self.coco_dir / f"{split}2017"
#         self.captions_path = self.coco_dir / "annotations" / f"captions_{split}2017.json"
#         self.instances_path = self.coco_dir / "annotations" / f"instances_{split}2017.json"
#         self.person_keypoints_path = self.coco_dir / "annotations" / f"person_keypoints_{split}2017.json"

#         self.captions_data = json.load(open(self.captions_path))
#         self.instances_data = json.load(open(self.instances_path))
#         self._process_captions()
#         # these other ones only have segmentation maps
#         # self.instances_data = json.load(open(self.instances))
#         # self.person_keypoints_data = json.load(open(self.person_keypoints))
#         self.text_tokenizer_kwargs = {
#             "return_tensors": "pt",
#             "truncation": True,
#             "padding": "max_length",
#             "max_length": 32,
#             "return_attention_mask": True,
#         }

#         # tasks available
#         self.tasks = TaskInterface()

#     def _process_captions(self) -> None:
#         self._image_info = {}
#         self.image_annotation = {}

#         # there are multiple captions per image
#         for annotation in self.captions_data["annotations"]:
#             if annotation["image_id"] not in self.image_annotation:
#                 self.image_annotation[annotation["image_id"]] = []
#             self.image_annotation[annotation["image_id"]].append(annotation)

#         self._dataset = []
#         for image_info in self.captions_data["images"]:
#             if image_info["id"] not in self._image_info:
#                 self._image_info[image_info["id"]] = image_info
#             else:
#                 raise ValueError("Duplicate image id! This should not happen.")

#             image_id = image_info["id"]
#             image_path = self.img_dir / image_info["file_name"]
#             captions = self.image_annotation.get(image_id, None)

#             self._dataset.append(
#                 {
#                     "image_id": image_id,
#                     "image_path": str(image_path),
#                     "caption": captions,
#                 }
#             )

#     def __len__(self):
#         return len(self._dataset)

#     def __getitem__(self, idx: int, **kwargs) -> Sample:
#         sample = super().__getitem__(idx, **kwargs)

#         item = self._dataset[idx]
#         self.extra_metadata(sample, item)

#         image = self.read_image(item["image_path"])
#         image = self.image_transform(ImageType(image))
#         # pick from one of the captions
#         caption = random.choice(item["caption"])
#         caption = TextType(caption["caption"])
#         # instruction = TextTypeRaw(f"Describe this image.")

#         sample.data = image
#         sample.target = caption

#         text_tokenizer_kwargs = {**self.text_tokenizer_kwargs, **kwargs.get("text_tokenizer_kwargs", {})}

#         if not kwargs.get("process_data", self.process_data):
#             data = self.tokenizers.image(sample.data)
#             sample.data = data

#         if not kwargs.get("process_target", self.process_target):
#             _caption = f"{self.tasks.caption(caption.data)}"

#             target = self.tokenizers.text.encode_plus(_caption, **text_tokenizer_kwargs)
#             sample.target = TextType(target["input_ids"])
#             sample.masks["target"] = target["attention_mask"]

#         return sample

#     def extra_metadata(self, sample: Sample, item: Dict[Any, Any]) -> dict:
#         sample.metadata.item_path = item["image_path"]
#         sample.metadata.image_id = item["image_id"]


# from generalist.generalist_tokenizers import image_tokenizers
# from generalist.generalist_tokenizers import text_tokenizers


# if __name__ == "__main__":
#     import time

#     device = "cuda"
#     image_tokenizer = image_tokenizers.ImageTokenizer(device=device)
#     text_tokenizer = text_tokenizers.TextTokenizerBert.from_pretrained("bert-base-uncased")
#     # CocoDatasetMultipleTask.use_tokenizers([image_tokenizer, text_tokenizer])

#     t1 = time.process_time()
#     dataset = CocoCaption(
#         # dataset = datasets.CocoCaptions(
#         root="/data/graham/datasets/coco/aokvqacoco/datasets/coco/val2017",
#         annFile="/data/graham/datasets/coco/aokvqacoco/datasets/coco/annotations/captions_val2017.json",
#     )

#     t2 = time.process_time()

#     # breakpoint()
#     # my_dataset = CocoDatasetMultipleTask(coco_dir="/data/graham/datasets/coco/aokvqacoco/datasets/coco", split="val")

#     t3 = time.process_time()

#     out1 = dataset[5]
#     t4 = time.process_time()
#     out2 = my_dataset[0]
#     t5 = time.process_time()

#     # my_out = dataset[0]
#     # out = dataset[0]

#     breakpoint()
