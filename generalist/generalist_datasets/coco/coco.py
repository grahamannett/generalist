import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torchvision
import torchvision.transforms as transforms
import pycocotools.mask as coco_mask

# from generalist.data_types.helper_types import Sample, SampleBuilder
from generalist.data_types.input_types import ImageType, TextType
from generalist.data_types.helper_types import SampleBuilderMixin
from generalist.generalist_datasets.coco.file_info import CocoFilepathsBase
from generalist.generalist_datasets.utils.tasks_utils import TaskInterface
from generalist.generalist_tokenizers import text_tokenizers

# from pycocotools import mask as coco_mask


class CocoCaptionTargetTranform:
    train = transforms.Compose([])
    val = transforms.Compose([])

    @staticmethod
    def use_text_tokenizer(text_tokenizer: text_tokenizers.TextTokenizer, text_tokenizer_kwargs: Dict[str, Any]):
        def _to_text_type(*args, **kwargs):
            caption = random.choice(args[0])
            caption = TaskInterface.caption(caption)
            caption = text_tokenizer.encode_plus(caption, **text_tokenizer_kwargs)
            return TextType(caption["input_ids"]), caption["attention_mask"]

        return _to_text_type

    @classmethod
    def get(cls, text_tokenizer: text_tokenizers.TextTokenizer, text_tokenizer_kwargs: Dict[str, Any]):
        cls.train.transforms.append(CocoCaptionTargetTranform.use_text_tokenizer(text_tokenizer, text_tokenizer_kwargs))
        _transforms = cls()
        return _transforms


class CocoImageTransforms:
    # potentially add these:

    # transforms.ColorJitter(brightness=[0.5, 1.3], contrast=[0.8, 1.5], saturation=[0.2, 1.5]),
    train = transforms.Compose(
        [
            transforms.Resize((320, 320)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            # transforms.Lambda(ImageType.transform),
            transforms.Lambda(ImageType.transform),
        ]
    )

    val = transforms.Compose(
        [
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Lambda(ImageType.transform),
        ]
    )


@dataclass
class CocoFilepaths(CocoFilepathsBase):
    coco_dir: str = None
    images_root: str = None
    captions_filepath: str = None
    instances_filepath: str = None
    person_keypoints_filepath: str = None

    def __post_init__(self):
        assert self.split in ["train", "val"]
        self.coco_dir = self.base_dir

        self.images_root: str = f"{self.coco_dir}/{self.split}2017"
        self.captions_filepath: str = f"{self.coco_dir}/annotations/captions_{self.split}2017.json"
        self.instances_filepath: str = f"{self.coco_dir}/annotations/instances_{self.split}2017.json"
        self.person_keypoints_filepath: str = f"{self.coco_dir}/annotations/person_keypoints_{self.split}2017.json"


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

    def __getitem__(self, idx: int, *args, **kwargs) -> Tuple[Any, Any]:
        image, (caption, caption_mask) = super().__getitem__(idx, *args, **kwargs)
        image_id = self.ids[idx]

        image_mask = torch.zeros(image.shape[-3:], dtype=torch.uint8)

        masks = {"data": image_mask, "target": caption_mask}

        sample_metadata = self.sample_builder.metadata(idx=idx, dataset_name=self.__class__.__name__, image_id=image_id)
        sample = self.sample_builder(data=image, target=caption, masks=masks, metadata=sample_metadata)
        return sample


# class CocoDetection(SampleBuilderMixin, torchvision.datasets.CocoDetection):
class CocoDetection(SampleBuilderMixin, torchvision.datasets.CocoDetection):
    """
    use something like
        coco_detection = CocoDetection(
        root=coco_filepaths.images_root,
        annFile=coco_filepaths.instances_filepath,
        transform=CocoImageTransforms.train,
        # transforms=CocoRegionTargetTransform.train
        # transforms=coco_detction_transforms,
    )

    # detection_sample = coco_detection[59537]
    """

    def __init__(self, return_masks: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx: int, **kwargs) -> Tuple[Any, Any]:
        # img, target = super().__getitem__(idx, *args, **kwargs)
        image, target = super(CocoDetection, self).__getitem__(index=idx)
        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": target}
        sample_metadata = self.sample_builder.metadata(idx=idx, image_id=image_id, dataset_name=self.__class__.__name__)

        image, target = self.prepare(image, target)

        _random_idx = random.randint(0, len(target["boxes"]) - 1)

        boxes, labels, image_mask, area, iscrowd = (
            target["boxes"][_random_idx],
            target["labels"][_random_idx],
            target["masks"][_random_idx],
            target["area"][_random_idx],
            target["iscrowd"][_random_idx],
        )

        category = self.coco.loadCats(labels.item())[0]["name"]

        _caption = TaskInterface.categorize_region(boxes.to(int).tolist(), category)
        masks = {"image_mask": None}
        # breakpoint()
        sample = self.sample_builder(data=image, target=_caption, masks=masks, metadata=sample_metadata)
        return sample


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

    def __call__(self, image: ImageType, target):

        if image.ndim == 3:
            image = image.unsqueeze(0)

        b, c, w, h = image.shape

        image_id = target["image_id"]
        # image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing

        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        breakpoint()
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


class CocoRegionTargetTransform:
    train = transforms.Compose([transforms.Lambda(ConvertCocoPolysToMask())])
    val = transforms.Compose([])
