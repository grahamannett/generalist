import unittest

from generalist.generalist_datasets.coco.coco import CocoCaption, CocoCaptionTargetTranform, CocoFilepaths, CocoImageTransforms
from generalist.generalist_datasets.hf.summary import SummaryTransforms, XSum
from generalist.generalist_datasets.utils.data_collate import collate_func_helper
from generalist.generalist_datasets.utils.multiple_datasets import BatchUniformDatasetSampler, CombinedDataset
from torch.utils.data import DataLoader

from tests.utils import TextTestMixin


class TestBatchSampler(TextTestMixin, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_batch_sampler(self):
        cfg = self.cfg
        device = self.cfg.device
        batch_size = 4
        text_tokenizer = self.text_tokenizer
        text_tokenizer_kwargs = self.text_tokenizer_kwargs

        coco_filepaths = CocoFilepaths(base_dir=cfg.coco_dir, split="train")

        coco_caption = CocoCaption(
            root=coco_filepaths.images_root,
            annFile=coco_filepaths.captions_filepath,
            transform=CocoImageTransforms.train,
            target_transform=CocoCaptionTargetTranform.get(
                text_tokenizer=text_tokenizer, text_tokenizer_kwargs=text_tokenizer_kwargs
            ).train,
        )

        summary_dataset = XSum(
            text_transform=SummaryTransforms.make_transforms(
                text_tokenizer=text_tokenizer, text_tokenizer_kwargs=text_tokenizer_kwargs
            ).train,
        )

        chained_dataset = CombinedDataset([coco_caption, summary_dataset])
        collate_fn = collate_func_helper(device=device, return_tensors="pt")

        batch_sampler = BatchUniformDatasetSampler(chained_dataset, batch_size=batch_size)
        train_dataloader = DataLoader(
            chained_dataset,
            # batch_size=batch_size,
            # shuffle=True,
            collate_fn=collate_fn,
            batch_sampler=batch_sampler,
        )

        for idx, data in enumerate(train_dataloader):
            metadata = data.metadata
            for m in metadata[1:]:
                self.assertEqual(m.dataset_name, metadata[0].dataset_name)

            # jsut checking 10 batches
            if idx > 10:
                break

