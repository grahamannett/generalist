import unittest

from hydra import initialize, compose
from generalist.generalist_datasets.coco.coco import CocoDetection, CocoFilepaths
from generalist.generalist_datasets.coco.panoptic import CocoPanoptic, CocoPanopticFilepaths
from generalist.generalist_tokenizers import text_tokenizers

from generalist.utils.utils import get_hostname


class TestCoco(unittest.TestCase):
    def setUp(self) -> None:
        """
        https://github.com/facebookresearch/hydra/blob/main/examples/advanced/hydra_app_example/tests/test_example.py
        """
        self.text_tokenizer = text_tokenizers.TextTokenizerBert.from_pretrained("bert-base-uncased")
        with initialize(version_base=None, config_path="../config"):

            # cfg = compose(config_name="", overrides=["app.user=test_user"])

            cfg = compose(config_name=get_hostname())
            # val is considerately faster than train
            self.split = "val"
            self.coco_filepaths = CocoFilepaths(base_dir=cfg.coco_dir, split=self.split)
            self.coco_panoptic_filepaths = CocoPanopticFilepaths(base_dir=cfg.coco_panoptic, split=self.split)

    @unittest.skip
    def test_coco_detection(self):

        coco_detection = CocoDetection(
            root=self.coco_filepaths.images_root, annFile=self.coco_filepaths.instances_filepath, return_masks=True
        )

        image, target = coco_detection[0]

        self.assertEqual(image.__class__.__name__, "Image")

    @unittest.skip
    def test_coco_panoptic(self):
        coco_panoptic = CocoPanoptic(
            img_folder=self.coco_filepaths.images_root,
            ann_folder=self.coco_panoptic_filepaths.annotations_folder,
            ann_file=self.coco_panoptic_filepaths.annotations_file,
            return_masks=True,
        )

        image, target = coco_panoptic[0]

        self.assertIsInstance(target, dict)
