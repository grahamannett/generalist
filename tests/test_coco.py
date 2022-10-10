import unittest

from hydra import initialize, compose
from generalist.generalist_datasets.coco.coco import CocoDetection, coco_get_filepaths

from generalist.utils.utils import get_hostname


class TestCoco(unittest.TestCase):
    def setUp(self) -> None:
        """
        https://github.com/facebookresearch/hydra/blob/main/examples/advanced/hydra_app_example/tests/test_example.py
        """
        with initialize(version_base=None, config_path="../config"):

            # cfg = compose(config_name="", overrides=["app.user=test_user"])

            cfg = compose(config_name=get_hostname())
            # val is considerately faster than train
            self.split = "val"
            self.coco_filepaths = coco_get_filepaths(coco_dir=cfg.coco_dir, split=self.split)

    def test_coco_detection(self):

        coco_detection = CocoDetection(
            root=self.coco_filepaths.images_root, annFile=self.coco_filepaths.instances_filepath, return_masks=True
        )
        coco_detection2 = CocoDetection(
            root=self.coco_filepaths.images_root, annFile=self.coco_filepaths.instances_filepath, return_masks=False
        )
        out = coco_detection[0]
        out_masks = coco_detection2[0]
        breakpoint()
