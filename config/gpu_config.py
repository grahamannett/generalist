from config.helper import ConfigInterface


class Config(ConfigInterface):
    DEVICE = device = "cuda:1"

    coco_dir = "/data/graham/datasets/coco/aokvqacoco/datasets/coco"
