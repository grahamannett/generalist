import os

from config.helper import ConfigInterface


# from config.dev import Config

class Config:
#
    DEVICE = device = "cuda:1"

    BASE_DATADIR = os.environ.get("BASE_DATADIR", __file__.removesuffix("config/__init__.py") + "datasets")

    DEFAULT_AOKVQA_DIR = f"{BASE_DATADIR}/aokvqa"
    DEFAULT_COCO_DIR = f"{BASE_DATADIR}/coco"
    DEFAULT_LOG_DIR = "./logs"
    DEFAULT_FEATURES_DIR = "/home/graham/code/torchtask/submodules/aokvqa/features"
    DEFAULT_PRETRAINED_MODELS_DIR = "/home/graham/code/torchtask/pretrained_models"

    # merge with env
    AOKVQA_DIR = os.environ.get("AOKVQA_DIR", DEFAULT_AOKVQA_DIR)
    COCO_DIR = os.environ.get("COCO_DIR", DEFAULT_COCO_DIR)
    LOG_DIR = os.environ.get("LOG_DIR", DEFAULT_LOG_DIR)
    FEATURES_DIR = os.environ.get("LOG_DIR", DEFAULT_FEATURES_DIR)
    PRETRAINED_MODELS_DIR = os.environ.get("PRETRAINED_MODELS_DIR", DEFAULT_PRETRAINED_MODELS_DIR)

    def __init__(self, config_name: str = "dev") -> None:
        config_name = os.environ.get("CONFIG_NAME", config_name)
        config_env = self._import_helper(config_name)
        self._merge_vals(config_env)

    def _merge_vals(self, config_env: ConfigInterface):
        for key, val in config_env.__dict__.items():
            if not key.endswith("__"):
                setattr(self, key, val)

    def _import_helper(self, config_name: str):
        exec(f"from config.{config_name} import Config as ConfigEnv")
        return locals().get("ConfigEnv", None)

config = Config()
device = config.device