import os

import pytomlpp as toml
from config.helper import ConfigInterface
from generalist.utils.utils import get_hostname

# from config.dev import Config


class Config:
    #
    DEVICE = device = "cuda:1"
    # DATASET RELATED
    BASE_DATADIR = os.environ.get("BASE_DATADIR", __file__.removesuffix("config/__init__.py") + "data")

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

    # MODEL RELATED
    EMBEDDING_DIM = 768

    def __init__(self, config_name: str = "dev") -> None:
        config_name = os.environ.get("CONFIG_NAME", config_name)
        config_env = self._import_helper(config_name)
        self._merge_vals(config_env)

    def _merge_vals(self, config_env: ConfigInterface):
        for key, val in config_env.__dict__.items():
            if not key.endswith("__"):
                setattr(self, key, val)

    def _import_helper(self, config_name: str):
        env = self._read_env_file()
        try:
            exec(f"from config.{config_name}_config import Config as ConfigEnv")
            return locals().get("ConfigEnv", None)
        except ModuleNotFoundError:
            raise ModuleNotFoundError(f"Config {config_name} not found, specify one from config/")

    def _read_env_file(self, env_file: str = "info.toml"):
        params = toml.load(env_file)
        env = params["config"].get(get_hostname(), params["config"]["default"])
        return env


config = Config()
device = config.device
