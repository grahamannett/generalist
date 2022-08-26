import os

import pytomlpp as toml
import torch
from generalist.utils.utils import get_hostname
from generalist.utils.device import get_device

from config.helper import ConfigInterface


class Config:
    #
    DEVICE = device = get_device()
    # DATASET RELATED
    BASE_DATADIR = os.environ.get("BASE_DATADIR", __file__.removesuffix("config/__init__.py") + "data")

    DEFAULT_AOKVQA_DIR = f"{BASE_DATADIR}/aokvqa"
    DEFAULT_COCO_DIR = f"{BASE_DATADIR}/coco"
    DEFAULT_LOG_DIR = "./logs"
    DEFAULT_FEATURES_DIR = "/home/graham/code/torchtask/submodules/aokvqa/features"
    DEFAULT_PRETRAINED_MODELS_DIR = "/home/graham/code/torchtask/pretrained_models"

    # merge with env
    aokvqa_dir = os.environ.get("AOKVQA_DIR", DEFAULT_AOKVQA_DIR)
    coco_dir = os.environ.get("COCO_DIR", DEFAULT_COCO_DIR)
    log_dir = os.environ.get("LOG_DIR", DEFAULT_LOG_DIR)
    features_dir = os.environ.get("LOG_DIR", DEFAULT_FEATURES_DIR)
    pretrained_models_dir = os.environ.get("PRETRAINED_MODELS_DIR", DEFAULT_PRETRAINED_MODELS_DIR)

    # MODEL RELATED
    embedding_dim = 768

    def __init__(self, config_name: str = "dev") -> None:
        config_name = os.environ.get("CONFIG_NAME", config_name)
        config_env = self._import_helper(config_name)
        self._merge_vals(config_env)

    def _merge_vals(self, config_env: ConfigInterface):
        for key, val in config_env.__dict__.items():
            if not key.endswith("__"):
                setattr(self, key, val)

    def _import_helper(self, config_name: str):
        try:
            exec(f"from config.{config_name}_config import Config as ConfigEnv")
            return locals().get("ConfigEnv", None)
        except ModuleNotFoundError:
            raise ModuleNotFoundError(f"Config {config_name} not found, specify one from config/")

    def _read_env_file(self, env_file: str = "info.toml"):
        params = toml.load(env_file)
        # breakpoint()
        env = params["config"].get(get_hostname(), params["config"]["default"])
        return env

    def put_env_into_config(self, env: dict):
        # for key, val in
        pass


config = Config()
# device = config.device
device = "cuda" if torch.cuda.is_available() else "cpu"
