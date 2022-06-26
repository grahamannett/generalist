import os


#
DEVICE = device = "cuda:1"
# DEVICE = device = "cpu"

BASE_DATADIR= os.environ.get("BASE_DATADIR")

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
