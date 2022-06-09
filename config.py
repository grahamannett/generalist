import os


#
# DEVICE = device = "cuda:1"
DEVICE = device = "cpu"


DEFAULT_AOKVQA_DIR = "/data/graham/datasets/aokvqa/"
DEFAULT_COCO_DIR = "/data/graham/datasets/coco/aokvqacoco/datasets/coco"
DEFAULT_LOG_DIR = "/home/graham/code/torchtask/logs"
DEFAULT_FEATURES_DIR = "/home/graham/code/torchtask/submodules/aokvqa/features"
DEFAULT_PRETRAINED_MODELS_DIR = "/home/graham/code/torchtask/pretrained_models"

# merge with env
AOKVQA_DIR = os.environ.get("AOKVQA_DIR", DEFAULT_AOKVQA_DIR)
COCO_DIR = os.environ.get("COCO_DIR", DEFAULT_COCO_DIR)
LOG_DIR = os.environ.get("LOG_DIR", DEFAULT_LOG_DIR)
FEATURES_DIR = os.environ.get("LOG_DIR", DEFAULT_FEATURES_DIR)
PRETRAINED_MODELS_DIR = os.environ.get("PRETRAINED_MODELS_DIR", DEFAULT_PRETRAINED_MODELS_DIR)
