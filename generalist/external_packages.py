from packagify import Packagify
import sys


submodules = Packagify("/home/graham/code/torchtask/submodules")

load_aokvqa, get_coco_path = submodules.import_module("aokvqa.load_aokvqa", ["load_aokvqa", "get_coco_path"])

AokvqaDataset, load_data, target_texts, prompt_text = submodules.import_module(
    "aokvqa.ClipCap.data", ["AokvqaDataset", "load_data", "target_texts", "prompt_text"]
)

load_model = submodules.import_module("aokvqa.ClipCap.model", ["load_model"])


class ExternalPackage:
    def __init__(self, module_file, objects):
        pass


# ClipCap = ExternalPackage(
#     "aokvqa.ClipCap.data", ["AokvqaDataset", "load_data", "target_texts", "prompt_text"]
# )
