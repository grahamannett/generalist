from typing import Any

from pathlib import Path
import torch

import datetime
import os


def get_hostname():
    if override_config := os.getenv("CONFIG_NAME"):
        return override_config

    import platform

    return platform.node()


def matplotlib_system_setup():
    import platform

    import matplotlib

    match platform.system().lower():
        case "darwin":
            matplotlib.use("MacOSX")
        case "linux":
            # not sure if i even need this but just doing for uniformity, might need to pass
            matplotlib.use("agg")


def _all_keys_match(batch):
    all_match = True
    _keys = list(batch[0].__annotations__.keys())
    for _batch in batch:
        if _keys != list(_batch.__annotations__.keys()):
            all_match = False
    return all_match, _keys


def save_checkpoint(
    model_save_dir: Path,
    obj: Any,
    filename: str = None,
):
    """helper function to checkpoint an object.  inteded for use with saving embedding_model/model

    Args:
        model_save_dir (Path): directory to save checkpoint
        obj (Any): object containing
        filename (str, optional): _description_. Defaults to None.
    """

    if filename is None:
        filename = datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S") + ".pt"

    torch.save(
        obj,
        model_save_dir / f"{filename}",
    )
