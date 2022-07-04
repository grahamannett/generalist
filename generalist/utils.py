import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", type=str, default=["xsum"])
    return parser.parse_args()


def matplotlib_system_setup():
    import platform
    import matplotlib

    match platform.system().lower():
        case "darwin":
            matplotlib.use("MacOSX")


def _all_keys_match(batch):
    all_match = True
    _keys = list(batch[0].__annotations__.keys())
    for _batch in batch:
        if _keys != list(_batch.__annotations__.keys()):
            all_match = False
    return all_match, _keys


def collate_fn(batch):

    batch_out = {
        "data": [b.data for b in batch],
        "label": [b.label for b in batch],
    }

    return batch_out
