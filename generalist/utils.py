
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
    # all_match, _keys = _all_keys_match(batch)
    # batch_out = {k: [] for k in _keys}

    # for _batch in batch:
    #     for key in _keys:
    #         batch_out[key].append(getattr(_batch, key))
    # breakpoint()

    batch_out = {
        "data": [b.data for b in batch],
        "label": [b.label for b in batch],
    }

    return batch_out