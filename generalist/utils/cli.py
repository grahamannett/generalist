import argparse


def train_get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", type=str, default=["xsum"])
    parser.add_argument("-bs", "--batch_size", type=int, default=8, dest="batch_size")
    parser.add_argument("--n_epochs", type=int, default=1, dest="n_epochs")
    parser.add_argument("--display", action=argparse.BooleanOptionalAction)
    return parser.parse_args()
