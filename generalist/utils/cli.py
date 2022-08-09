import argparse


def train_get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", type=str, default=["xsum"])
    parser.add_argument("--display", action=argparse.BooleanOptionalAction)

    # model/training parameters
    parser.add_argument("-bs", "--batch_size", type=int, default=8, dest="batch_size")
    parser.add_argument("-lr", "--learning_rate", type=float, default=5e-5, dest="lr")
    parser.add_argument("-ne", "--n_epochs", type=int, default=1, dest="n_epochs")

    return parser.parse_args()
