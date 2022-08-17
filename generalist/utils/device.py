import torch


def get_device():
    # from config import device
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    return device
