import torch
from generalist.models.model import GeneralistModel


def train():

    image = torch.rand(32, 3, 224, 224)
    text = "Hello World"

    model = GeneralistModel()

    input_data_text = {
        "data": text,
        "type": "text",
    }
    out2 = model(input_data_text)
    breakpoint()


if __name__ == "__main__":

    train()
