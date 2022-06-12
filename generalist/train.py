from generalist.generalist_datasets.aokvqa.aokvqa import AokvqaDataset
import torch
from generalist.models.model import EmbeddingModel, GeneralistModel

from tqdm import trange
from torchvision.io import read_image


def train():

    text = "Hello World"

    embedding_model = EmbeddingModel()
    model = GeneralistModel()

    image = torch.rand(1, 3, 224, 224)
    dataset = AokvqaDataset()
    data0 = dataset[0]

    img = read_image(data0.image_path)
    img = img / 255

    input_data_text = {
        "data": text,
        "type": "text",
    }

    embeds = embedding_model(input_data_text)
    out2 = model(input_data_text)


if __name__ == "__main__":

    train()
