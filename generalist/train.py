from multiprocessing.spawn import prepare
from generalist.generalist_datasets.aokvqa.aokvqa import AokvqaDataset
import torch
from generalist.generalist_tokenizers.prepare_data import PrepareData
from generalist.models.model import EmbeddingModel, GeneralistModel
from config import device

from tqdm import trange
from torch.utils.data import DataLoader
from torchvision.io import read_image


def train():

    text = "Hello World"

    lr = 5.0  # learning rate

    embedding_model = EmbeddingModel().to(device)
    model = GeneralistModel().to(device)

    dataprep = PrepareData()
    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        [
            {"params": embedding_model.parameters()},
            {"params": model.parameters()},
        ],
        lr=lr,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    dataset = AokvqaDataset()

    train_dataloader = DataLoader(dataset, 1, shuffle=True)

    for idx, batch in enumerate(train_dataloader):

        data = batch["data"]
        label = batch["label"]

        print(f"-> {data[1].data}")

        data = dataprep(data)
        data = embedding_model(data)

        out = model(data)
        _max_len = min(dataprep.tokenizer.model_max_length, out.shape[1])

        label = dataprep.tokenizer(
            label.data, padding="max_length", truncation=True, max_length=_max_len, return_tensors="pt"
        )
        label = {k: v.to(device) for k, v in label.items()}

        loss = loss_fn(out[0], label["input_ids"][0])

        print(f"got loss: {loss.item():.4f}")

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()


if __name__ == "__main__":

    train()
