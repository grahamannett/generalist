import torch
from config import config
from torch.utils.data import DataLoader
from torchvision.io import read_image
from tqdm import tqdm

from generalist.generalist_datasets.aokvqa.aokvqa import AokvqaDataset
from generalist.generalist_tokenizers.prepare_data import PrepareData
from generalist.models.model import EmbeddingModel, GeneralistModel
from generalist.utils import collate_fn

device = config.device


def train():

    lr = 5.0  # learning rate
    n_epochs = 1
    batch_size = 1

    embedding_model = EmbeddingModel(device=device).to(device)
    model = GeneralistModel(device=device).to(device)

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

    _ = dataset[0]

    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    for epoch in range(n_epochs):
        print(f"on epoch: {epoch}")

        pbar = tqdm(train_dataloader)
        running_loss = 0.0
        pbar.set_description(f"Epoch {epoch}|{n_epochs}")
        for idx, batch in enumerate(pbar):

            data = batch["data"]
            label = batch["label"]

            data_tokenized = dataprep(data)

            # breakpoint()
            data_embedded = embedding_model(data_tokenized)

            out = model(data_embedded)

            labels = dataprep.prepare_label(label, out)

            out = torch.cat(out, dim=1).squeeze(0).to(device)
            labels = torch.cat([l["input_ids"] for l in labels], dim=1).squeeze(0).to(device)
            loss = loss_fn(out, labels)
            breakpoint()

            # label = {k: v.to(device) for k, v in label.items()}
            # breakpoint()

            # loss = loss_fn(out[0], label["input_ids"][0].to(device))

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{running_loss:.3f}")

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()


if __name__ == "__main__":

    train()
