from base64 import encode
import torch
from config import config
from torch.utils.data import DataLoader
from torchvision.io import read_image
from tqdm import tqdm

from generalist.generalist_datasets.aokvqa.aokvqa import AokvqaDataset
from generalist.generalist_datasets.hf_datasets import SummarizationDataset
from generalist.generalist_tokenizers.prepare_data import PrepareData
from generalist.models.model import EmbeddingModel, GeneralistModel
from generalist.utils import collate_fn, get_args

device = config.device


def train():

    args = get_args()
    #

    lr = 5.0  # learning rate
    n_epochs = 1
    batch_size = 2

    embedding_model = EmbeddingModel().to(device)
    model = GeneralistModel().to(device)

    prepare_data = PrepareData(embedding_model=embedding_model, generalist_model=model, device=device)

    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        [
            {"params": embedding_model.parameters()},
            {"params": model.parameters()},
        ],
        lr=lr,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    # dataset = AokvqaDataset()
    dataset = SummarizationDataset()

    _ = dataset[0]

    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    for epoch in range(n_epochs):
        print(f"on epoch: {epoch}")

        pbar = tqdm(train_dataloader)
        running_loss = 0.0
        pbar.set_description(f"Epoch {epoch}|{n_epochs}")
        for idx, batch in enumerate(pbar):

            data, target = batch

            # the multi step process but explicit

            data_tokenized = prepare_data(data)
            data_embedded = embedding_model(data_tokenized)
            logits = model(data_embedded)

            encoded_targets = prepare_data.prepare_targets(target, logits)

            out = torch.cat(logits, dim=1).squeeze(0)
            encoded_targets = torch.cat(encoded_targets, dim=1).squeeze(0).to(device)

            if len(out) != len(encoded_targets):
                breakpoint()

            loss = loss_fn(out, encoded_targets)

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{running_loss:.3f}")

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()


if __name__ == "__main__":

    train()
