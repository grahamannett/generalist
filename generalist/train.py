import datetime
import logging
from pathlib import Path
from random import sample

import hydra
import torch
from omegaconf import DictConfig
from rich import print
from torch.utils.data import DataLoader

from generalist.data_types.input_types import ImageType, TextTypeRaw
from generalist.generalist_datasets import AokvqaDataset, CocoDataset, GeneralistDataset, MNISTDataset
from generalist.generalist_datasets.base import ChainedGenearlistDataset
from generalist.generalist_tokenizers import image_tokenizers, text_tokenizers
from generalist.models.model import EmbeddingModel, GeneralistModel
from generalist.models.output_model import GeneralClassificationOutput, GeneralOutput
from generalist.predict import ImageCaptionPrediction
from generalist.utils.display import GeneralistDisplay
from generalist.utils.utils import collate_func, get_hostname, save_checkpoint

log = logging.getLogger(__name__)


def train_step(embedding_model, genearlist_model, dataloader):
    pass


def manage_live(group):
    pass


# def get_dataset():
#     dataset = AokvqaDataset()
#     dataset = SummarizationDataset()
#     dataset = LanguageModelingDataset()
#     dataset = MNISTDataset(train=True, out_channels=3)
#     datasets = None

#     if datasets == None:
#         raise NotImplementedError("TODO")

#     return dataset

# datasets = ChainedGenearlistDataset(datasets=[dataset])


def pad_targets(targets, logits):
    # pad targets to match logits
    encoded_targets = [torch.nn.functional.pad(t, (0, logits.shape[1] - t.shape[-1], 0, 0), mode="constant", value=0) for t in targets]
    encoded_targets = torch.stack(encoded_targets)


@hydra.main(config_path=f"../config", config_name=get_hostname(), version_base=None)
def train(cfg: DictConfig):

    # print("Working directory : {}".format(os.getcwd()))
    model_save_dir = Path(cfg.model_save_dir)
    display_flag = cfg.display.display_flag
    device = cfg.device
    context_length = cfg.context_length

    learning_rate = cfg.training.learning_rate
    batch_size = cfg.training.batch_size
    n_epochs = cfg.training.n_epochs

    model_dim = cfg.model.model_dim

    image_tokenizer = image_tokenizers.ImageTokenizer(device=device)
    text_tokenizer = text_tokenizers.TextTokenizerBert.from_pretrained("bert-base-uncased")

    # text_tokenizer = TextTokenizerPretrained("BertTokenizer", pretrained_name_or_model="bert-base-uncased", device=device)

    embedding_model = EmbeddingModel(model_dim=model_dim)
    # output_model = GeneralClassificationOutput(model_dim=model_dim, num_classes=10, reduce_type="cls")
    output_model = GeneralOutput(model_dim=model_dim, output_dim=text_tokenizer.vocab_size)
    model = GeneralistModel(output_model=output_model, **cfg.model).to(device)

    start_tokens = torch.Tensor([text_tokenizer.cls_token_id]).to(device).to(int)

    embedding_model.to(device)
    model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        [
            {"params": embedding_model.parameters()},
            {"params": model.parameters()},
        ],
        lr=learning_rate,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    TextTypeRaw.tokenizer = text_tokenizer
    ImageType.tokenizer = image_tokenizer

    tokenizers = [image_tokenizer, text_tokenizer]

    # can just call this with the base class for all datasets
    GeneralistDataset.use_tokenizers(tokenizers)
    # or can call on a specific dataset
    MNISTDataset.use_tokenizers(tokenizers)

    coco_dataset = CocoDataset(coco_dir=cfg.coco_dir, device=device)

    dataset = coco_dataset

    # eval example
    out = dataset[1]
    _tmp_text_tokenizer_kwargs = {**dataset.text_tokenizer_kwargs, "max_length": -1}
    out_data = out.data
    out_target_true = out.target.data
    out_target_tokens = dataset.tokenizers.text(out.target.data, **_tmp_text_tokenizer_kwargs)

    _max_length = out_target_tokens.shape[-1]

    # out.data = out.data.to(device)
    # out.target = out.target.to(device)
    caption_preder = ImageCaptionPrediction(
        image_tokenizer=image_tokenizer, text_tokenizer=text_tokenizer, embedding_model=embedding_model, model=model, device=device
    )

    tokenized_image = out_data.to(device)
    tokenized_caption = out_target_tokens.to(device)
    generated_captions = []

    if cfg.display.initial_caption:
        initial_caption = caption_preder.make_caption(
            tokenized_image=tokenized_image,
            max_length=_max_length,
        )
        generated_captions.append(initial_caption)

    collate_fn = collate_func(device=device)

    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    # val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    display = GeneralistDisplay.make(display=display_flag, logger=log)
    display.manage()
    for epoch in range(n_epochs):
        # epoch_progress.update(epoch_task)

        running_loss = 0.0
        running_correct = 0
        running_total = 0
        display.update("epoch_progress", epoch)
        display.add_task("batch_progress", "[green]Batch", total=len(train_dataloader), running_loss=running_loss)

        model.train()

        for batch_idx, batch in enumerate(train_dataloader):

            data, target = batch.data, batch.target

            embedding = embedding_model(data)
            # encoded_target = torch.cat(target)

            # tgt_attention_mask = torch.zeros(encoded_target.shape, device=device).to(bool)
            # tgt_attention_mask[encoded_target != 0] = 1
            # tgt_attention_mask = ~tgt_attention_mask

            # embedded_tgt = embedding_model(encoded_target)
            # if batch_idx % 2 == 0:
            #     embedded_tgt = None
            # else:
            #     embedded_tgt = embedding_model(target)

            #

            # task_target = [TextTypeRaw("Describe the image") for _ in range(batch_size)]

            task_target_arr = [
                text_tokenizer._encode(
                    f"Caption: {t.data}",
                    return_tensors="pt",
                    truncation=True,
                    padding="max_length",
                    max_length=32,
                    return_attention_mask=True,
                )
                for t in target
            ]
            task_input_ids = torch.cat([task_target["input_ids"] for task_target in task_target_arr], dim=0).to(device)
            task_input_attention_mask = torch.cat([task_target["attention_mask"] for task_target in task_target_arr], dim=0).to(device)
            task_input_token_types = torch.cat([task_target["token_type_ids"] for task_target in task_target_arr], dim=0).to(device)

            task_input_attention_mask = ~task_input_attention_mask.to(bool)

            embedded_tgt = embedding_model.embed_data(task_input_ids, data_type="text")

            tgt_mask = model.get_tgt_mask(embedded_tgt=embedded_tgt)

            logits = model(embedding, embedded_tgt=embedded_tgt, tgt_key_padding_mask=task_input_attention_mask, tgt_mask=tgt_mask)
            # logits = model(embedding, embedded_tgt=embedded_tgt)

            loss = loss_fn(logits[:, :-1].permute(0, 2, 1), task_input_ids[:, 1:])

            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            try:
                test_decoded = text_tokenizer.batch_decode(logits.argmax(dim=-1))
                # test_actual = text_tokenizer.batch_decode(encoded_target)
                test_actual = [t.data for t in target]

            except IndexError:
                breakpoint()

            batch_correct = sum([1 if a == b else 0 for a, b in zip(test_decoded, test_actual)])
            batch_total = len(test_decoded)

            running_correct += batch_correct
            running_total += batch_total

            # breakpoint()
            if batch_idx % 50 == 0:
                decoded__ = text_tokenizer.batch_decode(logits.argmax(dim=-1)[0:5, 0:10])
                # actual__ = text_tokenizer.batch_decode(encoded_target[0:5, 0:10])
                # actual__ = [t.data for t in target]
                actual__ = [t.data for t in target]
                print(list(zip(decoded__, actual__)))

                # breakpoint()

            if batch_idx % 250 == 0:
                latest_caption = caption_preder.make_caption(
                    tokenized_image=tokenized_image,
                    max_length=context_length,
                )

                print(f"latest caption:{text_tokenizer.decode(latest_caption)}")
                print(f"true caption: {out_target_true}")

                generated_captions.append(latest_caption)

            acc = f"{(running_correct / running_total):0.3f}"

            display_vals = {
                "acc": acc,
                "batch_acc": batch_correct / batch_total,
                "batch_idx": batch_idx,
            }

            # if batch_idx % 50 == 0:
            #     display_vals["test_decoded"] = test_decoded[0][:75]
            #     display_vals["test_actual"] = test_actual[0][:75]

            display.update(
                "batch_progress",
                advance=1,
                batch_loss=f"{loss.item():0.3f}",
                running_loss=f"{running_loss:.3f}",
                test=display_vals,
            )

        # save_checkpoint(model_save_dir, model, embedding_model, optimizer, epoch)
        save_checkpoint(
            model_save_dir,
            obj={
                "model": model,
                "embedding_model": embedding_model,
                "optimizer": optimizer,
                "epoch": epoch,
                "tokenizers": {
                    "image_tokenizer": image_tokenizer,
                    "text_tokenizer": text_tokenizer,
                },
            },
            filename="latest.pt",
        )

        latest_caption = caption_preder.make_caption(tokenized_image=tokenized_image, max_length=context_length)
        generated_captions.append(latest_caption)

    captions_generated = text_tokenizer.batch_decode(generated_captions)
    print("--- --- --- captions_generated --- --- ---")
    print(captions_generated)

    display.manage("epoch", display.END)
    print("done with training")


if __name__ == "__main__":
    train()
