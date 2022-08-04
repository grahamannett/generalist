# from https://colab.research.google.com/drive/1F4Fn_8kF7FVyzyHOIpM9Sc3Q3AsLmcKT#scrollTo=npMiFNnRznc0

pretrained_model_name = "openai/imagegpt-small"

import argparse


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument("--model_type", type=str, default="imagegpt")
    return args.parse_args()


args = get_args()


from datasets import load_dataset

# load cifar10 (only small portion for demonstration purposes)
# train_ds, test_ds = load_dataset("mnist")
ds = load_dataset("mnist")
train_ds, test_ds = ds["train"], ds["test"]
# split up training into training + validation
splits = train_ds.train_test_split(test_size=0.1)
train_ds = splits["train"]
val_ds = splits["test"]

id2label = {idx: label for idx, label in enumerate(train_ds.features["label"].names)}
label2id = {label: idx for idx, label in id2label.items()}
print(id2label)
print(label2id)

from transformers import (
    ImageGPTConfig,
    ImageGPTFeatureExtractor,
    ImageGPTModel,
    PerceiverFeatureExtractor,
    ImageGPTForImageClassification,
    PerceiverForImageClassificationLearned,
)

# feature_extractor = PerceiverFeatureExtractor()


import numpy as np
import torchvision.transforms.functional as F


def get_perceiver():
    # perceiver
    feature_extractor = PerceiverFeatureExtractor.from_pretrained("deepmind/vision-perceiver-learned")
    model = PerceiverForImageClassificationLearned.from_pretrained(
        "deepmind/vision-perceiver-learned",
        num_labels=10,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    def preprocess_images(examples):
        images = [F.pil_to_tensor(i).repeat(3, 1, 1) for i in examples["image"]]
        examples["pixel_values"] = feature_extractor(images, return_tensors="pt").pixel_values
        return examples

    return model, feature_extractor, preprocess_images


def get_imagegpt():
    # imagegpt
    feature_extractor = ImageGPTFeatureExtractor.from_pretrained(pretrained_model_name)
    model = ImageGPTForImageClassification.from_pretrained(
        pretrained_model_name,
        num_labels=10,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    def preprocess_images(examples):
        images = [F.pil_to_tensor(i).repeat(3, 1, 1) for i in examples["image"]]
        examples["pixel_values"] = feature_extractor(images, return_tensors="pt").input_ids
        return examples

    return model, feature_extractor, preprocess_images


def get_model_specific(model_type: str):
    match model_type:
        case "imagegpt":
            func = get_imagegpt
        case "perceiver":
            func = get_perceiver
        case _:
            raise ValueError(f"Unknown model type: {model_type}")

    return func()


model, feature_extractor, preprocess_images = get_model_specific(args.model_type)
breakpoint()


train_ds.set_transform(preprocess_images)
val_ds.set_transform(preprocess_images)
test_ds.set_transform(preprocess_images)


import torch
from torch.utils.data import DataLoader


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


train_batch_size = 2
eval_batch_size = 1

train_dataloader = DataLoader(train_ds, shuffle=True, collate_fn=collate_fn, batch_size=train_batch_size)
val_dataloader = DataLoader(val_ds, collate_fn=collate_fn, batch_size=eval_batch_size)
test_dataloader = DataLoader(test_ds, collate_fn=collate_fn, batch_size=eval_batch_size)


batch = next(iter(train_dataloader))
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        print(k, v.shape)


print(next(iter(val_dataloader))["pixel_values"].shape)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

from sklearn.metrics import accuracy_score

# from tqdm.notebook import tqdm
from tqdm import tqdm
from transformers import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)


model.train()
for epoch in range(2):  # loop over the dataset multiple times
    print("Epoch:", epoch)
    pbar = tqdm(train_dataloader)

    running_correct = 0
    running_total = 0
    running_loss = 0
    for batch in pbar:
        # get the inputs;
        inputs = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        # evaluate
        predictions = outputs.logits.argmax(-1).cpu().detach().numpy()

        running_correct += (batch["labels"].numpy() == predictions).sum()
        running_total += len(batch["labels"])
        running_loss += loss.item()

        accuracy = running_correct / running_total
        # accuracy = accuracy_score(y_true=batch["labels"].numpy(), y_pred=predictions)
        # print(f"Loss: {loss.item()}, Accuracy: {accuracy}")
        pbar.set_postfix(loss=loss.item(), accuracy=accuracy, running_loss=running_loss)

# from tqdm.notebook import tqdm
from datasets import load_metric

accuracy = load_metric("accuracy")

model.eval()

pbar = tqdm(val_dataloader)
running_total = 0
running_correct = 0
for batch in pbar:
    # get the inputs;
    inputs = batch["pixel_values"].to(device)
    labels = batch["labels"].to(device)

    # forward pass
    outputs = model(inputs, labels=labels)
    logits = outputs.logits
    predictions = logits.argmax(-1).cpu().detach().numpy()
    references = batch["labels"].numpy()
    accuracy.add_batch(predictions=predictions, references=references)
    # running_correct += (batch["labels"].numpy() == predictions).sum()
    # running_total += len(batch["labels"])

final_score = accuracy.compute()
# final_score =
print("Accuracy on test set:", final_score)
