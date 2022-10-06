import json
from typing import Any, Dict

from torch.utils.data import Dataset


class EvalMixin:
    # def __init__(self, dataset: Dataset) -> None:
    #     self.dataset = dataset

    def save_result(self, result: Dict[Any, Any], file_path: str) -> None:
        with open(file_path, "w") as f:
            json.dump(result, f)


class CocoEval:
    def __init__(self, dataset, image_caption_generator):
        self.dataset = dataset
        self.image_caption_generator = image_caption_generator
        # self.embedding_model = embedding_model
        # self.model = model

    def make_captions_prediction(self, **kwargs):
        """make captions for each image in the dataset"""
        device = self.image_caption_generator.device
        results = []

        # for i, sample in enumerate(self.dataset):
        for i in range(len(self.dataset)):
            sample = self.dataset.__getitem__(i, text_tokenizer_kwargs={"max_length": -1})
            data, target, target_masks = sample.data, sample.target, sample.masks["target"]
            metadata = sample.metadata

            max_length = target.shape[-1]

            data = data.to(device)

            caption = self.image_caption_generator.make_caption(image=data, max_length=max_length)

            caption_result = {
                "image_id": metadata.image_id,
                "caption": self.image_caption_generator.text_tokenizer.decode(caption),
            }
            results.append(caption_result)

        return results

    def make_captions_baseline(self):
        """make captions for each image in the dataset"""
        device = self.image_caption_generator.device
        results = []

        # for i, sample in enumerate(self.dataset):
        for i in range(len(self.dataset)):
            sample = self.dataset.__getitem__(i, raw_target=True, text_tokenizer_kwargs={"max_length": -1})
            target = sample.target
            # data, target = sample.data, sample.target, sample.masks["target"]
            metadata = sample.metadata
            caption = target.data

            caption_result = {
                "image_id": metadata.image_id,
                "caption": caption,
            }
            results.append(caption_result)

        return results

    def save_results(self, results, path):
        with open(path, "w") as f:
            json.dump(results, f)
