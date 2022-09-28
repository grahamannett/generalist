import unittest
import torch
from generalist.generalist_tokenizers.image_tokenizers import ImageTokenizer

from generalist.data_types.input_types import ImageType
from generalist.generalist_embedding.image_embedding import ImageEmbeddingPath, ImagePath


class TestImages(unittest.TestCase):
    def setUp(self):
        self.image_tokenizer = ImageTokenizer()
        self.image_embedder = ImageEmbeddingPath(model_dim=768)

    def test_tokenizer(self):
        # 3x224x224
        image_tensor = torch.rand(3, 224, 224)
        image = ImageType(image_tensor)
        tokenized = self.image_tokenizer(image)
        self.assertEqual(tokenized.shape, torch.Size([1, 196, 768]))

        # 3x320x320
        image_tensor = torch.rand(3, 320, 320)
        image = ImageType(image_tensor)
        tokenized = self.image_tokenizer(image)
        self.assertEqual(tokenized.shape, torch.Size([1, 400, 768]))

    def test_embedding(self):
        import timeit

        # def _func

        # outtime = timeit.timeit(lambda: "-".join(map(str, range(100))), number=10000)
        # print(outtime)
        # pass
        image_tensor = torch.rand(3, 224, 224)
        image = ImageType(image_tensor)
        tokenized = self.image_tokenizer(image)

        def _func():
            embedded = self.image_embedder(tokenized)

        outtime = timeit.timeit(_func, number=10)
        breakpoint()
