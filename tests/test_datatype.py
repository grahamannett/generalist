import unittest
import torch
from generalist.data_types.input_types import ImageType, ImageTypeTensor, TextType


class TestImageType(unittest.TestCase):
    def test(self):
        image = ImageType.from_file("./tests/fixtures/img1.jpg")
        image_tensor = ImageType(torch.rand(3, 224, 224))

        self.assertIsInstance(image, ImageTypeTensor)
        self.assertIsInstance(image_tensor, ImageTypeTensor)


class TestTextType(unittest.TestCase):
    def test(self):
        base_str = "this is a string"
        text_str = TextType(base_str)
        text_tensor = TextType(torch.Tensor([1, 2, 3]))
        self.assertEqual(len(text_str), len(base_str))
        self.assertEqual(len(text_tensor), 3)

