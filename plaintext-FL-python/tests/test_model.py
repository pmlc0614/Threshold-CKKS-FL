import unittest

import torch

from plaintext_fl.model import (
    ResNet20,
    ResNet32,
    count_trainable_parameters,
    resnet20,
    resnet32,
)


class ResNet20Test(unittest.TestCase):
    def test_output_shape(self):
        model = resnet20()
        model.eval()

        inputs = torch.randn(
            4,
            3,
            32,
            32,
        )

        with torch.no_grad():
            outputs = model(inputs)

        self.assertEqual(
            tuple(outputs.shape),
            (4, 10),
        )

    def test_parameter_count(self):
        model = resnet20()

        self.assertEqual(
            count_trainable_parameters(model),
            272_474,
        )

    def test_factory_returns_resnet20(self):
        model = resnet20()

        self.assertIsInstance(
            model,
            ResNet20,
        )


class ResNet32Test(unittest.TestCase):
    def test_output_shape(self):
        model = resnet32()
        model.eval()

        inputs = torch.randn(
            2,
            3,
            32,
            32,
        )

        with torch.no_grad():
            outputs = model(inputs)

        self.assertEqual(
            tuple(outputs.shape),
            (2, 100),
        )

    def test_parameter_count(self):
        model = resnet32()

        self.assertEqual(
            count_trainable_parameters(model),
            472_756,
        )

    def test_factory_returns_resnet32(self):
        model = resnet32()

        self.assertIsInstance(
            model,
            ResNet32,
        )


if __name__ == "__main__":
    unittest.main()
