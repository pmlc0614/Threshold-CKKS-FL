import unittest

import torch
from torch import nn
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

from plaintext_fl.training import (
    build_optimizer,
    evaluate,
    learning_rate_at_step,
    resolve_device,
    train_one_epoch,
)


class TrainingTest(unittest.TestCase):
    def test_learning_rate_constant(self):
        result = learning_rate_at_step(
            initial_learning_rate=0.1,
            minimum_learning_rate=0.0,
            step=50,
            total_steps=200,
            schedule="constant",
        )

        self.assertEqual(
            result,
            0.1,
        )

    def test_cosine_learning_rate_starts_at_initial(self):
        result = learning_rate_at_step(
            initial_learning_rate=0.1,
            minimum_learning_rate=0.0,
            step=0,
            total_steps=200,
            schedule="cosine",
        )

        self.assertAlmostEqual(
            result,
            0.1,
        )

    def test_train_one_epoch_changes_parameters(self):
        torch.manual_seed(42)

        inputs = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [-1.0, -1.0],
            ]
        )

        targets = torch.tensor(
            [0, 1, 0, 1]
        )

        loader = DataLoader(
            TensorDataset(
                inputs,
                targets,
            ),
            batch_size=2,
            shuffle=False,
        )

        model = nn.Linear(
            2,
            2,
        )

        before = (
            model.weight.detach()
            .clone()
        )

        optimizer = build_optimizer(
            model=model,
            optimizer_name="sgd",
            learning_rate=0.1,
            momentum=0.0,
            weight_decay=0.0,
        )

        metrics = train_one_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            device=torch.device("cpu"),
        )

        self.assertEqual(
            metrics.sample_count,
            4,
        )

        self.assertFalse(
            torch.equal(
                before,
                model.weight.detach(),
            )
        )

    def test_evaluate_perfect_classifier(self):
        model = nn.Linear(
            2,
            2,
            bias=False,
        )

        with torch.no_grad():
            model.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                    ]
                )
            )

        inputs = torch.tensor(
            [
                [2.0, 0.0],
                [0.0, 2.0],
            ]
        )

        targets = torch.tensor(
            [0, 1]
        )

        loader = DataLoader(
            TensorDataset(
                inputs,
                targets,
            ),
            batch_size=2,
        )

        result = evaluate(
            model=model,
            loader=loader,
            device=torch.device("cpu"),
        )

        self.assertEqual(
            result.accuracy_percent,
            100.0,
        )

    def test_resolve_cpu(self):
        device = resolve_device(
            "cpu"
        )

        self.assertEqual(
            device.type,
            "cpu",
        )


if __name__ == "__main__":
    unittest.main()