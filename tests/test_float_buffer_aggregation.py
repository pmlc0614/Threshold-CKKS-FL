import unittest

import torch
from torch import nn

from plaintext_fl.state import (
    build_float_state_layout,
    clone_float_buffers,
    layout_numel,
    load_weighted_average_float_buffers,
    zero_nonparameter_entries,
)


class FloatBufferAggregationTest(
    unittest.TestCase
):
    def test_nonparameter_entries_are_zeroed(self):
        model = nn.Sequential(
            nn.Linear(4, 2),
            nn.BatchNorm1d(2),
        )

        layout = build_float_state_layout(
            model
        )

        values = torch.ones(
            layout_numel(layout),
            dtype=torch.float32,
        )

        result = zero_nonparameter_entries(
            flat_values=values,
            layout=layout,
        )

        for entry in layout:
            piece = result[
                entry.offset:
                entry.offset + entry.numel
            ]

            if entry.is_parameter:
                torch.testing.assert_close(
                    piece,
                    torch.ones_like(piece),
                )
            else:
                torch.testing.assert_close(
                    piece,
                    torch.zeros_like(piece),
                )

    def test_float_buffers_use_sample_weights(self):
        global_model = nn.BatchNorm1d(2)
        first_client = nn.BatchNorm1d(2)
        second_client = nn.BatchNorm1d(2)

        with torch.no_grad():
            first_client.running_mean.copy_(
                torch.tensor([1.0, 3.0])
            )
            first_client.running_var.copy_(
                torch.tensor([2.0, 4.0])
            )

            second_client.running_mean.copy_(
                torch.tensor([5.0, 7.0])
            )
            second_client.running_var.copy_(
                torch.tensor([6.0, 8.0])
            )

        load_weighted_average_float_buffers(
            model=global_model,
            client_buffers=[
                clone_float_buffers(
                    first_client
                ),
                clone_float_buffers(
                    second_client
                ),
            ],
            sample_counts=[1, 3],
        )

        torch.testing.assert_close(
            global_model.running_mean,
            torch.tensor([4.0, 6.0]),
        )

        torch.testing.assert_close(
            global_model.running_var,
            torch.tensor([5.0, 7.0]),
        )

        self.assertTrue(
            torch.all(
                global_model.running_var >= 0
            )
        )


if __name__ == "__main__":
    unittest.main()