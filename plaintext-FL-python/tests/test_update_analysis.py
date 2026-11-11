import unittest

import torch
from torch import nn

from plaintext_fl.state import (
    build_float_state_layout,
    clone_float_buffers,
    layout_numel,
)
from plaintext_fl.update_analysis import (
    add_masked_gaussian_noise,
    apply_buffer_delta_vector,
    buffer_delta_vector,
    buffer_values_vector,
    build_selected_parameter_mask,
    extract_parameter_values,
    gaussian_noise_std_from_exponent,
    resolve_gaussian_noise_std,
)


class UpdateAnalysisTest(unittest.TestCase):
    def test_extract_parameter_values_excludes_buffers(self):
        model = nn.Sequential(
            nn.Linear(4, 2),
            nn.BatchNorm1d(2),
        )

        layout = build_float_state_layout(
            model
        )

        values = torch.ones(
            layout_numel(layout)
        )

        parameters = extract_parameter_values(
            flat_values=values,
            layout=layout,
        )

        expected_numel = sum(
            parameter.numel()
            for parameter in model.parameters()
        )

        self.assertEqual(
            parameters.numel(),
            expected_numel,
        )

    def test_masked_noise_is_reproducible(self):
        values = torch.zeros(
            10,
            dtype=torch.float64,
        )

        mask = torch.tensor(
            [
                True, True, False, False, False,
                True, True, False, False, False,
            ]
        )

        first, first_noise = (
            add_masked_gaussian_noise(
                flat_values=values,
                coordinate_mask=mask,
                standard_deviation=0.1,
                seed=1234,
            )
        )

        second, second_noise = (
            add_masked_gaussian_noise(
                flat_values=values,
                coordinate_mask=mask,
                standard_deviation=0.1,
                seed=1234,
            )
        )

        torch.testing.assert_close(
            first,
            second,
            rtol=0,
            atol=0,
        )

        torch.testing.assert_close(
            first_noise,
            second_noise,
            rtol=0,
            atol=0,
        )

        torch.testing.assert_close(
            first_noise[~mask],
            torch.zeros(6, dtype=torch.float64),
        )

    def test_relative_noise_sigma(self):
        reference = torch.tensor(
            [-1.0, 1.0],
            dtype=torch.float64,
        )

        sigma = resolve_gaussian_noise_std(
            reference_values=reference,
            absolute_std=None,
            relative_std_ratio=0.1,
        )

        self.assertAlmostEqual(
            sigma,
            0.1,
        )

    def test_running_var_delta(self):
        model = nn.BatchNorm1d(2)

        previous = clone_float_buffers(
            model
        )

        with torch.no_grad():
            model.running_var.copy_(
                torch.tensor([2.0, 3.0])
            )

        current = clone_float_buffers(
            model
        )

        delta = buffer_delta_vector(
            previous_buffers=previous,
            current_buffers=current,
            name_suffix="running_var",
        )

        torch.testing.assert_close(
            delta,
            torch.tensor(
                [1.0, 2.0],
                dtype=torch.float64,
            ),
        )

    def test_noise_exponent(self):
        self.assertEqual(
            gaussian_noise_std_from_exponent(None),
            0.0,
        )

        self.assertEqual(
            gaussian_noise_std_from_exponent(12),
            2.0 ** -12,
        )

        self.assertEqual(
            gaussian_noise_std_from_exponent(
                12,
                reference_bound=0.5,
            ),
            2.0 ** -13,
        )

        with self.assertRaises(ValueError):
            gaussian_noise_std_from_exponent(-1)

        with self.assertRaises(ValueError):
            gaussian_noise_std_from_exponent(
                12,
                reference_bound=None,
            )

    def test_apply_running_delta_vector(self):
        model = nn.BatchNorm1d(2)
        previous = clone_float_buffers(model)

        apply_buffer_delta_vector(
            model=model,
            previous_buffers=previous,
            delta=torch.tensor(
                [0.25, -0.5],
                dtype=torch.float64,
            ),
            name_suffix="running_mean",
        )

        torch.testing.assert_close(
            model.running_mean,
            torch.tensor([0.25, -0.5]),
        )

        torch.testing.assert_close(
            model.running_var,
            torch.ones(2),
        )

    def test_buffer_values_use_delta_order(self):
        model = nn.Sequential(
            nn.BatchNorm1d(2),
            nn.BatchNorm1d(3),
        )

        previous = clone_float_buffers(model)
        expected_delta = torch.tensor(
            [0.1, 0.2, 0.3, 0.4, 0.5],
            dtype=torch.float64,
        )

        apply_buffer_delta_vector(
            model=model,
            previous_buffers=previous,
            delta=expected_delta,
            name_suffix="running_mean",
        )

        current = clone_float_buffers(model)
        actual_delta = buffer_delta_vector(
            previous_buffers=previous,
            current_buffers=current,
            name_suffix="running_mean",
        )

        actual_values = buffer_values_vector(
            buffers=current,
            name_suffix="running_mean",
        )

        torch.testing.assert_close(
            actual_delta,
            expected_delta,
        )

        torch.testing.assert_close(
            actual_values,
            expected_delta,
        )


if __name__ == "__main__":
    unittest.main()

