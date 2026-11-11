import tempfile
import unittest
from pathlib import Path

import torch

from plaintext_fl.model import resnet20
from plaintext_fl.model_adapter import build_model_protocol_layouts
from plaintext_fl.state import (
    apply_model_delta,
    build_float_state_layout,
    clone_nonfloat_state,
    compact_parameter_layout,
    compute_model_delta,
    flatten_float_state,
    layout_numel,
    layout_sha256,
    load_float_state,
    read_layout_json,
    write_layout_json,
)


class ModelStateTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

        self.model = resnet20()
        self.layout = build_float_state_layout(
            self.model
        )

    def test_float_state_size(self):
        self.assertEqual(
            layout_numel(self.layout),
            274_042,
        )

    def test_compact_parameter_layout_matches_encrypted_protocol(self):
        parameter_layout = compact_parameter_layout(self.layout)
        protocol_layout = build_model_protocol_layouts(self.model).parameter

        self.assertEqual(layout_numel(parameter_layout), 272_474)
        self.assertTrue(all(entry.is_parameter for entry in parameter_layout))
        self.assertEqual(
            tuple(
                (entry.name, entry.shape, entry.offset, entry.numel)
                for entry in parameter_layout
            ),
            tuple(
                (entry.name, entry.shape, entry.offset, entry.numel)
                for entry in protocol_layout.entries
            ),
        )

    def test_num_batches_tracked_is_excluded(self):
        layout_names = {
            entry.name
            for entry in self.layout
        }

        for state_name in self.model.state_dict():
            if state_name.endswith(
                "num_batches_tracked"
            ):
                self.assertNotIn(
                    state_name,
                    layout_names,
                )

    def test_flatten_and_restore_round_trip(self):
        original_flat = flatten_float_state(
            model=self.model,
            layout=self.layout,
        )

        restored_model = resnet20()

        load_float_state(
            model=restored_model,
            flat_state=original_flat,
            layout=self.layout,
        )

        restored_flat = flatten_float_state(
            model=restored_model,
            layout=self.layout,
        )

        torch.testing.assert_close(
            original_flat,
            restored_flat,
            rtol=0,
            atol=0,
        )

    def test_compute_model_delta(self):
        global_model = resnet20()
        local_model = resnet20()

        global_flat = flatten_float_state(
            global_model,
            self.layout,
        )

        modified_flat = (
            global_flat + 0.25
        )

        load_float_state(
            local_model,
            modified_flat,
            self.layout,
        )

        delta = compute_model_delta(
            local_model=local_model,
            global_model=global_model,
            layout=self.layout,
        )

        torch.testing.assert_close(
            delta,
            torch.full_like(
                delta,
                0.25,
            ),
            rtol=0,
            atol=1e-6,
        )

    def test_apply_model_delta(self):
        original_flat = flatten_float_state(
            self.model,
            self.layout,
        )

        delta = torch.full_like(
            original_flat,
            0.5,
        )

        apply_model_delta(
            model=self.model,
            delta=delta,
            layout=self.layout,
            scale=0.25,
        )

        updated_flat = flatten_float_state(
            self.model,
            self.layout,
        )

        expected = (
            original_flat + 0.125
        )

        torch.testing.assert_close(
            updated_flat,
            expected,
            rtol=0,
            atol=1e-6,
        )

    def test_layout_json_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = (
                Path(directory)
                / "model_layout.json"
            )

            write_layout_json(
                path=path,
                layout=self.layout,
            )

            restored_layout = read_layout_json(
                path
            )

            self.assertEqual(
                restored_layout,
                self.layout,
            )

            self.assertEqual(
                layout_sha256(restored_layout),
                layout_sha256(self.layout),
            )

    def test_nonfloat_state_is_cloned(self):
        nonfloat_state = clone_nonfloat_state(
            self.model
        )

        self.assertGreater(
            len(nonfloat_state),
            0,
        )

        for name, tensor in nonfloat_state.items():
            self.assertTrue(
                name.endswith(
                    "num_batches_tracked"
                )
            )

            self.assertFalse(
                tensor.is_floating_point()
            )


if __name__ == "__main__":
    unittest.main()
