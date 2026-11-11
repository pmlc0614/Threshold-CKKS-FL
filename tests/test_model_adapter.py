import copy
import unittest

import numpy as np
import torch
from torch import nn

from plaintext_fl.crypto_engine import CryptoEngineSetup, DecryptedRoundResult
from plaintext_fl.model_adapter import (
    apply_decrypted_result,
    build_model_protocol_layouts,
    candidate_from_models,
    compute_layout_delta,
    flatten_layout,
)
from plaintext_fl.model import resnet20, resnet32
from plaintext_fl.plaintext_engine import PlaintextCryptoEngine
from plaintext_fl.protocol_contract import (
    LayoutKind,
    RegisteredClientWeight,
    RoundPhase,
)
from plaintext_fl.round_orchestrator import FederatedRoundOrchestrator
from plaintext_fl.sparsify import (
    chunk_count,
    select_topk_chunks,
)


class TinyBatchNormModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.bn = nn.BatchNorm1d(2)


class UnsupportedFloatBufferModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.register_buffer("custom_float", torch.ones(1))
        self.bn = nn.BatchNorm1d(1)


class ModelAdapterTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = TinyBatchNormModel()
        self.layouts = build_model_protocol_layouts(
            self.model,
            model_name="tiny-bn",
        )

    def test_model_state_is_split_into_three_canonical_layouts(self):
        self.assertEqual(self.layouts.parameter.kind, LayoutKind.PARAMETER)
        self.assertEqual(self.layouts.running_mean.kind, LayoutKind.RUNNING_MEAN)
        self.assertEqual(
            self.layouts.running_variance.kind,
            LayoutKind.RUNNING_VARIANCE,
        )
        self.assertEqual(
            tuple(entry.name for entry in self.layouts.running_mean.entries),
            ("bn.running_mean",),
        )
        self.assertEqual(
            tuple(entry.name for entry in self.layouts.running_variance.entries),
            ("bn.running_var",),
        )
        self.assertEqual(self.layouts.parameter.total_numel, 10)
        self.assertEqual(self.layouts.running_mean.total_numel, 2)
        self.assertEqual(self.layouts.running_variance.total_numel, 2)
        self.assertEqual(len(self.layouts.parameter.sha256()), 64)

    def test_resnet20_layout_sizes_match_the_cifar10_target(self):
        layouts = build_model_protocol_layouts(resnet20())

        self.assertEqual(layouts.parameter.total_numel, 272_474)
        self.assertEqual(layouts.running_mean.total_numel, 784)
        self.assertEqual(layouts.running_variance.total_numel, 784)
        self.assertEqual(
            layouts.parameter.total_numel
            + layouts.running_mean.total_numel
            + layouts.running_variance.total_numel,
            274_042,
        )

    def test_resnet32_layout_sizes_match_the_cifar100_target(self):
        layouts = build_model_protocol_layouts(
            resnet32(),
            model_name="resnet32-cifar100",
        )

        self.assertEqual(layouts.parameter.total_numel, 472_756)
        self.assertEqual(layouts.running_mean.total_numel, 1_232)
        self.assertEqual(layouts.running_variance.total_numel, 1_232)
        self.assertEqual(
            layouts.parameter.total_numel
            + layouts.running_mean.total_numel
            + layouts.running_variance.total_numel,
            475_220,
        )

    def test_candidate_adapter_keeps_torch_out_of_engine_payloads(self):
        local_model = copy.deepcopy(self.model)
        with torch.no_grad():
            for parameter in local_model.parameters():
                parameter.add_(1.0)
            local_model.bn.running_mean.add_(2.0)
            local_model.bn.running_var.add_(3.0)

        parameter_delta = compute_layout_delta(
            local_model,
            self.model,
            self.layouts.parameter,
        )
        selection = select_topk_chunks(
            update=torch.from_numpy(parameter_delta.copy()),
            chunk_size=4,
            selected_chunk_count=2,
        )
        candidate = candidate_from_models(
            client_id=3,
            selection=selection,
            local_model=local_model,
            global_model=self.model,
            layouts=self.layouts,
        )

        self.assertIsInstance(candidate.selected_values, np.ndarray)
        self.assertEqual(candidate.selected_values.dtype, np.float32)
        np.testing.assert_array_equal(
            candidate.running_mean_delta,
            np.full(2, 2.0, dtype=np.float32),
        )
        np.testing.assert_array_equal(
            candidate.running_variance_delta,
            np.full(2, 3.0, dtype=np.float32),
        )

    def test_decrypted_streams_apply_to_only_their_model_state(self):
        before_parameters = flatten_layout(self.model, self.layouts.parameter)
        before_mean = flatten_layout(self.model, self.layouts.running_mean)
        before_variance = flatten_layout(
            self.model,
            self.layouts.running_variance,
        )
        before_batches = self.model.bn.num_batches_tracked.clone()
        result = DecryptedRoundResult(
            setup_id=bytes(16),
            round_id=0,
            accepted_actual_chunk_ids=(0,),
            cohort_sizes=(1, 0, 0),
            parameter_delta=np.ones(
                self.layouts.parameter.total_numel,
                dtype=np.float32,
            ),
            running_mean_delta=np.full(2, 2.0, dtype=np.float32),
            running_variance_delta=np.full(2, 3.0, dtype=np.float32),
        )

        apply_decrypted_result(self.model, self.layouts, result)

        np.testing.assert_allclose(
            flatten_layout(self.model, self.layouts.parameter),
            before_parameters + 1.0,
        )
        np.testing.assert_allclose(
            flatten_layout(self.model, self.layouts.running_mean),
            before_mean + 2.0,
        )
        np.testing.assert_allclose(
            flatten_layout(self.model, self.layouts.running_variance),
            before_variance + 3.0,
        )
        torch.testing.assert_close(
            self.model.bn.num_batches_tracked,
            before_batches,
        )

    def test_models_round_trip_through_engine_independent_orchestrator(self):
        local_one = copy.deepcopy(self.model)
        local_two = copy.deepcopy(self.model)
        with torch.no_grad():
            for parameter in local_one.parameters():
                parameter.add_(1.0)
            for parameter in local_two.parameters():
                parameter.add_(3.0)
            local_one.bn.running_mean.add_(2.0)
            local_two.bn.running_mean.add_(6.0)
            local_one.bn.running_var.add_(4.0)
            local_two.bn.running_var.add_(8.0)

        chunk_size = 4
        candidates = []
        for client_id, local_model in ((10, local_one), (20, local_two)):
            parameter_delta = compute_layout_delta(
                local_model,
                self.model,
                self.layouts.parameter,
            )
            selection = select_topk_chunks(
                update=torch.from_numpy(parameter_delta.copy()),
                chunk_size=chunk_size,
                selected_chunk_count=chunk_count(
                    self.layouts.parameter.total_numel,
                    chunk_size,
                ),
            )
            candidates.append(
                candidate_from_models(
                    client_id=client_id,
                    selection=selection,
                    local_model=local_model,
                    global_model=self.model,
                    layouts=self.layouts,
                )
            )

        before_parameters = flatten_layout(self.model, self.layouts.parameter)
        before_mean = flatten_layout(self.model, self.layouts.running_mean)
        before_variance = flatten_layout(
            self.model,
            self.layouts.running_variance,
        )
        engine = PlaintextCryptoEngine()
        orchestrator = FederatedRoundOrchestrator(engine)
        orchestrator.setup(
            CryptoEngineSetup(
                setup_id=bytes(range(16)),
                parameter_layout=self.layouts.parameter,
                running_mean_layout=self.layouts.running_mean,
                running_variance_layout=self.layouts.running_variance,
                clients=(
                    RegisteredClientWeight(client_id=10, example_count=1),
                    RegisteredClientWeight(client_id=20, example_count=3),
                ),
                chunk_size=chunk_size,
                minimum_cohort_size=2,
            )
        )

        orchestrator.run_round(
            0,
            candidates,
            apply_result=lambda result: apply_decrypted_result(
                self.model,
                self.layouts,
                result.decrypted,
            ),
        )

        np.testing.assert_allclose(
            flatten_layout(self.model, self.layouts.parameter),
            before_parameters + 2.5,
        )
        np.testing.assert_allclose(
            flatten_layout(self.model, self.layouts.running_mean),
            before_mean + 5.0,
        )
        np.testing.assert_allclose(
            flatten_layout(self.model, self.layouts.running_variance),
            before_variance + 7.0,
        )
        self.assertEqual(engine.phase, RoundPhase.COMMITTED)

    def test_unknown_float_buffer_is_rejected_explicitly(self):
        with self.assertRaisesRegex(ValueError, "custom_float"):
            build_model_protocol_layouts(UnsupportedFloatBufferModel())


if __name__ == "__main__":
    unittest.main()
