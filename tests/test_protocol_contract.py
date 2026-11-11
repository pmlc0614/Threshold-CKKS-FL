import hashlib
import json
import unittest
from pathlib import Path

from plaintext_fl.protocol_contract import (
    AcceptedChunkValues,
    BINARY_HEADER_SIZE,
    BinaryHeader,
    ClientSelectionRequest,
    LayoutEntry,
    LayoutKind,
    MessageType,
    RegisteredClientWeight,
    RoundPhase,
    TensorLayout,
    is_allowed_round_transition,
    quantize_registered_client_weights,
)


def make_layout(kind, name, shape):
    numel = 1
    for dimension in shape:
        numel *= dimension
    return TensorLayout(
        kind=kind,
        model_name="resnet20-cifar10",
        entries=(
            LayoutEntry(
                name=name,
                shape=shape,
                offset=0,
                numel=numel,
            ),
        ),
    )


class ProtocolContractTest(unittest.TestCase):
    def test_two_phase_messages_reject_duplicate_or_misaligned_values(self):
        parameter_hash = make_layout(
            LayoutKind.PARAMETER,
            "conv1.weight",
            (16, 3, 3, 3),
        ).sha256()
        duplicate_selection = ClientSelectionRequest(
            setup_id=bytes(16),
            round_id=1,
            client_id=2,
            parameter_layout_sha256=parameter_hash,
            selected_actual_chunk_ids=(4, 4),
        )
        with self.assertRaisesRegex(ValueError, "duplicates"):
            duplicate_selection.validate()

        misaligned_values = AcceptedChunkValues(
            setup_id=bytes(16),
            round_id=1,
            client_id=2,
            chunk_size=4,
            actual_chunk_ids=(4, 7),
            values=(1.0, 2.0, 3.0),
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            misaligned_values.validate()

    def test_repository_golden_vectors(self):
        document = json.loads(
            (
                Path(__file__).parents[1]
                / "interop"
                / "golden-v1.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(document["schema"], "fl-v17-contract-golden-v1")

        for vector in document["weights"]:
            clients = tuple(
                RegisteredClientWeight(client_id, example_count)
                for client_id, example_count in zip(
                    vector["client_ids"],
                    vector["example_counts"],
                    strict=True,
                )
            )
            self.assertEqual(
                quantize_registered_client_weights(
                    clients,
                    denominator=vector["denominator"],
                ),
                tuple(vector["expected_alphas"]),
            )

        for kind_name, vector in document["layouts"].items():
            entries = tuple(
                LayoutEntry(
                    name=entry["name"],
                    shape=tuple(entry["shape"]),
                    offset=entry["offset"],
                    numel=entry["numel"],
                    dtype=entry["dtype"],
                )
                for entry in vector["entries"]
            )
            layout = TensorLayout(
                kind=LayoutKind(kind_name),
                model_name=vector["model_name"],
                entries=entries,
            )
            self.assertEqual(layout.sha256(), vector["expected_sha256"])

    def test_largest_remainder_uses_client_id_tie_break(self):
        clients = (
            RegisteredClientWeight(30, 1),
            RegisteredClientWeight(10, 1),
            RegisteredClientWeight(20, 1),
        )
        self.assertEqual(
            quantize_registered_client_weights(clients),
            (42, 43, 43),
        )

    def test_zero_weight_setup_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "alpha_i >= 1"):
            quantize_registered_client_weights(
                (
                    RegisteredClientWeight(1, 1),
                    RegisteredClientWeight(2, 1000),
                )
            )

    def test_layout_kinds_have_distinct_hashes(self):
        parameter = make_layout(LayoutKind.PARAMETER, "conv1.weight", (16, 3, 3, 3))
        running_mean = make_layout(LayoutKind.RUNNING_MEAN, "bn1.running_mean", (16,))
        running_variance = make_layout(
            LayoutKind.RUNNING_VARIANCE,
            "bn1.running_var",
            (16,),
        )
        self.assertEqual(
            parameter.sha256(),
            "78500415f35edcad4ab057a2d2ecb5614f74eab963e536845b695110602ecaf6",
        )
        self.assertEqual(
            running_mean.sha256(),
            "d8204987221095582c7b13ada6e4b9b5fcad7dc2e47555d6ef81918a76ac86a4",
        )
        self.assertEqual(
            running_variance.sha256(),
            "c27aad243b2c8f93c795fb193743167e4ad75e153b7d54d69e8bdb672e51de87",
        )
        self.assertEqual(len({parameter.sha256(), running_mean.sha256(), running_variance.sha256()}), 3)

    def test_binary_header_round_trip(self):
        parameter = make_layout(LayoutKind.PARAMETER, "conv1.weight", (16, 3, 3, 3))
        running_mean = make_layout(LayoutKind.RUNNING_MEAN, "bn1.running_mean", (16,))
        running_variance = make_layout(
            LayoutKind.RUNNING_VARIANCE,
            "bn1.running_var",
            (16,),
        )
        payload = bytes((1, 2, 3, 4, 5))
        header = BinaryHeader.for_payload(
            message_type=MessageType.CLIENT_SELECTION,
            setup_id=bytes(range(16)),
            round_id=42,
            client_id=7,
            parameter_layout_sha256=parameter.sha256(),
            running_mean_layout_sha256=running_mean.sha256(),
            running_variance_layout_sha256=running_variance.sha256(),
            payload=payload,
        )
        encoded = header.to_bytes()
        self.assertEqual(len(encoded), BINARY_HEADER_SIZE)
        self.assertEqual(BinaryHeader.from_bytes(encoded), header)
        self.assertEqual(header.payload_sha256, hashlib.sha256(payload).hexdigest())
        self.assertEqual(
            encoded.hex(),
            "464c56313742494e0100000002000000000102030405060708090a0b0c0d0e0f"
            "2a000000000000000700000000000000"
            "78500415f35edcad4ab057a2d2ecb5614f74eab963e536845b695110602ecaf6"
            "d8204987221095582c7b13ada6e4b9b5fcad7dc2e47555d6ef81918a76ac86a4"
            "c27aad243b2c8f93c795fb193743167e4ad75e153b7d54d69e8bdb672e51de87"
            "0500000000000000"
            "74f81fe167d99b4cb41d6d0ccda82278caee9f3e2f25d5e5a3936ff3dcec60d0",
        )

    def test_round_state_machine_is_forward_only(self):
        self.assertTrue(
            is_allowed_round_transition(
                RoundPhase.ADMISSION_FINALIZED,
                RoundPhase.VALUES_COLLECTING,
            )
        )
        self.assertTrue(
            is_allowed_round_transition(
                RoundPhase.DECRYPTING,
                RoundPhase.ABORTED,
            )
        )
        self.assertFalse(
            is_allowed_round_transition(
                RoundPhase.COMMITTED,
                RoundPhase.ABORTED,
            )
        )


if __name__ == "__main__":
    unittest.main()
    ClientSelectionRequest,
