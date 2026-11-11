"""Production rounds must not invoke the privileged plaintext reference helpers."""
import unittest
from unittest.mock import patch
import torch
from plaintext_fl.encrypted_training import (EncryptedSparseRoundConfig,
    run_encrypted_full_round, run_encrypted_sparse_round)
from plaintext_fl.plaintext_engine import PlaintextCryptoEngine
from tests.test_encrypted_sparse import TinyClassifier, make_loaders


class RuntimeReferenceTest(unittest.TestCase):
    def test_reference_helpers_are_not_called_in_normal_rounds(self):
        for mode, function, helper in (
            ('FULL', run_encrypted_full_round, '_aggregate_full_reference'),
            ('SPARSE', run_encrypted_sparse_round, '_aggregate_admitted_reference'),
        ):
            with self.subTest(mode=mode):
                torch.manual_seed(17)
                config = EncryptedSparseRoundConfig(chunk_size=8, selected_chunk_count=2,
                    minimum_cohort_size=2, local_epochs=1, learning_rate=.01,
                    momentum=0, weight_decay=0, aggregate_update_bound=10,
                    aggregate_auxiliary_update_bound=10, protocol_mode=mode,
                    compute_plaintext_reference=False)
                with patch('plaintext_fl.encrypted_training.' + helper,
                           side_effect=AssertionError('reference called in production')):
                    result = function(global_model=TinyClassifier(),
                        client_loaders=make_loaders(), sample_counts=(4,4),
                        engine=PlaintextCryptoEngine(), config=config,
                        device=torch.device('cpu'))
                self.assertIsNone(result.plaintext_aggregate)
                self.assertIsNone(result.plaintext_aggregate_maxima)
                self.assertIsNotNone(result.decrypted_aggregate_maxima)


if __name__ == '__main__': unittest.main()
