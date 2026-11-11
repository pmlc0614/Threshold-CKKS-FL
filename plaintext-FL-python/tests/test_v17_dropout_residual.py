from dataclasses import replace
import unittest
import torch
from tests.test_encrypted_sparse import TinyClassifier, make_loaders
from plaintext_fl.encrypted_training import EncryptedSparseRoundConfig, run_encrypted_sparse_round
from plaintext_fl.plaintext_engine import PlaintextCryptoEngine


class DropoutResidualTest(unittest.TestCase):
    def test_offline_client_discards_tentative_residual_before_rejoin(self):
        torch.manual_seed(17)
        model = TinyClassifier()
        engine = PlaintextCryptoEngine(experimental_dynamic_cohort=True)
        config = EncryptedSparseRoundConfig(chunk_size=8, selected_chunk_count=1,
            minimum_cohort_size=3, local_epochs=1, learning_rate=0.01,
            momentum=0, weight_decay=0, aggregate_update_bound=10,
            aggregate_auxiliary_update_bound=10)
        arguments = dict(global_model=model, client_loaders=make_loaders()*2,
                         sample_counts=(4,4,4,4), engine=engine,
                         device=torch.device('cpu'))
        first = run_encrypted_sparse_round(**arguments, config=config)
        previous = tuple(value.clone() for value in first.residuals)
        dropped = run_encrypted_sparse_round(**arguments, config=replace(config, round_id=1),
            residuals=first.residuals, survivor_client_ids=(1,2,3))
        self.assertEqual(torch.count_nonzero(dropped.residuals[0]), 0)
        for original, saved in zip(first.residuals, previous):
            torch.testing.assert_close(original, saved)
        rejoined = run_encrypted_sparse_round(**arguments, config=replace(config, round_id=2),
                                             residuals=dropped.residuals)
        self.assertEqual(rejoined.result.survivor_client_ids, (0,1,2,3))


if __name__ == '__main__': unittest.main()
