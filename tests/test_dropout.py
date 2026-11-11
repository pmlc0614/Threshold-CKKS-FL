import random
import unittest

from plaintext_fl.dropout import build_dropout_schedule


class DropoutScheduleTest(unittest.TestCase):
    def test_schedule_is_fixed_distinct_and_rng_isolated(self):
        random.seed(91)
        state = random.getstate()
        first = build_dropout_schedule(
            total_rounds=200,
            dropout_round_count=10,
            client_count=8,
            seed=42,
        )
        self.assertEqual(state, random.getstate())
        second = build_dropout_schedule(
            total_rounds=200,
            dropout_round_count=10,
            client_count=8,
            seed=42,
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 10)
        self.assertEqual(len({event.round_id for event in first}), 10)
        for event in first:
            self.assertEqual(len(event.survivor_client_ids), 7)
            self.assertNotIn(
                event.dropped_client_id, event.survivor_client_ids
            )

    def test_invalid_schedule_dimensions_are_rejected(self):
        with self.assertRaises(ValueError):
            build_dropout_schedule(
                total_rounds=5,
                dropout_round_count=6,
                client_count=8,
                seed=0,
            )


if __name__ == "__main__":
    unittest.main()
