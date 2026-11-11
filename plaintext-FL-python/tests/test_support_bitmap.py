import unittest

from plaintext_fl.support_bitmap import (
    SUPPORT_CODEC_CODE,
    SUPPORT_CODEC_NAME,
    bitmap_byte_count,
    decode_support_bitmap,
    encode_support_bitmap,
    intersect_support_bitmaps,
)
from plaintext_fl.plaintext_communication import (
    measure_plaintext_round_communication,
)


class SupportBitmapTest(unittest.TestCase):
    def test_all_byte_values_round_trip(self):
        for byte_value in range(256):
            positions = tuple(
                bit_index
                for bit_index in range(8)
                if byte_value & (0x80 >> bit_index)
            )
            encoded = encode_support_bitmap(8, positions)
            self.assertEqual(encoded, bytes((byte_value,)))
            self.assertEqual(
                decode_support_bitmap(
                    8,
                    encoded,
                    expected_popcount=len(positions),
                ),
                positions,
            )

    def test_normative_vectors(self):
        self.assertEqual(SUPPORT_CODEC_CODE, 1)
        self.assertEqual(SUPPORT_CODEC_NAME, "CHUNK-BITMAP-MSB0-V1")
        self.assertEqual(encode_support_bitmap(10, (0, 3, 8)), b"\x90\x80")
        self.assertEqual(
            decode_support_bitmap(10, b"\x90\x80", expected_popcount=3),
            (0, 3, 8),
        )
        self.assertEqual(encode_support_bitmap(1, ()), b"\x00")
        self.assertEqual(encode_support_bitmap(1, (0,)), b"\x80")
        self.assertEqual(encode_support_bitmap(8, range(8)), b"\xff")
        self.assertEqual(encode_support_bitmap(9, (0, 8)), b"\x80\x80")
        self.assertEqual(bitmap_byte_count(231), 29)

    def test_rejects_padding_length_popcount_and_duplicates(self):
        with self.assertRaises(ValueError):
            decode_support_bitmap(10, b"\x90\x81")
        with self.assertRaises(ValueError):
            decode_support_bitmap(10, b"\x90")
        with self.assertRaises(ValueError):
            decode_support_bitmap(10, b"\x90\x80", expected_popcount=2)
        with self.assertRaises(ValueError):
            encode_support_bitmap(10, (3, 3))

    def test_intersection_is_canonical_upload_support(self):
        selected = encode_support_bitmap(10, (0, 3, 8))
        admitted = encode_support_bitmap(10, (3, 8, 9))
        self.assertEqual(
            decode_support_bitmap(
                10,
                intersect_support_bitmaps(10, selected, admitted),
            ),
            (3, 8),
        )

    def test_plaintext_sparse_accounting_uses_four_bitmaps_and_b_padding(self):
        report = measure_plaintext_round_communication(
            profile="SPARSE",
            cohort_sizes=(3, 0, 3),
            minimum_cohort_size=3,
            client_count=3,
            selected_chunk_count=2,
            parameter_numel=10,
            chunk_size=4,
            auxiliary_numel=2,
        )
        self.assertEqual(report["support_bitmap_bytes"], 1)
        self.assertEqual(report["selection_bitmap_bytes"], 3)
        self.assertEqual(report["upload_support_bitmap_bytes"], 3)
        self.assertEqual(report["admission_bitmap_bytes"], 3)
        self.assertEqual(report["result_bitmap_bytes"], 3)
        self.assertEqual(report["accepted_wire_value_numel"], 24)
        self.assertEqual(report["accepted_aggregate_value_numel"], 8)
        self.assertEqual(report["accepted_only_fp32_uplink_bytes"], 126)
        self.assertEqual(report["accepted_only_logical_downlink_bytes"], 42)
        self.assertEqual(report["accepted_only_unicast_downlink_bytes"], 126)


if __name__ == "__main__":
    unittest.main()
