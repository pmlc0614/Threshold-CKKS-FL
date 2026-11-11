import json
import unittest
from dataclasses import replace
from pathlib import Path

from plaintext_fl import wire_contract as wire
from plaintext_fl.native_engine import NativeCryptoEngine, native_binding_available
from tests.test_native_engine import make_options, make_setup, make_full_setup

VECTORS = json.loads((Path(__file__).parents[1] / 'interop/SD_V17_BYTE_VECTORS.json').read_text(encoding='utf-8'))


def native_input(case):
    value = dict(case['input'])
    for field in ('seed', 'setup_id', 'index_key'):
        value[field] = bytes.fromhex(value[field])
    return value


class WireReferenceTests(unittest.TestCase):
    def test_fixed_primitive_vectors(self):
        for case in VECTORS['cases']:
            with self.subTest(case=case['name']):
                p, expected = native_input(case), case['expected']
                tags = {}
                for path in ('FULL-MODEL', 'SP-MODEL', 'AUX'):
                    tags['local/' + path] = wire.local_a_tag(p['setup_id'], p['round_id'], p['client_id'], p['block'], path)
                    tags['offset/' + path] = wire.dd_tag(p['setup_id'], p['round_id'], p['block'], path)
                    tags['dither/' + path] = wire.dd_tag(p['setup_id'], p['round_id'], p['block'], path, p['client_id'])
                tags['PK-P1'] = b'PK-P1' + p['setup_id']
                tags['M2R-A'] = b'M2R-A' + p['setup_id'] + wire.be64(p['block']) + wire.be64(p['client_id'])
                self.assertEqual({k: v.hex() for k, v in tags.items()}, expected['tags'])
                for name, tag in tags.items():
                    self.assertEqual(list(wire.uniform_polynomial(p['seed'], tag, p['dimension'], p['prime'])), expected['masks'][name])
                self.assertEqual(wire.index_seed(p['index_key'], p['setup_id'], p['round_id'], p['chunks']).hex(), expected['index_seed'])
                self.assertEqual(list(wire.index_permutation(p['index_key'], p['setup_id'], p['round_id'], p['chunks'])), expected['permutation'])
                self.assertEqual(wire.pack_codes(p['codes'], p['bits']).hex(), expected['packed'])
                self.assertEqual(list(wire.unpack_codes(bytes.fromhex(expected['packed']), len(p['codes']), p['bits'])), expected['unpacked'])

    def test_public_setup_golden(self):
        v = VECTORS['public_setup']
        payload = wire.serialize_public_setup(bytes.fromhex(v['setup_id']), bytes.fromhex(v['crs_seed']), bytes.fromhex(v['descriptor']), v['external_ids'], [bytes.fromhex(h) for h in v['layout_hashes']])
        self.assertEqual(payload.hex(), v['expected'])
        parsed = wire.parse_public_setup(payload)
        self.assertEqual(parsed['external_ids'], (30, 10, 20, 40))
        self.assertEqual(parsed['descriptor'].hex(), v['descriptor'])

    def test_noncanonical_public_setup_rejected(self):
        valid = bytes.fromhex(VECTORS['public_setup']['expected'])
        # Roster starts after magic/version/IDs/descriptor length/descriptor/count.
        roster = 8 + 8 + 16 + 16 + 8 + 3 + 8
        wrong_position = bytearray(valid); wrong_position[roster + 7] = 1
        duplicate = bytearray(valid); duplicate[roster + 24:roster + 32] = duplicate[roster + 8:roster + 16]
        wrong_schema = bytearray(valid); wrong_schema[15] = 2
        for bad in (valid[:-1], valid + b'\0', bytes(wrong_position), bytes(duplicate), bytes(wrong_schema), b''):
            with self.subTest(payload=bad.hex()), self.assertRaises(ValueError):
                wire.parse_public_setup(bad)

    def test_invalid_width_padding_and_unsigned_integer(self):
        for value in (-1, 2**64, True, 1.5):
            with self.assertRaises(ValueError): wire.be64(value)
        for payload, count, bits in ((b'\x81', 1, 1), (b'', 1, 3), (b'\0', 1, 64)):
            with self.assertRaises(ValueError): wire.unpack_codes(payload, count, bits)
        with self.assertRaises(ValueError): wire.local_a_tag(bytes(16), 0, 0, 0, 'MODEL')


@unittest.skipUnless(native_binding_available(), 'native binding unavailable')
class NativeWireTests(unittest.TestCase):
    def test_native_matches_all_fixed_vectors(self):
        import ciphertext_fl_cpp as native
        for case in VECTORS['cases']:
            with self.subTest(case=case['name']):
                self.assertEqual(native._wire_vectors(native_input(case)), case['expected'])

    def test_native_rejects_malformed_payloads(self):
        import ciphertext_fl_cpp as native
        base = native_input(VECTORS['cases'][0])
        for override in ({'bitmap_override': b'\x90\xc1'}, {'bitmap_override': b'\x90'},
                         {'packed_override': bytes.fromhex(VECTORS['cases'][0]['expected']['packed'])[:-1]},
                         {'packed_override': bytes.fromhex(VECTORS['cases'][0]['expected']['packed'])[:-1] + b'\x01'},
                         {'labels': [0, 0]}, {'seed': bytes(15)}):
            with self.subTest(override=override), self.assertRaises((ValueError, RuntimeError)):
                native._wire_vectors({**base, **override})

    def test_actual_public_setup_all_modes_and_ranks(self):
        for mode, rank in (('SPARSE', 4), ('SPARSE', 1), ('FULL', 1)):
            with self.subTest(mode=mode, rank=rank):
                options = replace(make_options(), protocol_mode=mode, mlwe_rank=rank)
                setup = make_setup()
                if rank == 1:
                    options = replace(options, selected_chunks_per_client=None, selection_ratio=1.0)
                    setup = replace(setup, chunk_size=1024)
                if mode == 'FULL': setup = make_full_setup()
                engine = NativeCryptoEngine(options)
                engine.setup(setup)
                parsed = wire.parse_public_setup(engine.public_setup_binding)
                self.assertEqual(parsed['external_ids'], setup.client_ids)
                self.assertEqual(parsed['external_ids'][0], 30)
                self.assertEqual(parsed['descriptor'], wire.canonical_parameter_descriptor(engine.effective_protocol))
                self.assertEqual(len(parsed['crs_seed']), 16)
                self.assertEqual(parsed['layout_hashes'], tuple(bytes.fromhex(layout.sha256()) for layout in (setup.parameter_layout, setup.running_mean_layout, setup.running_variance_layout)))

    def test_adapter_rejects_tampered_native_setup(self):
        for field in ('setup_descriptor', 'public_setup_binding', 'crs_seed'):
            with self.subTest(field=field):
                engine = NativeCryptoEngine(make_options())
                backend = engine._backend
                class Tampered:
                    def setup(self, *args, **kwargs):
                        value = backend.setup(*args, **kwargs)
                        raw = bytes(value[field]); value[field] = raw[:-1] + bytes([raw[-1] ^ 1])
                        return value
                engine._backend = Tampered()
                with self.assertRaisesRegex(RuntimeError, 'mismatch'): engine.setup(make_setup())
                self.assertIsNone(engine.public_setup_binding)


if __name__ == '__main__':
    unittest.main()
