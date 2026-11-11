import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from plaintext_fl import encrypted_full, encrypted_sparse, encrypted_training
from plaintext_fl.protocol_contract import BinaryHeader

ROOT=Path(__file__).parents[1]/'plaintext_fl/profiles'
class ModeEntryPointTest(unittest.TestCase):
    def test_mode_defaults(self):
        self.assertEqual(encrypted_training._load_cli_profile(None,'FULL').mode,'FULL')
        self.assertEqual(encrypted_training._load_cli_profile(None,'SPARSE').mode,'SPARSE')
    def test_wrong_mode_fails_before_dataset_loading(self):
        for mode,profile in [('FULL','resnet20-cifar10-sp-k64-rho020-shared-v17.json'),('SPARSE','resnet20-cifar10-full-shared-v17.json')]:
            args=SimpleNamespace(crypto_parameter_profile=ROOT/profile)
            with patch.object(encrypted_training,'_parse_args',return_value=args), patch.object(encrypted_training,'_load_encrypted_dataset') as load:
                with self.assertRaisesRegex(ValueError,'requires a'):
                    encrypted_training.main(expected_mode=mode)
                load.assert_not_called()
    def test_entrypoints_enforce_mode(self):
        for module,mode in [(encrypted_full,'FULL'),(encrypted_sparse,'SPARSE')]:
            with patch.object(module,'run_training') as run:
                module.main()
                run.assert_called_once_with(expected_mode=mode)
    def test_old_wire_header_rejected(self):
        import json
        record=json.loads((ROOT.parents[1]/'interop/golden-v1.json').read_text())
        current=bytes.fromhex(record['binary_header']['expected_hex'])
        BinaryHeader.from_bytes(current)
        with self.assertRaises(ValueError):
            BinaryHeader.from_bytes(b'FLV12BIN'+current[8:])
