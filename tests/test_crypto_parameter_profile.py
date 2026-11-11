import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from plaintext_fl.crypto_parameter_profile import load_crypto_parameter_profile
from plaintext_fl.native_engine import NativeCryptoEngineOptions
from plaintext_fl.model import resnet20, resnet32
from plaintext_fl.model_adapter import build_model_protocol_layouts


class CryptoParameterProfileTest(unittest.TestCase):
    def test_large_integer_and_decimal_string_scales(self):
        for as_string in (False, True):
            document = copy.deepcopy(load_crypto_parameter_profile().document)
            model, aux = 2**87+1, 2**81
            document["scaling"] = {"model_factor": str(model) if as_string else model,
                                   "aux_factor": str(aux) if as_string else aux}
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/"profile.json"
                path.write_text(json.dumps(document),encoding="utf-8")
                profile=load_crypto_parameter_profile(path)
                options=profile.native_options(master_seed=bytes(32))
                self.assertEqual(options.model_scaling_factor,model)
                self.assertEqual(options.aux_scaling_factor,aux)

    def test_noncanonical_or_oversized_scale_strings(self):
        for bad in ("01", "+12", "1e26", " 12", str(2**256), True):
            document = copy.deepcopy(load_crypto_parameter_profile().document)
            document["scaling"]["model_factor"]=bad
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/"profile.json"
                path.write_text(json.dumps(document),encoding="utf-8")
                with self.subTest(scale=bad), self.assertRaises(ValueError):
                    load_crypto_parameter_profile(path)

    def test_default_profile_matches_resnet20(self):
        profile = load_crypto_parameter_profile()
        self.assertEqual(profile.protocol_version, "v17")
        self.assertEqual(profile.q_modulus_bits, (48, 48))
        self.assertEqual(profile.selected_chunk_count(), 426)
        profile.validate_layouts(build_model_protocol_layouts(resnet20()))

    def test_all_packaged_profiles_use_v17_and_match_layouts(self):
        paths = list((Path(__file__).parents[1] / "plaintext_fl/profiles").glob("*.json"))
        self.assertGreaterEqual(len(paths), 4)
        for path in paths:
            with self.subTest(profile=path.name):
                profile = load_crypto_parameter_profile(path)
                self.assertEqual(profile.protocol_version, "v17")
                self.assertEqual(len(profile.q_modulus_bits), 2)
                self.assertEqual(profile.hybrid_large_digits, 2)
                model = resnet20() if profile.model_name == "resnet20-cifar10" else resnet32()
                profile.validate_layouts(build_model_protocol_layouts(model, model_name=profile.model_name))
                options = profile.native_options(master_seed=bytes(32))
                effective = options.to_binding_dict()
                sparse = profile.mode == "SPARSE"
                effective.update(dropout_attempt_cap=1,
                                 support_codec="CHUNK-BITMAP-MSB0-V1" if sparse else "NONE",
                                 support_codec_code=int(sparse))
                options.validate_effective_protocol(effective)
                effective["compression_cd"] += 1
                with self.assertRaisesRegex(RuntimeError, "effective protocol mismatch"):
                    options.validate_effective_protocol(effective)

    def test_retired_versions_rejected_at_profile_and_adapter_boundaries(self):
        for version in ("v12", "v23", "v27", "v28", "v99"):
            with self.subTest(version=version):
                with self.assertRaisesRegex(ValueError, "must be v17"):
                    NativeCryptoEngineOptions(protocol_version=version).validate()
                document = copy.deepcopy(load_crypto_parameter_profile().document)
                document["protocol_version"] = version
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "profile.json"
                    path.write_text(json.dumps(document), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "must use protocol v17"):
                        load_crypto_parameter_profile(path)

    def test_single_limb_rejected(self):
        document = copy.deepcopy(load_crypto_parameter_profile().document)
        document["ring"].update(q_modulus_bits=[60], hybrid_large_digits=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly two Q primes"):
                load_crypto_parameter_profile(path)

    def test_full_profile_uses_direct_rlwe_and_full_support(self):
        path = Path(__file__).parents[1] / "plaintext_fl/profiles/resnet32-cifar100-full-shared-v17.json"
        profile = load_crypto_parameter_profile(path)
        self.assertEqual(profile.mode, "FULL")
        self.assertEqual(profile.mlwe_rank, 1)
        self.assertEqual(profile.chunk_size, 8192)
        self.assertEqual(profile.selected_chunk_count(), 58)
        self.assertFalse(profile.native_options(master_seed=bytes(32)).experimental_dynamic_cohort)

    def test_crypto_threads_validated(self):
        self.assertEqual(NativeCryptoEngineOptions(crypto_threads=4).to_binding_dict()["crypto_threads"], 4)
        for count in (0, 257):
            with self.assertRaises(ValueError):
                NativeCryptoEngineOptions(crypto_threads=count).validate()
