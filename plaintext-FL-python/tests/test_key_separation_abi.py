"""Prevent loading native binaries predating Full-SD key separation."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from plaintext_fl.native_engine import _load_native_module, NativeBindingUnavailableError


class KeySeparationAbiTests(unittest.TestCase):
    def test_old_native_module_is_rejected(self):
        old = SimpleNamespace(BINDING_ABI_VERSION=18, PROTOCOL_VERSION="v17")
        with patch("plaintext_fl.native_engine.importlib.import_module", return_value=old):
            with self.assertRaises(NativeBindingUnavailableError):
                _load_native_module()

    def test_key_separation_module_is_accepted(self):
        current = SimpleNamespace(BINDING_ABI_VERSION=22, PROTOCOL_VERSION="v17")
        with patch("plaintext_fl.native_engine.importlib.import_module", return_value=current):
            self.assertIs(_load_native_module(), current)
