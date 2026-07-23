import os
import tempfile
import unittest
from argparse import Namespace
from unittest.mock import Mock

from src.geneformer_assets import (
    get_model_spec,
    resolve_geneformer_assets,
    resolve_token_dictionary,
)


def args(**overrides):
    values = {
        "scfm_mode": "online_frozen",
        "scfm_model_path": None,
        "scfm_model_repo": "ctheodoris/Geneformer",
        "scfm_model_subfolder": "Geneformer-V1-10M",
        "scfm_token_dictionary_path": None,
        "scfm_token_dictionary_file": "dict.pkl",
        "hf_cache_dir": None,
        "scfm_model_version": "V1",
    }
    values.update(overrides)
    return Namespace(**values)


class GeneformerAssetsTest(unittest.TestCase):
    def test_local_checkpoint_takes_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            resolved = resolve_geneformer_assets(
                args(scfm_model_path=directory), require_token_dictionary=False
            )
            self.assertEqual(resolved.model_path, directory)
            self.assertIsNone(resolved.model_repo)

    def test_local_dictionary_does_not_download(self):
        with tempfile.NamedTemporaryFile() as handle:
            downloader = Mock(side_effect=AssertionError("must not download"))
            resolved = resolve_token_dictionary(
                args(scfm_token_dictionary_path=handle.name),
                downloader=downloader,
            )
            self.assertEqual(resolved, handle.name)
            downloader.assert_not_called()

    def test_missing_local_assets_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "does not exist"):
            resolve_geneformer_assets(
                args(scfm_model_path="/definitely/missing"),
                require_token_dictionary=False,
            )

    def test_precomputed_needs_no_assets(self):
        resolved = resolve_geneformer_assets(
            args(scfm_mode="precomputed"), require_token_dictionary=False
        )
        self.assertIsNone(resolved.model_path)
        self.assertIsNone(resolved.token_dictionary_path)

    def test_v1_contract(self):
        spec = get_model_spec("V1")
        self.assertEqual(spec.maximum_sequence_length, 2048)
        self.assertFalse(spec.special_tokens)


if __name__ == "__main__":
    unittest.main()
