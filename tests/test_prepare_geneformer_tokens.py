import os
import pickle
import tempfile
import unittest

import pandas as pd
import torch

from src.prepare_geneformer_tokens import (
    prepare_geneformer_tokens,
    validate_token_artifact,
)


class PrepareGeneformerTokensTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.expression_path = os.path.join(self.temp.name, "expression.csv")
        self.mapping_path = os.path.join(self.temp.name, "mapping.csv")
        self.dictionary_path = os.path.join(self.temp.name, "tokens.pkl")
        self.output_path = os.path.join(self.temp.name, "tokens.pt")
        genes = ["A", "B", "C", "D", "E"]
        pd.DataFrame(
            {
                "gene": genes,
                "cell1": [5.0, 5.0, 9.0, 8.0, 0.0],
                "cell2": [1.0, 3.0, 4.0, 2.0, 0.0],
            }
        ).to_csv(self.expression_path, index=False)
        pd.DataFrame(
            {
                "gene": genes,
                "ensembl_id": [
                    "ENSG00000000001",
                    "ENSG00000000002",
                    "ENSG",
                    "ENSG00000000004",
                    "ENSG00000000005",
                ],
            }
        ).to_csv(self.mapping_path, index=False)
        with open(self.dictionary_path, "wb") as handle:
            pickle.dump(
                {
                    "ENSG00000000001": 11,
                    "ENSG00000000002": 12,
                    "ENSG00000000005": 15,
                },
                handle,
            )

    def tearDown(self):
        self.temp.cleanup()

    def test_rank_tokenization_preserves_original_indices(self):
        artifact, summary = prepare_geneformer_tokens(
            self.expression_path,
            self.mapping_path,
            self.output_path,
            self.dictionary_path,
        )
        self.assertTrue(os.path.isfile(summary))
        self.assertEqual(artifact["input_ids"][0, :2].tolist(), [11, 12])
        self.assertEqual(artifact["gene_index_map"][0, :2].tolist(), [0, 1])
        self.assertEqual(artifact["gene_index_map"][1, :2].tolist(), [1, 0])
        self.assertFalse(artifact["tokenizable_gene_mask"][2])
        self.assertFalse(artifact["tokenizable_gene_mask"][3])
        self.assertTrue((artifact["gene_index_map"][artifact["attention_mask"] == 0] == -1).all())
        self.assertFalse(artifact["metadata"]["special_tokens_added"])
        loaded = torch.load(self.output_path, weights_only=False)
        self.assertTrue(torch.equal(loaded["input_ids"], artifact["input_ids"]))
        self.assertEqual(validate_token_artifact(loaded)["cell_count"], 2)

    def test_mapping_order_mismatch_fails(self):
        mapping = pd.read_csv(self.mapping_path)
        mapping.loc[[0, 1], "gene"] = ["B", "A"]
        mapping.to_csv(self.mapping_path, index=False)
        with self.assertRaisesRegex(ValueError, "gene order differ"):
            prepare_geneformer_tokens(
                self.expression_path,
                self.mapping_path,
                self.output_path,
                self.dictionary_path,
            )


if __name__ == "__main__":
    unittest.main()
