import os
import sys
import tempfile
import types
import unittest
from argparse import Namespace
from unittest.mock import patch

import torch
import torch.nn as nn

from src.models_cell_guided_graph import CellGuidedGraphScRegNet
from src.scfm_encoder import ScFMEncoder
from src.train_cell_guided_graph import check_trainable_gradients_after_backward


def make_args(**overrides):
    values = {
        "scfm_mode": "precomputed",
        "scfm_model_path": None,
        "scfm_tokenized_path": None,
        "scfm_output_layer": "last_hidden",
        "scfm_pooling": "gene",
        "train_scfm_top_layers": 0,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "lora_target_modules": "query,value",
        "cache_online_scfm_outputs": False,
        "max_scfm_cells": 0,
    }
    values.update(overrides)
    return Namespace(**values)


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])

    def forward(self, hidden):
        for layer in self.layer:
            hidden = torch.tanh(layer(hidden))
        return hidden


class FakeScFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=4)
        self.embeddings = nn.Embedding(16, 4)
        self.encoder = FakeEncoder()

    def forward(self, input_ids, **_):
        hidden = self.encoder(self.embeddings(input_ids))
        return types.SimpleNamespace(
            last_hidden_state=hidden, hidden_states=(hidden,)
        )


class FakeAutoModel:
    @staticmethod
    def from_pretrained(_):
        return FakeScFM()


class ScFMEncoderTest(unittest.TestCase):
    def test_precomputed_mode_returns_input_without_backbone(self):
        encoder = ScFMEncoder(make_args(), torch.device("cpu"))
        embeddings = torch.randn(3, 5)
        self.assertIs(
            encoder({"precomputed_embeddings": embeddings, "num_genes": 3}),
            embeddings,
        )
        self.assertFalse(encoder.backbone_loaded)

    def test_online_mode_fails_loudly_when_paths_are_missing(self):
        with self.assertRaisesRegex(ValueError, "scfm_model_path"):
            ScFMEncoder(
                make_args(scfm_mode="online_lora"), torch.device("cpu")
            )

    def test_online_mode_rejects_missing_token_to_gene_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            tokenized_path = os.path.join(directory, "tokens.pt")
            torch.save({"input_ids": torch.tensor([[1, 2]])}, tokenized_path)
            with self.assertRaisesRegex(ValueError, "token-to-gene mapping"):
                ScFMEncoder(
                    make_args(
                        scfm_mode="online_topk",
                        scfm_model_path="fake-model",
                        scfm_tokenized_path=tokenized_path,
                        train_scfm_top_layers=1,
                    ),
                    torch.device("cpu"),
                )

    def test_online_topk_preserves_gradient_to_real_backbone_output(self):
        with tempfile.TemporaryDirectory() as directory:
            tokenized_path = os.path.join(directory, "tokens.pt")
            torch.save(
                {
                    "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
                    "attention_mask": torch.ones(2, 3, dtype=torch.long),
                    "gene_indices": torch.tensor([[0, 1, 2], [0, 1, 2]]),
                },
                tokenized_path,
            )
            fake_transformers = types.ModuleType("transformers")
            fake_transformers.AutoModel = FakeAutoModel
            args = make_args(
                scfm_mode="online_topk",
                scfm_model_path="fake-local-model",
                scfm_tokenized_path=tokenized_path,
                train_scfm_top_layers=1,
            )
            with patch.dict(sys.modules, {"transformers": fake_transformers}):
                encoder = ScFMEncoder(args, torch.device("cpu"))
                gene_embeddings = encoder({"num_genes": 3})
                model = CellGuidedGraphScRegNet(
                    num_genes=3,
                    scfm_dim=4,
                    latent_dim=4,
                    gnn_hidden_dims=[4],
                    dropout=0.0,
                    detach_scfm_input=False,
                )
                output = model(
                    gene_embeddings,
                    torch.eye(3),
                    torch.tensor([[0, 1], [1, 2]]),
                    raw_expr=torch.randn(2, 3),
                    tf_indices=torch.tensor([0, 1]),
                )
                output["logits"].sum().backward()
                check_trainable_gradients_after_backward(model, args, encoder)

            self.assertEqual(gene_embeddings.shape, (3, 4))
            self.assertTrue(gene_embeddings.requires_grad)
            top_layer = encoder.model.encoder.layer[-1]
            self.assertTrue(
                any(
                    parameter.grad is not None
                    and parameter.grad.abs().sum() > 0
                    for parameter in top_layer.parameters()
                )
            )
            self.assertTrue(
                all(
                    not parameter.requires_grad
                    for parameter in encoder.model.embeddings.parameters()
                )
            )


if __name__ == "__main__":
    unittest.main()
