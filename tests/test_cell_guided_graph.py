import unittest
from unittest.mock import Mock

import torch

from src.models_cell_guided_graph import CellGuidedGraphScRegNet
from src.train_cell_guided_graph import compute_sparse_loss, log_adj_stats


class CellGuidedGraphSmokeTest(unittest.TestCase):
    def _run_constructor(self, constructor_type):
        torch.manual_seed(0)
        num_genes, latent_dim = 10, 8
        model = CellGuidedGraphScRegNet(
            num_genes=num_genes,
            scfm_dim=12,
            latent_dim=latent_dim,
            condition_hidden_dim=16,
            gnn_hidden_dims=[16, latent_dim],
            link_hidden_dim=16,
            dropout=0.0,
            graph_alpha=0.8,
            graph_constructor_type=constructor_type,
        )
        tf_indices = torch.tensor([0, 2, 4])
        edge_pairs = torch.tensor([[0, 1], [2, 3], [4, 5], [0, 6]])
        output = model(
            scfm_gene_emb=torch.randn(num_genes, 12),
            prior_adjacency=torch.eye(num_genes).to_sparse(),
            edge_pairs=edge_pairs,
            raw_expr=torch.randn(6, num_genes),
            tf_indices=tf_indices,
        )
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            output["logits"], labels
        ) + 0.01 * output["A_ctx"].mean()
        loss.backward()

        self.assertEqual(output["z_ctx"].shape, (num_genes, latent_dim))
        self.assertEqual(output["A_ctx"].shape, (num_genes, num_genes))
        self.assertEqual(output["z_graph"].shape, (num_genes, latent_dim))
        self.assertEqual(output["probabilities"].shape, (len(edge_pairs),))
        self.assertIs(output["pred"], output["probabilities"])
        self.assertEqual(output["candidate_mask"].shape, (num_genes, num_genes))
        self.assertEqual(output["candidate_mask"].dtype, torch.bool)
        self.assertTrue(output["z_ctx"].requires_grad)
        self.assertTrue(output["A_ctx"].requires_grad)
        self.assertTrue(
            any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.graph_constructor.parameters())
        )
        self.assertTrue(
            any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.cell_m.parameters())
        )
        non_tf = torch.tensor([1, 3, 5, 6, 7, 8, 9])
        self.assertTrue(torch.equal(output["A_ctx"][non_tf], torch.zeros_like(output["A_ctx"][non_tf])))

    def test_mlp_constructor_gradient_path(self):
        self._run_constructor("mlp")

    def test_bilinear_constructor_gradient_path(self):
        self._run_constructor("bilinear")

    def test_hard_topk_is_rejected_during_training(self):
        model = CellGuidedGraphScRegNet(6, 4, latent_dim=4, gnn_hidden_dims=[4])
        with self.assertRaisesRegex(RuntimeError, "cannot be used while training"):
            model(
                torch.randn(6, 4),
                torch.eye(6),
                torch.tensor([[0, 1]]),
                tf_indices=torch.tensor([0]),
                hard_topk_eval_only=1,
            )

    def test_adjacency_stats_name_all_and_candidate_regions(self):
        test_logger = Mock()
        adjacency = torch.tensor([[9.0, 0.2], [0.8, 7.0]])
        candidate_mask = torch.tensor([[True, True], [False, False]])
        log_adj_stats(test_logger, "A_ctx", adjacency, candidate_mask)

        all_format, *all_args = test_logger.info.call_args_list[0].args
        candidate_format, *candidate_args = test_logger.info.call_args_list[1].args
        self.assertEqual(all_args[0], "A_ctx_all_offdiag")
        self.assertEqual(candidate_args[0], "A_ctx_candidate")
        self.assertEqual(all_args[1], "A_ctx_all_offdiag")
        self.assertAlmostEqual(all_args[2], 0.5)
        self.assertEqual(candidate_args[1], "A_ctx_candidate")
        self.assertAlmostEqual(candidate_args[2], 0.2)
        self.assertIn("density_%s@0.1", all_format)
        self.assertIn("density_%s@0.9", candidate_format)

    def test_sparse_loss_uses_candidate_without_diagonal(self):
        adjacency = torch.tensor([[100.0, 0.2], [0.8, 200.0]], requires_grad=True)
        candidate_mask = torch.tensor([[True, True], [False, False]])
        sparse_loss, basis = compute_sparse_loss(adjacency, candidate_mask)
        self.assertEqual(basis, "candidate")
        self.assertAlmostEqual(sparse_loss.item(), 0.2)
        sparse_loss.backward()
        self.assertTrue(adjacency.grad[0, 1] > 0)
        self.assertEqual(adjacency.grad[0, 0].item(), 0.0)

    def test_sparse_loss_falls_back_to_all_offdiag(self):
        adjacency = torch.tensor([[100.0, 0.2], [0.8, 200.0]])
        sparse_loss, basis = compute_sparse_loss(adjacency)
        self.assertEqual(basis, "all_offdiag")
        self.assertAlmostEqual(sparse_loss.item(), 0.5)

    def test_fixed_fusion_preserves_scalar_alpha_formula(self):
        model = CellGuidedGraphScRegNet(
            6,
            4,
            latent_dim=4,
            gnn_hidden_dims=[4],
            dropout=0.0,
            graph_alpha=0.8,
            graph_fusion_type="fixed",
        )
        output = model(
            torch.randn(6, 4),
            torch.eye(6),
            torch.tensor([[0, 1], [2, 3]]),
            tf_indices=torch.tensor([0, 2]),
        )
        expected = 0.8 * output["A_prior"] + 0.2 * output["A_ctx"]
        self.assertTrue(torch.allclose(output["A_final"], expected))
        self.assertNotIn("gate", output)

    def test_edge_gate_is_differentiable_and_initialized_from_alpha(self):
        model = CellGuidedGraphScRegNet(
            8,
            6,
            latent_dim=4,
            gnn_hidden_dims=[4],
            link_hidden_dim=8,
            dropout=0.0,
            graph_alpha=0.8,
            graph_fusion_type="edge_gate",
            gate_hidden_dim=5,
            gate_temperature=1.0,
            gate_init_from_alpha=True,
        )
        output = model(
            torch.randn(8, 6),
            torch.eye(8),
            torch.tensor([[0, 1], [2, 3], [0, 4]]),
            raw_expr=torch.randn(3, 8),
            tf_indices=torch.tensor([0, 2]),
        )
        self.assertEqual(output["gate"].shape, (8, 8))
        self.assertTrue(output["gate"].requires_grad)
        self.assertTrue(torch.allclose(output["gate"], torch.full((8, 8), 0.8)))
        self.assertFalse(torch.allclose(output["A_final"], output["A_prior"]))
        self.assertFalse(torch.allclose(output["A_final"], output["A_ctx"]))
        output["logits"].sum().backward()
        self.assertTrue(
            any(
                parameter.grad is not None and parameter.grad.abs().sum() > 0
                for parameter in model.edge_gate.parameters()
            )
        )
        self.assertTrue(
            any(
                parameter.grad is not None and parameter.grad.abs().sum() > 0
                for parameter in model.graph_constructor.parameters()
            )
        )


if __name__ == "__main__":
    unittest.main()
