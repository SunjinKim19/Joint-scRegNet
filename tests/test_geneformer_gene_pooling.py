import unittest

import torch

from src.scfm_encoder import GeneRepresentationPooler


class GenePoolingTest(unittest.TestCase):
    def test_mean_pooling_matches_original_formula(self):
        pooler = GeneRepresentationPooler(num_genes=3, hidden_dim=2)
        hidden = torch.tensor(
            [[[1.0, 3.0], [4.0, 8.0]], [[3.0, 5.0], [6.0, 10.0]]],
            requires_grad=True,
        )
        attention = torch.ones(2, 2, dtype=torch.long)
        mapping = torch.tensor([[0, 1], [0, 1]])
        gene_sum, observation_count = pooler.accumulate(
            hidden, attention, mapping
        )
        pooled = gene_sum / observation_count.clamp_min(1)
        self.assertTrue(
            torch.equal(pooled[0], hidden[:, 0].sum(dim=0) / 2)
        )
        self.assertTrue(
            torch.equal(pooled[1], hidden[:, 1].sum(dim=0) / 2)
        )

    def test_scatter_pooling_padding_fallback_and_gradients(self):
        pooler = GeneRepresentationPooler(num_genes=4, hidden_dim=2)
        hidden = torch.tensor(
            [[[1.0, 2.0], [10.0, 20.0]], [[3.0, 4.0], [30.0, 40.0]]],
            requires_grad=True,
        )
        attention = torch.tensor([[1, 1], [1, 0]])
        mapping = torch.tensor([[0, 1], [0, -1]])
        output = pooler(hidden, attention, mapping)
        self.assertEqual(output.shape, (4, 2))
        self.assertTrue(torch.allclose(output[0], torch.tensor([2.0, 3.0])))
        self.assertTrue(torch.allclose(output[1], torch.tensor([10.0, 20.0])))
        self.assertTrue(torch.allclose(output[2], pooler.fallback_gene_embeddings.weight[2]))
        output.sum().backward()
        self.assertTrue(hidden.grad.abs().sum() > 0)
        self.assertTrue(pooler.fallback_gene_embeddings.weight.grad[2].abs().sum() > 0)
        self.assertTrue(torch.isfinite(output).all())

    def test_multiple_batch_accumulation_matches_single_batch(self):
        pooler = GeneRepresentationPooler(3, 2)
        hidden = torch.randn(4, 2, 2)
        attention = torch.ones(4, 2, dtype=torch.long)
        mapping = torch.tensor([[0, 1], [1, 2], [0, 2], [2, 1]])
        whole_sum, whole_count = pooler.accumulate(hidden, attention, mapping)
        first = pooler.accumulate(hidden[:2], attention[:2], mapping[:2])
        second = pooler.accumulate(hidden[2:], attention[2:], mapping[2:])
        combined = pooler.finalize(first[0] + second[0], first[1] + second[1])
        whole = pooler.finalize(whole_sum, whole_count)
        self.assertTrue(torch.allclose(combined, whole))

    def test_expression_weighted_pooling_shape_sanitization_and_gradient(self):
        pooler = GeneRepresentationPooler(num_genes=910, hidden_dim=256)
        hidden = torch.randn(2, 3, 256, requires_grad=True)
        attention = torch.ones(2, 3, dtype=torch.long)
        mapping = torch.tensor([[0, 1, 2], [0, 1, 2]])
        expression_weights = torch.tensor(
            [[1.0, float("nan"), -2.0], [3.0, 2.0, float("inf")]]
        )
        weighted_sum, weight_sum = pooler.accumulate_expression_weighted(
            hidden, attention, mapping, expression_weights
        )
        pooled = weighted_sum / weight_sum.clamp_min(1e-8)
        observed = weight_sum.squeeze(-1).gt(0)
        output = torch.where(
            observed.unsqueeze(-1),
            pooled,
            pooler.fallback_gene_embeddings.weight,
        )
        self.assertEqual(output.shape, (910, 256))
        self.assertTrue(
            torch.allclose(
                output[0], (hidden[0, 0] + 3 * hidden[1, 0]) / 4
            )
        )
        self.assertTrue(torch.equal(output[1], hidden[1, 1]))
        self.assertTrue(
            torch.equal(
                output[2], pooler.fallback_gene_embeddings.weight[2]
            )
        )
        self.assertTrue(output.requires_grad)
        output.sum().backward()
        self.assertTrue(torch.isfinite(hidden.grad).all())
        self.assertGreater(hidden.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
