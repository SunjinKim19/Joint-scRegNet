import unittest

import torch

from src.scfm_encoder import GeneRepresentationPooler


class GenePoolingTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
