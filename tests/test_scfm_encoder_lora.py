import importlib.util
import os
import tempfile
import unittest
from argparse import Namespace

import torch
import torch.nn.functional as F

from src.models_cell_guided_graph import CellGuidedGraphScRegNet
from src.scfm_encoder import ScFMEncoder
from src.train_cell_guided_graph import (
    _gradient_diagnostics,
    check_trainable_gradients_after_backward,
)


ONLINE_DEPENDENCIES_AVAILABLE = (
    importlib.util.find_spec("transformers") is not None
    and importlib.util.find_spec("peft") is not None
)


def make_lora_args(model_path, tokenized_path):
    return Namespace(
        scfm_mode="online_lora",
        scfm_model_path=model_path,
        scfm_model_repo=None,
        scfm_model_subfolder=None,
        hf_cache_dir=None,
        scfm_model_version="V1",
        scfm_tokenized_path=tokenized_path,
        scfm_output_layer="last_hidden",
        scfm_pooling="gene",
        train_scfm_top_layers=0,
        lora_rank=4,
        lora_r=None,
        lora_alpha=8,
        lora_dropout=0.0,
        lora_target_modules="query,value",
        cache_online_scfm_outputs=False,
        max_scfm_cells=0,
        scfm_cell_batch_size=1,
        scfm_cell_sampling="all",
        scfm_seed=7,
        gradient_checkpointing=False,
    )


@unittest.skipUnless(
    ONLINE_DEPENDENCIES_AVAILABLE,
    "actual tiny-model LoRA test requires transformers and peft",
)
class ScFMEncoderLoraTest(unittest.TestCase):
    def test_actual_peft_forward_pooling_and_backward(self):
        from peft import PeftModel
        from transformers import BertConfig, BertModel

        torch.manual_seed(7)
        with tempfile.TemporaryDirectory() as directory:
            BertModel(
                BertConfig(
                    vocab_size=32,
                    hidden_size=16,
                    num_hidden_layers=2,
                    num_attention_heads=4,
                    intermediate_size=32,
                    max_position_embeddings=32,
                )
            ).save_pretrained(directory)
            tokenized_path = os.path.join(directory, "tokens.pt")
            torch.save(
                {
                    "input_ids": torch.tensor(
                        [[1, 2, 3, 4], [5, 6, 7, 0]], dtype=torch.long
                    ),
                    "attention_mask": torch.tensor(
                        [[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.long
                    ),
                    "gene_index_map": torch.tensor(
                        [[0, 1, 2, 3], [0, 1, 2, -1]], dtype=torch.long
                    ),
                    "metadata": {"original_gene_count": 5},
                },
                tokenized_path,
            )
            args = make_lora_args(directory, tokenized_path)
            encoder = ScFMEncoder(args, torch.device("cpu"))
            self.assertIsInstance(encoder.model, PeftModel)

            wrapper_forward_calls = []
            hook = encoder.model.register_forward_hook(
                lambda *_: wrapper_forward_calls.append(1)
            )
            gene_embeddings = encoder({"num_genes": 5})
            hook.remove()
            self.assertEqual(len(wrapper_forward_calls), 2)
            self.assertEqual(gene_embeddings.shape, (5, 16))
            self.assertEqual(encoder.last_diagnostics["pooled_gene_count"], 4)
            self.assertEqual(encoder.last_diagnostics["fallback_gene_count"], 1)

            downstream = CellGuidedGraphScRegNet(
                num_genes=5,
                scfm_dim=16,
                latent_dim=8,
                condition_hidden_dim=8,
                gnn_hidden_dims=[8],
                link_hidden_dim=8,
                dropout=0.0,
                detach_scfm_input=False,
            )
            lora_parameters = [
                parameter
                for _, parameter in encoder.lora_named_parameters(
                    trainable_only=True
                )
            ]
            optimizer = torch.optim.AdamW(
                [
                    {"params": downstream.parameters()},
                    {"params": encoder.fallback_parameters()},
                    {"params": lora_parameters},
                ],
                lr=1e-3,
            )
            optimizer_ids = {
                id(parameter)
                for group in optimizer.param_groups
                for parameter in group["params"]
            }
            self.assertTrue(
                encoder.lora_parameter_ids_from_forward_model()
                <= optimizer_ids
            )

            output = downstream(
                gene_embeddings,
                torch.eye(5),
                torch.tensor([[0, 1], [1, 2], [2, 3], [0, 4]]),
                raw_expr=torch.randn(3, 5),
                tf_indices=torch.tensor([0, 1, 2]),
            )
            loss = F.binary_cross_entropy_with_logits(
                output["logits"], torch.tensor([1.0, 0.0, 1.0, 0.0])
            )
            loss.backward()
            check_trainable_gradients_after_backward(downstream, args, encoder)

            raw_hidden_grad_sum = sum(
                hidden.grad.abs().sum().item()
                for hidden in encoder._debug_backbone_hiddens
            )
            lora_diagnostics = _gradient_diagnostics(lora_parameters)
            downstream_grad_sum = _gradient_diagnostics(
                parameter
                for parameter in downstream.parameters()
                if parameter.requires_grad
            )["absolute_sum"]
            self.assertGreater(raw_hidden_grad_sum, 0)
            self.assertTrue(
                all(
                    torch.isfinite(hidden.grad).all()
                    for hidden in encoder._debug_backbone_hiddens
                )
            )
            self.assertGreater(lora_diagnostics["nonzero_count"], 0)
            self.assertGreater(lora_diagnostics["absolute_sum"], 0)
            self.assertGreater(downstream_grad_sum, 0)
            self.assertTrue(
                all(
                    not parameter.requires_grad
                    for name, parameter in encoder.model.named_parameters()
                    if "lora_" not in name
                )
            )
            print(
                "LORA_TEST_METRICS "
                f"raw_hidden_grad_sum={raw_hidden_grad_sum:.12g} "
                "lora_nonzero_grad_parameter_count="
                f"{lora_diagnostics['nonzero_count']} "
                f"lora_total_grad_sum={lora_diagnostics['absolute_sum']:.12g} "
                f"downstream_grad_sum={downstream_grad_sum:.12g}"
            )


if __name__ == "__main__":
    unittest.main()
