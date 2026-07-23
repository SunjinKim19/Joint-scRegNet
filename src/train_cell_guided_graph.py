"""Train the serial Cell-M -> GraphConstructor -> Graph-M scRegNet model."""

import gc
import logging
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Keep --help available on systems where optional legacy GNN packages are absent.
if __name__ == "__main__" and any(arg in ("-h", "--help") for arg in sys.argv[1:]):
    from src.args import parse_args as _parse_args

    _parse_args()

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.args import parse_args, save_args
from src.device_utils import get_device
from src.scfm_encoder import ScFMEncoder
from src.train_condition_joint import (
    focal_bce_with_logits,
    prepare_condition_bundles,
    requested_cell_types,
    safe_evaluation,
    to_binary_label,
)
from src.utils import set_logging, set_seed

logger = logging.getLogger(__name__)


def make_offdiag_mask(A):
    """Return a boolean mask for all entries except a square diagonal."""
    if not isinstance(A, torch.Tensor):
        raise TypeError("adjacency must be a torch.Tensor")
    if A.dim() != 2:
        raise ValueError(f"adjacency must be 2-D, got shape {tuple(A.shape)}")
    mask = torch.ones(A.shape, dtype=torch.bool, device=A.device)
    if A.size(0) == A.size(1):
        mask.fill_diagonal_(False)
    return mask


def get_candidate_mask(candidate_mask, A):
    """Validate a candidate mask and exclude diagonal self-loops from it."""
    if candidate_mask is None:
        return None
    if not isinstance(candidate_mask, torch.Tensor):
        raise TypeError("candidate_mask must be a torch.Tensor")
    if candidate_mask.shape != A.shape:
        raise ValueError(
            f"candidate_mask shape {tuple(candidate_mask.shape)} does not match "
            f"adjacency shape {tuple(A.shape)}"
        )
    return candidate_mask.to(device=A.device, dtype=torch.bool) & make_offdiag_mask(A)


def compute_sparse_loss(A_ctx, candidate_mask=None):
    """Compute sparsity loss on candidate off-diagonal entries when available."""
    candidate_offdiag = get_candidate_mask(candidate_mask, A_ctx)
    if candidate_offdiag is not None:
        selected_mask = candidate_offdiag
        basis = "candidate"
    else:
        selected_mask = make_offdiag_mask(A_ctx)
        basis = "all_offdiag"
    if not selected_mask.any():
        raise ValueError(f"Cannot compute sparse loss: {basis} mask is empty")
    return A_ctx[selected_mask].mean(), basis


@torch.no_grad()
def log_adj_stats(logger, name, A, candidate_mask=None):
    """Log unambiguous adjacency statistics, excluding square diagonals."""
    if not isinstance(A, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    adjacency = A.to_dense() if A.is_sparse else A
    if adjacency.dim() != 2:
        raise ValueError(f"{name} must be 2-D, got shape {tuple(adjacency.shape)}")

    all_mask = make_offdiag_mask(adjacency)

    def log_region(region_name, region_mask):
        values = adjacency[region_mask]
        if values.numel() == 0:
            logger.info("%s | empty region", region_name)
            return
        logger.info(
            "%s | mean_%s=%.6f min_%s=%.6f max_%s=%.6f "
            "density_%s@0.1=%.6f density_%s@0.5=%.6f density_%s@0.9=%.6f",
            region_name,
            region_name,
            values.mean().item(),
            region_name,
            values.min().item(),
            region_name,
            values.max().item(),
            region_name,
            (values > 0.1).float().mean().item(),
            region_name,
            (values > 0.5).float().mean().item(),
            region_name,
            (values > 0.9).float().mean().item(),
        )

    log_region(f"{name}_all_offdiag", all_mask)
    if candidate_mask is not None:
        candidate_offdiag = get_candidate_mask(candidate_mask, adjacency)
        log_region(f"{name}_candidate", candidate_offdiag)


def _has_nonzero_finite_gradient(parameters):
    return any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and parameter.grad.abs().sum().item() > 0
        for parameter in parameters
    )


def check_trainable_gradients_after_backward(model, args, scfm_encoder=None):
    """Verify the normal training backward reached the intended parameters."""
    downstream_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not _has_nonzero_finite_gradient(downstream_parameters):
        raise RuntimeError("No non-zero gradient reached downstream model parameters")

    if args.scfm_mode == "precomputed":
        if scfm_encoder is not None and scfm_encoder.backbone_loaded:
            raise RuntimeError("precomputed mode unexpectedly loaded an scFM backbone")
        adapter_parameters = [
            parameter
            for parameter in model.cell_m.scfm_adapter.parameters()
            if parameter.requires_grad
        ]
        if adapter_parameters and not _has_nonzero_finite_gradient(adapter_parameters):
            raise RuntimeError("No non-zero gradient reached the Cell-M adapter")
        logger.info(
            "Post-backward gradient check passed: scfm_mode=precomputed, "
            "scFM_backbone_loaded=False, Cell-M_adapter_trainable=%s, "
            "Cell-M_adapter_grad=%s, downstream_grad=True",
            bool(adapter_parameters),
            _has_nonzero_finite_gradient(adapter_parameters)
            if adapter_parameters
            else None,
        )
        return

    if scfm_encoder is None or not scfm_encoder.backbone_loaded:
        raise RuntimeError(f"scfm_mode={args.scfm_mode} has no loaded scFM backbone")
    scfm_trainable = [
        parameter
        for parameter in scfm_encoder.parameters()
        if parameter.requires_grad
    ]
    if args.scfm_mode == "online_frozen":
        if scfm_trainable:
            raise RuntimeError("online_frozen has trainable scFM parameters")
        logger.info(
            "Post-backward gradient check passed: scfm_mode=online_frozen, "
            "scFM_trainable_parameters=0, downstream_grad=True"
        )
        return
    if not scfm_trainable:
        raise RuntimeError(
            f"scfm_mode={args.scfm_mode} created zero trainable scFM parameters"
        )
    if not _has_nonzero_finite_gradient(scfm_trainable):
        raise RuntimeError(
            f"True fine-tuning check failed: no non-zero gradient reached "
            f"{args.scfm_mode} scFM parameters. Ensure scFM outputs are not detached."
        )
    logger.info(
        "Post-backward gradient check passed: scfm_mode=%s, "
        "scFM_trainable_grad=True, downstream_grad=True",
        args.scfm_mode,
    )


class CellGuidedGraphTrainer:
    def __init__(self, args):
        self.args = args
        self.device = get_device(args.device)
        self.model = None
        self.scfm_encoder = None
        self._logged_forward_shapes = False
        self._checked_gradients = False
        self._checked_post_backward_gradients = False

    def get_model(self, bundles):
        from src.models_cell_guided_graph import CellGuidedGraphScRegNet

        if not self.args.use_cell_guided_graph:
            raise ValueError(
                "train_cell_guided_graph.py requires --use_cell_guided_graph true; "
                "use train_condition_joint.py for the legacy baseline."
            )
        if self.args.gnn_type != "GCN":
            logger.warning(
                "The differentiable Graph-M currently uses weighted GCN-style layers; "
                "--gnn_type %s is treated as GCN.",
                self.args.gnn_type,
            )
        if (
            self.args.scfm_mode == "precomputed"
            and self.args.train_scfm_top_layers > 0
        ):
            logger.warning(
                "Precomputed scFM embeddings do not expose transformer layers; "
                "--train_scfm_top_layers is retained for CLI compatibility but cannot be applied."
            )
        if self.args.scfm_mode == "precomputed" and not self.args.freeze_scfm:
            logger.warning(
                "--freeze_scfm false cannot unfreeze a precomputed embedding tensor; "
                "the Cell-M adapter remains the trainable scFM-facing component."
            )
        ref = bundles[0]
        num_genes = ref.data_feature2.size(0)
        if self.args.scfm_mode == "precomputed":
            scfm_dim = ref.data_feature1.size(1)
            self.scfm_encoder.output_dim = scfm_dim
        else:
            scfm_dim = self.scfm_encoder.output_dim
        hidden_dims = [self.args.gnn_dim_hidden] * self.args.gnn_num_layers
        model = CellGuidedGraphScRegNet(
            num_genes=num_genes,
            scfm_dim=scfm_dim,
            latent_dim=self.args.latent_dim,
            condition_hidden_dim=self.args.condition_hidden_dim,
            gnn_hidden_dims=hidden_dims,
            link_hidden_dim=self.args.link_hidden_dim,
            dropout=self.args.dropout,
            graph_alpha=self.args.graph_alpha,
            graph_constructor_type=self.args.graph_constructor_type,
            graph_fusion_type=self.args.graph_fusion_type,
            gate_hidden_dim=self.args.gate_hidden_dim,
            gate_dropout=self.args.gate_dropout,
            gate_temperature=self.args.gate_temperature,
            gate_init_from_alpha=self.args.gate_init_from_alpha,
            scfm_tune_mode="adapter",
            detach_scfm_input=self.args.scfm_mode
            not in ("online_lora", "online_topk"),
        ).to(self.device)
        if not self.args.train_scfm_adapter:
            for parameter in model.cell_m.scfm_adapter.parameters():
                parameter.requires_grad_(False)
        if self.args.scfm_mode == "precomputed":
            logger.info(
                "scFM embeddings are precomputed/frozen=True; "
                "Cell-M adapter trainable=%s",
                self.args.train_scfm_adapter,
            )
        return model

    def build_optimizer(self):
        downstream_parameters = [
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        ]
        if not downstream_parameters:
            raise RuntimeError("No trainable downstream parameters were found")
        parameter_groups = [
            {
                "params": downstream_parameters,
                "lr": self.args.downstream_lr,
                "weight_decay": self.args.downstream_weight_decay,
            }
        ]
        logger.info(
            "Optimizer group downstream: parameters=%d lr=%.6g weight_decay=%.6g",
            sum(parameter.numel() for parameter in downstream_parameters),
            self.args.downstream_lr,
            self.args.downstream_weight_decay,
        )
        scfm_parameters = [
            parameter
            for parameter in self.scfm_encoder.parameters()
            if parameter.requires_grad
        ]
        if self.args.scfm_mode in ("online_lora", "online_topk"):
            if not scfm_parameters:
                raise RuntimeError(
                    f"scfm_mode={self.args.scfm_mode} has zero trainable scFM "
                    "parameters; refusing to claim fine-tuning."
                )
            parameter_groups.append(
                {
                    "params": scfm_parameters,
                    "lr": self.args.scfm_lr,
                    "weight_decay": self.args.scfm_weight_decay,
                }
            )
            logger.info(
                "Optimizer group scFM: parameters=%d lr=%.6g weight_decay=%.6g",
                sum(parameter.numel() for parameter in scfm_parameters),
                self.args.scfm_lr,
                self.args.scfm_weight_decay,
            )
        elif scfm_parameters:
            raise RuntimeError(
                f"scfm_mode={self.args.scfm_mode} unexpectedly has trainable "
                "backbone parameters"
            )
        return getattr(optim, self.args.optimizer_name)(parameter_groups)

    def link_loss(self, logits, targets, pos_weight):
        if self.args.loss_type == "bce":
            return F.binary_cross_entropy_with_logits(logits, targets)
        if self.args.loss_type == "pos_weight_bce":
            return F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=pos_weight
            )
        return focal_bce_with_logits(
            logits, targets, self.args.focal_alpha, self.args.focal_gamma
        )

    def forward(self, bundle, edge_pairs, evaluation=False):
        scfm_gene_embeddings = self.scfm_encoder(
            {
                "precomputed_embeddings": bundle.data_feature1,
                "num_genes": bundle.data_feature2.size(0),
                "cell_type": bundle.cell_type,
            }
        )
        if scfm_gene_embeddings.dim() != 2:
            raise ValueError(
                "ScFMEncoder must return [num_genes, scfm_dim], got "
                f"{tuple(scfm_gene_embeddings.shape)}"
            )
        if scfm_gene_embeddings.size(0) != bundle.data_feature2.size(0):
            raise ValueError(
                "Online scFM gene count does not match the downstream graph: "
                f"{scfm_gene_embeddings.size(0)} vs {bundle.data_feature2.size(0)}"
            )
        output = self.model(
            scfm_gene_emb=scfm_gene_embeddings,
            prior_adjacency=bundle.adj,
            edge_pairs=edge_pairs,
            raw_expr=bundle.raw_expr,
            tf_indices=bundle.tf_indices,
            hard_topk_eval_only=(
                self.args.hard_topk_eval_only if evaluation else 0
            ),
        )
        if not self._logged_forward_shapes:
            logger.info("Cell-guided graph forward shape summary")
            logger.info("  raw expression shape: %s", tuple(bundle.raw_expr.shape))
            logger.info(
                "  scFM output shape: %s", tuple(scfm_gene_embeddings.shape)
            )
            logger.info(
                "  scFM output requires_grad: %s",
                scfm_gene_embeddings.requires_grad,
            )
            logger.info(
                "  Cell-M input shape: %s", tuple(scfm_gene_embeddings.shape)
            )
            logger.info("  z_ctx shape: %s", tuple(output["z_ctx"].shape))
            logger.info("  A_ctx shape: %s", tuple(output["A_ctx"].shape))
            logger.info("  A_prior shape: %s", tuple(output["A_prior"].shape))
            logger.info("  A_final shape: %s", tuple(output["A_final"].shape))
            if output.get("gate") is not None:
                logger.info("  gate shape: %s", tuple(output["gate"].shape))
            logger.info("  z_graph shape: %s", tuple(output["z_graph"].shape))
            logger.info("  edge_pairs shape: %s", tuple(edge_pairs.shape))
            logger.info("  prediction shape: %s", tuple(output["probabilities"].shape))
            self._logged_forward_shapes = True
        return output

    def loss(self, output, labels, bundle):
        bce_loss = self.link_loss(output["logits"], labels, bundle.pos_weight)
        sparse_loss_used, sparse_loss_basis = compute_sparse_loss(
            output["A_ctx"], output.get("candidate_mask")
        )
        total_loss = bce_loss + self.args.lambda_sparse * sparse_loss_used
        return total_loss, bce_loss, sparse_loss_used, sparse_loss_basis

    def verify_gradient_path(self, output):
        """Check autograd connectivity without invoking backward or reading grads."""
        if self._checked_gradients:
            return
        prediction = output.get("prediction")
        if prediction is None or not prediction.requires_grad:
            raise RuntimeError("Gradient path broken: prediction must require gradients")
        if not output["z_ctx"].requires_grad:
            raise RuntimeError("Gradient path broken: z_ctx must require gradients")
        a_ctx_should_require_grad = (
            self.args.graph_fusion_type == "edge_gate"
            or self.args.graph_alpha < 1.0
        )
        if a_ctx_should_require_grad and not output["A_ctx"].requires_grad:
            raise RuntimeError(
                "Gradient path broken: A_ctx must require gradients for the "
                f"{self.args.graph_fusion_type} fusion path"
            )
        gate_requires_grad = None
        if self.args.graph_fusion_type == "edge_gate":
            gate = output.get("gate")
            if gate is None or not gate.requires_grad:
                raise RuntimeError(
                    "Gradient path broken: edge_gate output must require gradients"
                )
            gate_requires_grad = gate.requires_grad
            if torch.allclose(output["A_final"], output["A_prior"]):
                raise RuntimeError("edge_gate produced A_final identical to A_prior")
            if torch.allclose(output["A_final"], output["A_ctx"]):
                raise RuntimeError("edge_gate produced A_final identical to A_ctx")
        logger.info(
            "Non-invasive gradient path check passed: prediction.requires_grad=%s, "
            "A_ctx.requires_grad=%s, z_ctx.requires_grad=%s, gate.requires_grad=%s",
            prediction.requires_grad,
            output["A_ctx"].requires_grad,
            output["z_ctx"].requires_grad,
            gate_requires_grad,
        )
        self._checked_gradients = True

    @torch.no_grad()
    def evaluate_bundle(self, bundle, edge_data):
        self.model.eval()
        self.scfm_encoder.eval()
        output = self.forward(bundle, edge_data[:, :2], evaluation=True)
        labels = to_binary_label(edge_data[:, -1])
        return safe_evaluation(output["probabilities"], labels)

    def evaluate_all(self, bundles, split_name):
        metrics = {}
        for bundle in bundles:
            edge_data = bundle.valid_data if split_name == "valid" else bundle.test_data
            auc, aupr = self.evaluate_bundle(bundle, edge_data)
            metrics[bundle.cell_type] = {"auroc": auc, "auprc": aupr}
        macro_auc = float(np.nanmean([item["auroc"] for item in metrics.values()]))
        macro_aupr = float(np.nanmean([item["auprc"] for item in metrics.values()]))
        return metrics, macro_auc, macro_aupr

    def train(self):
        bundles = prepare_condition_bundles(self.args, self.device)
        if self.args.scfm_mode != "precomputed" and len(bundles) != 1:
            raise ValueError(
                "Online scFM mode currently accepts one tokenized condition at a "
                "time. Run one --cell_type per tokenized input."
            )
        # A_prior is built only from the optimization subset's positive labels.
        # TODO: prefer a wholly external biological prior when one is available.
        self.scfm_encoder = ScFMEncoder(self.args, self.device).to(self.device)
        self.model = self.get_model(bundles)
        optimizer = self.build_optimizer()
        amp_enabled = self.args.amp and self.device.type == "cuda"
        if self.args.amp and not amp_enabled:
            logger.warning("--amp requested without CUDA; AMP is disabled.")
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        optimized_parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        total_scfm, trainable_scfm = self.scfm_encoder.parameter_counts()
        trainable_downstream = sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )
        logger.info(
            "scfm_mode=%s backbone_loaded=%s trainable_scfm_params=%d/%d "
            "trainable_downstream_params=%d",
            self.args.scfm_mode,
            self.scfm_encoder.backbone_loaded,
            trainable_scfm,
            total_scfm,
            trainable_downstream,
        )
        loaders = {
            bundle.cell_type: DataLoader(
                bundle.train_dataset,
                batch_size=self.args.batch_size,
                shuffle=True,
            )
            for bundle in bundles
        }
        checkpoint_path = os.path.join(
            self.args.ckpt_dir, f"cell_guided_graph_seed{self.args.random_seed}.pt"
        )
        best_score = float("-inf")
        best_valid_auc = best_valid_aupr = float("nan")
        patience_count = 0
        saved_checkpoint = False

        for epoch in tqdm(range(self.args.gnn_epochs)):
            self.model.train()
            self.scfm_encoder.train()
            totals = {"total": 0.0, "bce": 0.0, "sparse": 0.0}
            sparse_loss_basis = None
            steps = 0
            for bundle_index, bundle in enumerate(bundles):
                for batch_index, (edge_pairs, train_y) in enumerate(
                    loaders[bundle.cell_type]
                ):
                    edge_pairs = edge_pairs.to(self.device)
                    labels = to_binary_label(train_y.to(self.device))
                    optimizer.zero_grad(set_to_none=True)
                    try:
                        with torch.amp.autocast(
                            device_type=self.device.type, enabled=amp_enabled
                        ):
                            output = self.forward(bundle, edge_pairs)
                            (
                                total_loss,
                                bce_loss,
                                sparse_loss_used,
                                batch_sparse_loss_basis,
                            ) = self.loss(output, labels, bundle)
                    except torch.cuda.OutOfMemoryError as exc:
                        raise RuntimeError(
                            "Training ran out of GPU memory. Reduce "
                            "--max_scfm_cells, prefer online_lora over "
                            "online_topk, lower --batch_size, or disable --amp "
                            "only if mixed precision is unstable."
                        ) from exc
                    if epoch == 0 and bundle_index == 0 and batch_index == 0:
                        candidate_mask = output.get("candidate_mask")
                        if output.get("gate") is not None:
                            log_adj_stats(
                                logger, "Gate", output["gate"], candidate_mask
                            )
                        log_adj_stats(
                            logger, "A_ctx", output["A_ctx"], candidate_mask
                        )
                        log_adj_stats(
                            logger, "A_prior", output["A_prior"], candidate_mask
                        )
                        log_adj_stats(
                            logger, "A_final", output["A_final"], candidate_mask
                        )
                    self.verify_gradient_path(output)
                    if sparse_loss_basis is None:
                        sparse_loss_basis = batch_sparse_loss_basis
                    elif sparse_loss_basis != batch_sparse_loss_basis:
                        raise RuntimeError(
                            "Sparse loss basis changed within an epoch: "
                            f"{sparse_loss_basis} -> {batch_sparse_loss_basis}"
                        )
                    scaler.scale(total_loss).backward()
                    scaler.unscale_(optimizer)
                    if (
                        epoch == 0
                        and bundle_index == 0
                        and batch_index == 0
                        and not self._checked_post_backward_gradients
                    ):
                        check_trainable_gradients_after_backward(
                            self.model, self.args, self.scfm_encoder
                        )
                        self._checked_post_backward_gradients = True
                    if self.args.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            optimized_parameters, self.args.grad_clip_norm
                        )
                    scaler.step(optimizer)
                    scaler.update()
                    totals["total"] += total_loss.item()
                    totals["bce"] += bce_loss.item()
                    totals["sparse"] += sparse_loss_used.item()
                    steps += 1

            should_evaluate = (epoch + 1) % self.args.gnn_eval_interval == 0 or epoch + 1 == self.args.gnn_epochs
            if not should_evaluate:
                continue
            valid_metrics, macro_auc, macro_aupr = self.evaluate_all(bundles, "valid")
            current = macro_aupr if self.args.early_stop_metric == "auprc" else macro_auc
            comparison = float("-inf") if np.isnan(current) else current
            if not saved_checkpoint or comparison > best_score + self.args.min_delta:
                best_score = comparison
                best_valid_auc, best_valid_aupr = macro_auc, macro_aupr
                patience_count = 0
                self.args.ckpt_name = checkpoint_path
                checkpoint = {"model": self.model.state_dict()}
                if self.args.scfm_mode != "precomputed":
                    checkpoint["scfm_encoder"] = self.scfm_encoder.state_dict()
                torch.save(
                    checkpoint
                    if self.args.scfm_mode != "precomputed"
                    else checkpoint["model"],
                    checkpoint_path,
                )
                save_args(self.args, self.args.ckpt_dir)
                saved_checkpoint = True
            else:
                patience_count += 1
            denom = max(1, steps)
            logger.info(
                "Epoch %03d | total_loss=%.4f | bce_loss=%.4f | "
                "sparse_loss_used=%.4f | sparse_loss_basis=%s | "
                "lambda_sparse=%.6g | valid_AUROC=%.4f | valid_AUPRC=%.4f | "
                "patience=%d/%d",
                epoch + 1,
                totals["total"] / denom,
                totals["bce"] / denom,
                totals["sparse"] / denom,
                sparse_loss_basis,
                self.args.lambda_sparse,
                macro_auc,
                macro_aupr,
                patience_count,
                self.args.patience,
            )
            for cell_type, metric in valid_metrics.items():
                logger.info("  valid[%s] AUROC=%.4f AUPRC=%.4f", cell_type, metric["auroc"], metric["auprc"])
            if patience_count >= self.args.patience:
                logger.info("Early stopping triggered at epoch %d", epoch + 1)
                break

        if saved_checkpoint:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if self.args.scfm_mode == "precomputed":
                self.model.load_state_dict(checkpoint)
            else:
                self.model.load_state_dict(checkpoint["model"])
                self.scfm_encoder.load_state_dict(checkpoint["scfm_encoder"])
        test_metrics, test_auc, test_aupr = self.evaluate_all(bundles, "test")
        for cell_type, metric in test_metrics.items():
            logger.info("Final test[%s] AUROC=%.4f AUPRC=%.4f", cell_type, metric["auroc"], metric["auprc"])
        logger.info(
            "Best validation macro AUROC/AUPRC: %.4f/%.4f | Final test macro AUROC/AUPRC: %.4f/%.4f",
            best_valid_auc,
            best_valid_aupr,
            test_auc,
            test_aupr,
        )
        return test_auc, test_aupr


def main():
    set_logging()
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = "./out/cell_guided_graph"
    if args.ckpt_dir is None:
        args.ckpt_dir = os.path.join(args.output_dir, "ckpt")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    set_seed(args.random_seed)
    logger.critical(
        "Training serial CellGuidedGraphScRegNet on %s with scFM=%s, "
        "scfm_mode=%s, "
        "constructor=%s, graph_fusion_type=%s, alpha=%.3f",
        ",".join(requested_cell_types(args)),
        args.llm_type,
        args.scfm_mode,
        args.graph_constructor_type,
        args.graph_fusion_type,
        args.graph_alpha,
    )
    trainer = CellGuidedGraphTrainer(args)
    result = trainer.train()
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return result


if __name__ == "__main__":
    final_auc, final_aupr = main()
    print(final_auc, final_aupr)
