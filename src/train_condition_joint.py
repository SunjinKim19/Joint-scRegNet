import gc
import logging
import os
import sys
from dataclasses import dataclass

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Keep CLI discovery usable even when optional GNN dependencies are absent.
if __name__ == "__main__" and any(arg in ("-h", "--help") for arg in sys.argv[1:]):
    from src.args import parse_args as _parse_args

    _parse_args()

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.args import parse_args, save_args
from src.device_utils import get_device
from src.utils import (
    Evaluation,
    adj2saprse_tensor,
    load_data,
    scRNADataset,
    set_logging,
    set_seed,
)

logger = logging.getLogger(__name__)


@dataclass
class ConditionBundle:
    cell_type: str
    condition_id: int
    train_dataset: scRNADataset
    valid_data: torch.Tensor
    test_data: torch.Tensor
    adj: torch.Tensor
    data_feature1: torch.Tensor
    data_feature2: torch.Tensor
    raw_expr: torch.Tensor
    gene_names: list
    tf_names: list
    tf_indices: torch.Tensor
    target_names: list
    train_full_shape: tuple
    pos_weight: torch.Tensor


def to_binary_label(y):
    y = y.float()
    if y.dim() == 2:
        if y.size(1) == 2:
            return y[:, 1].contiguous()
        if y.size(1) == 1:
            return y[:, 0].contiguous()
    return y.view(-1).contiguous()


def focal_bce_with_logits(logits, targets, alpha=0.75, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    prob = torch.sigmoid(logits)
    pt = prob * targets + (1.0 - prob) * (1.0 - targets)
    return (alpha * (1.0 - pt).pow(gamma) * bce).mean()


def available_cell_types(args):
    if not os.path.isdir(args.data_folder):
        return []
    cell_types = []
    for name in sorted(os.listdir(args.data_folder)):
        candidate = os.path.join(args.data_folder, name, f"TFs+{args.num_TF}")
        if os.path.isdir(candidate):
            cell_types.append(name)
    return cell_types


def requested_cell_types(args):
    if getattr(args, "cell_types", ""):
        return [item.strip() for item in args.cell_types.split(",") if item.strip()]
    return [args.cell_type]


def _validate_condition_path(args, cell_type):
    path = os.path.join(args.data_folder, cell_type, f"TFs+{args.num_TF}")
    if not os.path.isdir(path):
        choices = ", ".join(available_cell_types(args)) or "none found"
        raise FileNotFoundError(
            f"Cell type '{cell_type}' with TFs+{args.num_TF} was not found at "
            f"{path}. Available cell types for this num_TF: {choices}."
        )
    required = [
        "BL--ExpressionData.csv",
        "Train_set.csv",
        "Test_set.csv",
        "TF.csv",
        "Target.csv",
    ]
    missing = [name for name in required if not os.path.isfile(os.path.join(path, name))]
    if missing:
        raise FileNotFoundError(f"Missing required files under {path}: {missing}")
    return path


def _get_embeddings(args, cell_type, data_input):
    """
    Mirrors Trainer._get_embeddings from src/train.py without importing train.py,
    because train.py imports torch_geometric at module import time.
    """
    if args.llm_type == "Geneformer":
        scfm_embs = os.path.join(args.scFM_folder, "Geneformer")
        emb_file = os.path.join(
            scfm_embs, f"{cell_type}_{args.num_TF}_gene_embeddings.csv"
        )
        final_file = os.path.join(scfm_embs, f"{cell_type}_{args.num_TF}.csv")
        embs = pd.read_csv(emb_file)
        final_df = pd.read_csv(final_file)
        zero = np.zeros((len(embs.columns) - 1,), dtype=np.float32)
        emb_lookup = {
            row.iloc[0]: row.iloc[1:].to_numpy(dtype=np.float32)
            for _, row in embs.iterrows()
        }
        gene_embeddings = []
        for _, row in final_df.iterrows():
            ensembl_id = row["ensembl_id"]
            gene_embeddings.append(emb_lookup.get(ensembl_id, zero))
        gene_embeddings = np.asarray(gene_embeddings, dtype=np.float32)

    elif args.llm_type == "scBERT":
        scfm_embs = os.path.join(args.scFM_folder, "scBERT")
        cell_embeddings_arr = np.load(
            os.path.join(scfm_embs, f"{cell_type}_{args.num_TF}_cell_embeddings.npy")
        )
        gene_embeddings = np.mean(cell_embeddings_arr, axis=0).astype(np.float32)

    elif args.llm_type == "scFoundation":
        scfm_embs = os.path.join(args.scFM_folder, "scFoundation")
        gene_list_df = pd.read_csv(
            os.path.join(scfm_embs, "OS_scRNA_gene_index.19264.tsv"),
            header=0,
            delimiter="\t",
        )
        gene_list = list(gene_list_df["gene_name"])
        zero = np.zeros(512, dtype=np.float32)
        gene_embeddings = np.load(
            os.path.join(
                scfm_embs,
                f"genemodule_{cell_type}_{args.num_TF}_singlecell_gene_embedding_f2_resolution.npy",
            )
        )
        pooled_gene_embeddings = np.mean(gene_embeddings, axis=0)
        final_gene_embeddings = []
        for gene_name in data_input.index:
            try:
                final_gene_embeddings.append(
                    pooled_gene_embeddings[gene_list.index(gene_name)]
                )
            except ValueError:
                final_gene_embeddings.append(zero)
        gene_embeddings = np.asarray(final_gene_embeddings, dtype=np.float32)
    else:
        raise ValueError(
            f"Unsupported llm_type '{args.llm_type}'. Expected Geneformer, scBERT, or scFoundation."
        )

    return torch.from_numpy(gene_embeddings).float()


def split_train_valid(train_load, args, device):
    raw_data = np.asarray(train_load.train_set)
    labels = raw_data[:, -1].astype(np.int64)
    indices = np.arange(len(raw_data))
    try:
        train_idx, valid_idx = train_test_split(
            indices,
            test_size=0.1,
            random_state=args.random_seed,
            shuffle=True,
            stratify=labels,
        )
    except ValueError as exc:
        logger.warning(
            "Stratified validation split unavailable for %s (%s); using a "
            "deterministic random 90/10 split.",
            getattr(args, "cell_type", "condition"),
            exc,
        )
        generator = np.random.default_rng(args.random_seed)
        shuffled = generator.permutation(indices)
        valid_size = max(1, int(round(0.1 * len(shuffled))))
        valid_idx, train_idx = shuffled[:valid_size], shuffled[valid_size:]

    if len(train_idx) == 0 or len(valid_idx) == 0:
        raise ValueError("Training data is too small to create a 90/10 validation split")

    train_data = raw_data[train_idx]
    valid_data = torch.as_tensor(raw_data[valid_idx], device=device)
    return scRNADataset(train_data, train_load.num_gene, flag=train_load.flag), valid_data


def load_condition_bundle(args, cell_type, condition_id, device):
    path = _validate_condition_path(args, cell_type)
    exp_file = os.path.join(path, "BL--ExpressionData.csv")
    train_file = os.path.join(path, "Train_set.csv")
    test_file = os.path.join(path, "Test_set.csv")
    tf_file = os.path.join(path, "TF.csv")
    target_file = os.path.join(path, "Target.csv")

    data_input = pd.read_csv(exp_file, index_col=0)
    train_data = pd.read_csv(train_file, index_col=0).values
    test_data = pd.read_csv(test_file, index_col=0).values
    tf_df = pd.read_csv(tf_file, index_col=0)
    target_df = pd.read_csv(target_file, index_col=0)
    tf_indices = torch.from_numpy(tf_df["index"].values.astype(np.int64)).to(device)

    loader = load_data(data_input)
    feature2 = torch.from_numpy(loader.exp_data()).float()
    scfm_mode = getattr(args, "scfm_mode", "precomputed")
    if scfm_mode == "precomputed":
        feature1 = _get_embeddings(args, cell_type, data_input)
        if args.llm_type == "scBERT":
            feature1 = feature1[:-1]
    else:
        # Online scFM modes obtain their feature tensor from ScFMEncoder inside
        # the training loop and must not depend on a precomputed CSV.
        feature1 = torch.empty((feature2.size(0), 0), dtype=torch.float32)

    if feature1.size(0) != feature2.size(0):
        raise ValueError(
            f"Feature gene-count mismatch for {cell_type}: scFM={feature1.size(0)} "
            f"but expression={feature2.size(0)}. A shared gene index mapping is required."
        )

    data_feature1 = feature1.to(device)
    data_feature2 = feature2.to(device)
    raw_expr = feature2.t().contiguous().to(device)
    gene_num = feature2.shape[0]

    full_train_load = scRNADataset(train_data, gene_num, flag=args.flag)
    train_dataset, valid_data = split_train_valid(full_train_load, args, device)

    # Current scRegNet builds adj from positive training labels. Rebuild from the
    # optimization subset to avoid validation-edge leakage.
    adj = train_dataset.Adj_Generate(tf_indices, loop=args.loop)
    adj = adj2saprse_tensor(adj).to(device)
    test_tensor = torch.as_tensor(test_data, device=device)

    raw_train_labels = torch.as_tensor(
        train_dataset.train_set[:, -1], dtype=torch.float32, device=device
    )
    num_pos = raw_train_labels.sum()
    num_neg = raw_train_labels.numel() - num_pos
    pos_weight = (num_neg / num_pos.clamp_min(1.0)).reshape(1)

    return ConditionBundle(
        cell_type=cell_type,
        condition_id=condition_id,
        train_dataset=train_dataset,
        valid_data=valid_data,
        test_data=test_tensor,
        adj=adj,
        data_feature1=data_feature1,
        data_feature2=data_feature2,
        raw_expr=raw_expr,
        gene_names=list(data_input.index),
        tf_names=list(tf_df.iloc[:, 0]) if len(tf_df.columns) else [],
        tf_indices=tf_indices,
        target_names=list(target_df.iloc[:, 0]) if len(target_df.columns) else [],
        train_full_shape=tuple(train_data.shape),
        pos_weight=pos_weight,
    )


def prepare_condition_bundles(args, device):
    cell_types = requested_cell_types(args)
    bundles = [
        load_condition_bundle(args, cell_type, condition_id, device)
        for condition_id, cell_type in enumerate(cell_types)
    ]
    validate_condition_compatibility(bundles)
    return bundles


def validate_condition_compatibility(bundles):
    if not bundles:
        raise ValueError("No condition datasets were requested.")
    ref = bundles[0]
    mismatches = []
    for bundle in bundles[1:]:
        if bundle.data_feature1.shape != ref.data_feature1.shape:
            mismatches.append(
                f"{bundle.cell_type} scFM {tuple(bundle.data_feature1.shape)} "
                f"!= {ref.cell_type} {tuple(ref.data_feature1.shape)}"
            )
        if bundle.data_feature2.shape != ref.data_feature2.shape:
            mismatches.append(
                f"{bundle.cell_type} expression {tuple(bundle.data_feature2.shape)} "
                f"!= {ref.cell_type} {tuple(ref.data_feature2.shape)}"
            )
        if bundle.gene_names != ref.gene_names:
            logger.warning(
                "Gene name order differs between %s and %s. This first "
                "multi-condition implementation requires compatible dimensions "
                "but does not remap gene identities.",
                ref.cell_type,
                bundle.cell_type,
            )
    if mismatches:
        raise ValueError(
            "Condition feature dimensions are incompatible. A shared gene index "
            "mapping is required before joint multi-condition training:\n"
            + "\n".join(mismatches)
        )


def tensor_shape(value):
    if isinstance(value, torch.Tensor):
        return tuple(value.shape)
    return None


def adj_edge_count(adj):
    if isinstance(adj, torch.Tensor) and adj.is_sparse:
        return adj._nnz()
    if isinstance(adj, torch.Tensor) and adj.dim() == 2 and adj.size(0) == 2:
        return adj.size(1)
    if isinstance(adj, torch.Tensor):
        return int((adj != 0).sum().item())
    return 0


def positive_ratio(edge_tensor):
    labels = to_binary_label(edge_tensor[:, -1])
    return float(labels.mean().item()) if labels.numel() else float("nan")


def log_condition_shape_summary(bundles):
    logger.info("Condition shape summary")
    for bundle in bundles:
        logger.info("Condition: %s", bundle.cell_type)
        logger.info(
            "  scfm_gene_emb / data_feature1: %s",
            tuple(bundle.data_feature1.shape),
        )
        logger.info(
            "  graph_node_features / data_feature2: %s",
            tuple(bundle.data_feature2.shape),
        )
        logger.info(
            "  adj: shape=%s sparse=%s edge_count=%s",
            tuple(bundle.adj.shape),
            bundle.adj.is_sparse if isinstance(bundle.adj, torch.Tensor) else False,
            adj_edge_count(bundle.adj),
        )
        train_edges = bundle.train_dataset.train_set[:, :2]
        train_labels = bundle.train_dataset.train_set[:, -1]
        logger.info(
            "  train edges: %s labels=%s positive_ratio=%.4f",
            tuple(train_edges.shape),
            tuple(train_labels.shape),
            float(np.mean(train_labels)),
        )
        logger.info(
            "  valid edges: %s positive_ratio=%.4f",
            tuple(bundle.valid_data[:, :2].shape),
            positive_ratio(bundle.valid_data),
        )
        logger.info(
            "  test edges: %s positive_ratio=%.4f",
            tuple(bundle.test_data[:, :2].shape),
            positive_ratio(bundle.test_data),
        )
        logger.info("  raw_expr / cell x gene: %s", tuple(bundle.raw_expr.shape))


def count_parameters(module, trainable_only=False):
    if module is None:
        return 0
    parameters = module.parameters()
    if trainable_only:
        return sum(p.numel() for p in parameters if p.requires_grad)
    return sum(p.numel() for p in parameters)


def log_model_parameter_summary(model):
    logger.info("Model parameter summary")
    logger.info("  total parameters: %d", count_parameters(model))
    logger.info("  trainable parameters: %d", count_parameters(model, True))
    modules = [
        ("CellConditionM", model.cell_condition_m),
        ("GraphM", model.graph_m),
        ("InferM", model.infer_m),
        ("LinkDecoder", model.link_decoder),
        ("ContextDecoder", model.context_decoder),
    ]
    if model.condition_classifier is not None:
        modules.append(("ConditionClassifier", model.condition_classifier))
    for name, module in modules:
        logger.info("  %s trainable parameters: %d", name, count_parameters(module, True))


def safe_evaluation(scores, labels):
    try:
        auc, aupr, _ = Evaluation(
            y_pred=scores.view(-1, 1),
            y_true=labels,
            flag=False,
        )
        return auc, aupr
    except ValueError as exc:
        logger.warning("Evaluation failed: %s", exc)
        return float("nan"), float("nan")


class ConditionJointTrainer:
    def __init__(self, args):
        self.args = args
        self.device = get_device(args.device)
        self.model = None

    def get_model(self, bundles):
        if self.args.gnn_type != "GCN":
            logger.warning(
                "ConditionJointScRegNet currently implements GCNConv only; "
                "requested --gnn_type %s will use GCN.",
                self.args.gnn_type,
            )
        if self.args.scfm_tune_mode in ("top", "full"):
            logger.warning(
                "Full scFM fine-tuning is not available because the current "
                "pipeline provides precomputed embeddings only."
            )
        try:
            from src.models_condition_joint import ConditionJointScRegNet
        except ModuleNotFoundError as exc:
            if exc.name == "torch_geometric":
                raise ModuleNotFoundError(
                    "torch_geometric is required to train ConditionJointScRegNet "
                    "because Graph-M uses GCNConv."
                ) from exc
            raise

        ref = bundles[0]
        gnn_hidden_dims = getattr(self.args, "gnn_hidden_dims", None)
        if gnn_hidden_dims is None:
            gnn_hidden_dims = [self.args.gnn_dim_hidden] * self.args.gnn_num_layers
        return ConditionJointScRegNet(
            num_genes=ref.data_feature2.size(0),
            graph_input_dim=ref.data_feature2.size(1),
            scfm_dim=ref.data_feature1.size(1),
            gnn_hidden_dims=gnn_hidden_dims,
            latent_dim=self.args.condition_joint_latent_dim,
            condition_hidden_dim=self.args.condition_hidden_dim,
            link_hidden_dim=self.args.link_hidden_dim,
            dropout=self.args.dropout,
            infer_fusion_mode=self.args.infer_fusion_mode,
            infer_layers=self.args.infer_layers,
            infer_heads=self.args.infer_heads,
            infer_dropout=self.args.infer_dropout,
            context_recon_target=self.args.context_recon_target,
            scfm_tune_mode=self.args.scfm_tune_mode,
            num_conditions=len(bundles),
        ).to(self.device)

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

    def compute_losses(self, output, labels, bundle, condition_loss_enabled):
        link_loss = self.link_loss(output["logits"], labels, bundle.pos_weight)
        zero = output["logits"].new_tensor(0.0)

        context_loss = zero
        if (
            self.args.context_recon_target != "none"
            and output["context_recon"] is not None
            and output["context_target"] is not None
        ):
            context_loss = F.mse_loss(output["context_recon"], output["context_target"])

        condition_loss = zero
        if condition_loss_enabled and output["condition_logits"] is not None:
            condition_target = torch.tensor(
                [bundle.condition_id], dtype=torch.long, device=self.device
            )
            condition_loss = F.cross_entropy(
                output["condition_logits"].view(1, -1), condition_target
            )

        graph_pre_loss = zero
        if self.args.lambda_graph_pre > 0:
            graph_pre_loss = F.mse_loss(
                F.normalize(output["z_graph"], p=2, dim=1),
                F.normalize(output["z_gene_ctx"].detach(), p=2, dim=1),
            )

        total_loss = (
            link_loss
            + self.args.lambda_context * context_loss
            + self.args.lambda_condition * condition_loss
            + self.args.lambda_graph_pre * graph_pre_loss
        )
        return total_loss, link_loss, context_loss, condition_loss, graph_pre_loss

    @torch.no_grad()
    def evaluate_bundle(self, bundle, edge_data):
        self.model.eval()
        output = self.model(
            scfm_gene_emb=bundle.data_feature1,
            graph_node_features=bundle.data_feature2,
            adj=bundle.adj,
            edge_pairs=edge_data[:, :2],
            raw_expr=bundle.raw_expr,
            condition_id=bundle.condition_id,
        )
        labels = to_binary_label(edge_data[:, -1])
        scores = torch.sigmoid(output["logits"])
        return safe_evaluation(scores, labels)

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
        log_condition_shape_summary(bundles)

        condition_loss_enabled = len(bundles) > 1 and self.args.lambda_condition > 0
        if not condition_loss_enabled:
            if len(bundles) == 1:
                logger.info("Only one condition detected; condition loss disabled.")
            elif self.args.lambda_condition <= 0:
                logger.info("lambda_condition <= 0; condition loss disabled.")

        self.model = self.get_model(bundles)
        log_model_parameter_summary(self.model)
        optimizer = getattr(optim, self.args.optimizer_name)(
            self.model.parameters(),
            lr=self.args.gnn_lr,
            weight_decay=self.args.gnn_weight_decay,
        )
        train_loaders = {
            bundle.cell_type: DataLoader(
                bundle.train_dataset, batch_size=self.args.batch_size, shuffle=True
            )
            for bundle in bundles
        }

        best_score = float("-inf")
        best_valid_auc = float("nan")
        best_valid_aupr = float("nan")
        patience_count = 0
        checkpoint_path = os.path.join(
            self.args.ckpt_dir, f"condition_joint_seed{self.args.random_seed}.pt"
        )
        saved_checkpoint = False

        for epoch in tqdm(range(self.args.gnn_epochs)):
            self.model.train()
            running = {
                "total": 0.0,
                "link": 0.0,
                "context": 0.0,
                "condition": 0.0,
                "graph_pre": 0.0,
            }
            steps = 0

            for bundle in bundles:
                for train_x, train_y in train_loaders[bundle.cell_type]:
                    train_x = train_x.to(self.device)
                    labels = to_binary_label(train_y.to(self.device))
                    optimizer.zero_grad()
                    output = self.model(
                        scfm_gene_emb=bundle.data_feature1,
                        graph_node_features=bundle.data_feature2,
                        adj=bundle.adj,
                        edge_pairs=train_x,
                        raw_expr=bundle.raw_expr,
                        condition_id=bundle.condition_id,
                    )
                    (
                        loss,
                        link_loss,
                        context_loss,
                        condition_loss,
                        graph_pre_loss,
                    ) = self.compute_losses(output, labels, bundle, condition_loss_enabled)
                    loss.backward()
                    optimizer.step()

                    running["total"] += loss.item()
                    running["link"] += link_loss.item()
                    running["context"] += context_loss.item()
                    running["condition"] += condition_loss.item()
                    running["graph_pre"] += graph_pre_loss.item()
                    steps += 1

            should_evaluate = (
                (epoch + 1) % self.args.gnn_eval_interval == 0
                or epoch + 1 == self.args.gnn_epochs
            )
            if not should_evaluate:
                continue

            valid_metrics, macro_auc, macro_aupr = self.evaluate_all(bundles, "valid")
            current_score = (
                macro_aupr if self.args.early_stop_metric == "auprc" else macro_auc
            )
            score_for_compare = (
                float("-inf") if np.isnan(current_score) else current_score
            )
            improved = (
                not saved_checkpoint
                or score_for_compare > best_score + self.args.min_delta
            )
            if improved:
                best_score = score_for_compare
                best_valid_auc = macro_auc
                best_valid_aupr = macro_aupr
                patience_count = 0
                self.args.ckpt_name = checkpoint_path
                torch.save(self.model.state_dict(), checkpoint_path)
                save_args(self.args, self.args.ckpt_dir)
                saved_checkpoint = True
                logger.info(
                    "Best validation updated: macro_AUROC=%.4f macro_AUPRC=%.4f",
                    best_valid_auc,
                    best_valid_aupr,
                )
            else:
                patience_count += 1

            denom = max(1, steps)
            logger.info(
                "Epoch %03d | train_loss=%.4f | link_loss=%.4f | "
                "context_loss=%.4f | condition_loss=%.4f | graph_pre_loss=%.4f | "
                "valid_AUROC=%.4f | valid_AUPRC=%.4f | patience=%d/%d",
                epoch + 1,
                running["total"] / denom,
                running["link"] / denom,
                running["context"] / denom,
                running["condition"] / denom,
                running["graph_pre"] / denom,
                macro_auc,
                macro_aupr,
                patience_count,
                self.args.patience,
            )
            for cell_type, metric in valid_metrics.items():
                logger.info(
                    "  valid[%s] AUROC=%.4f AUPRC=%.4f",
                    cell_type,
                    metric["auroc"],
                    metric["auprc"],
                )

            if patience_count >= self.args.patience:
                logger.info(
                    "Early stopping triggered at epoch %d. Best macro AUROC/AUPRC: %.4f/%.4f",
                    epoch + 1,
                    best_valid_auc,
                    best_valid_aupr,
                )
                break

        if saved_checkpoint:
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        test_metrics, test_macro_auc, test_macro_aupr = self.evaluate_all(bundles, "test")
        for cell_type, metric in test_metrics.items():
            logger.info(
                "Final test[%s] AUROC=%.4f AUPRC=%.4f",
                cell_type,
                metric["auroc"],
                metric["auprc"],
            )
        logger.info(
            "Best validation macro AUROC/AUPRC: %.4f/%.4f | Final test macro AUROC/AUPRC: %.4f/%.4f",
            best_valid_auc,
            best_valid_aupr,
            test_macro_auc,
            test_macro_aupr,
        )
        return test_macro_auc, test_macro_aupr


def main():
    set_logging()
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = "./out/condition_joint"
    if args.ckpt_dir is None:
        args.ckpt_dir = os.path.join(args.output_dir, "ckpt")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    set_seed(random_seed=args.random_seed)
    logger.critical(
        "Training ConditionJointScRegNet on %s with scFM=%s, Infer-M=%s",
        ",".join(requested_cell_types(args)),
        args.llm_type,
        args.infer_fusion_mode,
    )
    trainer = ConditionJointTrainer(args)
    test_auc, test_aupr = trainer.train()
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return test_auc, test_aupr


if __name__ == "__main__":
    final_auc, final_aupr = main()
    print(final_auc, final_aupr)
