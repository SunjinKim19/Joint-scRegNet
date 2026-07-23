import gc
import logging
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Argument discovery should work without importing optional GNN dependencies.
if __name__ == "__main__" and any(arg in ("-h", "--help") for arg in sys.argv[1:]):
    from src.args import parse_args as _parse_args

    _parse_args()

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.args import parse_args, save_args
from src.device_utils import get_device
from src.models_cross_attention import CrossAttentionScRegNet
from src.train import Trainer as BaseTrainer
from src.utils import (
    Evaluation,
    adj2saprse_tensor,
    scRNADataset,
    set_logging,
    set_seed,
)

logger = logging.getLogger(__name__)


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


class CrossAttentionTrainer(BaseTrainer):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self._device = get_device(args.device)

    @property
    def device(self):
        return self._device

    def get_model(self, data_feature1, data_feature2):
        if data_feature1.size(0) != data_feature2.size(0):
            raise ValueError(
                "scFM and expression features must contain the same number of genes; "
                f"got {data_feature1.size(0)} and {data_feature2.size(0)}"
            )
        gnn_hidden_dims = getattr(self.args, "gnn_hidden_dims", None)
        if gnn_hidden_dims is None:
            gnn_hidden_dims = [self.args.gnn_dim_hidden] * self.args.gnn_num_layers
        return CrossAttentionScRegNet(
            num_genes=data_feature2.size(0),
            expr_input_dim=data_feature2.size(1),
            scfm_dim=data_feature1.size(1),
            gnn_hidden_dims=gnn_hidden_dims,
            latent_dim=self.args.cross_latent_dim,
            link_hidden_dim=self.args.link_hidden_dim,
            fusion_mode=self.args.fusion_mode,
            fusion_layers=self.args.fusion_layers,
            fusion_heads=self.args.fusion_heads,
            fusion_dropout=self.args.fusion_dropout,
            dropout=self.args.dropout,
            directed_link_predictor=self.args.directed_link_predictor,
        ).to(self.device)

    def _split_train_valid(self, train_load):
        raw_data = np.asarray(train_load.train_set)
        labels = raw_data[:, -1].astype(np.int64)
        indices = np.arange(len(raw_data))
        try:
            train_idx, valid_idx = train_test_split(
                indices,
                test_size=0.1,
                random_state=self.args.random_seed,
                shuffle=True,
                stratify=labels,
            )
        except ValueError as exc:
            logger.warning(
                "Stratified validation split unavailable (%s); using a deterministic "
                "random 90/10 split.",
                exc,
            )
            generator = np.random.default_rng(self.args.random_seed)
            shuffled = generator.permutation(indices)
            valid_size = max(1, int(round(0.1 * len(shuffled))))
            valid_idx, train_idx = shuffled[:valid_size], shuffled[valid_size:]
        if len(train_idx) == 0 or len(valid_idx) == 0:
            raise ValueError("Training data is too small to create a 90/10 validation split")
        train_data, valid_data = raw_data[train_idx], raw_data[valid_idx]
        return (
            scRNADataset(train_data, train_load.num_gene, flag=self.args.flag),
            torch.as_tensor(valid_data, device=self.device),
        )

    def _link_loss(self, logits, targets, pos_weight):
        if self.args.loss_type == "bce":
            return F.binary_cross_entropy_with_logits(logits, targets)
        if self.args.loss_type == "pos_weight_bce":
            return F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=pos_weight
            )
        return focal_bce_with_logits(
            logits, targets, self.args.focal_alpha, self.args.focal_gamma
        )

    @torch.no_grad()
    def _evaluate(self, edge_data, adj, data_feature1, data_feature2):
        self.model.eval()
        edge_data = edge_data.to(self.device)
        labels = to_binary_label(edge_data[:, -1])
        logits = self.model(
            data_feature2, adj, edge_data[:, :2], data_feature1
        )
        return Evaluation(
            y_pred=torch.sigmoid(logits).view(-1, 1),
            y_true=labels,
            flag=False,
        )[:2]

    def train(self):
        train_load, test_data, _, data_feature1, data_feature2 = self._prepare_data()
        train_dataset, valid_data = self._split_train_valid(train_load)

        # Rebuild topology from the optimization subset to avoid validation-edge leakage.
        adj = train_dataset.Adj_Generate(
            torch.empty(0, dtype=torch.long, device=self.device), loop=self.args.loop
        )
        adj = adj2saprse_tensor(adj).to(self.device)
        data_feature1 = data_feature1.to(self.device)
        data_feature2 = data_feature2.to(self.device)
        test_data = test_data.to(self.device)

        self.model = self.get_model(data_feature1, data_feature2)
        optimizer = getattr(optim, self.args.optimizer_name)(
            self.model.parameters(),
            lr=self.args.gnn_lr,
            weight_decay=self.args.gnn_weight_decay,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=self.args.batch_size, shuffle=True
        )
        raw_labels = torch.as_tensor(
            train_dataset.train_set[:, -1], dtype=torch.float32, device=self.device
        )
        num_pos = raw_labels.sum()
        num_neg = raw_labels.numel() - num_pos
        pos_weight = (num_neg / num_pos.clamp_min(1.0)).reshape(1)

        best_score = float("-inf")
        best_valid_auc = float("nan")
        best_valid_aupr = float("nan")
        patience_count = 0
        checkpoint_path = os.path.join(
            self.args.ckpt_dir, f"cross_attention_seed{self.args.random_seed}.pt"
        )

        for epoch in tqdm(range(self.args.gnn_epochs)):
            self.model.train()
            running_loss = 0.0
            for train_x, train_y in train_loader:
                train_x = train_x.to(self.device)
                train_y = to_binary_label(train_y.to(self.device))
                optimizer.zero_grad()
                logits = self.model(data_feature2, adj, train_x, data_feature1)
                loss = self._link_loss(logits, train_y, pos_weight)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            should_evaluate = (
                (epoch + 1) % self.args.gnn_eval_interval == 0
                or epoch + 1 == self.args.gnn_epochs
            )
            if not should_evaluate:
                continue

            valid_auc, valid_aupr = self._evaluate(
                valid_data, adj, data_feature1, data_feature2
            )
            current_score = (
                valid_aupr
                if self.args.early_stop_metric == "auprc"
                else valid_auc
            )
            improved = current_score > best_score + self.args.min_delta
            if improved:
                best_score = current_score
                best_valid_auc, best_valid_aupr = valid_auc, valid_aupr
                patience_count = 0
                self.args.ckpt_name = checkpoint_path
                torch.save(self.model.state_dict(), checkpoint_path)
                save_args(self.args, self.args.ckpt_dir)
            else:
                patience_count += 1

            logger.info(
                "Epoch %03d | train_loss=%.4f | valid_AUROC=%.4f | "
                "valid_AUPRC=%.4f | best_%s=%.4f | patience=%d/%d",
                epoch + 1,
                running_loss / max(1, len(train_loader)),
                valid_auc,
                valid_aupr,
                self.args.early_stop_metric,
                best_score,
                patience_count,
                self.args.patience,
            )
            if patience_count >= self.args.patience:
                break

        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device)
        )
        test_auc, test_aupr = self._evaluate(
            test_data, adj, data_feature1, data_feature2
        )
        logger.info(
            "Best validation AUROC/AUPRC: %.4f/%.4f | Final test AUROC/AUPRC: %.4f/%.4f",
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
        args.output_dir = "./out/cross_attention"
    if args.ckpt_dir is None:
        args.ckpt_dir = os.path.join(args.output_dir, "ckpt")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    set_seed(random_seed=args.random_seed)
    logger.critical(
        "Training cross-attention scRegNet on %s with scFM=%s, fusion=%s",
        args.dataset,
        args.llm_type,
        args.fusion_mode,
    )
    trainer = CrossAttentionTrainer(args)
    test_auc, test_aupr = trainer.train()
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return test_auc, test_aupr


if __name__ == "__main__":
    final_auc, final_aupr = main()
    print(final_auc, final_aupr)
