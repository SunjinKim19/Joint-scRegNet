# src/train_joint.py

import os
import gc
import logging
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Keep CLI discovery usable even when optional training dependencies are absent.
if __name__ == "__main__" and any(arg in ("-h", "--help") for arg in sys.argv[1:]):
    from src.args import parse_args as _parse_args

    _parse_args()

from src.train import Trainer as BaseTrainer
from src.models_joint import JointInferScRegNet
from src.utils import Evaluation, set_logging, set_seed
from src.args import save_args, parse_args
from src.device_utils import get_device

logger = logging.getLogger(__name__)

def to_binary_label(y):
    """
    scRegNet label이 [batch], [batch, 1], [batch, 2] 중 어떤 형태로 와도
    BCEWithLogitsLoss에 맞는 [batch] binary label로 변환한다.

    [batch, 2] one-hot label이면 positive class인 두 번째 column을 사용한다.
    """
    y = y.float()

    if y.dim() == 2:
        if y.size(1) == 2:
            return y[:, 1].contiguous()
        if y.size(1) == 1:
            return y[:, 0].contiguous()

    return y.view(-1).contiguous()

class JointInferTrainer(BaseTrainer):
    """
    기존 scRegNet Trainer를 상속해서
    model과 training objective만 GeSubNet-style joint inference로 교체한다.
    """

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self._device = get_device(args.device)

    @property
    def device(self):
        return self._device

    def get_model(self, data_feature1, data_feature2):
        num_genes = data_feature2.size(0)
        expr_input_dim = data_feature2.size(1)
        scfm_dim = data_feature1.size(1)

        # 기존 args에 없으면 기본값 사용
        latent_dim = getattr(self.args, "joint_latent_dim", 128)
        cell_hidden_dim = getattr(self.args, "cell_hidden_dim", 256)
        link_hidden_dim = getattr(self.args, "link_hidden_dim", 128)

        gnn_hidden_dims = getattr(self.args, "gnn_hidden_dims", None)
        if gnn_hidden_dims is None:
            gnn_hidden_dims = [self.args.gnn_dim_hidden] * self.args.gnn_num_layers

        model = JointInferScRegNet(
            num_genes=num_genes,
            expr_input_dim=expr_input_dim,
            scfm_dim=scfm_dim,
            gnn_hidden_dims=gnn_hidden_dims,
            cell_hidden_dim=cell_hidden_dim,
            latent_dim=latent_dim,
            link_hidden_dim=link_hidden_dim,
            dropout=self.args.dropout,
            max_recon_cells=self.args.max_recon_cells,
        ).to(self.device)

        return model

    def train(self):
        best_score = -1.0
        best_auc = 0.0
        best_aupr = 0.0
        accumulate_patience = 0

        train_load, test_data, adj, data_feature1, data_feature2 = self._prepare_data()

        self.model = self.get_model(
            data_feature1=data_feature1,
            data_feature2=data_feature2,
        )

        optimizer = getattr(optim, self.args.optimizer_name)(
            self.model.parameters(),
            lr=self.args.gnn_lr,
            weight_decay=self.args.gnn_weight_decay,
        )

        lambda_recon = getattr(self.args, "lambda_recon", 0.01)
        lambda_align = getattr(self.args, "lambda_align", 0.01)
        link_criterion = torch.nn.BCEWithLogitsLoss()
        recon_criterion = torch.nn.MSELoss()

        train_loader = DataLoader(
            train_load, batch_size=self.args.batch_size, shuffle=True
        )

        for epoch in tqdm(range(self.args.gnn_epochs)):
            self.model.train()
            running_loss = 0.0
            running_link_loss = 0.0
            running_recon_loss = 0.0
            running_align_loss = 0.0

            for train_x, train_y in train_loader:
                optimizer.zero_grad()

                train_x = train_x.to(self.device)
                train_y = to_binary_label(train_y.to(self.device))

                logits, aux = self.model(
                    x_gene_expr=data_feature2,
                    adj=adj,
                    edge_pairs=train_x,
                    scfm_emb=data_feature1,
                )

                if logits.shape != train_y.shape:
                    raise RuntimeError(
                        f"Shape mismatch: logits={logits.shape}, train_y={train_y.shape}"
                    )

                # 1. 기존 scRegNet의 link prediction loss
                link_loss = link_criterion(logits, train_y)

                # 2. GeSubNet Infer-M 스타일 reconstruction loss
                # x_recon:  [num_cells, num_genes]
                # x_target: [num_cells, num_genes]
                recon_loss = recon_criterion(
                    aux["x_recon"],
                    aux["x_target"],
                )

                # 3. scFM prior와 GNN representation alignment
                # scFM은 frozen external embedding으로 보고,
                # GNN representation이 scFM context를 따라가도록 유도
                z_graph_norm = F.normalize(aux["z_graph"], p=2, dim=1)
                z_scfm_norm = F.normalize(aux["z_scfm"].detach(), p=2, dim=1)

                align_loss = F.mse_loss(
                    z_graph_norm,
                    z_scfm_norm,
                )

                loss = (
                    link_loss
                    + lambda_recon * recon_loss
                    + lambda_align * align_loss
                )

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_link_loss += link_loss.item()
                running_recon_loss += recon_loss.item()
                running_align_loss += align_loss.item()

            if (epoch + 1) % self.args.gnn_eval_interval == 0:
                self.model.eval()

                with torch.no_grad():
                    test_pairs = test_data[:, :2]
                    test_labels = to_binary_label(test_data[:, 2:].to(self.device))

                    test_logits, _ = self.model(
                        x_gene_expr=data_feature2,
                        adj=adj,
                        edge_pairs=test_pairs,
                        scfm_emb=data_feature1,
                    )

                    test_score = torch.sigmoid(test_logits).view(-1, 1)

                    auc, aupr, _ = Evaluation(
                        y_pred=test_score,
                        y_true=test_labels,
                        flag=False,
                    )

                num_batches = max(1, len(train_loader))
                avg_loss = running_loss / num_batches
                avg_link = running_link_loss / num_batches
                avg_recon = running_recon_loss / num_batches
                avg_align = running_align_loss / num_batches

                logger.info(
                    f"Epoch {epoch + 1:03d} | "
                    f"loss={avg_loss:.4f} | "
                    f"link={avg_link:.4f} | "
                    f"recon={avg_recon:.4f} | "
                    f"align={avg_align:.4f} | "
                    f"AUROC={auc:.4f} | AUPRC={aupr:.4f}"
                )

                # early stopping 기준 선택
                if self.args.early_stop_metric == "auprc":
                    current_score = aupr
                else:
                    current_score = auc

                improved = current_score > best_score + self.args.min_delta

                if improved:
                    accumulate_patience = 0
                    best_score = current_score
                    best_auc = auc
                    best_aupr = aupr

                    self.args.ckpt_name = os.path.join(
                        self.args.ckpt_dir,
                        f"joint_model_seed{self.args.random_seed}.pt",
                    )

                    os.makedirs(self.args.ckpt_dir, exist_ok=True)
                    torch.save(self.model.state_dict(), self.args.ckpt_name)
                    save_args(self.args, self.args.ckpt_dir)

                    logger.info(
                        f"New best model saved | "
                        f"metric={self.args.early_stop_metric} | "
                        f"score={best_score:.4f} | "
                        f"AUROC={best_auc:.4f} | AUPRC={best_aupr:.4f}"
                    )
                else:
                    accumulate_patience += 1
                    logger.info(
                        f"No improvement | "
                        f"patience={accumulate_patience}/{self.args.patience} | "
                        f"best_{self.args.early_stop_metric}={best_score:.4f}"
                    )

                if accumulate_patience >= self.args.patience:
                    logger.info(
                        f"Early stopping triggered at epoch {epoch + 1}. "
                        f"Best AUROC={best_auc:.4f}, Best AUPRC={best_aupr:.4f}"
                    )
                    break

        logger.info(f"best_auroc: {best_auc:.4f}, best_auprc: {best_aupr:.4f}")
        return best_auc, best_aupr


def main():
    set_logging()
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    device = get_device(args.device)
    args.device = str(device)

    logger.critical(
        f"Training JointInfer-scRegNet on {args.dataset}, "
        f"scFM={args.llm_type}, GNN={args.gnn_type}"
    )

    set_seed(random_seed=args.random_seed)

    trainer = JointInferTrainer(args)
    auroc, auprc = trainer.train()

    return auroc, auprc


if __name__ == "__main__":
    auroc, auprc = main()
    print(auroc, auprc)
