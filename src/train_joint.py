# src/train_joint.py

import os
import gc
import logging
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.train import Trainer as BaseTrainer
from src.models_joint import JointInferScRegNet
from src.utils import Evaluation, set_logging, set_seed
from src.args import save_args, parse_args

logger = logging.getLogger(__name__)


class JointInferTrainer(BaseTrainer):
    """
    기존 scRegNet Trainer를 상속해서
    model과 training objective만 GeSubNet-style joint inference로 교체한다.
    """

    def get_model(self, data_feature1, data_feature2):
        num_genes = data_feature2.size(0)
        expr_input_dim = data_feature2.size(1)
        scfm_dim = data_feature1.size(1)

        # 기존 args에 없으면 기본값 사용
        latent_dim = getattr(self.args, "joint_latent_dim", 128)
        cell_hidden_dim = getattr(self.args, "cell_hidden_dim", 256)
        link_hidden_dim = getattr(self.args, "link_hidden_dim", 128)

        model = JointInferScRegNet(
            num_genes=num_genes,
            expr_input_dim=expr_input_dim,
            scfm_dim=scfm_dim,
            gnn_hidden_dims=self.args.gnn_hidden_dims,
            cell_hidden_dim=cell_hidden_dim,
            latent_dim=latent_dim,
            link_hidden_dim=link_hidden_dim,
            dropout=self.args.dropout,
        ).to(self.device)

        return model

    def train(self):
        max_auc = 0.0
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

        lambda_recon = getattr(self.args, "lambda_recon", 0.1)
        lambda_align = getattr(self.args, "lambda_align", 0.01)

        for epoch in tqdm(range(self.args.gnn_epochs)):
            self.model.train()
            running_loss = 0.0
            running_link_loss = 0.0
            running_recon_loss = 0.0
            running_align_loss = 0.0

            for train_x, train_y in DataLoader(
                train_load,
                batch_size=self.args.batch_size,
                shuffle=True,
            ):
                optimizer.zero_grad()

                train_x = train_x.to(self.device)
                train_y = train_y.to(self.device).float().view(-1)

                logits, aux = self.model(
                    x_gene_expr=data_feature2,
                    adj=adj,
                    edge_pairs=train_x,
                    scfm_emb=data_feature1,
                )

                # 1. 기존 scRegNet의 link prediction loss
                link_loss = F.binary_cross_entropy_with_logits(
                    logits,
                    train_y,
                )

                # 2. GeSubNet Infer-M 스타일 reconstruction loss
                # x_recon:  [num_cells, num_genes]
                # x_target: [num_cells, num_genes]
                recon_loss = F.mse_loss(
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
                    test_labels = test_data[:, -1]

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

                avg_loss = running_loss / max(1, len(train_load))
                avg_link = running_link_loss / max(1, len(train_load))
                avg_recon = running_recon_loss / max(1, len(train_load))
                avg_align = running_align_loss / max(1, len(train_load))

                logger.info(
                    f"Epoch {epoch + 1:03d} | "
                    f"loss={avg_loss:.4f} | "
                    f"link={avg_link:.4f} | "
                    f"recon={avg_recon:.4f} | "
                    f"align={avg_align:.4f} | "
                    f"AUROC={auc:.4f} | AUPRC={aupr:.4f}"
                )

                if auc > max_auc:
                    accumulate_patience = 0
                    max_auc = auc
                    best_aupr = aupr

                    self.args.ckpt_name = os.path.join(
                        self.args.ckpt_dir,
                        f"joint_model_seed{self.args.random_seed}.pt",
                    )

                    torch.save(
                        self.model.state_dict(),
                        self.args.ckpt_name,
                    )

                    save_args(self.args, self.args.ckpt_dir)

                else:
                    accumulate_patience += 1

                if accumulate_patience >= 10:
                    break

        logger.info(f"best_auroc: {max_auc:.4f}, best_auprc: {best_aupr:.4f}")
        return max_auc, best_aupr


def main():
    set_logging()
    args = parse_args()

    logger.critical(
        f"Training JointInfer-scRegNet on {args.dataset}, "
        f"scFM={args.llm_type}, GNN={args.gnn_type}"
    )

    set_seed(random_seed=args.random_seed)

    trainer = JointInferTrainer(args)
    auroc, auprc = trainer.train()

    del trainer
    torch.cuda.empty_cache()
    gc.collect()

    return auroc, auprc


if __name__ == "__main__":
    auroc, auprc = main()
    print(auroc, auprc)