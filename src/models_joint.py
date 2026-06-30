# src/models_joint.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


def to_edge_index(adj):
    """
    scRegNet의 adj는 torch sparse tensor 형태로 들어올 수 있으므로
    PyG GCNConv가 사용할 수 있는 edge_index 형태로 변환한다.
    """
    if isinstance(adj, torch.Tensor) and adj.layout != torch.strided:
        return adj.coalesce().indices().long()

    if isinstance(adj, torch.Tensor) and adj.dim() == 2 and adj.size(0) == 2:
        return adj.long()

    if isinstance(adj, torch.Tensor) and adj.dim() == 2 and adj.size(0) == adj.size(1):
        return adj.nonzero(as_tuple=False).T.contiguous().long()

    raise ValueError(
        "adj must be a square dense/sparse adjacency tensor or edge_index [2, E]."
    )


class CellExpressionEncoder(nn.Module):
    """
    GeSubNet의 Patient-M에 해당하는 부분.
    단, 여기서는 scRegNet에 맞게 cell-by-gene expression matrix를
    cell latent representation으로 변환한다.

    입력:
        X_cell_gene: [num_cells, num_genes]

    출력:
        z_cell: [num_cells, latent_dim]
    """

    def __init__(self, num_genes, hidden_dim, latent_dim, dropout=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_genes, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )

    def forward(self, x_cell_gene):
        return self.encoder(x_cell_gene)


class GraphEncoderGCN(nn.Module):
    """
    scRegNet의 GNN branch에 해당하는 부분.
    GeSubNet의 Graph-M처럼 prior regulatory graph를 학습한다.
    """

    def __init__(self, input_dim, hidden_dims, dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList()

        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.convs.append(GCNConv(prev_dim, hidden_dim))
            prev_dim = hidden_dim

        self.dropout = dropout
        self.output_dim = prev_dim

    def forward(self, x_gene_feature, edge_index):
        x = x_gene_feature

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)

            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x


class JointInferScRegNet(nn.Module):
    """
    scRegNet + GeSubNet-style Infer-M.

    기존 scRegNet:
        scFM embedding -> 따로 추출
        GNN embedding  -> 따로 추출
        마지막에 concat 후 link prediction

    제안 구조:
        1. scFM gene embedding을 graph latent space로 projection
        2. GNN으로 regulatory topology embedding 학습
        3. cell expression latent와 gene graph latent를 곱해서 expression 재구성
        4. reconstruction loss가 GNN encoder를 업데이트
        5. 최종 link prediction 수행
    """

    def __init__(
        self,
        num_genes,
        expr_input_dim,
        scfm_dim,
        gnn_hidden_dims,
        cell_hidden_dim=256,
        latent_dim=128,
        link_hidden_dim=128,
        dropout=0.2,
        max_recon_cells=256,
    ):
        super().__init__()

        self.num_genes = num_genes
        self.latent_dim = latent_dim
        self.max_recon_cells = max_recon_cells

        # Graph-M
        self.graph_encoder = GraphEncoderGCN(
            input_dim=expr_input_dim,
            hidden_dims=gnn_hidden_dims,
            dropout=dropout,
        )

        graph_out_dim = self.graph_encoder.output_dim

        # scRegNet의 scFM branch를 graph latent space로 맞춤
        self.scfm_projector = nn.Sequential(
            nn.Linear(scfm_dim, latent_dim),
            nn.ReLU(),
            nn.LayerNorm(latent_dim),
        )

        self.graph_projector = nn.Sequential(
            nn.Linear(graph_out_dim, latent_dim),
            nn.ReLU(),
            nn.LayerNorm(latent_dim),
        )

        # Patient-M 대체 모듈
        self.cell_encoder = CellExpressionEncoder(
            num_genes=num_genes,
            hidden_dim=cell_hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout,
        )

        # scFM과 GNN gene embedding을 단순 concat하지 않고 gated fusion
        self.gate = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Sigmoid(),
        )

        # Link predictor
        # pair representation = [zi, zj, zi*zj, |zi-zj|]
        self.link_predictor = nn.Sequential(
            nn.Linear(latent_dim * 4, link_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(link_hidden_dim, link_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(link_hidden_dim // 2, 1),
        )

    def encode_gene(self, x_gene_expr, adj, scfm_emb):
        edge_index = to_edge_index(adj)

        z_graph_raw = self.graph_encoder(x_gene_expr, edge_index)
        z_graph = self.graph_projector(z_graph_raw)

        z_scfm = self.scfm_projector(scfm_emb)

        gate = self.gate(torch.cat([z_graph, z_scfm], dim=1))

        # scFM prior가 graph representation을 보정하도록 함
        z_gene = gate * z_graph + (1.0 - gate) * z_scfm

        return z_gene, z_graph, z_scfm

    def reconstruct_expression(self, x_gene_expr, z_gene):
        """
        GeSubNet Infer-M의 핵심 아이디어 반영.

        scRegNet의 expression feature는 [num_genes, num_cells] 형태이므로
        이를 transpose하여 [num_cells, num_genes]로 바꾼다.

        z_cell: [num_cells, latent_dim]
        z_gene: [num_genes, latent_dim]

        x_recon = z_cell @ z_gene.T
                = [num_cells, num_genes]
        """
        x_cell_gene = x_gene_expr.T
        if (
            self.training
            and self.max_recon_cells is not None
            and self.max_recon_cells > 0
            and x_cell_gene.size(0) > self.max_recon_cells
        ):
            cell_indices = torch.randperm(
                x_cell_gene.size(0), device=x_cell_gene.device
            )[: self.max_recon_cells]
            x_cell_gene = x_cell_gene[cell_indices]
        z_cell = self.cell_encoder(x_cell_gene)
        x_recon = torch.matmul(z_cell, z_gene.T)

        return x_recon, x_cell_gene, z_cell

    def decode_links(self, z_gene, edge_pairs):
        src = edge_pairs[:, 0].long()
        dst = edge_pairs[:, 1].long()

        zi = z_gene[src]
        zj = z_gene[dst]

        pair_feat = torch.cat(
            [
                zi,
                zj,
                zi * zj,
                torch.abs(zi - zj),
            ],
            dim=1,
        )

        logits = self.link_predictor(pair_feat).view(-1)
        return logits

    def forward(self, x_gene_expr, adj, edge_pairs, scfm_emb):
        z_gene, z_graph, z_scfm = self.encode_gene(
            x_gene_expr=x_gene_expr,
            adj=adj,
            scfm_emb=scfm_emb,
        )

        logits = self.decode_links(z_gene, edge_pairs)

        x_recon, x_cell_gene, z_cell = self.reconstruct_expression(
            x_gene_expr=x_gene_expr,
            z_gene=z_gene,
        )

        aux = {
            "z_gene": z_gene,
            "z_graph": z_graph,
            "z_scfm": z_scfm,
            "z_cell": z_cell,
            "x_recon": x_recon,
            "x_target": x_cell_gene,
        }

        return logits, aux

    @torch.no_grad()
    def predict_all_edges(self, x_gene_expr, adj, scfm_emb, threshold=0.5):
        """
        학습 후 전체 gene pair에 대해 link probability를 계산한다.
        """
        self.eval()

        z_gene, _, _ = self.encode_gene(x_gene_expr, adj, scfm_emb)
        num_genes = z_gene.size(0)

        pairs = torch.combinations(
            torch.arange(num_genes, device=z_gene.device),
            r=2,
        )

        logits = self.decode_links(z_gene, pairs)
        probs = torch.sigmoid(logits)

        selected = pairs[probs >= threshold]

        return selected, probs
