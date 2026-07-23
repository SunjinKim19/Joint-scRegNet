import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


def to_edge_index(adj):
    """Convert a sparse adjacency or an existing edge index for PyG."""
    if isinstance(adj, torch.Tensor) and adj.is_sparse:
        return adj.coalesce().indices().long()
    if isinstance(adj, torch.Tensor) and adj.dim() == 2 and adj.size(0) == 2:
        return adj.long()
    raise ValueError(
        "adj must be a torch sparse adjacency tensor or edge_index with shape [2, E]"
    )


class GraphEncoderGCN(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=0.2):
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one GCN output dimension")
        self.convs = nn.ModuleList()
        for hidden_dim in hidden_dims:
            self.convs.append(GCNConv(input_dim, hidden_dim))
            input_dim = hidden_dim
        self.output_dim = input_dim
        self.dropout = dropout

    def forward(self, x_gene_expr, edge_index):
        x = x_gene_expr
        for index, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if index < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class _AttentionUpdate(nn.Module):
    def __init__(self, latent_dim, num_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            latent_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(latent_dim)
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 4, latent_dim),
        )
        self.norm2 = nn.LayerNorm(latent_dim)

    def forward(self, query, context):
        attn_out, _ = self.attn(
            query.unsqueeze(0), context.unsqueeze(0), context.unsqueeze(0)
        )
        updated = self.norm1(query + self.dropout(attn_out.squeeze(0)))
        return self.norm2(updated + self.dropout(self.ffn(updated)))


class CrossModelAttentionBlock(nn.Module):
    def __init__(self, latent_dim, num_heads=4, dropout=0.2, mode="gnn_to_scfm"):
        super().__init__()
        if mode not in ("gnn_to_scfm", "bidirectional"):
            raise ValueError("attention mode must be gnn_to_scfm or bidirectional")
        if latent_dim % num_heads != 0:
            raise ValueError("latent_dim must be divisible by num_heads")
        self.mode = mode
        self.gnn_update = _AttentionUpdate(latent_dim, num_heads, dropout)
        self.scfm_update = (
            _AttentionUpdate(latent_dim, num_heads, dropout)
            if mode == "bidirectional"
            else None
        )

    def forward(self, z_scfm, z_gnn):
        # Both directions read the same pre-update inputs in bidirectional mode.
        next_gnn = self.gnn_update(z_gnn, z_scfm)
        next_scfm = (
            self.scfm_update(z_scfm, z_gnn)
            if self.scfm_update is not None
            else z_scfm
        )
        return next_scfm, next_gnn


class CrossAttentionScRegNet(nn.Module):
    def __init__(
        self,
        num_genes,
        expr_input_dim,
        scfm_dim,
        gnn_hidden_dims,
        latent_dim=128,
        link_hidden_dim=128,
        fusion_mode="gnn_to_scfm",
        fusion_layers=1,
        fusion_heads=4,
        fusion_dropout=0.2,
        dropout=0.2,
        directed_link_predictor=True,
    ):
        super().__init__()
        self.num_genes = num_genes
        self.directed_link_predictor = bool(directed_link_predictor)
        self.graph_encoder = GraphEncoderGCN(
            expr_input_dim, gnn_hidden_dims, dropout=dropout
        )
        self.graph_projector = nn.Sequential(
            nn.Linear(self.graph_encoder.output_dim, latent_dim),
            nn.ReLU(),
            nn.LayerNorm(latent_dim),
        )
        self.scfm_projector = nn.Sequential(
            nn.Linear(scfm_dim, latent_dim), nn.ReLU(), nn.LayerNorm(latent_dim)
        )

        attention_layers = 0 if fusion_mode == "gated" else fusion_layers
        self.fusion_blocks = nn.ModuleList(
            CrossModelAttentionBlock(
                latent_dim, fusion_heads, fusion_dropout, mode=fusion_mode
            )
            for _ in range(attention_layers)
        )
        self.gate = nn.Sequential(nn.Linear(latent_dim * 2, latent_dim), nn.Sigmoid())

        if self.directed_link_predictor:
            self.src_proj = nn.Linear(latent_dim, latent_dim)
            self.dst_proj = nn.Linear(latent_dim, latent_dim)
            self.interaction_proj = nn.Linear(latent_dim, latent_dim)
            pair_dim = latent_dim * 5
        else:
            pair_dim = latent_dim * 4
        self.link_predictor = nn.Sequential(
            nn.Linear(pair_dim, link_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(link_hidden_dim, 1),
        )

    def encode_gene(self, x_gene_expr, adj, scfm_emb):
        edge_index = to_edge_index(adj)
        z_graph_raw = self.graph_encoder(x_gene_expr, edge_index)
        z_gnn = self.graph_projector(z_graph_raw)
        z_scfm = self.scfm_projector(scfm_emb)
        for block in self.fusion_blocks:
            z_scfm, z_gnn = block(z_scfm, z_gnn)
        gate = self.gate(torch.cat([z_gnn, z_scfm], dim=-1))
        return gate * z_gnn + (1.0 - gate) * z_scfm

    def decode_links(self, z_gene, edge_pairs):
        src = edge_pairs[:, 0].long()
        dst = edge_pairs[:, 1].long()
        if self.directed_link_predictor:
            h_src = self.src_proj(z_gene[src])
            h_dst = self.dst_proj(z_gene[dst])
            product = h_src * h_dst
            interaction = self.interaction_proj(product)
            pair_feat = torch.cat(
                [h_src, h_dst, product, h_src - h_dst, interaction], dim=-1
            )
        else:
            h_src, h_dst = z_gene[src], z_gene[dst]
            pair_feat = torch.cat(
                [h_src, h_dst, h_src * h_dst, torch.abs(h_src - h_dst)], dim=-1
            )
        return self.link_predictor(pair_feat).view(-1)

    def forward(self, x_gene_expr, adj, edge_pairs, scfm_emb):
        z_gene = self.encode_gene(x_gene_expr, adj, scfm_emb)
        return self.decode_links(z_gene, edge_pairs)
