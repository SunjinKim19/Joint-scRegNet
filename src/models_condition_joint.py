import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


def to_edge_index(adj):
    """Convert adjacency-like inputs to a PyG edge_index tensor."""
    if isinstance(adj, torch.Tensor) and adj.is_sparse:
        return adj.coalesce().indices().long()
    if isinstance(adj, torch.Tensor) and adj.dim() == 2 and adj.size(0) == 2:
        return adj.long()
    if (
        isinstance(adj, torch.Tensor)
        and adj.dim() == 2
        and adj.size(0) == adj.size(1)
    ):
        return adj.nonzero(as_tuple=False).t().contiguous().long()
    raise ValueError(
        "adj must be a torch sparse adjacency tensor, edge_index [2, E], "
        "or dense square adjacency [G, G]."
    )


class CellConditionM(nn.Module):
    """
    scRegNet analogue of GeSubNet's Patient/Cell/Condition-M.

    This module produces context embeddings only. It does not perform TF-target
    link prediction.
    """

    def __init__(
        self,
        scfm_dim,
        num_genes,
        hidden_dim=256,
        latent_dim=128,
        dropout=0.2,
        scfm_tune_mode="adapter",
    ):
        super().__init__()
        if scfm_tune_mode in ("top", "full"):
            warnings.warn(
                "Full/top scFM fine-tuning is not available because the current "
                "pipeline provides precomputed embeddings only; falling back to "
                "the trainable adapter.",
                RuntimeWarning,
            )
            scfm_tune_mode = "adapter"
        if scfm_tune_mode not in ("frozen_embedding", "adapter"):
            raise ValueError(
                "scfm_tune_mode must be one of frozen_embedding, adapter, top, full"
            )
        self.scfm_tune_mode = scfm_tune_mode
        self.scfm_adapter = nn.Sequential(
            nn.Linear(scfm_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.cell_encoder = nn.Sequential(
            nn.Linear(num_genes, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.cell_to_gene_context = nn.Linear(latent_dim, latent_dim)
        self.condition_norm = nn.LayerNorm(latent_dim)

    def forward(self, scfm_gene_emb, raw_expr=None, condition_id=None):
        # Precomputed scFM embeddings are treated as frozen external inputs.
        scfm_input = scfm_gene_emb.detach()
        z_gene_ctx = self.scfm_adapter(scfm_input)

        z_cell = None
        if raw_expr is not None:
            z_cell = self.cell_encoder(raw_expr.float())
            cell_context = self.cell_to_gene_context(z_cell.mean(dim=0))
            z_gene_ctx = self.condition_norm(z_gene_ctx + cell_context.unsqueeze(0))

        z_condition = z_gene_ctx.mean(dim=0)
        return {
            "z_gene_ctx": z_gene_ctx,
            "z_condition": z_condition,
            "z_cell": z_cell,
        }


class GraphM(nn.Module):
    """Graph-M: trainable GCN encoder over the fixed TF-target prior graph."""

    def __init__(self, input_dim, hidden_dims, latent_dim=128, dropout=0.2):
        super().__init__()
        if not hidden_dims:
            hidden_dims = [latent_dim]
        self.convs = nn.ModuleList()
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.convs.append(GCNConv(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        self.projector = nn.Sequential(
            nn.Linear(prev_dim, latent_dim),
            nn.ReLU(),
            nn.LayerNorm(latent_dim),
        )
        self.dropout = dropout

    def forward(self, graph_node_features, adj):
        edge_index = to_edge_index(adj)
        x = graph_node_features.float()
        for index, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if index < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return self.projector(x)


class _AttentionUpdate(nn.Module):
    def __init__(self, latent_dim, num_heads=4, dropout=0.2):
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


class _BidirectionalInferBlock(nn.Module):
    def __init__(self, latent_dim, num_heads=4, dropout=0.2):
        super().__init__()
        self.ctx_from_graph = _AttentionUpdate(latent_dim, num_heads, dropout)
        self.graph_from_ctx = _AttentionUpdate(latent_dim, num_heads, dropout)

    def forward(self, z_gene_ctx, z_graph):
        next_ctx = self.ctx_from_graph(z_gene_ctx, z_graph)
        next_graph = self.graph_from_ctx(z_graph, z_gene_ctx)
        return next_ctx, next_graph


class InferM(nn.Module):
    """Joint inference module where Cell/Condition-M and Graph-M interact."""

    def __init__(
        self,
        latent_dim=128,
        fusion_mode="bidirectional",
        num_layers=1,
        num_heads=4,
        dropout=0.2,
    ):
        super().__init__()
        if fusion_mode not in ("gated", "bidirectional"):
            raise ValueError("infer fusion mode must be gated or bidirectional")
        if latent_dim % num_heads != 0:
            raise ValueError("latent_dim must be divisible by infer_heads")
        self.fusion_mode = fusion_mode
        attention_layers = 0 if fusion_mode == "gated" else num_layers
        self.blocks = nn.ModuleList(
            _BidirectionalInferBlock(latent_dim, num_heads, dropout)
            for _ in range(attention_layers)
        )
        self.gate = nn.Sequential(nn.Linear(latent_dim * 2, latent_dim), nn.Sigmoid())

    def forward(self, z_gene_ctx, z_graph):
        z_gene_ctx_updated = z_gene_ctx
        z_graph_updated = z_graph
        for block in self.blocks:
            z_gene_ctx_updated, z_graph_updated = block(
                z_gene_ctx_updated, z_graph_updated
            )
        gate = self.gate(torch.cat([z_graph_updated, z_gene_ctx_updated], dim=-1))
        z_joint = gate * z_graph_updated + (1.0 - gate) * z_gene_ctx_updated
        return z_joint, z_gene_ctx_updated, z_graph_updated


class DirectedLinkDecoder(nn.Module):
    """Directed TF-source / target-destination link decoder."""

    def __init__(self, latent_dim=128, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.src_proj = nn.Linear(latent_dim, latent_dim)
        self.dst_proj = nn.Linear(latent_dim, latent_dim)
        self.interaction_proj = nn.Linear(latent_dim, latent_dim)
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim * 5, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, z_joint, edge_pairs):
        src = edge_pairs[:, 0].long()
        dst = edge_pairs[:, 1].long()
        h_src = self.src_proj(z_joint[src])
        h_dst = self.dst_proj(z_joint[dst])
        interaction_base = h_src * h_dst
        pair_feat = torch.cat(
            [
                h_src,
                h_dst,
                interaction_base,
                h_src - h_dst,
                self.interaction_proj(interaction_base),
            ],
            dim=-1,
        )
        return self.predictor(pair_feat).view(-1)


class ContextDecoder(nn.Module):
    """Reconstruct scFM context from the joint Infer-M embedding."""

    def __init__(
        self,
        latent_dim=128,
        scfm_dim=512,
        hidden_dim=256,
        dropout=0.2,
        target="projected_scfm",
    ):
        super().__init__()
        if target not in ("none", "scfm", "projected_scfm"):
            raise ValueError(
                "context reconstruction target must be none, scfm, or projected_scfm"
            )
        self.target = target
        if target == "none":
            self.decoder = None
        else:
            output_dim = scfm_dim if target == "scfm" else latent_dim
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )

    def forward(self, z_joint):
        if self.decoder is None:
            return None
        return self.decoder(z_joint)


class ConditionClassifier(nn.Module):
    def __init__(self, latent_dim=128, num_conditions=1):
        super().__init__()
        self.classifier = nn.Linear(latent_dim, num_conditions)

    def forward(self, z_condition):
        return self.classifier(z_condition)


class ConditionJointScRegNet(nn.Module):
    """GeSubNet-inspired Cell/Condition-M + Graph-M + Infer-M scRegNet."""

    def __init__(
        self,
        num_genes,
        graph_input_dim,
        scfm_dim,
        gnn_hidden_dims,
        latent_dim=128,
        condition_hidden_dim=256,
        link_hidden_dim=128,
        dropout=0.2,
        infer_fusion_mode="bidirectional",
        infer_layers=1,
        infer_heads=4,
        infer_dropout=0.2,
        context_recon_target="projected_scfm",
        scfm_tune_mode="adapter",
        num_conditions=1,
    ):
        super().__init__()
        self.num_genes = num_genes
        self.context_recon_target = context_recon_target
        self.cell_condition_m = CellConditionM(
            scfm_dim=scfm_dim,
            num_genes=num_genes,
            hidden_dim=condition_hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout,
            scfm_tune_mode=scfm_tune_mode,
        )
        self.graph_m = GraphM(
            input_dim=graph_input_dim,
            hidden_dims=gnn_hidden_dims,
            latent_dim=latent_dim,
            dropout=dropout,
        )
        self.infer_m = InferM(
            latent_dim=latent_dim,
            fusion_mode=infer_fusion_mode,
            num_layers=infer_layers,
            num_heads=infer_heads,
            dropout=infer_dropout,
        )
        self.link_decoder = DirectedLinkDecoder(
            latent_dim=latent_dim,
            hidden_dim=link_hidden_dim,
            dropout=dropout,
        )
        self.context_decoder = ContextDecoder(
            latent_dim=latent_dim,
            scfm_dim=scfm_dim,
            hidden_dim=condition_hidden_dim,
            dropout=dropout,
            target=context_recon_target,
        )
        self.condition_classifier = (
            ConditionClassifier(latent_dim, num_conditions)
            if num_conditions > 1
            else None
        )

    def _context_target(self, scfm_gene_emb, z_gene_ctx):
        if self.context_recon_target == "none":
            return None
        if self.context_recon_target == "scfm":
            return scfm_gene_emb.detach()
        if self.context_recon_target == "projected_scfm":
            return z_gene_ctx.detach()
        raise ValueError(f"Unknown context target: {self.context_recon_target}")

    def forward(
        self,
        scfm_gene_emb,
        graph_node_features,
        adj,
        edge_pairs,
        raw_expr=None,
        condition_id=None,
    ):
        condition_out = self.cell_condition_m(
            scfm_gene_emb=scfm_gene_emb,
            raw_expr=raw_expr,
            condition_id=condition_id,
        )
        z_gene_ctx = condition_out["z_gene_ctx"]
        z_graph = self.graph_m(graph_node_features, adj)
        z_joint, z_gene_ctx_updated, z_graph_updated = self.infer_m(
            z_gene_ctx, z_graph
        )
        logits = self.link_decoder(z_joint, edge_pairs)
        context_recon = self.context_decoder(z_joint)
        condition_logits = None
        if self.condition_classifier is not None:
            condition_logits = self.condition_classifier(z_joint.mean(dim=0))
        return {
            "logits": logits,
            "z_gene_ctx": z_gene_ctx,
            "z_condition": condition_out["z_condition"],
            "z_cell": condition_out["z_cell"],
            "z_graph": z_graph,
            "z_gene_ctx_updated": z_gene_ctx_updated,
            "z_graph_updated": z_graph_updated,
            "z_joint": z_joint,
            "context_recon": context_recon,
            "context_target": self._context_target(scfm_gene_emb, z_gene_ctx),
            "condition_logits": condition_logits,
        }
