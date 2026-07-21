"""Serial Cell-M -> soft graph construction -> Graph-M link predictor."""

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


class CellM(nn.Module):
    """Produce per-gene context embeddings from scFM and raw expression inputs."""

    def __init__(self, scfm_dim, num_genes, hidden_dim=256, latent_dim=128, dropout=0.2, scfm_tune_mode="adapter"):
        super().__init__()
        if scfm_tune_mode in ("top", "full"):
            warnings.warn(
                "Only precomputed scFM embeddings are available; using a trainable adapter.",
                RuntimeWarning,
            )
            scfm_tune_mode = "adapter"
        if scfm_tune_mode not in ("frozen_embedding", "adapter"):
            raise ValueError("scfm_tune_mode must be frozen_embedding or adapter")
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

    def forward(self, scfm_gene_emb, raw_expr=None):
        # The upstream scFM tensor is frozen; gradients still train this Cell-M adapter.
        z_ctx = self.scfm_adapter(scfm_gene_emb.detach())
        z_cell = None
        if raw_expr is not None:
            if raw_expr.dim() != 2 or raw_expr.size(1) != self.cell_encoder[0].in_features:
                raise ValueError(
                    "raw_expr must be [num_cells, num_genes], got "
                    f"{tuple(raw_expr.shape)}"
                )
            z_cell = self.cell_encoder(raw_expr.float())
            cell_context = self.cell_to_gene_context(z_cell.mean(dim=0))
            z_ctx = self.condition_norm(z_ctx + cell_context.unsqueeze(0))
        return {"z_ctx": z_ctx, "z_cell": z_cell}


class DirectedDecoder(nn.Module):
    """Score ordered source-target pairs from Graph-M output only."""

    def __init__(self, latent_dim=128, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.src_proj = nn.Linear(latent_dim, latent_dim)
        self.dst_proj = nn.Linear(latent_dim, latent_dim)
        self.interaction_proj = nn.Linear(latent_dim, latent_dim)
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim * 5, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(1, hidden_dim // 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(1, hidden_dim // 2), 1),
        )

    def forward(self, z_graph, edge_pairs):
        source = edge_pairs[:, 0].long()
        target = edge_pairs[:, 1].long()
        z_source = self.src_proj(z_graph[source])
        z_target = self.dst_proj(z_graph[target])
        interaction = z_source * z_target
        pair_features = torch.cat(
            (
                z_source,
                z_target,
                interaction,
                z_source - z_target,
                self.interaction_proj(interaction),
            ),
            dim=-1,
        )
        return self.predictor(pair_features).view(-1)


def adjacency_to_dense(adjacency, num_genes, device, dtype):
    """Return a differentiable dense [N, N] adjacency with clear validation."""
    if not isinstance(adjacency, torch.Tensor):
        raise TypeError("adjacency must be a torch.Tensor")
    dense = adjacency.to_dense() if adjacency.is_sparse else adjacency
    if dense.shape != (num_genes, num_genes):
        raise ValueError(
            f"adjacency must have shape [{num_genes}, {num_genes}], got {tuple(dense.shape)}"
        )
    return dense.to(device=device, dtype=dtype)


class GraphConstructor(nn.Module):
    """Construct directed TF->target soft edge probabilities from gene context."""

    def __init__(self, latent_dim=128, constructor_type="mlp", hidden_dim=128):
        super().__init__()
        if constructor_type not in ("mlp", "bilinear"):
            raise ValueError("constructor_type must be 'mlp' or 'bilinear'")
        self.constructor_type = constructor_type
        if constructor_type == "bilinear":
            self.bilinear_weight = nn.Parameter(torch.empty(latent_dim, latent_dim))
            nn.init.xavier_uniform_(self.bilinear_weight)
        else:
            self.score_mlp = nn.Sequential(
                nn.Linear(latent_dim * 4, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

    def _score_source_chunk(self, source, targets):
        if self.constructor_type == "bilinear":
            return (source @ self.bilinear_weight) @ targets.t()
        num_source, num_target = source.size(0), targets.size(0)
        src = source[:, None, :].expand(num_source, num_target, -1)
        dst = targets[None, :, :].expand(num_source, num_target, -1)
        features = torch.cat((src, dst, src * dst, src - dst), dim=-1)
        return self.score_mlp(features).squeeze(-1)

    def forward(self, z_ctx, tf_indices=None):
        if z_ctx.dim() != 2:
            raise ValueError(f"z_ctx must be [N, d], got {tuple(z_ctx.shape)}")
        num_genes = z_ctx.size(0)
        if tf_indices is None:
            tf_indices = torch.arange(num_genes, device=z_ctx.device)
        tf_indices = tf_indices.to(device=z_ctx.device, dtype=torch.long).view(-1)
        if tf_indices.numel() == 0:
            raise ValueError("tf_indices must contain at least one regulator index")
        if tf_indices.min() < 0 or tf_indices.max() >= num_genes:
            raise IndexError("tf_indices contains an index outside z_ctx")

        # Only TF rows receive learned scores. Non-TF genes cannot be regulators.
        # Chunk TF sources to keep the MLP implementation practical for N~910.
        score_chunks = [
            self._score_source_chunk(z_ctx[index_chunk], z_ctx)
            for index_chunk in tf_indices.split(32)
        ]
        tf_probabilities = torch.sigmoid(torch.cat(score_chunks, dim=0))
        source_mask = F.one_hot(tf_indices, num_classes=num_genes).to(z_ctx.dtype).t()
        return source_mask @ tf_probabilities

    @torch.no_grad()
    def hard_topk(self, adjacency, k):
        """Hard TF-row top-k view for evaluation/visualization, never training."""
        if k <= 0 or k >= adjacency.size(1):
            return adjacency.clone()
        _, indices = adjacency.topk(k, dim=1)
        mask = torch.zeros_like(adjacency).scatter_(1, indices, 1.0)
        return adjacency * mask


class WeightedGraphLayer(nn.Module):
    """Dense directed message passing that preserves gradients to edge weights."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.self_linear = nn.Linear(input_dim, output_dim)
        self.message_linear = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, node_features, adjacency):
        # adjacency[source, target]: each target aggregates incoming TF messages.
        incoming_degree = adjacency.sum(dim=0).clamp_min(1e-6).unsqueeze(-1)
        messages = adjacency.t() @ self.message_linear(node_features)
        return self.self_linear(node_features) + messages / incoming_degree


class WeightedGraphM(nn.Module):
    """Graph-M over z_ctx node features and the differentiable A_final."""

    def __init__(self, latent_dim=128, hidden_dims=None, dropout=0.2):
        super().__init__()
        hidden_dims = hidden_dims or [latent_dim, latent_dim]
        dims = [latent_dim] + list(hidden_dims)
        self.layers = nn.ModuleList(
            WeightedGraphLayer(dims[index], dims[index + 1])
            for index in range(len(dims) - 1)
        )
        self.output_projector = nn.Linear(dims[-1], latent_dim)
        self.output_norm = nn.LayerNorm(latent_dim)
        self.dropout = dropout

    def forward(self, node_features, adjacency):
        if node_features.dim() != 2 or adjacency.shape != (
            node_features.size(0),
            node_features.size(0),
        ):
            raise ValueError(
                "Graph-M expects node_features [N, d] and adjacency [N, N]; "
                f"got {tuple(node_features.shape)} and {tuple(adjacency.shape)}"
            )
        x = node_features
        for layer in self.layers:
            x = layer(x, adjacency)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.output_norm(self.output_projector(x))


class CellGuidedGraphScRegNet(nn.Module):
    """End-to-end serial model with no Infer-M or z_joint fusion path."""

    def __init__(
        self,
        num_genes,
        scfm_dim,
        latent_dim=128,
        condition_hidden_dim=256,
        gnn_hidden_dims=None,
        link_hidden_dim=128,
        dropout=0.2,
        graph_alpha=0.8,
        graph_constructor_type="mlp",
        scfm_tune_mode="adapter",
    ):
        super().__init__()
        if not 0.0 <= graph_alpha <= 1.0:
            raise ValueError(f"graph_alpha must be in [0, 1], got {graph_alpha}")
        self.num_genes = num_genes
        # Kept as a buffer so a future implementation can replace it with a gate.
        self.register_buffer("graph_alpha", torch.tensor(float(graph_alpha)))
        self.cell_m = CellM(
            scfm_dim=scfm_dim,
            num_genes=num_genes,
            hidden_dim=condition_hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout,
            scfm_tune_mode=scfm_tune_mode,
        )
        self.graph_constructor = GraphConstructor(
            latent_dim=latent_dim,
            constructor_type=graph_constructor_type,
            hidden_dim=link_hidden_dim,
        )
        self.graph_m = WeightedGraphM(latent_dim, gnn_hidden_dims, dropout)
        self.decoder = DirectedDecoder(latent_dim, link_hidden_dim, dropout)

    def forward(
        self,
        scfm_gene_emb,
        prior_adjacency,
        edge_pairs,
        raw_expr=None,
        tf_indices=None,
        hard_topk_eval_only=0,
    ):
        if scfm_gene_emb.size(0) != self.num_genes:
            raise ValueError(
                f"scFM embeddings contain {scfm_gene_emb.size(0)} genes; expected {self.num_genes}"
            )
        edge_pairs = edge_pairs.long()
        if edge_pairs.dim() != 2 or edge_pairs.size(1) != 2:
            raise ValueError(f"edge_pairs must be [B, 2], got {tuple(edge_pairs.shape)}")
        if edge_pairs.numel() and (edge_pairs.min() < 0 or edge_pairs.max() >= self.num_genes):
            raise IndexError("edge_pairs contains a gene index outside the model gene set")

        cell_output = self.cell_m(scfm_gene_emb, raw_expr=raw_expr)
        z_ctx = cell_output["z_ctx"]
        a_ctx = self.graph_constructor(z_ctx, tf_indices=tf_indices)
        if tf_indices is None:
            candidate_mask = torch.ones(
                (self.num_genes, self.num_genes),
                dtype=torch.bool,
                device=z_ctx.device,
            )
        else:
            tf_indices_for_mask = tf_indices.to(
                device=z_ctx.device, dtype=torch.long
            ).view(-1)
            source_is_tf = F.one_hot(
                tf_indices_for_mask, num_classes=self.num_genes
            ).sum(dim=0).bool()
            candidate_mask = source_is_tf[:, None].expand(
                self.num_genes, self.num_genes
            )
        a_ctx_for_graph = a_ctx
        if hard_topk_eval_only:
            if self.training:
                raise RuntimeError("hard_topk_eval_only cannot be used while training")
            a_ctx_for_graph = self.graph_constructor.hard_topk(
                a_ctx, hard_topk_eval_only
            )
        a_prior = adjacency_to_dense(
            prior_adjacency, self.num_genes, z_ctx.device, z_ctx.dtype
        )
        a_final = self.graph_alpha * a_prior + (1.0 - self.graph_alpha) * a_ctx_for_graph
        z_graph = self.graph_m(node_features=z_ctx, adjacency=a_final)
        logits = self.decoder(z_graph, edge_pairs)
        pred = torch.sigmoid(logits)
        return {
            "logits": logits,
            "pred": pred,
            "probabilities": pred,
            "z_ctx": z_ctx,
            "A_ctx": a_ctx,
            "A_prior": a_prior,
            "A_final": a_final,
            "z_graph": z_graph,
            "z_cell": cell_output["z_cell"],
            "candidate_mask": candidate_mask,
        }
