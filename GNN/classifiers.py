import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import (
    GATConv,
    GCNConv,
    GINConv,
    SAGEConv,
    ensure_graph_dependencies,
    global_mean_pool,
)


class GraphClassifierHead(nn.Module):
    def __init__(self, in_dim, classes=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, classes),
        )

    def forward(self, x):
        return self.net(x)


class GCNClassifier(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, num_layers=2, dropout=0.2, classes=256):
        super().__init__()
        ensure_graph_dependencies('gcn')
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.classifier = GraphClassifierHead(hidden_dim, classes=classes)

    def forward(self, data):
        if isinstance(data, torch.Tensor):
            return self._forward_dense(data)
        # PyG Batch/Data path (kept for compatibility)
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        return self.classifier(x)

    def _forward_dense(self, x):
        # Optimized path for complete-graph traces: input x is (B, 700) or (B, 700, 1)
        # Replicates GCNConv math exactly.
        # Because the topology is a fixed complete graph for *every* sample, after the
        # first layer every node embedding becomes identical (global mean of transformed
        # features). We therefore drop the N dimension early and stay in (B, F) space.
        # This keeps memory minimal and compute tiny.
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (B, N, 1)
        B, N, Fin = x.shape
        for i, conv in enumerate(self.convs):
            W = conv.lin.weight
            x = torch.matmul(x, W.t())  # (B, N, Fout)
            mean_x = x.mean(dim=1)  # (B, Fout)  -- same for all nodes
            if conv.bias is not None:
                mean_x = mean_x + conv.bias
            x = mean_x
            if i < len(self.convs) - 1:
                # Only need to re-expand if more layers follow (they will mean again anyway)
                x = x.unsqueeze(1).expand(-1, N, -1)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        # x is now either (B, F) or we ensure it
        if x.dim() == 3:
            x = x.mean(dim=1)
        return self.classifier(x)


class GATClassifier(nn.Module):
    """GAT for ASCAD sample-as-node traces.

    Full softmax attention on complete K_700 collapses to ~uniform 1/N weights
    (every node ≈ global mean → CE stuck at ln(256)). The dense training path
    therefore uses **local temporal attention** (banded window ±attn_window)
    which matches the 1-D trace structure and keeps attention non-uniform.

    Also: residual self projection + LayerNorm (GraphSAGE-style identity path).
    The PyG complete-graph path is kept for API compat only; training uses dense.
    """

    def __init__(
        self,
        input_dim=1,
        hidden_dim=128,
        num_layers=2,
        dropout=0.2,
        heads=4,
        classes=256,
        attn_window=32,
        use_residual=True,
    ):
        super().__init__()
        ensure_graph_dependencies('gat')
        self.dropout = dropout
        self.use_residual = bool(use_residual)
        # Legacy checkpoints (no residual/LN) were trained with full-K attention.
        self.attn_window = (
            (int(attn_window) if attn_window is not None else None)
            if self.use_residual
            else None
        )
        self.convs = nn.ModuleList()
        self.res_projs = nn.ModuleList()
        self.norms = nn.ModuleList()
        # Attention dropout inside GATConv is often too strong on small windows; keep 0
        # and apply feature dropout after the layer instead.
        self.convs.append(GATConv(input_dim, hidden_dim, heads=heads, dropout=0.0, concat=False))
        if self.use_residual:
            self.res_projs.append(
                nn.Identity() if input_dim == hidden_dim else nn.Linear(input_dim, hidden_dim)
            )
            self.norms.append(nn.LayerNorm(hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_dim, hidden_dim, heads=heads, dropout=0.0, concat=False))
            if self.use_residual:
                self.res_projs.append(nn.Identity())
                self.norms.append(nn.LayerNorm(hidden_dim))
        self.classifier = GraphClassifierHead(hidden_dim, classes=classes)

    def forward(self, data):
        if isinstance(data, torch.Tensor):
            return self._forward_dense(data)
        # PyG Batch/Data path
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.use_residual:
            for conv, res_proj, norm in zip(self.convs, self.res_projs, self.norms):
                residual = res_proj(x)
                x = conv(x, edge_index)
                x = x + residual
                x = norm(x)
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        else:
            for conv in self.convs:
                x = conv(x, edge_index)
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        return self.classifier(x)

    def _forward_dense(self, x):
        # Local-window multi-head attention over time (default); optional full K_N if window is None.
        # Implementation is target-chunked over a *source band* only — O(B · N · window · H · C),
        # not O(B · N²), so it stays trainable at benchmark batch sizes.
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        B, N, _ = x.shape
        window = self.attn_window
        # Target chunk: keeps peak activation memory modest under AMP.
        tgt_chunk = 64 if (window is not None and window < N) else 96
        layer_iter = (
            zip(self.convs, self.res_projs, self.norms)
            if self.use_residual
            else ((conv, None, None) for conv in self.convs)
        )
        for conv, res_proj, norm in layer_iter:
            gat = conv
            H, C = gat.heads, gat.out_channels
            curr_fin = x.size(-1)
            residual = res_proj(x) if res_proj is not None else None  # (B, N, C)
            x_flat = x.view(-1, curr_fin)
            x_proj = gat.lin(x_flat).view(B, N, H, C)  # (B, N, H, C)
            alpha_src = (x_proj * gat.att_src).sum(dim=-1)  # (B, N, H)
            alpha_dst = (x_proj * gat.att_dst).sum(dim=-1)  # (B, N, H)

            out = torch.empty(B, N, C, device=x.device, dtype=x.dtype)
            for start in range(0, N, tgt_chunk):
                end = min(start + tgt_chunk, N)
                ch = end - start
                a_dst = alpha_dst[:, start:end].unsqueeze(2)  # (B, ch, 1, H)

                if window is None or window >= N:
                    a_src = alpha_src.unsqueeze(1)  # (B, 1, N, H)
                    alpha = F.leaky_relu(a_dst + a_src, gat.negative_slope)
                    alpha = F.softmax(alpha, dim=2)
                    alpha = F.dropout(alpha, p=self.dropout, training=self.training)
                    chunk_out = torch.einsum('b t j h, b j h c -> b t h c', alpha, x_proj)
                else:
                    w = window
                    j0 = max(0, start - w)
                    j1 = min(N, end + w)
                    a_src = alpha_src[:, j0:j1].unsqueeze(1)  # (B, 1, J, H)
                    alpha = F.leaky_relu(a_dst + a_src, gat.negative_slope)  # (B, ch, J, H)
                    # Mask positions outside each target's [i-w, i+w] band.
                    tgt = torch.arange(start, end, device=x.device).view(ch, 1)
                    src = torch.arange(j0, j1, device=x.device).view(1, -1)
                    valid = (src >= (tgt - w)) & (src <= (tgt + w))  # (ch, J)
                    alpha = alpha.masked_fill(~valid.view(1, ch, -1, 1), torch.finfo(alpha.dtype).min)
                    alpha = F.softmax(alpha, dim=2)
                    alpha = F.dropout(alpha, p=self.dropout, training=self.training)
                    chunk_out = torch.einsum(
                        'b t j h, b j h c -> b t h c',
                        alpha,
                        x_proj[:, j0:j1],
                    )
                out[:, start:end, :] = chunk_out.mean(dim=2)  # mean heads

            if gat.bias is not None:
                out = out + gat.bias.view(1, 1, -1)
            if residual is not None:
                out = out + residual
                out = norm(out)
            x = F.elu(out)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.mean(dim=1)  # global_mean_pool
        return self.classifier(x)


def _gin_mlp(in_dim, hidden_dim, use_layernorm=True):
    """2-layer MLP (+ optional LayerNorm) for GIN updates."""
    layers = [
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
    ]
    if use_layernorm:
        layers.append(nn.LayerNorm(hidden_dim))
    return nn.Sequential(*layers)


class GINClassifier(nn.Module):
    """GIN on complete K_N traces.

    Dense path uses *mean*-normalized aggregation (not raw sum) so scales stay finite
    under AMP + large batch on N=700. Also:
      - train_eps=True (learnable epsilon)
      - LayerNorm inside each GIN MLP (modern)
      - residual around the MLP when dims match (or after a projection)
    """

    def __init__(
        self,
        input_dim=1,
        hidden_dim=128,
        num_layers=2,
        dropout=0.2,
        classes=256,
        use_residual=True,
        use_layernorm=True,
    ):
        super().__init__()
        ensure_graph_dependencies('gin')
        self.dropout = dropout
        self.use_residual = bool(use_residual)
        self.convs = nn.ModuleList()
        self.res_projs = nn.ModuleList()
        self.convs.append(
            GINConv(_gin_mlp(input_dim, hidden_dim, use_layernorm=use_layernorm), train_eps=True)
        )
        if self.use_residual:
            self.res_projs.append(
                nn.Identity() if input_dim == hidden_dim else nn.Linear(input_dim, hidden_dim)
            )
        for _ in range(num_layers - 1):
            self.convs.append(
                GINConv(_gin_mlp(hidden_dim, hidden_dim, use_layernorm=use_layernorm), train_eps=True)
            )
            if self.use_residual:
                self.res_projs.append(nn.Identity())
        self.classifier = GraphClassifierHead(hidden_dim, classes=classes)

    def forward(self, data):
        if isinstance(data, torch.Tensor):
            return self._forward_dense(data)
        # PyG Batch/Data path
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.use_residual:
            for conv, res_proj in zip(self.convs, self.res_projs):
                residual = res_proj(x)
                x = conv(x, edge_index)
                x = x + residual
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        else:
            for conv in self.convs:
                x = conv(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        return self.classifier(x)

    def _forward_dense(self, x):
        # Optimized path: mean-agg GIN for complete graph + optional residual + train_eps.
        # Raw sum over N=700 → ~700x blowup → inf/nan (esp. AMP). Mean keeps scale sane
        # while mean_all + (1+eps)*x still preserves per-node differences.
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        B, N, _ = x.shape
        layer_iter = (
            zip(self.convs, self.res_projs)
            if self.use_residual
            else ((conv, None) for conv in self.convs)
        )
        for conv, res_proj in layer_iter:
            mlp = conv.nn
            eps = conv.eps if hasattr(conv, 'eps') else 0.0
            if isinstance(eps, torch.Tensor):
                eps = eps.to(dtype=x.dtype, device=x.device)
            curr_fin = x.size(-1)
            residual = res_proj(x) if res_proj is not None else None
            # Normalized aggregation for stability on dense complete graph
            mean_all = x.mean(dim=1, keepdim=True)
            out = mean_all + (1.0 + eps) * x
            out = out.view(B * N, curr_fin)
            out = mlp(out)
            out = out.view(B, N, -1)
            if residual is not None:
                out = out + residual
            x = F.relu(out)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.mean(dim=1)
        return self.classifier(x)


class GraphSAGEClassifier(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, num_layers=2, dropout=0.2, classes=256):
        super().__init__()
        ensure_graph_dependencies('graphsage')
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.classifier = GraphClassifierHead(hidden_dim, classes=classes)

    def forward(self, data):
        if isinstance(data, torch.Tensor):
            return self._forward_dense(data)
        # PyG Batch/Data path (kept for compatibility)
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        return self.classifier(x)

    def _forward_dense(self, x):
        # Optimized path: replicates SAGEConv (mean aggr + lin_l(mean) + lin_r(self), optional norm) for complete.
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        B, N, Fin = x.shape
        for conv in self.convs:
            mean_all = x.mean(dim=1, keepdim=True)
            mean_neigh = mean_all.expand(B, N, -1)
            out = conv.lin_l(mean_neigh)
            if getattr(conv, 'root_weight', True) and hasattr(conv, 'lin_r'):
                out = out + conv.lin_r(x)
            if getattr(conv, 'normalize', False):
                out = F.normalize(out, p=2., dim=-1)
            x = out
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.mean(dim=1)
        return self.classifier(x)


def _path_mean_neighbors(x, mask=None):
    """Mean of left/right neighbors on a path graph. x: (B, N, F), mask: (B, N) bool/float."""
    B, N, F = x.shape
    if N == 1:
        return x

    left = torch.zeros_like(x)
    right = torch.zeros_like(x)
    left[:, 1:, :] = x[:, :-1, :]
    right[:, :-1, :] = x[:, 1:, :]

    if mask is None:
        # Degree: ends have 1 neighbor, middle 2
        deg = torch.full((B, N, 1), 2.0, device=x.device, dtype=x.dtype)
        deg[:, 0, :] = 1.0
        deg[:, -1, :] = 1.0
        return (left + right) / deg.clamp(min=1.0)

    m = mask.to(dtype=x.dtype).unsqueeze(-1)  # (B, N, 1)
    x_m = x * m
    left = torch.zeros_like(x)
    right = torch.zeros_like(x)
    left[:, 1:, :] = x_m[:, :-1, :]
    right[:, :-1, :] = x_m[:, 1:, :]
    left_ok = torch.zeros((B, N, 1), device=x.device, dtype=x.dtype)
    right_ok = torch.zeros((B, N, 1), device=x.device, dtype=x.dtype)
    left_ok[:, 1:, :] = m[:, :-1, :]
    right_ok[:, :-1, :] = m[:, 1:, :]
    deg = (left_ok + right_ok).clamp(min=1.0)
    return (left + right) / deg


def _complete_mean_neighbors(x, mask=None):
    """Mean of all other real nodes (complete graph without self)."""
    if mask is None:
        # mean of others = (sum_all - self) / (N-1)
        N = x.size(1)
        if N <= 1:
            return x
        total = x.sum(dim=1, keepdim=True)
        return (total - x) / float(N - 1)

    m = mask.to(dtype=x.dtype).unsqueeze(-1)
    x_m = x * m
    total = x_m.sum(dim=1, keepdim=True)
    deg = m.sum(dim=1, keepdim=True) - m  # exclude self if real
    # nodes with no other neighbors: fall back to self
    neigh = (total - x_m) / deg.clamp(min=1.0)
    isolated = (deg.squeeze(-1) <= 0).unsqueeze(-1)
    return torch.where(isolated, x_m, neigh)


def _combine_product_neighbors(x, mask=None):
    """Cancellation-style messages: mean of (x_i ⊙ x_j) over neighbors j≠i.

    Classical HO combining uses products of multi-time leakages; here each
    node aggregates elementwise products with other real nodes.
    """
    if mask is None:
        # For each i: mean_j≠i (x_i * x_j) = x_i * mean_j≠i x_j
        mean_others = _complete_mean_neighbors(x, mask=None)
        return x * mean_others

    m = mask.to(dtype=x.dtype).unsqueeze(-1)
    x_m = x * m
    mean_others = _complete_mean_neighbors(x, mask=mask)
    return x_m * mean_others


def pairwise_product_affinity(x, mask=None):
    """Pairwise combine affinity A_ij = mean_f (x_i,f * x_j,f), i≠j.

    Returns
    -------
    A : (B, N, N) signed scores; diagonal zeroed; pad rows/cols zeroed if mask given.
    """
    # x: (B, N, F) → A_ij = (1/F) sum_f x_if x_jf = (x @ x^T) / F
    B, N, F = x.shape
    if mask is not None:
        m = mask.to(dtype=x.dtype).unsqueeze(-1)
        x = x * m
    A = torch.matmul(x, x.transpose(1, 2)) / float(max(F, 1))
    eye = torch.eye(N, device=x.device, dtype=x.dtype).unsqueeze(0)
    A = A * (1.0 - eye)
    if mask is not None:
        mm = mask.to(dtype=x.dtype)
        A = A * mm.unsqueeze(2) * mm.unsqueeze(1)
    return A


def row_normalize_abs_affinity(A, eps=1e-8):
    """Row-normalize |A| so each real row sums to 1 (soft weighted adjacency)."""
    W = A.abs()
    denom = W.sum(dim=-1, keepdim=True).clamp(min=eps)
    return W / denom


def _weighted_combine_neighbors(x, mask=None):
    """Weighted cancellation graph.

    1) Build soft adjacency Ã from |mean_f(x_i ⊙ x_j)| (row-normalized).
    2) Message: product combine, aggregated with Ã:
         neigh_i = sum_j Ã_ij (x_i ⊙ x_j) = x_i ⊙ (sum_j Ã_ij x_j)
    """
    A = pairwise_product_affinity(x, mask=mask)
    W = row_normalize_abs_affinity(A)
    # weighted mean of other node features: (B, N, F)
    weighted_others = torch.matmul(W, x)
    if mask is not None:
        weighted_others = weighted_others * mask.to(dtype=x.dtype).unsqueeze(-1)
    return x * weighted_others


def _masked_mean_pool(x, mask=None):
    """x: (B, N, F), mask: (B, N) → (B, F)."""
    if mask is None:
        return x.mean(dim=1)
    m = mask.to(dtype=x.dtype).unsqueeze(-1)
    denom = m.sum(dim=1).clamp(min=1.0)
    return (x * m).sum(dim=1) / denom


class SegmentGraphSAGEClassifier(nn.Module):
    """GraphSAGE on fixed-N segment nodes.

    ``edge_mode``:
      - path: temporal neighbors only (Plan B default)
      - relation / complete: mean over all other real nodes
      - combine: uniform product messages (legacy)
      - weighted: soft A_ij from product affinity + product messages (Plan C default)

    Input tensor shapes accepted by forward:
      - (B, N, F) node features only (no mask → all real)
      - tuple/list (x, mask) with mask (B, N) bool/float
    """

    def __init__(
        self,
        input_dim=12,
        hidden_dim=64,
        num_layers=2,
        dropout=0.2,
        classes=256,
        num_nodes=32,
        head_style='compact',
        edge_mode='path',
    ):
        super().__init__()
        ensure_graph_dependencies('seg_graphsage')
        self.input_dim = int(input_dim)
        self.num_nodes = int(num_nodes)
        self.dropout = dropout
        self.edge_mode = (edge_mode or 'path').lower()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(self.input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        if head_style == 'ascad':
            self.classifier = GraphClassifierHead(hidden_dim, classes=classes)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, classes),
            )

    def forward(self, data, mask=None):
        if isinstance(data, (tuple, list)):
            x, mask = data[0], data[1]
            return self._forward_dense(x, mask)
        if isinstance(data, torch.Tensor):
            return self._forward_dense(data, mask)
        # PyG path
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        return self.classifier(x)

    def _neighbor_aggregate(self, x, mask=None):
        mode = self.edge_mode
        if mode in ('path',):
            return _path_mean_neighbors(x, mask)
        if mode in ('relation', 'complete'):
            return _complete_mean_neighbors(x, mask)
        if mode in ('combine',):
            return _combine_product_neighbors(x, mask)
        if mode in ('weighted', 'weighted_combine', 'product_adj'):
            return _weighted_combine_neighbors(x, mask)
        raise ValueError("Unknown edge_mode '%s'" % mode)

    def _forward_dense(self, x, mask=None):
        if x.dim() == 2:
            # (B, N*F) not supported; require (B, N, F)
            raise ValueError(
                "SegmentGraphSAGE expects (B, N, F) features; got 2D tensor %s" % (tuple(x.shape),)
            )
        if mask is not None and mask.dim() == 1:
            mask = mask.unsqueeze(0).expand(x.size(0), -1)
        for conv in self.convs:
            mean_neigh = self._neighbor_aggregate(x, mask)
            out = conv.lin_l(mean_neigh)
            if getattr(conv, 'root_weight', True) and hasattr(conv, 'lin_r'):
                out = out + conv.lin_r(x)
            if getattr(conv, 'normalize', False):
                out = F.normalize(out, p=2., dim=-1)
            x = F.relu(out)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if mask is not None:
                x = x * mask.to(dtype=x.dtype).unsqueeze(-1)
        pooled = _masked_mean_pool(x, mask)
        return self.classifier(pooled)

    # Back-compat alias
    def _forward_path(self, x, mask=None):
        return self._forward_dense(x, mask)
