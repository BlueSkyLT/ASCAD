"""Segment-as-node graph construction for ASCAD traces.

Each trace is split into waveform segments (find_segments_v2 by default).
Each segment becomes one node with a fixed-length feature vector
(segment_to_fixed_vector). Variable segment counts are padded/cropped to a
fixed N for dense batched training. Topology is a path graph (time order).

Does not modify ASCAD HDF5 files. Optional feature cache writes NPZ under a
caller-chosen directory (typically beside trained models).
"""

from __future__ import annotations

import hashlib
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from utils.find_segments import (
    find_segments,
    find_segments_v2,
    segment_to_fixed_vector,
)

# Defaults match T-003 Plan B first cut.
DEFAULT_THRESHOLD = 10
DEFAULT_OUT_DIM = 12
DEFAULT_NUM_NODES = 32
DEFAULT_SEGMENTER = "v2"  # "v2" | "v1" | "uniform" | "snr"
DEFAULT_EDGE_MODE = "path"  # "path" | "relation" | "complete" | "combine"


def _segment_bounds(
    trace: np.ndarray,
    segmenter: str = DEFAULT_SEGMENTER,
    threshold: int = DEFAULT_THRESHOLD,
    num_uniform: Optional[int] = None,
    fixed_windows: Optional[Sequence] = None,
    **seg_kwargs,
) -> list:
    trace = np.asarray(trace, dtype=np.float32).ravel()
    n = len(trace)
    if n == 0:
        return []
    if segmenter == "snr":
        # Fixed PoI windows from profiling SNR peaks (same for every trace).
        if not fixed_windows:
            raise ValueError(
                "segmenter='snr' requires fixed_windows from profiling SNR peak discovery"
            )
        return [(int(s), int(e)) for s, e in fixed_windows]
    if segmenter == "v2":
        return find_segments_v2(trace, threshold=threshold, **seg_kwargs)
    if segmenter == "v1":
        return find_segments(trace, threshold=threshold)
    if segmenter == "uniform":
        k = int(num_uniform or DEFAULT_NUM_NODES)
        k = max(1, min(k, n))
        edges = np.linspace(0, n, k + 1, dtype=int)
        segs = []
        for i in range(k):
            s, e = int(edges[i]), int(edges[i + 1]) - 1
            if e < s:
                e = s
            segs.append((s, e))
        return segs
    raise ValueError(f"Unknown segmenter '{segmenter}' (use v1|v2|uniform|snr)")


def _merge_shortest(bounds: list, target_n: int) -> list:
    """Merge adjacent segments until len(bounds) <= target_n."""
    bounds = list(bounds)
    while len(bounds) > target_n:
        lengths = [e - s + 1 for s, e in bounds]
        # Prefer merging the shortest pair of adjacent segments
        best_i = 0
        best_len = lengths[0] + lengths[1]
        for i in range(1, len(bounds) - 1):
            pair = lengths[i] + lengths[i + 1]
            if pair < best_len:
                best_len = pair
                best_i = i
        s0, _ = bounds[best_i]
        _, e1 = bounds[best_i + 1]
        bounds[best_i] = (s0, e1)
        del bounds[best_i + 1]
    return bounds


def trace_to_fixed_segment_features(
    trace: np.ndarray,
    num_nodes: int = DEFAULT_NUM_NODES,
    out_dim: int = DEFAULT_OUT_DIM,
    segmenter: str = DEFAULT_SEGMENTER,
    threshold: int = DEFAULT_THRESHOLD,
    center: bool = False,
    **seg_kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert one trace to fixed-N node features + boolean mask.

    Returns
    -------
    x : float32 (num_nodes, out_dim)
    mask : bool (num_nodes,)  True = real node, False = pad
    """
    t = np.asarray(trace, dtype=np.float32).ravel()
    if center and t.size:
        t = t - float(t.mean())

    bounds = _segment_bounds(
        t, segmenter=segmenter, threshold=threshold, num_uniform=num_nodes, **seg_kwargs
    )
    if not bounds:
        bounds = [(0, max(0, len(t) - 1))]

    if len(bounds) > num_nodes:
        bounds = _merge_shortest(bounds, num_nodes)

    feats = []
    for s, e in bounds:
        seg = t[s : e + 1]
        feats.append(segment_to_fixed_vector(seg, out_dim=out_dim))
    n_real = len(feats)
    x = np.zeros((num_nodes, out_dim), dtype=np.float32)
    mask = np.zeros((num_nodes,), dtype=np.bool_)
    if n_real > 0:
        x[:n_real] = np.stack(feats, axis=0)
        mask[:n_real] = True
    # If somehow empty, keep one dummy node so graph is non-empty
    if not mask.any():
        mask[0] = True
    return x, mask


def traces_to_segment_batch(
    traces: np.ndarray,
    num_nodes: int = DEFAULT_NUM_NODES,
    out_dim: int = DEFAULT_OUT_DIM,
    segmenter: str = DEFAULT_SEGMENTER,
    threshold: int = DEFAULT_THRESHOLD,
    center: bool = False,
    show_progress: bool = False,
    **seg_kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Batch-convert traces (n, L) → features (n, N, F) and masks (n, N)."""
    traces = np.asarray(traces)
    n = traces.shape[0]
    X = np.zeros((n, num_nodes, out_dim), dtype=np.float32)
    M = np.zeros((n, num_nodes), dtype=np.bool_)
    step = max(1, n // 10)
    for i in range(n):
        X[i], M[i] = trace_to_fixed_segment_features(
            traces[i],
            num_nodes=num_nodes,
            out_dim=out_dim,
            segmenter=segmenter,
            threshold=threshold,
            center=center,
            **seg_kwargs,
        )
        if show_progress and (i + 1) % step == 0:
            print(f"  segment featurize {i + 1}/{n}")
    return X, M


def build_path_edge_index(num_nodes: int, bidirectional: bool = True) -> torch.Tensor:
    """COO edge_index for a path of length num_nodes (optionally undirected)."""
    if num_nodes <= 1:
        return torch.zeros((2, 0), dtype=torch.long)
    src = list(range(num_nodes - 1))
    dst = list(range(1, num_nodes))
    if bidirectional:
        src = src + dst
        dst = dst + list(range(num_nodes - 1))
    return torch.tensor([src, dst], dtype=torch.long)


def build_complete_edge_index(num_nodes: int, include_self: bool = False) -> torch.Tensor:
    """COO edge_index for complete graph on num_nodes (optional self-loops)."""
    if num_nodes <= 0:
        return torch.zeros((2, 0), dtype=torch.long)
    src, dst = [], []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j and not include_self:
                continue
            src.append(i)
            dst.append(j)
    if not src:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor([src, dst], dtype=torch.long)


def build_edge_index(num_nodes: int, edge_mode: str = DEFAULT_EDGE_MODE) -> torch.Tensor:
    """Dispatch fixed topology by edge_mode name (binary skeleton only).

    Weighted Plan C edges are feature-dependent and live in the classifier
    (``edge_mode='weighted'``), not in a static edge_index.
    """
    mode = (edge_mode or DEFAULT_EDGE_MODE).lower()
    if mode in ("path",):
        return build_path_edge_index(num_nodes, bidirectional=True)
    if mode in ("relation", "complete", "combine", "weighted", "weighted_combine", "product_adj"):
        return build_complete_edge_index(num_nodes, include_self=False)
    raise ValueError(
        "Unknown edge_mode '%s' (use path|relation|complete|combine|weighted)" % edge_mode
    )


def weighted_adjacency_numpy(features: np.ndarray, row_normalize: bool = True) -> np.ndarray:
    """Build product-affinity adjacency for one graph.

    Parameters
    ----------
    features : (N, F) node feature matrix for one trace
    row_normalize : if True, return Ã with rows summing to 1 over |A|

    Returns
    -------
    A : (N, N) float32
        Signed affinity mean_f(x_i ⊙ x_j) if row_normalize=False;
        row-normalized abs weights if row_normalize=True (plot / message graph).
    """
    X = np.asarray(features, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("features must be (N, F)")
    n, f = X.shape
    A = (X @ X.T) / float(max(f, 1))
    np.fill_diagonal(A, 0.0)
    if not row_normalize:
        return A.astype(np.float32)
    W = np.abs(A)
    denom = W.sum(axis=1, keepdims=True)
    denom = np.maximum(denom, 1e-8)
    return (W / denom).astype(np.float32)


def segment_config_fingerprint(
    num_nodes: int,
    out_dim: int,
    segmenter: str,
    threshold: int,
    center: bool,
    n_traces: int,
    trace_len: int,
    split: str,
    extra: str = "",
) -> str:
    key = (
        f"n{num_nodes}_d{out_dim}_{segmenter}_th{threshold}_c{int(center)}"
        f"_nt{n_traces}_L{trace_len}_{split}_{extra}"
    )
    return hashlib.md5(key.encode()).hexdigest()[:12]


def cache_path_for(
    cache_dir: str,
    database_path: str,
    split: str,
    num_nodes: int,
    out_dim: int,
    segmenter: str,
    threshold: int,
    center: bool,
    n_traces: int,
    trace_len: int,
    extra: str = "",
) -> str:
    base = os.path.basename(database_path).replace(".h5", "")
    fp = segment_config_fingerprint(
        num_nodes, out_dim, segmenter, threshold, center, n_traces, trace_len, split, extra=extra
    )
    return os.path.join(cache_dir, f"segfeat_{base}_{split}_{fp}.npz")


def standardize_segment_features(
    X: np.ndarray,
    mask: np.ndarray,
    mu: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score features using real (non-pad) nodes. Returns (Xz, mu, std)."""
    X = np.asarray(X, dtype=np.float32)
    m = np.asarray(mask, dtype=np.float32)[..., None]
    if mu is None or std is None:
        denom = m.sum(axis=(0, 1)).clip(min=1.0)
        mu = (X * m).sum(axis=(0, 1)) / denom
        var = ((X - mu) ** 2 * m).sum(axis=(0, 1)) / denom
        std = np.sqrt(var).clip(min=1e-6).astype(np.float32)
        mu = mu.astype(np.float32)
    Xz = ((X - mu) / std) * m
    # keep pads at 0
    return Xz.astype(np.float32), mu, std


def load_or_build_segment_features(
    traces: np.ndarray,
    cache_path: Optional[str] = None,
    num_nodes: int = DEFAULT_NUM_NODES,
    out_dim: int = DEFAULT_OUT_DIM,
    segmenter: str = DEFAULT_SEGMENTER,
    threshold: int = DEFAULT_THRESHOLD,
    center: bool = False,
    show_progress: bool = True,
    standardize: bool = True,
    stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    **seg_kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load cached (X, mask) or build and optionally save NPZ.

    If standardize=True, z-score using real nodes. Pass stats=(mu, std) from
    profiling when transforming attack traces so scales match.
    """
    if cache_path and os.path.isfile(cache_path):
        data = np.load(cache_path)
        X, M = data["X"], data["mask"]
        print(f"Loaded segment features cache: {cache_path}  X={X.shape}")
    else:
        print(
            f"Building segment features: n={traces.shape[0]} L={traces.shape[1]} "
            f"N={num_nodes} F={out_dim} segmenter={segmenter} th={threshold}"
        )
        X, M = traces_to_segment_batch(
            traces,
            num_nodes=num_nodes,
            out_dim=out_dim,
            segmenter=segmenter,
            threshold=threshold,
            center=center,
            show_progress=show_progress,
            **seg_kwargs,
        )
        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            np.savez_compressed(cache_path, X=X, mask=M)
            print(f"Saved segment features cache: {cache_path}")

    if standardize:
        mu_std = stats
        if mu_std is not None:
            X, _, _ = standardize_segment_features(X, M, mu=mu_std[0], std=mu_std[1])
        else:
            X, mu, std = standardize_segment_features(X, M)
            # stash for callers via attribute (train script may recompute)
            load_or_build_segment_features.last_stats = (mu, std)
        print("Applied segment feature standardization (z-score on real nodes)")
    return X, M


def segment_count_stats(
    traces: Sequence[np.ndarray],
    segmenter: str = DEFAULT_SEGMENTER,
    threshold: int = DEFAULT_THRESHOLD,
    max_traces: int = 500,
    center: bool = False,
    **seg_kwargs,
) -> dict:
    """Quick segment-count summary for planning N."""
    n = min(len(traces), max_traces)
    counts = []
    for i in range(n):
        t = np.asarray(traces[i], dtype=np.float32).ravel()
        if center and t.size:
            t = t - float(t.mean())
        bounds = _segment_bounds(t, segmenter=segmenter, threshold=threshold, **seg_kwargs)
        counts.append(len(bounds) if bounds else 1)
    arr = np.asarray(counts, dtype=np.float64)
    return {
        "n_traces": n,
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "p95": float(np.percentile(arr, 95)),
        "frac_le_32": float((arr <= 32).mean()),
    }
