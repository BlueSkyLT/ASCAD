"""SCA signal-to-noise ratio (SNR) helpers for PoI / segment construction.

SNR here is the standard side-channel definition per time sample:

    SNR[t] = Var_c(E[X[:,t] | z=c]) / E_c(Var[X[:,t] | z=c])

High SNR marks good points of interest (PoIs). Peak indices are not stored
in ASCAD HDF5 and must be recomputed on profiling data.

Does not read or write dataset files itself — callers pass arrays.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.signal import find_peaks
except ImportError:  # pragma: no cover
    find_peaks = None


def snr_vector(
    traces: np.ndarray,
    labels: np.ndarray,
    eps: float = 1e-10,
) -> np.ndarray:
    """Per-sample SNR over time.

    Parameters
    ----------
    traces : (N, T) float-like
    labels : (N,) intermediate values (e.g. S-box class 0..255)

    Returns
    -------
    snr : (T,) float64
    """
    traces = np.asarray(traces)
    labels = np.asarray(labels).ravel()
    if traces.ndim != 2:
        raise ValueError("traces must be 2-D (N, T), got shape %s" % (traces.shape,))
    if labels.shape[0] != traces.shape[0]:
        raise ValueError("labels length %d != n_traces %d" % (labels.shape[0], traces.shape[0]))

    n, t_len = traces.shape
    out = np.zeros(t_len, dtype=np.float64)
    classes = np.unique(labels)
    # Pre-group indices for speed
    class_idx = {int(c): np.where(labels == c)[0] for c in classes}
    for t in range(t_len):
        col = traces[:, t].astype(np.float64, copy=False)
        means = []
        vars_ = []
        for c, idx in class_idx.items():
            if idx.size == 0:
                continue
            vals = col[idx]
            means.append(vals.mean())
            vars_.append(vals.var())
        if not means:
            out[t] = 0.0
            continue
        out[t] = float(np.var(means) / (np.mean(vars_) + eps))
    return out


def find_snr_peaks(
    snr: np.ndarray,
    min_distance: int = 10,
    height: Optional[float] = None,
    prominence: Optional[float] = None,
    max_peaks: Optional[int] = None,
) -> np.ndarray:
    """Return peak sample indices of an SNR curve (sorted ascending in time).

    If scipy is unavailable, falls back to top-k local maxima by simple scan.
    """
    snr = np.asarray(snr, dtype=np.float64).ravel()
    if snr.size == 0:
        return np.zeros(0, dtype=np.int64)

    if height is None:
        # Relative to max: keep peaks that are at least 10% of max (or absolute tiny floor)
        height = max(1e-12, 0.1 * float(np.max(snr)))
    if prominence is None:
        prominence = max(1e-12, 0.05 * float(np.max(snr)))

    if find_peaks is not None:
        peaks, props = find_peaks(
            snr,
            distance=max(1, int(min_distance)),
            height=height,
            prominence=prominence,
        )
        peaks = np.asarray(peaks, dtype=np.int64)
        if max_peaks is not None and peaks.size > max_peaks:
            # Keep highest SNR peaks
            heights = snr[peaks]
            order = np.argsort(-heights)[: int(max_peaks)]
            peaks = np.sort(peaks[order])
        return peaks

    # Fallback: simple local max + distance greedy
    cand = []
    for i in range(1, snr.size - 1):
        if snr[i] >= snr[i - 1] and snr[i] >= snr[i + 1] and snr[i] >= height:
            cand.append(i)
    cand = sorted(cand, key=lambda i: -snr[i])
    selected: List[int] = []
    for i in cand:
        if all(abs(i - j) >= min_distance for j in selected):
            selected.append(i)
        if max_peaks is not None and len(selected) >= max_peaks:
            break
    return np.array(sorted(selected), dtype=np.int64)


def windows_from_peaks(
    peaks: Sequence[int],
    half_width: int,
    trace_len: int,
) -> List[Tuple[int, int]]:
    """Inclusive (start, end) windows centered on each peak."""
    w = max(0, int(half_width))
    L = int(trace_len)
    bounds = []
    for p in peaks:
        p = int(p)
        s = max(0, p - w)
        e = min(L - 1, p + w)
        if e < s:
            e = s
        bounds.append((s, e))
    return bounds


def peak_stability(
    traces: np.ndarray,
    labels: np.ndarray,
    n_splits: int = 2,
    seed: int = 0,
    min_distance: int = 10,
    max_peaks: int = 16,
    match_tol: int = 5,
) -> Dict[str, float]:
    """Compare SNR peaks on disjoint profiling halves.

    Returns Jaccard-like agreement: fraction of peaks in split A that have a
    match in split B within ``match_tol`` samples (and vice versa, averaged).
    """
    n = traces.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    fold = n // n_splits
    peak_sets = []
    for s in range(n_splits):
        idx = perm[s * fold : (s + 1) * fold] if s < n_splits - 1 else perm[s * fold :]
        snr = snr_vector(traces[idx], labels[idx])
        peaks = find_snr_peaks(snr, min_distance=min_distance, max_peaks=max_peaks)
        peak_sets.append(set(int(p) for p in peaks))

    def match_frac(a: set, b: set) -> float:
        if not a:
            return 1.0 if not b else 0.0
        hits = 0
        for p in a:
            if any(abs(p - q) <= match_tol for q in b):
                hits += 1
        return hits / len(a)

    scores = []
    for i in range(len(peak_sets)):
        for j in range(i + 1, len(peak_sets)):
            scores.append(0.5 * (match_frac(peak_sets[i], peak_sets[j]) + match_frac(peak_sets[j], peak_sets[i])))

    all_peaks = [sorted(s) for s in peak_sets]
    return {
        "n_splits": float(n_splits),
        "mean_match_frac": float(np.mean(scores)) if scores else 0.0,
        "n_peaks_split0": float(len(all_peaks[0])) if all_peaks else 0.0,
        "n_peaks_split1": float(len(all_peaks[1])) if len(all_peaks) > 1 else 0.0,
        "peaks_split0": all_peaks[0] if all_peaks else [],
        "peaks_split1": all_peaks[1] if len(all_peaks) > 1 else [],
    }


def discover_snr_windows(
    traces: np.ndarray,
    labels: np.ndarray,
    half_width: int = 12,
    min_distance: int = 10,
    max_peaks: int = 16,
    height: Optional[float] = None,
    prominence: Optional[float] = None,
) -> Dict:
    """Profiling-only: SNR → peaks → fixed windows for all traces.

    Returns dict with snr, peaks, windows, and summary stats.
    """
    snr = snr_vector(traces, labels)
    peaks = find_snr_peaks(
        snr,
        min_distance=min_distance,
        height=height,
        prominence=prominence,
        max_peaks=max_peaks,
    )
    windows = windows_from_peaks(peaks, half_width=half_width, trace_len=traces.shape[1])
    return {
        "snr": snr,
        "peaks": peaks.astype(np.int64),
        "windows": windows,
        "half_width": int(half_width),
        "min_distance": int(min_distance),
        "max_peaks": int(max_peaks),
        "n_peaks": int(peaks.size),
        "snr_max": float(snr.max()) if snr.size else 0.0,
        "snr_mean": float(snr.mean()) if snr.size else 0.0,
    }
