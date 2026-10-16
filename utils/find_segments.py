# How many "segments" are in the trace? One segment goes from a local minimum to the next local minimum. 

import numpy as np
import matplotlib.pyplot as plt

# For improved segmentation (v2)
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

def find_segments(trace, threshold=10):
    """
    Identify segments in the trace based on local minima (define as derivative < 0 and value < 0).

    Parameters:
        trace (np.ndarray): The input trace data.
        threshold (float): Minimum time difference between local minima to consider them as separate segments.
    """
    segments = []
    start_idx = 0
    for i in range(1, len(trace) - 1):
        if trace[i] <= trace[i - 1] and trace[i] <= trace[i + 1]:  # Local minimum
            if i - start_idx >= threshold:
                # ensure value is below 0
                if trace[i] <= 0:
                    segments.append((start_idx, i))
                    start_idx = i
    # Add the last segment if it wasn't added
    if start_idx < len(trace) - 1:
        segments.append((start_idx, len(trace) - 1))
    return segments


# how to use the function:
# Find segments in the selected trace
"""
segments = find_segments(trace, threshold=10)
print(f'\nIdentified {len(segments)} segments in trace {trace_idx}.')   

# plot the segments on the trace
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(samples, trace, linewidth=0.8, color='#2E86AB')
for start, end in segments:
    ax.axvspan(start, end, color='orange', alpha=0.3)
ax.set_title(f'Trace #{trace_idx} with Identified Segments', fontsize=12, fontweight='bold', pad=15)
ax.set_xlabel('Sample Index', fontsize=11)
ax.set_ylabel('Power Consumption (a.u.)', fontsize=11)
plt.tight_layout()
plt.show()
"""


def segment_to_fixed_vector(segment: np.ndarray, out_dim: int = 12) -> np.ndarray:
    """Simple filter: resample a variable-length segment to a fixed-length feature vector.

    Uses linear interpolation on the normalized time axis [0,1].
    - Preserves approximate shape + amplitude of the waveform between minima.
    - Independent of original length (as long as >=1 sample).
    - No extra dependencies (pure numpy).
    - Small fixed dimension suitable for node features in a GNN.

    Returns: float32 array of shape (out_dim,).
    """
    seg = np.asarray(segment, dtype=np.float32).ravel()
    n = seg.shape[0]
    if n == 0:
        return np.zeros(out_dim, dtype=np.float32)
    if n == 1:
        return np.full(out_dim, seg[0], dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, n)
    x_new = np.linspace(0.0, 1.0, out_dim)
    return np.interp(x_new, x_old, seg).astype(np.float32)


def trace_to_segment_features(trace: np.ndarray, threshold: int = 10,
                              out_dim: int = 12) -> np.ndarray:
    """Convenience: run find_segments then featurize every segment.
    Returns: (num_segments, out_dim) array, or (0, out_dim) if none.
    """
    segs = find_segments(trace, threshold=threshold)
    if not segs:
        return np.empty((0, out_dim), dtype=np.float32)
    feats = [segment_to_fixed_vector(trace[s:e+1], out_dim) for s, e in segs]
    return np.stack(feats, axis=0)


# -------------------------------------------------------------------------
# v2: Better segment detection using wavelet-inspired ideas
# -------------------------------------------------------------------------

def find_segments_v2(trace, threshold=10, prominence=None, smooth_scale=None):
    """
    Improved segment finder (v2).

    Wavelet transform thinking:
    - We apply a smoothing filter (uniform_filter1d) that acts similarly
      to a low-pass / scaling function in a discrete wavelet transform.
      This looks at the signal at a coarser "scale" (the approximation
      coefficients), suppressing high-frequency noise while preserving
      the main "dips" or events.
    - We then locate significant local minima in this smoothed view using
      scipy.signal.find_peaks(..., prominence=...).
    - Prominence replaces the brittle "value < 0" test: it measures how
      much a minimum stands out relative to its local surroundings.
    - The `distance` parameter in find_peaks directly enforces a minimum
      separation between detected minima (hence minimum segment length).

    Result: far fewer spurious short segments, segments respect the
    threshold more strictly, and detection is more robust across
    different trace offsets/scales (important for desync and v2 data).

    Parameters:
        trace: 1D array
        threshold: minimum distance between minima (min segment "width")
        prominence: how prominent a dip must be. If None, auto ~0.25 * std
        smooth_scale: size of smoothing window. If None, ~ threshold/2
    """
    trace = np.asarray(trace, dtype=np.float32).ravel()
    n = len(trace)
    if n <= 1:
        return [(0, n - 1)]

    if smooth_scale is None:
        smooth_scale = max(3, threshold // 2)

    # Wavelet approximation step: smooth (low-pass / scaling)
    smoothed = uniform_filter1d(trace, size=int(smooth_scale), mode="nearest")

    # Detect local minima in the smoothed (coarse-scale) signal
    neg = -smoothed
    if prominence is None:
        prominence = max(1e-6, 0.25 * np.std(trace))

    # find_peaks on negated smoothed = local minima, with distance constraint
    minima, _ = find_peaks(neg, distance=threshold, prominence=prominence)

    segments = []
    start = 0
    for m in minima:
        seg_len = m - start
        if seg_len >= threshold:
            segments.append((start, m))
            start = m

    # Handle the tail
    if n - 1 > start:
        tail_len = n - 1 - start
        if tail_len >= threshold or len(segments) == 0:
            segments.append((start, n - 1))
        elif segments:
            # Merge short tail into the previous segment
            ps, _ = segments[-1]
            segments[-1] = (ps, n - 1)
        else:
            segments.append((start, n - 1))

    if not segments:
        segments = [(0, n - 1)]

    return segments


def trace_to_segment_features_v2(trace: np.ndarray, threshold: int = 10,
                                 out_dim: int = 12, **kwargs) -> np.ndarray:
    """Convenience wrapper using the v2 segmenter."""
    segs = find_segments_v2(trace, threshold=threshold, **kwargs)
    if not segs:
        return np.empty((0, out_dim), dtype=np.float32)
    feats = [segment_to_fixed_vector(trace[s:e+1], out_dim) for s, e in segs]
    return np.stack(feats, axis=0)
