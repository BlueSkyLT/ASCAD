"""T-039 (2026-10-28, CPU, diagnostic). Write a copy of a cached-heads npz with every C head's per-trace log-probabilities
projected onto [1, bit0..bit7] (bit-linear) or the 9 Hamming-weight indicators, then re-normalised. alpha, beta and pi outputs
are copied unchanged. Scoring the copy with `experiments.w6_nochead_eval --npz` estimates what a bit-linear / HW C head would
give on the attack if it learned the projected part of the trained head and nothing else (an offline upper/lower bound, not a
trained constrained head). No labels are read. `t039_controls.py --project-c` applies the same projection to its
constant-trace null.
Usage: python experiments/t039_c_projection_npz.py IN.npz OUT.npz {bit,hw}
"""
import sys
import numpy as np


def projector(kind):
    v = np.arange(256)
    bits = ((v[:, None] >> np.arange(8)) & 1).astype(np.float64)
    X = np.concatenate([np.ones((256, 1)), bits], 1) if kind == "bit" else np.eye(9)[bits.sum(1).astype(int)]
    return X @ np.linalg.pinv(X)                                            # (256, 256)


def _logp(p):
    lp = np.log(np.clip(np.asarray(p, dtype=np.float64), 1e-12, None))
    return lp - lp.mean(1, keepdims=True)


def matched_temperature(heads, kind="bit"):
    """Global factor t such that t * (centred C log-probs) has the same mean spread over the 256 values as their `kind`
    projection: the structure-free control for the projection (it flattens by the same amount, keeps every direction)."""
    P = projector(kind)
    lps = [_logp(p) for k, p in heads.items() if k.startswith("sbox_")]
    return float(np.mean([(lp @ P).std(1).mean() for lp in lps]) / np.mean([lp.std(1).mean() for lp in lps]))


def project_c(heads, kind):
    """Return a copy of a {name: (n, 256 or 16)} dict with every sbox_* posterior projected and re-normalised.
    kind "bit" / "hw": projection; "temp:<t>": log-probs multiplied by t (no projection)."""
    if kind.startswith("temp:"):
        t = float(kind.split(":", 1)[1])
        P = t * np.eye(256)
    else:
        P = projector(kind)
    out = dict(heads)
    for k, p in heads.items():
        if k.startswith("sbox_"):
            lp = _logp(p) @ P
            lp -= lp.max(1, keepdims=True)
            q = np.exp(lp)
            out[k] = (q / q.sum(1, keepdims=True)).astype(np.asarray(p).dtype)
    return out


if __name__ == "__main__":
    src, dst, kind = sys.argv[1], sys.argv[2], sys.argv[3]
    d = np.load(src)
    heads = {k: d[k] for k in d.files}
    if kind.startswith("tempmatch:"):             # tempmatch:bit -> temp:<t matched to the bit projection's spread>
        kind = f"temp:{matched_temperature(heads, kind.split(':', 1)[1]):.6f}"
    np.savez_compressed(dst, **project_c(heads, kind))
    print(f"{src} -> {dst} ({kind})")
