"""T-039 (2026-11-01, CPU, diagnostic; user question: without pi labels, do the 16 C heads output the same thing?).
On cached head posteriors (attack traces 0..N-1, the already-used half), per model:
  JS     mean pairwise Jensen-Shannon divergence between the 16 slot heads' distributions (nats; 0 = identical; ln2 max)
  dev    mean |log p_slot - log pbar| over values (pbar = slot-mean; 0 = identical)
  corr   mean pairwise correlation of the centred log-probs across the 256 values
  own    mean log(256 p_slot(C_slot)) - mean log(256 p_slot(random))  : does head s score ITS slot's value?
  other  the same at another slot's value C_t (t != s)                 : or any of the 16 values equally?
own > other means the head is slot-specific; own = other means it only knows the unordered set of 16 values.
Labels (sbox_masked = value processed in each slot) are read only for own / other.
Usage: python experiments/t039_c_head_identity.py LABEL=heads.npz ... [--n 2000]
"""
import sys
import numpy as np
sys.path.insert(0, "/home/lantian/ASCAD")
from experiments.w1_timesnet_old_head_accs import load_attack_split

n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 2000
_, lab = load_attack_split(n)
C = np.asarray(lab["sbox_masked"][:n]).astype(np.int64)                  # (n, 16): value in slot s
rng = np.random.default_rng(0)
rnd = rng.integers(0, 256, (n, 16))
other = C[:, (np.arange(16) + 1 + rng.integers(0, 15, 16)) % 16]         # another slot's value, per slot
iu = np.triu_indices(16, 1)
print(f"n = {n} attack traces (0..{n - 1})")
print(f"{'model':<20}{'JS':>9}{'dev':>9}{'corr':>8}{'own':>9}{'other':>9}{'own-other':>11}")
for spec in [a for a in sys.argv[1:] if "=" in a]:
    label, path = spec.split("=", 1)
    d = np.load(path)
    P = np.stack([d[f"sbox_{i}_output"][:n] for i in range(16)], axis=1).astype(np.float64)   # (n, 16, 256)
    P = np.clip(P, 1e-12, None)
    P /= P.sum(-1, keepdims=True)
    L = np.log(P)
    pbar = P.mean(1)
    dev = np.abs(L - np.log(pbar)[:, None, :]).mean()
    js = []
    for i, j in zip(*iu):
        m = 0.5 * (P[:, i] + P[:, j])
        js.append(0.5 * ((P[:, i] * (L[:, i] - np.log(m))).sum(-1) + (P[:, j] * (L[:, j] - np.log(m))).sum(-1)).mean())
    Lc = L - L.mean(-1, keepdims=True)
    Ln = Lc / np.maximum(np.linalg.norm(Lc, axis=-1, keepdims=True), 1e-12)
    corr = np.einsum("nsv,ntv->nst", Ln, Ln)[:, iu[0], iu[1]].mean()
    lp = L + np.log(256)
    at = lambda idx: np.take_along_axis(lp, idx[:, :, None], 2)[..., 0].mean()
    own, oth, base = at(C), at(other), at(rnd)
    print(f"{label:<20}{np.mean(js):9.4f}{dev:9.4f}{corr:8.3f}{own - base:9.4f}{oth - base:9.4f}{own - oth:11.4f}")
