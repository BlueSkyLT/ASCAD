"""T-039 (2026-10-28, CPU, diagnostic). Is the signal in the learned C heads bit-linear or Hamming-weight shaped, and how much noise would
a structured C head remove? Take the slot-mean C posterior (what the uniform-routing decoder reads) on the first N attack traces, lp =
log(256 P). Signal = mean lp at the 16 true C values minus mean lp at 16 random values (the 'C information', real heads ~0.03).
Project lp (per trace, over the 256 values) onto (a) the 9 Hamming-weight indicators, (b) [1, bit0..bit7]; report the signal and the
spread (sd over the 256 values) of the original and projected log-probs. Attack labels are used only to locate the true C values.
Usage: python experiments/t039_c_structure_projection.py LABEL=heads.npz ... [--n 2000]"""
import sys
import numpy as np
sys.path.insert(0, "/home/lantian/ASCAD")
from experiments.w1_timesnet_old_head_accs import load_attack_split

n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 2000
_, lab = load_attack_split(5000)
C_true = np.asarray(lab["sbox_masked"][:n]).astype(np.int64)             # (n, 16) the 16 masked values (slot order irrelevant)
v = np.arange(256)
bits = ((v[:, None] >> np.arange(8)) & 1).astype(np.float64)              # (256, 8)
hw = bits.sum(1).astype(int)
H = np.eye(9)[hw]                                                          # (256, 9) Hamming-weight indicators
B = np.concatenate([np.ones((256, 1)), bits], axis=1)                      # (256, 9) constant + bits
proj = lambda X: X @ np.linalg.pinv(X)                                     # (256, 256) projector on the column space
PH, PB = proj(H), proj(B)
rng = np.random.default_rng(0)
print(f"n = {n}; C information = mean lp at the 16 true values minus at 16 random values (nat)")
print(f"{'model':<18}{'proj':<10}{'info':>8}{'sd over v':>11}{'info/sd':>9}")
for spec in [a for a in sys.argv[1:] if "=" in a]:
    label, path = spec.split("=", 1)
    d = np.load(path)
    P = np.mean([d[f"sbox_{i}_output"][:n] for i in range(16)], axis=0)    # (n, 256) slot-mean posterior
    lp = np.log(256 * np.clip(P, 1e-12, None))
    lp = lp - lp.mean(1, keepdims=True)
    rnd = rng.integers(0, 256, (n, 16))
    for name, L in (("original", lp), ("HW", lp @ PH), ("bit-linear", lp @ PB)):
        t = np.take_along_axis(L, C_true, 1).mean()
        r = np.take_along_axis(L, rnd, 1).mean()
        sd = L.std(1).mean()
        print(f"{label:<18}{name:<10}{t - r:8.4f}{sd:11.4f}{(t - r) / sd:9.3f}")
