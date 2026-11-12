"""T-039: perfect-reading limit — traces needed to isolate each key byte with vs without knowing pi (CPU, 5000 attack traces, 500 orderings)."""
import sys, numpy as np
sys.path.insert(0, '.')
from ASCAD_generate import AES_Sbox
from MultiResNet.decoder_loss import GF_MUL
from experiments.w1_timesnet_old_head_accs import load_attack_split
N = 5000
lab = load_attack_split(N)[1]
sb = np.asarray(AES_Sbox, dtype=np.int64); gm = GF_MUL
x = (lab["plaintext"][:N].astype(np.int64) ^ lab["key"][:N].astype(np.int64))      # (n,16) sbox inputs
S = sb[x]; a = lab["alpha"][:N]; be = lab["beta"][:N]
C = lab["sbox_masked"][:N]; perm = lab["perm_index"][:N]
# label identity check: C_slot = alpha (x) S[perm[slot]] xor beta
pred = gm[a[:, None], np.take_along_axis(S, perm, 1)] ^ be[:, None]
print("identity C_slot = a(x)S[pi(slot)]^b holds on", (pred == C).all(1).mean() * 100, "% of traces")
print("alpha == 0 on", (a == 0).mean() * 100, "% of traces (no information in those)")
H = np.arange(256)
# V[n,b,h] = value predicted by hypothesis h for byte b
V = gm[a[:, None, None], sb[x[:, :, None] ^ H[None, None, :]]] ^ be[:, None, None]
# with pi: compare to the C of the slot that holds byte b
slot_of = np.argsort(perm, 1)                      # slot_of[n,b] = slot s with perm[s]=b
C_b = np.take_along_axis(C, slot_of, 1)            # (n,16)
pass_pi = (V == C_b[:, :, None])                   # (n,16,256)
# without pi: value must appear among the 16 C values of the trace
pass_nopi = (V[:, :, :, None] == C[:, None, None, :]).any(-1)
print("false-pass rate (h != 0): with pi %.4f (=1/256 %.4f), without pi %.4f (~15.5/256 %.4f)" %
      (pass_pi[:, :, 1:].mean(), 1/256, pass_nopi[:, :, 1:].mean(), 15.5/256))
print("true key passes: with pi", pass_pi[:, :, 0].mean(), "without pi", pass_nopi[:, :, 0].mean())
rng = np.random.default_rng(0)
def traces_to_unique(P, order):
    """P (n,16,256) bool; smallest n with only h=0 surviving, per byte (inf if never)."""
    alive = np.ones((16, 256), bool); out = np.full(16, np.inf)
    for t, i in enumerate(order, 1):
        alive &= P[i]
        done = (alive.sum(1) == 1) & np.isinf(out)
        out[done] = t
        if (~np.isinf(out)).all(): break
    return out
res = {"with pi": [], "without pi": []}
for _ in range(500):
    order = rng.permutation(N)
    res["with pi"].append(traces_to_unique(pass_pi, order)); res["without pi"].append(traces_to_unique(pass_nopi, order))
for k, v in res.items():
    v = np.array(v)
    full = v.max(1)       # traces until ALL 16 bytes unique (worst byte)
    print(f"{k:11s}: per-byte traces to a unique survivor mean {v.mean():.2f} median {np.median(v):.0f} | all 16 bytes: mean {full.mean():.2f}, median {np.median(full):.0f}, 95th pct {np.percentile(full,95):.0f}, max {full.max():.0f} (500 random orderings)")
