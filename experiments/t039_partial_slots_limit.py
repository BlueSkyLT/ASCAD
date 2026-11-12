"""T-039: exact alpha, beta and C at only m random slots per trace -> traces until every key byte ranks 0.

Exact readings (as t039_perfect_reading_limit.py). pi known: a slot fixes its byte exactly (coupon collector).
pi unknown: score(b, h) = number of observed slot values equal to the value h predicts for byte b
(monotone in the likelihood of the mixture over which byte the slot holds); success = true h strictly above
every wrong h on all 16 bytes (worst-tie)."""
import sys, numpy as np
sys.path.insert(0, '.')
from ASCAD_generate import AES_Sbox
from MultiResNet.decoder_loss import GF_MUL
from experiments.w1_timesnet_old_head_accs import load_attack_split
N, NMAX, NORD = 5000, 3000, 40
lab = load_attack_split(N)[1]
sb = np.asarray(AES_Sbox, dtype=np.int64)
x = lab["plaintext"][:N].astype(np.int64) ^ lab["key"][:N].astype(np.int64)
a, be, C = lab["alpha"][:N], lab["beta"][:N], lab["sbox_masked"][:N]
H = np.arange(256)
V = (GF_MUL[a[:, None, None], sb[x[:, :, None] ^ H[None, None, :]]] ^ be[:, None, None]).astype(np.uint8)  # (n,16 byte,256)
rng = np.random.default_rng(1)
MS = (1, 2, 4, 8, 16)

def coupon(m, reps=20000):
    out = np.empty(reps)
    for r in range(reps):
        seen = np.zeros(16, bool); t = 0
        while not seen.all():
            t += 1; seen[rng.choice(16, m, replace=False)] = True
        out[r] = t
    return out

print("pi KNOWN (exact: each observed slot fixes its byte; coupon collector over bytes):")
for m in MS:
    c = coupon(m); print(f"  m={m:2d} slots/trace: mean {c.mean():6.2f}  median {np.median(c):4.0f}  95th pct {np.percentile(c,95):4.0f}")

res = {m: [] for m in MS}
for o in range(NORD):
    idx = rng.permutation(N)[:NMAX]
    Vo, Co = V[idx], C[idx].astype(np.uint8)
    M = (Vo[:, None, :, :] == Co[:, :, None, None])               # (t, slot, byte, h)
    for m in MS:
        sel = np.zeros((NMAX, 16), bool)
        for t in range(NMAX):
            sel[t, rng.choice(16, m, replace=False)] = True
        cnt = np.cumsum(np.einsum("ts,tsbh->tbh", sel.astype(np.float32), M.astype(np.float32)), 0)  # (t,16,256)
        wrong = cnt[:, :, 1:].max(2)
        ok = (cnt[:, :, 0] > wrong).all(1)
        # first t after which success is permanent would be stricter; report first success
        res[m].append(int(np.argmax(ok)) + 1 if ok.any() else np.inf)
print(f"pi UNKNOWN (count of matches, {NORD} random orderings, cap {NMAX}):")
for m in MS:
    v = np.array(res[m]); fin = np.isfinite(v)
    print(f"  m={m:2d} slots/trace: mean {v[fin].mean():7.1f}  median {np.median(v):6.0f}  95th pct {np.percentile(v,95):6.0f}  (never within {NMAX}: {int((~fin).sum())}/{NORD})")
