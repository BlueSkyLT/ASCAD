"""T-039 (2026-10-19, CPU, seconds). Which decoder-loss gradients exist at a given 'informative/uniform' state of alpha, beta, C?

Synthetic traces with the verified identity C_slot = alpha (x) S[pi(slot)] xor beta, pi fixed to the truth (isolates the
three shares). 'I' = logits 6*onehot(truth) (+ small noise), 'W' = weak 1.5*onehot, 'U' = exactly uniform. Reports the mean |dNLL/dlogits|
per head group (float64).
"""
import itertools
import sys
import numpy as np
import torch
sys.path.insert(0, "/home/lantian/ASCAD")
from MultiResNet.decoder_loss import GF_MUL, decoder_nll

torch.manual_seed(0)
rng = np.random.default_rng(0)
n = 256
gf = torch.tensor(GF_MUL, dtype=torch.long)
S = rng.integers(0, 256, (n, 16))
a = rng.integers(1, 256, n); b = rng.integers(0, 256, n)
pi = np.stack([rng.permutation(16) for _ in range(n)])
C = np.stack([GF_MUL[a[i], S[i][pi[i]]] ^ b[i] for i in range(n)])        # (n, slot)
R = torch.zeros(n, 16, 16, dtype=torch.float64)                            # R[slot, byte]
for i in range(n):
    R[i, torch.arange(16), torch.tensor(pi[i])] = 1.0


def logits(truth, state):
    shape = truth.shape + (256,)
    z = torch.zeros(shape, dtype=torch.float64)
    if state != "U":
        z.scatter_(-1, torch.tensor(truth)[..., None], {"I": 6.0, "W": 1.5}[state])
        z = z + 0.01 * torch.randn(shape, dtype=torch.float64)
    return z.requires_grad_(True)


print(f"{'alpha beta C':<14}{'NLL/16':>9}{'|g_alpha|':>12}{'|g_beta|':>12}{'|g_C|':>12}")
for sa, sb, sc in itertools.product("IU", repeat=3):
    za, zb, zc = logits(a, sa), logits(b, sb), logits(C, sc)
    nll = decoder_nll(torch.softmax(za, -1), torch.softmax(zb, -1), torch.softmax(zc, -1), R,
                      torch.tensor(S), gf).mean()
    nll.backward()
    g = lambda z: z.grad.abs().mean().item()
    print(f"{sa+'  '+sb+'  '+sc:<14}{nll.item()/16:9.3f}{g(za):12.2e}{g(zb):12.2e}{g(zc):12.2e}")
print("-- C weak (W), alpha sharp, beta sharp / uniform")
for sa, sb, sc in [("I", "I", "W"), ("I", "U", "W"), ("I", "W", "U"), ("I", "W", "W")]:
    za, zb, zc = logits(a, sa), logits(b, sb), logits(C, sc)
    nll = decoder_nll(torch.softmax(za, -1), torch.softmax(zb, -1), torch.softmax(zc, -1), R,
                      torch.tensor(S), gf).mean()
    nll.backward()
    g = lambda z: z.grad.abs().mean().item()
    print(f"{sa+'  '+sb+'  '+sc:<14}{nll.item()/16:9.3f}{g(za):12.2e}{g(zb):12.2e}{g(zc):12.2e}")
