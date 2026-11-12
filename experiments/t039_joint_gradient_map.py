"""T-039 (2026-10-28, CPU). Per-byte-marginal decoder loss vs the JOINT-over-16-bytes likelihood: which gives the alpha / beta
heads a gradient, and does it point at the truth?

Synthetic traces with the verified identity C_slot = alpha (x) S[pi(slot)] xor beta, uniform routing (R = 1/16, the 3headResNet
setting). Head states: 'I' = logits 6*onehot(truth), 'W' = 1.5*onehot, 'U' = pure noise 0.01*randn (a near-uniform start, not an
exact zero), 'Cset' = every C head the SAME weak distribution on the 16 true values (what RCpi / 3head C heads actually encode:
the unordered set, no slot), 'Cslot' = slot-specific 6*onehot. Reports per method: NLL/16, mean |dNLL/dlogits| for alpha, beta,
C, and cos(-grad, onehot(truth) - softmax) for alpha and beta (does a gradient step move the head toward the truth?).
"""
import sys
import numpy as np
import torch

sys.path.insert(0, "/home/lantian/ASCAD")
from MultiResNet.decoder_loss import GF_MUL, decoder_joint_nll, decoder_nll

torch.manual_seed(0)
rng = np.random.default_rng(0)
n = 64
gf = torch.tensor(GF_MUL, dtype=torch.long)
S = rng.integers(0, 256, (n, 16))
a = rng.integers(1, 256, n)
b = rng.integers(0, 256, n)
pi = np.stack([rng.permutation(16) for _ in range(n)])
Cv = np.stack([GF_MUL[a[i], S[i][pi[i]]] ^ b[i] for i in range(n)])      # (n, slot)
R = torch.full((n, 16, 16), 1 / 16, dtype=torch.float64)
St = torch.tensor(S)


def head(truth, state):
    z = 0.01 * torch.randn(truth.shape + (256,), dtype=torch.float64)
    if state in ("I", "W"):
        z.scatter_(-1, torch.tensor(truth)[..., None], {"I": 6.0, "W": 1.5}[state])
    return z.requires_grad_(True)


def c_head(state):
    if state == "Cslot":
        return head(Cv, "I")
    z = 0.01 * torch.randn(n, 16, 256, dtype=torch.float64)
    if state == "Cset":
        w = torch.zeros(n, 256, dtype=torch.float64)
        for i in range(n):
            w[i, torch.tensor(Cv[i])] = 1.5
        z = z + w[:, None, :]
    return z.requires_grad_(True)


def toward(z, truth):
    g = -z.grad
    t = torch.zeros_like(z)
    t.scatter_(-1, torch.tensor(truth)[..., None], 1.0)
    t = t - torch.softmax(z.detach(), -1)
    return torch.nn.functional.cosine_similarity(g, t, dim=-1).mean().item()


rows = [("I", "U", "Cset"), ("I", "W", "Cset"), ("I", "I", "Cset"), ("U", "I", "Cset"), ("U", "U", "Cset"),
        ("I", "U", "U"), ("U", "U", "U"), ("I", "I", "U"), ("I", "I", "Cslot"), ("U", "U", "Cslot")]
print(f"{'alpha beta C':<16}{'method':<10}{'NLL/16':>8}{'|g_a|':>10}{'|g_b|':>10}{'|g_C|':>10}{'cos_a':>8}{'cos_b':>8}")
for sa, sb, sc in rows:
    for name in ("per-byte", "joint"):
        torch.manual_seed(1)                                   # identical heads for both methods
        za, zb, zc = head(a, sa), head(b, sb), c_head(sc)
        pa, pb, pc = (torch.softmax(z, -1) for z in (za, zb, zc))
        nll = (decoder_nll(pa, pb, pc, R, St, gf) if name == "per-byte" else decoder_joint_nll(pa, pb, pc, St, gf)).mean()
        nll.backward()
        g = lambda z: z.grad.abs().mean().item()
        print(f"{sa + '  ' + sb + '  ' + sc:<16}{name:<10}{nll.item() / 16:8.3f}{g(za):10.2e}{g(zb):10.2e}{g(zc):10.2e}"
              f"{toward(za, a):8.3f}{toward(zb, b):8.3f}")
