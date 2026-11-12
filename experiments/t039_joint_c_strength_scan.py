"""T-039 (2026-10-28, CPU). How informative must the C heads be before the JOINT decoder loss gives alpha / beta a gradient that
points at the truth? Synthetic traces (same generator as t039_joint_gradient_map.py), alpha and beta uniform-noise, every C head =
the same distribution putting logit s on the 16 true C values ('Cset'). Reports, per scale s, the C information (mean log(256 P) at
the 16 true values minus at random values; the real RCpi/3head heads measure 0.03, ledger row 'RCpi C heads') and, for per-byte and
joint losses, cos(-grad, onehot(truth) - p) for alpha and beta."""
import sys
import numpy as np
import torch
sys.path.insert(0, "/home/lantian/ASCAD")
from MultiResNet.decoder_loss import GF_MUL, decoder_joint_nll, decoder_nll
NOISE = float(sys.argv[1]) if len(sys.argv) > 1 else 0.01
rng = np.random.default_rng(0); n = 64
gf = torch.tensor(GF_MUL, dtype=torch.long)
S = rng.integers(0, 256, (n, 16)); a = rng.integers(1, 256, n); b = rng.integers(0, 256, n)
pi = np.stack([rng.permutation(16) for _ in range(n)])
Cv = np.stack([GF_MUL[a[i], S[i][pi[i]]] ^ b[i] for i in range(n)])
R = torch.full((n, 16, 16), 1 / 16, dtype=torch.float64); St = torch.tensor(S)
def toward(z, truth):
    t = torch.zeros_like(z); t.scatter_(-1, torch.tensor(truth)[..., None], 1.0); t = t - torch.softmax(z.detach(), -1)
    return torch.nn.functional.cosine_similarity(-z.grad, t, dim=-1).mean().item()
print(f"C-logit noise sd = {NOISE}")
print(f"{'scale s':>8}{'C info':>8} | {'per-byte cos_a':>15}{'cos_b':>8} | {'joint cos_a':>12}{'cos_b':>8}{'|g_a|':>10}")
for s in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 1.5):
    res = {}
    for name in ("per-byte", "joint"):
        torch.manual_seed(1)
        za = (0.01 * torch.randn(n, 256, dtype=torch.float64)).requires_grad_(True)
        zb = (0.01 * torch.randn(n, 256, dtype=torch.float64)).requires_grad_(True)
        w = torch.zeros(n, 256, dtype=torch.float64)
        for i in range(n): w[i, torch.tensor(Cv[i])] = s
        zc = (NOISE * torch.randn(n, 16, 256, dtype=torch.float64) + w[:, None, :]).requires_grad_(True)
        pa, pb, pc = (torch.softmax(z, -1) for z in (za, zb, zc))
        nll = (decoder_nll(pa, pb, pc, R, St, gf) if name == "per-byte" else decoder_joint_nll(pa, pb, pc, St, gf)).mean()
        nll.backward()
        res[name] = (toward(za, a), toward(zb, b), za.grad.abs().mean().item(), pc.detach())
    pc = res["joint"][3][:, 0]                                   # (n, 256)
    lp = torch.log(256 * pc)
    true_lp = np.mean([lp[i, torch.tensor(Cv[i])].mean().item() for i in range(n)])
    rand_lp = np.mean([lp[i, torch.randint(0, 256, (16,))].mean().item() for i in range(n)])
    print(f"{s:8.2f}{true_lp - rand_lp:8.3f} | {res['per-byte'][0]:15.3f}{res['per-byte'][1]:8.3f} | "
          f"{res['joint'][0]:12.3f}{res['joint'][1]:8.3f}{res['joint'][2]:10.2e}")
