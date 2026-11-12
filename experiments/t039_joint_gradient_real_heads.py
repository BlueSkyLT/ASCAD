"""T-039 (2026-10-28, CPU, diagnostic only). Per-byte-marginal vs JOINT decoder loss, on the REAL cached head posteriors of a
trained model (uniform routing), first N attack traces. Attack-split labels (alpha, beta, S) are read for the gradient direction
only; nothing is trained. Three states per model: as trained; beta replaced by uniform (the state a model is in just after its
beta CE is switched off); alpha and beta both uniform with the real C heads kept.
Reports NLL/16, mean |dNLL/dlogit| for alpha, beta, C, and cos(-grad, onehot(truth) - p) for alpha and beta.
Usage: python experiments/t039_joint_gradient_real_heads.py LABEL=path_to_heads.npz [LABEL=...] [--n 512]
"""
import sys
import numpy as np
import torch

sys.path.insert(0, "/home/lantian/ASCAD")
from ASCAD_generate import AES_Sbox
from experiments.w1_timesnet_old_head_accs import load_attack_split
from MultiResNet.decoder_loss import GF_MUL, decoder_joint_nll, decoder_nll

args = [x for x in sys.argv[1:] if "=" in x]
n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 512
_, lab = load_attack_split(5000)
pt = np.asarray(lab["plaintext"][:n], dtype=np.uint8)
ky = np.asarray(lab["key"][:n], dtype=np.uint8)
S = torch.tensor(AES_Sbox[pt ^ ky].astype(np.int64))
a_true = np.asarray(lab["alpha"][:n]).astype(np.int64).reshape(-1)
b_true = np.asarray(lab["beta"][:n]).astype(np.int64).reshape(-1)
gf = torch.tensor(GF_MUL, dtype=torch.long)
R = torch.full((n, 16, 16), 1 / 16, dtype=torch.float64)


def logit(p):
    return torch.log(torch.as_tensor(p, dtype=torch.float64).clamp_min(1e-12)).clone().requires_grad_(True)


def toward(z, truth):
    t = torch.zeros_like(z)
    t.scatter_(-1, torch.tensor(truth)[..., None], 1.0)
    t = t - torch.softmax(z.detach(), -1)
    return torch.nn.functional.cosine_similarity(-z.grad, t, dim=-1).mean().item()


print(f"n = {n} attack traces")
print(f"{'model / state':<34}{'method':<10}{'NLL/16':>8}{'|g_a|':>10}{'|g_b|':>10}{'|g_C|':>10}{'cos_a':>8}{'cos_b':>8}")
for spec in args:
    label, path = spec.split("=", 1)
    d = np.load(path)
    A = d["alpha_output"][:n]
    B = d["beta_output"][:n]
    C = np.stack([d[f"sbox_{i}_output"][:n] for i in range(16)], axis=1)
    U = np.full_like(A, 1 / 256)
    for state, (pa0, pb0) in {"as trained": (A, B), "beta -> uniform": (A, U), "alpha, beta -> uniform": (U, U)}.items():
        for name in ("per-byte", "joint"):
            za, zb, zc = logit(pa0), logit(pb0), logit(C)
            pa, pb, pc = (torch.softmax(z, -1) for z in (za, zb, zc))
            nll = (decoder_nll(pa, pb, pc, R, S, gf) if name == "per-byte" else decoder_joint_nll(pa, pb, pc, S, gf)).mean()
            nll.backward()
            g = lambda z: z.grad.abs().mean().item()
            print(f"{label + ' / ' + state:<34}{name:<10}{nll.item() / 16:8.3f}{g(za):10.2e}{g(zb):10.2e}{g(zc):10.2e}"
                  f"{toward(za, a_true):8.3f}{toward(zb, b_true):8.3f}")
