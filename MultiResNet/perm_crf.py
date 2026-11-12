"""T-041 B3: permutation CRF over the 16 pi heads.

Model: P(sigma | x) = exp(sum_s L[s, sigma(s)]) / perm(exp L), sigma in S_16, L = the 16 x 16 pi logits
(rows = slots, columns = bytes). Compared with 16 independent row cross-entropies, the numerator is the
same; only the normaliser changes, from all 16^16 slot->byte maps to the 16! permutations. Hence
sum_s CE_s = crf_nll + perm_collision: the CE sum already contains the CRF NLL plus a label-free term whose
gradient (row softmax - mu) makes each row agree on its own with the joint marginal.

  crf_nll(L, perm)       = -sum_s L[s, perm[s]] + log perm(exp L)      (per trace)
  perm_collision(L)      = sum_s logsumexp_b L[s, b] - log perm(exp L)
                         = -log P_indep(the 16 independent row softmaxes draw a valid permutation) >= 0
  marginals(L)           = d log perm(exp L) / dL = P(sigma(s) = b | x), differentiable (create_graph)

log perm is a forward DP over the subsets M of already-used bytes, one popcount layer per slot:
  f_k[M] = logsumexp_{b in M} ( f_{k-1}[M \\ b] + L[k-1, b] ),   |M| = k,   log perm = f_16[full].
Each layer is one gather + logsumexp with no in-place writes, so autograd (and double backward) works.
"""
import itertools
import math

import numpy as np
import torch

N = 16
_TABLES = {}


def _tables(n, device):
    key = (n, str(device))
    if key not in _TABLES:
        masks = np.arange(1 << n)
        pc = np.array([bin(m).count("1") for m in masks])
        layer_of = [masks[pc == k] for k in range(n + 1)]
        pos = np.zeros(1 << n, dtype=np.int64)
        for k in range(n + 1):
            pos[layer_of[k]] = np.arange(len(layer_of[k]))
        src, byte = [None], [None]
        for k in range(1, n + 1):
            M = layer_of[k]
            bits = np.array([[b for b in range(n) if (m >> b) & 1] for m in M], dtype=np.int64)  # (|L_k|, k)
            src.append(torch.as_tensor(pos[M[:, None] ^ (1 << bits)], device=device))
            byte.append(torch.as_tensor(bits, device=device))
        _TABLES[key] = (src, byte)
    return _TABLES[key]


def log_perm(L):
    """log perm(exp L) for L of shape (B, n, n); float32/64, differentiable."""
    B, n, _ = L.shape
    src, byte = _tables(n, L.device)
    f = L.new_zeros(B, 1)                                    # layer 0: the empty set
    for k in range(1, n + 1):
        cand = f[:, src[k]] + L[:, k - 1, :][:, byte[k]]     # (B, |L_k|, k)
        f = torch.logsumexp(cand, dim=-1)
    return f[:, 0]


def crf_nll(L, perm):
    """Per-trace permutation-CRF negative log-likelihood. L (B, n, n) logits, perm (B, n) long."""
    gold = L.gather(2, perm[:, :, None]).squeeze(-1).sum(1)
    return log_perm(L) - gold


def perm_collision(L):
    """Per-trace D = sum_s logsumexp_b L[s, b] - log perm(exp L) >= 0. Uses no labels."""
    return torch.logsumexp(L, dim=-1).sum(1) - log_perm(L)


def marginals(L, create_graph=True):
    """Exact P(sigma(s) = b | x) as d log perm / dL; differentiable w.r.t. L when create_graph."""
    if not L.requires_grad:
        L = L.detach().requires_grad_(True)
        create_graph = False
    lz = log_perm(L)
    (mu,) = torch.autograd.grad(lz.sum(), L, create_graph=create_graph)
    return mu


def _brute(L):
    B, n, _ = L.shape
    perms = torch.as_tensor(list(itertools.permutations(range(n))), device=L.device)   # (n!, n)
    scores = L[:, torch.arange(n), :][:, :, perms].diagonal(dim1=1, dim2=3).sum(-1) if False else \
        torch.stack([L[:, torch.arange(n), p].sum(-1) for p in perms], 1)              # (B, n!)
    lz = torch.logsumexp(scores, 1)
    w = torch.softmax(scores, 1)
    mu = torch.zeros_like(L)
    for i, p in enumerate(perms):
        mu[:, torch.arange(n), p] += w[:, i:i + 1]
    return lz, mu


def self_test(device="cpu"):
    g = torch.Generator().manual_seed(0)
    L6 = torch.randn(5, 6, 6, generator=g, dtype=torch.float64).to(device) * 2
    lz_b, mu_b = _brute(L6)
    e_lz = (log_perm(L6) - lz_b).abs().max().item()
    e_mu = (marginals(L6.clone().requires_grad_(True)) - mu_b).abs().max().item()
    L16 = torch.zeros(1, N, N, dtype=torch.float64, device=device)
    e_unif = abs(log_perm(L16).item() - math.lgamma(N + 1))
    Lg = (torch.randn(2, 5, 5, generator=g, dtype=torch.float64) * 1.5).to(device).requires_grad_(True)
    ok_gc = torch.autograd.gradcheck(lambda x: marginals(x).sum(dim=(1, 2)) * 0 + (marginals(x) ** 2).sum(dim=(1, 2)),
                                     (Lg,), eps=1e-6, atol=1e-5)
    perm = torch.stack([torch.randperm(6, generator=g) for _ in range(5)]).to(device)
    ce = torch.nn.functional.cross_entropy(L6.reshape(-1, 6), perm.reshape(-1), reduction="none").view(5, 6).sum(1)
    e_dec = (ce - crf_nll(L6, perm) - perm_collision(L6)).abs().max().item()
    print(f"perm_crf self-test: log_perm vs brute {e_lz:.1e}, marginals vs brute {e_mu:.1e}, "
          f"uniform 16x16 log 16! err {e_unif:.1e}, double-backward gradcheck {ok_gc}, "
          f"CE = CRF + D err {e_dec:.1e}")
    assert e_lz < 1e-10 and e_mu < 1e-10 and e_unif < 1e-9 and ok_gc and e_dec < 1e-10


if __name__ == "__main__":
    self_test()
