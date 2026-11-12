"""T-039 E6: differentiable affine-masking decoder for ASCADv2, used as a training loss.

The label identity (checked on the extracted set) is  C_slot = alpha (x) S[pi(slot)] xor beta,
with (x) multiplication in GF(2^8) (AES polynomial) and pi(slot) = perm_index[slot] the byte
processed in that slot. The attack decoder (`experiments/w2_decoders`) marginalises the heads:

    q_b(v) = sum_slot R[slot, b] * sum_{a, d} P_alpha(a) P_beta(d) P_C[slot](a (x) v xor d)

This module computes -log(q_b(S_b) / sum_v q_b(v)) for the TRUE S_b only, which needs no 256x256
table per slot:
  U[slot]       = P_C[slot] (xor-conv) P_beta          (Walsh-Hadamard, exact)
  T[slot, b]    = sum_a P_alpha(a) U[slot, a (x) S_b]  (gather)
  Z[slot]       = sum_v T_full[slot, v] = (1 - P_alpha(0)) + P_alpha(0) * 256 * U[slot, 0]
because v -> a (x) v is a bijection for a != 0 and maps everything to 0 for a = 0.
"""
import numpy as np
import torch


def _gmul(a, b):
    p = 0
    for _ in range(8):
        if b & 1:
            p ^= a
        hi = a & 0x80
        a = (a << 1) & 0xFF
        if hi:
            a ^= 0x1B
        b >>= 1
    return p


GF_MUL = np.array([[_gmul(a, b) for b in range(256)] for a in range(256)], dtype=np.int64)


def fwht(x):
    """Unnormalised Walsh-Hadamard transform along the last axis (length 256)."""
    shape = x.shape
    n = shape[-1]
    h = 1
    while h < n:
        x = x.reshape(*shape[:-1], n // (2 * h), 2, h)
        a, b = x[..., 0, :], x[..., 1, :]
        x = torch.stack((a + b, a - b), dim=-2)
        h *= 2
    return x.reshape(shape)


def xor_conv(p, q):
    """r[j] = sum_d q[d] p[j xor d]; p (..., 256), q broadcastable."""
    return fwht(fwht(p) * fwht(q)) / p.shape[-1]


def head_probs(outputs, n_slots=16, pi_routing="row"):
    """Softmax of the 34 heads in float32: alpha (n,256), beta (n,256), C (n,16,256), R (n,16,16).

    pi_routing="row": R = per-slot softmax (every run before T-041). "crf": R = exact permutation
    marginals of the pi logits (T-041 B3), differentiable; also works under torch.no_grad.
    pi_routing="uniform": R = 1/16 everywhere (RC-pi-U); the pi logits are not read, so the pi heads get no gradient.
    """
    pa = torch.softmax(outputs["alpha_output"].float(), dim=-1)
    pb = torch.softmax(outputs["beta_output"].float(), dim=-1)
    pc = torch.stack([torch.softmax(outputs[f"sbox_{i}_output"].float(), dim=-1)
                      for i in range(n_slots)], dim=1)
    if pi_routing == "uniform":
        return pa, pb, pc, torch.full((pa.shape[0], n_slots, n_slots), 1.0 / n_slots, device=pa.device)
    Lpi = torch.stack([outputs[f"permind_{i}_output"].float() for i in range(n_slots)], dim=1)
    if pi_routing == "crf":
        from MultiResNet.perm_crf import marginals
        if torch.is_grad_enabled() and Lpi.requires_grad:
            R = marginals(Lpi, create_graph=True)
        else:
            with torch.enable_grad():
                R = marginals(Lpi.detach().requires_grad_(True), create_graph=False).detach()
    else:
        R = torch.softmax(Lpi, dim=-1)
    return pa, pb, pc, R


def decoder_joint_nll(pa, pb, pc, S, gf_mul, eps=1e-30, chunk=16):
    """Per-trace -log of the JOINT likelihood of all 16 S values under one shared (alpha, beta) (uniform routing).

    Model: p(S_1..S_16 | x) = sum_{a=1..255, d} P_alpha'(a) P_beta(d) prod_b Pbar_C(a (x) S_b xor d), with Pbar_C the
    slot-mean of the 16 C heads (what uniform pi routing gives) and P_alpha' = P_alpha restricted to a != 0 and
    renormalised (alpha is never 0). Each factor sums to 1 over S_b (a != 0, v -> a (x) v is a bijection), so this is a
    proper pmf. `decoder_nll` instead sums the 16 per-byte marginals, i.e. treats the bytes as independent although they
    share one (alpha, beta); the two coincide when P_alpha and P_beta are one-hot. Uniform heads give 16 ln 256.
    S (n,16) long, pc (n,16,256). Cost: n * 16 * 255 * 256 gathers; `chunk` traces at a time bounds the memory.
    """
    pcm = pc.mean(dim=1)                                          # (n, 256)
    logpc = torch.log(pcm.clamp_min(eps))
    pa1 = pa[:, 1:] / (1.0 - pa[:, 0:1]).clamp_min(eps)           # (n, 255): a = 1..255
    logprior = torch.log(pa1.clamp_min(eps))[:, :, None] + torch.log(pb.clamp_min(eps))[:, None, :]   # (n, 255, 256)
    d = torch.arange(256, device=S.device)
    out = []
    for i in range(0, S.shape[0], chunk):
        Sc = S[i:i + chunk]
        m = Sc.shape[0]
        mul = gf_mul[1:][:, Sc].permute(1, 2, 0)                  # (m, byte, a): a (x) S_b for a = 1..255
        idx = mul[..., None] ^ d                                  # (m, byte, a, d)
        lp = torch.gather(logpc[i:i + m, None, None, :].expand(m, 16, 255, 256), 3, idx)
        out.append(-torch.logsumexp((lp.sum(dim=1) + logprior[i:i + m]).flatten(1), dim=1))
    return torch.cat(out)


def decoder_nll(pa, pb, pc, R, S, gf_mul, eps=1e-30):
    """Per-trace sum over the 16 bytes of -log posterior(S_b); S (n,16) long, gf_mul (256,256) long.

    Returns (n,) tensor. Uniform heads give 16*ln(256); oracle one-hot heads give 0.
    """
    n = S.shape[0]
    U = xor_conv(pc, pb[:, None, :])                         # (n, slot, 256)
    idx = gf_mul[:, S].permute(1, 2, 0)                      # (n, byte, a): a (x) S_b
    Ug = torch.gather(U[:, :, None, :].expand(n, 16, 16, 256), 3,
                      idx[:, None, :, :].expand(n, 16, 16, 256))   # (n, slot, byte, a)
    T = torch.einsum("na,nsba->nsb", pa, Ug)                  # (n, slot, byte)
    Z = (1.0 - pa[:, 0:1]) + pa[:, 0:1] * 256.0 * U[:, :, 0]  # (n, slot)
    q = (R * T).sum(dim=1)                                    # (n, byte)
    zb = (R * Z[:, :, None]).sum(dim=1)                       # (n, byte)
    return -(torch.log(q.clamp_min(eps)) - torch.log(zb.clamp_min(eps))).sum(dim=1)
