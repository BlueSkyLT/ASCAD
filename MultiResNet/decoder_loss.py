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


def head_probs(outputs, n_slots=16):
    """Softmax of the 34 heads in float32: alpha (n,256), beta (n,256), C (n,16,256), R (n,16,16)."""
    pa = torch.softmax(outputs["alpha_output"].float(), dim=-1)
    pb = torch.softmax(outputs["beta_output"].float(), dim=-1)
    pc = torch.stack([torch.softmax(outputs[f"sbox_{i}_output"].float(), dim=-1)
                      for i in range(n_slots)], dim=1)
    R = torch.stack([torch.softmax(outputs[f"permind_{i}_output"].float(), dim=-1)
                     for i in range(n_slots)], dim=1)
    return pa, pb, pc, R


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
