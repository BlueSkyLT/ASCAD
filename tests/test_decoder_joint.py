"""decoder_joint_nll: brute-force agreement, uniform value, and equality with the per-byte decoder at one-hot alpha, beta."""
import math
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/lantian/ASCAD")
from MultiResNet.decoder_loss import GF_MUL, decoder_joint_nll, decoder_nll

GF = torch.tensor(GF_MUL, dtype=torch.long)


def _heads(n, seed):
    g = torch.Generator().manual_seed(seed)
    pa = torch.softmax(torch.randn(n, 256, generator=g, dtype=torch.float64), -1)
    pb = torch.softmax(torch.randn(n, 256, generator=g, dtype=torch.float64), -1)
    pc = torch.softmax(torch.randn(n, 16, 256, generator=g, dtype=torch.float64), -1)
    S = torch.randint(0, 256, (n, 16), generator=g)
    return pa, pb, pc, S


def test_matches_brute_force():
    pa, pb, pc, S = _heads(2, 0)
    got = decoder_joint_nll(pa, pb, pc, S, GF, chunk=1)
    for i in range(2):
        pcm = pc[i].mean(0).numpy()
        pa1 = pa[i, 1:].numpy() / (1.0 - pa[i, 0].item())
        total = 0.0
        for a in range(1, 256):
            term = np.ones(256)
            for b in range(16):
                term = term * pcm[GF_MUL[a, S[i, b].item()] ^ np.arange(256)]
            total += pa1[a - 1] * float((pb[i].numpy() * term).sum())
        assert abs(got[i].item() - (-math.log(total))) < 1e-9, (got[i].item(), -math.log(total))


def test_uniform_is_16_ln_256():
    n = 3
    pa = torch.full((n, 256), 1 / 256, dtype=torch.float64)
    pb = pa.clone()
    pc = torch.full((n, 16, 256), 1 / 256, dtype=torch.float64)
    S = torch.randint(0, 256, (n, 16))
    got = decoder_joint_nll(pa, pb, pc, S, GF)
    assert torch.allclose(got, torch.full((n,), 16 * math.log(256), dtype=torch.float64), atol=1e-9)


def test_equals_per_byte_when_alpha_beta_are_one_hot():
    n = 3
    _, _, pc, S = _heads(n, 1)
    pa = torch.zeros(n, 256, dtype=torch.float64)
    pb = torch.zeros(n, 256, dtype=torch.float64)
    pa[torch.arange(n), torch.tensor([5, 77, 200])] = 1.0
    pb[torch.arange(n), torch.tensor([9, 0, 255])] = 1.0
    R = torch.full((n, 16, 16), 1 / 16, dtype=torch.float64)
    joint = decoder_joint_nll(pa, pb, pc, S, GF)
    per_byte = decoder_nll(pa, pb, pc, R, S, GF)
    assert torch.allclose(joint, per_byte, atol=1e-8), (joint, per_byte)


if __name__ == "__main__":
    test_matches_brute_force()
    test_uniform_is_16_ln_256()
    test_equals_per_byte_when_alpha_beta_are_one_hot()
    print("3 passed")
