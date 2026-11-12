"""T-039 RC-pi-U: --uniform-pi routes the decoder loss with R = 1/16 and never reads the pi heads."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from MultiResNet.decoder_loss import GF_MUL, decoder_nll, head_probs


def _outputs(n=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    keys = ["alpha_output", "beta_output"] + [f"sbox_{i}_output" for i in range(16)]
    out = {k: torch.randn(n, 256, generator=g, requires_grad=True) for k in keys}
    for i in range(16):
        out[f"permind_{i}_output"] = torch.randn(n, 16, generator=g, requires_grad=True)
    return out


def test_uniform_routing_equals_zero_pi_logits():
    out = _outputs()
    S = torch.randint(0, 256, (4, 16))
    gf = torch.as_tensor(GF_MUL)
    zero = {k: (torch.zeros_like(v) if k.startswith("permind_") else v) for k, v in out.items()}
    ref = decoder_nll(*head_probs(zero, pi_routing="row"), S, gf)
    got = decoder_nll(*head_probs(out, pi_routing="uniform"), S, gf)
    assert torch.allclose(ref, got, atol=1e-5)


def test_uniform_routing_gives_pi_heads_no_gradient_and_others_some():
    out = _outputs()
    S = torch.randint(0, 256, (4, 16))
    decoder_nll(*head_probs(out, pi_routing="uniform"), S, torch.as_tensor(GF_MUL)).mean().backward()
    assert all(out[f"permind_{i}_output"].grad is None for i in range(16))
    assert out["alpha_output"].grad is not None and out["sbox_0_output"].grad is not None


def test_row_routing_still_trains_pi():
    out = _outputs()
    S = torch.randint(0, 256, (4, 16))
    decoder_nll(*head_probs(out, pi_routing="row"), S, torch.as_tensor(GF_MUL)).mean().backward()
    assert all(out[f"permind_{i}_output"].grad is not None for i in range(16))
