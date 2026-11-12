"""T-039 3headResNet: one C head, no pi heads, same 34-key forward dict."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from MultiResNet.model import CompactMultiResNet


def _model(three_head):
    return CompactMultiResNet(input_dim=15000, depth=19, bottleneck_dim=0, head_hidden_dim=64,
                              three_head=three_head).eval()


def test_param_counts():
    assert sum(p.numel() for p in _model(False).parameters()) == 15_646_944
    assert sum(p.numel() for p in _model(True).parameters()) == 8_264_096


def test_forward_keeps_the_34_key_layout():
    m = _model(True)
    out = m(torch.randn(2, 15000))
    assert len(out) == 34
    assert all(torch.equal(out[f"sbox_{i}_output"], out["sbox_0_output"]) for i in range(16))
    assert all(out[f"permind_{i}_output"].shape == (2, 16) and not out[f"permind_{i}_output"].any()
               for i in range(16))
    assert torch.allclose(torch.softmax(out["permind_3_output"], -1), torch.full((2, 16), 1 / 16))


def test_one_c_head_gets_gradient_from_all_slots():
    m = _model(True).train()
    out = m(torch.randn(4, 15000))
    sum(out[f"sbox_{i}_output"].sum() for i in range(16)).backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in m.sbox[0].parameters())
    assert not hasattr(m, "permind")
