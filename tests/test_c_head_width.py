"""CompactMultiResNet(c_head_hidden_dim): only the C head changes width; default reproduces the old model; forward layout unchanged."""
import sys
import torch
sys.path.insert(0, "/home/lantian/ASCAD")
from MultiResNet.model import CompactMultiResNet


def _n(m):
    return sum(p.numel() for p in m.parameters())


def test_default_is_unchanged():
    a = CompactMultiResNet(input_dim=15000, depth=19, bottleneck_dim=0, head_hidden_dim=64, three_head=True)
    assert _n(a) == 8_264_096


def test_only_c_head_widens():
    base = CompactMultiResNet(input_dim=15000, depth=19, bottleneck_dim=0, head_hidden_dim=64, three_head=True)
    wide = CompactMultiResNet(input_dim=15000, depth=19, bottleneck_dim=0, head_hidden_dim=64, three_head=True,
                              c_head_hidden_dim=256)
    c_base = sum(p.numel() for p in base.sbox.parameters())
    c_wide = sum(p.numel() for p in wide.sbox.parameters())
    assert c_wide > c_base
    assert _n(wide) - _n(base) == c_wide - c_base              # nothing else changed
    assert sum(p.numel() for p in wide.alpha.parameters()) == sum(p.numel() for p in base.alpha.parameters())
    wide.eval()
    out = wide(torch.randn(2, 15000))
    assert len(out) == 34 and out["sbox_0_output"].shape == (2, 256)


def test_sixteen_head_model_widens_all_c_heads():
    base = CompactMultiResNet(input_dim=15000, depth=19, bottleneck_dim=0, head_hidden_dim=64)
    wide = CompactMultiResNet(input_dim=15000, depth=19, bottleneck_dim=0, head_hidden_dim=64, c_head_hidden_dim=128)
    assert len(wide.sbox) == 16
    assert sum(p.numel() for p in wide.sbox.parameters()) > sum(p.numel() for p in base.sbox.parameters())


if __name__ == "__main__":
    test_default_is_unchanged(); test_only_c_head_widens(); test_sixteen_head_model_widens_all_c_heads(); print("3 passed")
