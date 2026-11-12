"""CompactMultiResNet(c_head_form): "bit" logits are bit-linear (softmax = product of 8 bit probabilities), "hw" logits depend on
the Hamming weight only, "full" reproduces the old model; only the C head changes; forward layout unchanged."""
import sys
import torch
sys.path.insert(0, "/home/lantian/ASCAD")
from MultiResNet.model import CompactMultiResNet

V = torch.arange(256)
BITS = ((V[:, None] >> torch.arange(8)) & 1).double()


def _model(form):
    torch.manual_seed(0)
    m = CompactMultiResNet(input_dim=15000, depth=19, bottleneck_dim=0, head_hidden_dim=64, three_head=True, c_head_form=form)
    return m.eval()


def _n(m):
    return sum(p.numel() for p in m.parameters())


def test_full_is_unchanged():
    assert _n(_model("full")) == 8_264_096


def test_bit_form_factorises():
    m = _model("bit")
    full = _model("full")
    assert _n(full) - _n(m) == (64 + 1) * (256 - 8)             # only the C output layer (weights + bias) shrinks
    out = m(torch.randn(3, 15000))
    assert len(out) == 34 and out["sbox_7_output"].shape == (3, 256)
    L = out["sbox_0_output"].double()
    w = L[:, 1 << torch.arange(8)] - L[:, :1]                   # per-bit weights read back from the logits
    assert torch.allclose(L - L[:, :1], w @ BITS.t(), atol=1e-4)
    p = torch.softmax(L, -1)
    pb = torch.sigmoid(w)                                        # P(bit_i = 1)
    prod = torch.prod(torch.where(BITS.bool()[None], pb[:, None, :], 1 - pb[:, None, :]), dim=-1)
    assert torch.allclose(p, prod, atol=1e-6)


def test_hw_form_depends_on_weight_only():
    out = _model("hw")(torch.randn(2, 15000))["sbox_0_output"]
    hw = BITS.sum(1).long()
    for h in range(9):
        vals = out[:, hw == h]
        assert torch.allclose(vals, vals[:, :1].expand_as(vals))


if __name__ == "__main__":
    test_full_is_unchanged(); test_bit_form_factorises(); test_hw_form_depends_on_weight_only(); print("3 passed")
