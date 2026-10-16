#!/usr/bin/env python3
"""Unit tests for CompactMultiResNet."""

import sys
import os
import torch

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from MultiResNet.model import MultiResNet, CompactMultiResNet


def test_compact_multiresnet():
    print("Testing CompactMultiResNet instantiation and properties...")
    
    # 1. Instantiate
    model = CompactMultiResNet(input_dim=15000, depth=19, without_permind=False, bottleneck_dim=512, head_hidden_dim=256)
    
    # 2. Count parameters
    total_p = sum(p.numel() for p in model.parameters())
    trunk_p = sum(p.numel() for n, p in model.named_parameters() if any(k in n for k in ["stem", "stacks", "pool"]))
    bot_p = sum(p.numel() for n, p in model.named_parameters() if "bottleneck" in n)
    heads_p = sum(p.numel() for n, p in model.named_parameters() if any(k in n for k in ["alpha", "beta", "sbox", "permind"]))
    
    print(f"Total params: {total_p:,} ({total_p/1e6:.2f}M)")
    print(f"Trunk params: {trunk_p:,} ({trunk_p/1e6:.2f}M)")
    print(f"Bottleneck params: {bot_p:,} ({bot_p/1e6:.2f}M)")
    print(f"34 Heads params: {heads_p:,} ({heads_p/1e6:.2f}M)")
    
    assert total_p < 16_000_000, f"Expected < 16M params, got {total_p:,}"
    assert trunk_p == 7_525_472, f"Expected exact trunk params 7,525,472, got {trunk_p:,}"
    
    # 3. Check baseline for comparison
    base_m = MultiResNet(input_dim=15000, depth=19, without_permind=False)
    base_p = sum(p.numel() for p in base_m.parameters())
    reduction = (base_p - total_p) / base_p * 100.0
    print(f"Baseline params: {base_p:,} ({base_p/1e6:.2f}M)")
    print(f"Reduction: {reduction:.2f}%")
    assert reduction > 88.0, f"Expected >88% reduction, got {reduction:.2f}%"
    
    # 4. Forward pass
    x = torch.randn(2, 1, 15000)
    out = model(x)
    assert len(out) == 34, f"Expected 34 output heads, got {len(out)}"
    assert out["alpha_output"].shape == (2, 256)
    assert out["beta_output"].shape == (2, 256)
    for i in range(16):
        assert out[f"sbox_{i}_output"].shape == (2, 256)
        assert out[f"permind_{i}_output"].shape == (2, 16)
        
    # 5. Backward pass
    loss = sum(v.sum() for v in out.values())
    loss.backward()
    for n, p in model.named_parameters():
        assert p.grad is not None, f"Parameter {n} has no gradient!"
        
    print("Variant A (shared bottleneck) tests PASSED!")

    # 6. Test Variant B: Direct Unshared Slim Heads (bottleneck_dim=None / 0, head_hidden_dim=64)
    print("\nTesting Variant B: Direct Unshared Slim Heads (bottleneck_dim=0, head_hidden_dim=64)...")
    model_b = CompactMultiResNet(input_dim=15000, depth=19, without_permind=False, bottleneck_dim=0, head_hidden_dim=64)
    total_b = sum(p.numel() for p in model_b.parameters())
    bot_b = sum(p.numel() for n, p in model_b.named_parameters() if "bottleneck" in n)
    heads_b = sum(p.numel() for n, p in model_b.named_parameters() if any(k in n for k in ["alpha", "beta", "sbox", "permind"]))
    print(f"Variant B total: {total_b:,} ({total_b/1e6:.2f}M)")
    print(f"Variant B bottleneck params: {bot_b:,}")
    print(f"Variant B heads params: {heads_b:,} ({heads_b/1e6:.2f}M)")
    assert bot_b == 0, f"Expected 0 bottleneck params for Variant B, got {bot_b}"
    assert total_b == 15_646_944, f"Expected exact 15,646,944 params, got {total_b:,}"
    reduction_b = (base_p - total_b) / base_p * 100.0
    print(f"Variant B reduction: {reduction_b:.2f}%")
    assert reduction_b > 88.0, f"Expected >88% reduction, got {reduction_b:.2f}%"

    out_b = model_b(x)
    assert len(out_b) == 34
    loss_b = sum(v.sum() for v in out_b.values())
    loss_b.backward()
    for n, p in model_b.named_parameters():
        assert p.grad is not None, f"Parameter {n} has no gradient in Variant B!"

    print("ALL TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_compact_multiresnet()
