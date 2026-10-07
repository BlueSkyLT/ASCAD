# Key Elements for Supervised Prediction of α, β, π, C

**Context:** MultiResNet learns α/β/π well and nominally learns C (0.69% top-1, −0.09 nats). TimesNet learns α and π weakly but fails on C (0.40% top-1, −0.073 nats). GPAM learns α very well (100%), π at 86.5%, β at 10–14%, C at chance (~0.6%). What's the difference?

---

## MultiResNet (137M params, learns α/β/π, nominally C)

**Architecture:**
- **Trunk:** 1 stem conv → 9 residual stacks (each with 1–2 blocks) → AvgPool(k=4) → flatten
- **Feature bottleneck:** ~12.8k flat features (all information compressed to this point)
- **Heads:** 34 independent dense head branches, all **reading from the same flat vector**
  - α: 256-class (global, per-byte)
  - β: 256-class (global, per-byte)
  - C (sbox): 16 × 256-class, one per slot (slot-indexed masked S-box)
  - π: 16 × 16-class, one per slot (byte-to-slot permutation)

**Key strength:** 
- **Flat feature vector forces all 34 heads to share the same representation.** They all have equal access to the same 12.8k features and must cooperate.
- **Simple, deterministic routing.** Each head reads `(B, 12.8k)` and outputs `(B, 256)` or `(B, 16)`.
- **One global α works for all 16 slots.** No position-dependent branching.

**What enables C learning?**
- The flat vector is **position-agnostic.** C's slot-indexed label (sbox_0 through sbox_15) can be learned because the 16 C-heads each read the *same* 12.8k features but produce *different* outputs via independent dense layers (each head has its own `fc → relu → bn → out`).
- C gets the **same information density** as α/β/π but learns only at chance because: (1) C is slot-indexed (depends on π, which is only 67% learnable), (2) C's signal is small (SNR 0.072), (3) C is a function of α/β (not independent).
- But **learning still nominally happens** in the loss (CE term goes to ~0.69% top-1).

---

## TimesNet (8.4M params, learns α, weak π, fails C)

**Architecture:**
- **Trunk:** 1 embedding layer → N TimesNet blocks (each: FFT-based period detection → per-period Inception conv → residual) → temporal_pool (AvgPool over time or adaptive segmentation) → optional segment projection → flatten
- **Feature bottleneck:** Smaller flat vector (depends on pool configuration)
- **Heads:** 18 independent dense heads, same as MultiResNet structure
  - α: 256-class
  - β: 256-class (added in fair retrain, originally omitted)
  - π: 16 × 16-class (added in fair retrain)
  - (No C heads in the fair retrain; test C from cache)

**Key difference:**
- **Position-aware intermediate representations.** Before the final pool, TimesNet maintains `(B, seq_len, d_model)` with learned attention and temporal structure.
- **Pool destroys position information.** The temporal_pool function (mean/max over time, or adaptive segmentation) *collapses* all position-dependent information into a flat vector.
- **Stride + pool combination.** With `input_stride=5` and `pool_segments=32`, the model sees only 3,000 samples (15k / 5 stride = 3k) and then pools into 32 segments. Each segment is ~94 samples.

**Why C fails on TimesNet:**
- C is **slot-indexed** (`sbox_0` at sample 5552, `sbox_1` at sample 5925, etc.; ~373 samples apart).
- With stride=5, these samples appear at indices ~1110, ~1185, ... in the strided trace (15000 / 5 = 3000 samples).
- With pool_segments=32, each segment is 3000/32 ≈ 94 samples. The C slots can fall into *different segments* or be compressed by the pool.
- **The pool cannot preserve slot-to-slot relationships** that are separated by 370 raw samples (74 strided samples; crossing segment boundaries in the 94-sample pool).
- π (byte-to-slot routing) learns at 18% attack accuracy (vs 67% supervised) — near-random. Without π, the 16 C-heads don't know *which* slot each output should predict.
- Result: **C heads output at chance despite reading the pooled vector**, because the vector has lost the position information needed to index slots correctly.

---

## GPAM (8.4M params, learns α at 100%, π at 86.5%, β at 10–14%, C at ~0.6%)

**Architecture:**
- **Tokenization:** 400-sample patches (no stride) → embed to d_model=192
- **Trunk:** 3 full-attention GAU (Gated Attention Units) blocks over the sequence
- **Position-aware head:** Each head reads `(n_tokens, d_model)` and **preserves token-level information**
  - Head output: mean over filters (not over tokens) → `(d_model,)` — each head sees all token positions

**Key strength:**
- **Attention preserves position.** Each token interacts with every other token; the model can track which tokens hold which samples.
- **Position-aware readout.** Even after mean pooling the filters, the model knows (from the attention weights) which tokens/positions were relevant.
- **400-sample tokens are coarse but capture α/β cleanly.** α leaks at sample 867 (within one 400-sample window); β at 338 (same window).
- **C (slot-indexed) can still be recovered** because 16 C heads can each attend to different slot windows. The slot separation (~370 raw samples) is within one or two token boundaries, and attention is fine-grained.

**Why GPAM learns α better:**
- **Supervised α training is not a bottleneck in GPAM.** α is 100% on attack split (oracle reading).
- **The breakthrough is π learning.** GPAM reaches 86.5% π accuracy (vs ~67% MultiResNet, ~18% TimesNet fair retrain). This is because the attention-pooled representation naturally captures *which-sample-at-which-position*, i.e., the permutation.
- **With π learned, C becomes tractable.** True π + α + β → oracle C. But GPAM's C head scores only at chance (0.6%), same as MultiResNet baseline (0.69%).

---

## Comparison Table

| Element | MultiResNet | TimesNet | GPAM |
|---------|-------------|----------|------|
| **Trunk type** | ResNet (conv stacks) | TimesNet (FFT + Inception) | Attention (GAU blocks) |
| **Intermediate shape** | (B, C=256, L=…) → (B, 256, 94) | (B, seq_len, d=128) → pool → (B, pool_dim) | (B, n_tokens, d=192) → (B, d) per head |
| **Feature bottleneck** | Global flat 12.8k | Pooled/segmented flat | Per-head, position-agnostic |
| **Head input** | (B, 12.8k) | (B, flat_dim) | (B, 192) |
| **α accuracy (attack)** | 100% | 87% (fair retrain) | 100% (oracle reading) |
| **π accuracy (attack)** | 67% | 18% (fair retrain) | 86.5% |
| **β accuracy (attack)** | 5.84% | varies (18% no-C) | 10–14% |
| **C accuracy (attack)** | 0.69% (−0.09 nats) | 0.40% (−0.073 nats) | ~0.6% |
| **Key recovery (z)** | +9.5 / 16/16 | −0.107 / 0/16 | varies (not v2 eval) |
| **Key enabler for C?** | Shared flat features; 16 C-heads read same vector, output different predictions | **FAILS:** position info lost in pool; C slots compressed/collapsed | Attention preserves position; but C still at chance (slot-indexing issue remains) |

---

## Why C Stays at Chance Across All Models

**The fundamental problem: C is slot-indexed, not byte-indexed.**

- **MultiResNet:** All 16 C-heads read the same 12.8k flat vector. To predict sbox_i correctly, the head must:
  1. Know which byte is in which slot (i.e., π[i])
  2. Know α and β for that byte
  3. Compute C = α ⊗ S(p⊕k) ⊕ β
  - But the head only sees `(B, 12.8k)` and outputs `(B, 256)`. It has no mechanism to *know* which slot it's supposed to predict. The **label** is slot-specific (sbox_0, sbox_1, ...), but the **input** is position-agnostic.
  - Result: **Each C head must learn the slot mapping implicitly from the loss gradient.** This is hard because the loss mixes all 16 slot predictions into one cross-entropy term.

- **TimesNet:** Worse. The pool collapses position information entirely. The flat vector loses which samples held which slots.
  - Result: Even with position information present in intermediate layers, the head cannot recover it from a position-agnostic flat input.

- **GPAM:** Preserves position in the attention mechanism, but each head still reads a global summary (mean over filters, not over tokens).
  - Result: Better π learning (86.5%), but C still at chance because the slot-indexing problem remains.

**What C would need:** A head that **explicitly reads slot-position samples** (e.g., samples 5552, 5925, ... for slots 0, 1, ...) or a mechanism that **learns to route inputs to C-heads by slot index** (e.g., a position-conditional readout). MultiResNet's flat-feature heads cannot do this.

---

## Bottom Line

| Target | How It's Learned | Why It Works | Where It Fails |
|--------|-----------------|--------------|----------------|
| **α** | Global 256-way classification on the shared flat features | α leaks strongly (SNR 2.0, 871 samples, 100% supervised accuracy); global label means all samples vote for one of 256 classes | Label-free discovery (kurtosis, variance fail; necessary condition ~100k traces per R10) |
| **β** | Global 256-way classification on the shared flat features | β leaks moderately (SNR 1.4, 55 samples, 5.8% supervised accuracy); same routing as α | β's weak signal means the 16 β-heads receive noisy gradients; GPAM reaches only 10–14% even with attention |
| **π** | 16 × 16-way classification per slot | π is a control-flow label (which byte in which slot); attention models (GPAM 86.5%) learn this better than flat-feature models (MultiResNet 67%, TimesNet 18%) because attention preserves position information | Label-free discovery (never tried; R11 parked); π not essential for key (12/16 with uniform π); not a binding constraint |
| **C** | 16 × 256-way classification per slot, indexed by sample position (5552, 5925, ...) | **DOES NOT WORK.** C is the masked S-box value at slot i, a function of α, β, π at that slot. All models score at chance (0.4–0.7% top-1, −0.09 nats). | **C is not an observation; it's a computed function of the masks.** All models fail to learn it because the head structure cannot preserve slot-position information (MultiResNet loses it in the flat bottleneck; TimesNet loses it in the pool; GPAM's per-head design doesn't index by slot). Oracle C lifts z +14 (carriage works), but no training objective makes the heads learn C from the labels. |

---

## Next Steps Implied

1. **C cannot be fixed by training alone.** Neither down-weighting (T-031 N2/N4/N4b), contrastive loss (CL-0 falsified), nor architecture change (TimesNet w=0.5 flagged) will make C learnable.

2. **The key carry is already there (oracle C proves it).** The working baseline (z +9.5, 16/16) uses C's slot-indexed predictions as a routing mechanism, and that works even when C is at chance.

3. **Non-worst-case depends on label-free α.** R10 bounds it at ~100k traces minimum (with oracle π/positions). Label-free π is not yet tried; label-free C is ruled out (not learnable).

4. **Worst-case optimization:** w=0.5 on MultiResNet is adopted (T31-N4b). β improves to ~8%, attack stays 16/16, z +9.984.
