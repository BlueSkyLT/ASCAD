# T-031 Contrastive Learning: Why C is Learnable by MultiResNet but Not TimesNet

**Date:** 2026-10-08  
**Status:** Plan (not yet executed)  
**Scope:** ASCADv2, 50k profiling / 5k attack traces  
**Question:** Can contrastive learning reveal why C is learnable in one model but not another, and can it improve C learning without mask labels?

---

## Context

**C4 Oracle Head Swaps (2026-10-08):**
- **TimesNet baseline:** z −0.1066 (median), 0/16 rank-0
- **+ true C head:** z +14.2108, 16/16 rank-0 (40 traces)
- **+ true α/β/π (C left to model):** z +0.0766, 0/16 rank-0

**Interpretation:** C is the single binding constraint. TimesNet's trunk and α/β/π heads are fine; its C heads cannot learn the slot-aligned label. MultiResNet learned C; TimesNet did not. This gap is not architectural (both have trunks + heads) or data-related (same 50k traces).

**Prior observations:**
- C scores −0.09 nats (below uniform) on both models despite being key-carrying
- C is slot-indexed `sbox_masked[:,slot]` at fixed sample positions (5552, 5925, ...), unlike α/β/π which are byte-global
- MultiResNet: 137M params, simple ResNet trunk → flat 12.8k features → 34 heads
- TimesNet: 8.4M params, attention+pool → position-aware token outputs → 18 heads

**Hypothesis:** C's leakage is primarily **relational** (slot-to-slot consistency) or **higher-order** (products of traces), not a first-order 256-class label. Contrastive learning could exploit that by avoiding explicit class labels.

---

## Three Contrastive Approaches

### Approach 1: CL-α (Label-Free α via Contrastive Clustering)

**Goal:** Learn α without mask labels, using the fact that traces with the same key byte have the same α.

**Design:**
- **Positive pairs:** Two traces with the same key byte k, any plaintext p (same α)
- **Negative pairs:** Traces with different key bytes (different α)
- **Encoder:** Shared trunk (MultiResNet or TimesNet) up to the penultimate layer
- **Loss:** InfoNCE or simpler cross-entropy on the contrastive embedding
  ```
  L_cl = -log( exp(sim(z_i, z_pos) / τ) / Σ_j exp(sim(z_i, z_j) / τ) )
  ```
- **Temperature τ:** 0.07 (standard for SimCLR)
- **Batch construction:** Stratified by key byte, ensuring 50% same-key pairs per batch
- **Evaluation gate:** After contrastive pretraining, train a linear probe on α labels (50k traces); must reach ≥ 50% top-1 (phase 1 bar for oracle clustering)

**Why this matters:**
- R1–R10 measured that α *can* be read from the SNR>0.1 band (93.95% supervised MLP)
- But label-free clustering on the full trace failed (variance, kurtosis: ~2% matched accuracy)
- Contrastive learning aligns the **entire trunk** to preserve α-relevant structure, not just band selection
- If CL-α reaches 50%+, it unblocks pseudo-labeling for the full pipeline

**Cost:** GPU ~30 min (50k traces, 15 epochs, contrastive + linear probe)

### Approach 2: CL-C (Label-Free C via Slot-Aligned Leakage Contrast)

**Goal:** Learn C without slot-indexed labels, exploiting the fact that C = α ⊗ S(p⊕k) ⊕ β is deterministic given plaintext, key, and (the learned α, β).

**Design:**
1. **Supervised baseline:** Train MultiResNet on the standard setup (mask labels, C included), freeze trunk
2. **Contrastive head on top of trunk:** Replace the standard C head with a contrastive objective
   - **Positive:** Same plaintext + key → same C value (C is deterministic)
   - **Negative:** Different plaintext, same key (different S-box value) → different C
3. **Hybrid loss:** λ₁ × (standard CE on α/β/π) + λ₂ × (contrastive loss on C)
4. **Evaluation:** Does the contrastive C head learn C better than the standard CE head?

**Variant CL-C2 (Label-Free Fully):**
- Compute true α/β from the cache (oracle), use them to generate synthetic C = α ⊗ S(p⊕k) ⊕ β
- Train with contrastive loss on the **computed** C values (no mask labels after initialization)
- Evaluate: Can the model learn to reproduce computed C without storing it as a direct label?

**Why this matters:**
- C's label is unusually structured: it's not independent of α/β but a deterministic function
- Standard CE treats it as an unrelated 256-way classification; contrastive loss treats it as a relational property
- If this works, it suggests C's learning failure is label-structure, not capacity or data scarcity

**Cost:** GPU ~20 min per variant (same as N4: 50k, 15 epochs)

### Approach 3: CL-Bridge (Why TimesNet Cannot Learn C)

**Goal:** Understand the architectural gap by contrasting the trunks of MultiResNet (learns C) vs TimesNet (does not).

**Design:**
1. Train both models jointly on a shared profiling set, with a **trunk-level contrastive loss**:
   ```
   L_bridge = -log( exp(sim(trunk_mr, trunk_tn) / τ) / norm )
   ```
   where `trunk_mr` and `trunk_tn` are the penultimate activations of each trunk on the same trace
2. **Hypothesis:** If the trunks can be forced to represent the same information, TimesNet's C head will learn
3. **Evaluation:** Train with and without the bridge loss; measure C-head accuracy, key recovery z, traces-to-rank-0

**Alternative CL-Bridge2:**
- Use contrastive loss to align **head outputs**, not trunk representations
- Does a contrastive C-head loss (on computed C values) make TimesNet's C head learnable?

**Why this matters:**
- Pinpoints whether the gap is in feature learning (trunk) or head training (loss/optimization)
- If bridge loss + contrastive C-head fixes TimesNet, we know the problem is training protocol, not architecture

**Cost:** GPU ~45 min (50k traces, 15 epochs, two models, contrastive supervision)

---

## Execution Plan (Staged, with Gates)

### Stage 1: CL-α Probe (CPU prep + ~30 min GPU)

**Pre-run (CPU only):**
1. Write `experiments/cl_alpha_dataset.py`: stratified batch loader that ensures 50% same-key pairs per batch on 50k traces
2. Unit test on 1k traces to verify batch composition and positive/negative pair counts
3. Compute embedding dimension needed: measure trunk output size (MultiResNet: 12.8k, TimesNet: depends on architecture)

**GPU run:**
1. Implement InfoNCE loss in `ASCAD_train_models.py` (or reuse `torch.nn.CrossEntropyLoss` with cosine similarity)
2. Train contrastive model 15 epochs, 50k traces, batch 64, τ=0.07, lr=1e-3
3. Freeze trunk, train linear probe on α labels for 5 epochs (supervised validation)
4. **Gate:** Linear probe must reach ≥50% top-1 on α on the 5k attack split
   - **PASS:** Proceed to CL-C
   - **FAIL:** α is not encodable via contrastive learning; revisit design

**Report:** `docs/tasks/T031-reports/CL-α-contrastive-probe.md`

### Stage 2: CL-C Hybrid (if CL-α passes; ~20 min GPU)

**Pre-run (CPU):**
1. Verify MultiResNet baseline (reproduce w=0.5 from N4b: z +9.984 ±0.549)
2. Implement hybrid loss: λ₁ CE(α/β/π) + λ₂ contrastive(C)
3. Freeze λ₂ at 0.25 initially (conservative mix)

**GPU run:**
1. Train MultiResNet with hybrid loss, same setup as N4b (w=0.5 baseline for α/β/π part)
2. Measure per-epoch C-head accuracy vs standard CE-only baseline
3. **Gate:** C-head accuracy must improve by ≥1.5 pp over standard CE (to exceed measurement noise)
   - **PASS:** Contrastive C learning is effective; consider adding to production pipeline
   - **FAIL:** Contrastive loss on C does not help; C's learning failure is not structure-related

**Report:** `docs/tasks/T031-reports/CL-C-hybrid-loss.md`

### Stage 3: CL-Bridge (parallel to Stage 2, if time permits; ~45 min GPU)

**Pre-run (CPU):**
1. Build dual-trunk trainer that loads both MultiResNet and TimesNet with shared data loader
2. Implement trunk-level InfoNCE loss (same embedding space)
3. Prepare joint loss: λ₁ task-specific (standard CE) + λ₂ contrastive bridge

**GPU run:**
1. Train both models jointly for 15 epochs
2. Measure: (a) each model's C-head accuracy, (b) key recovery z, (c) alignment quality (cosine similarity of trunk outputs)
3. **Gate:** TimesNet's z must improve by >0.5 nats over baseline (z −0.1066 → ≥+0.4) with bridge loss
   - **PASS:** Architectural gap is bridgeable; trunk alignment is the lever
   - **FAIL:** Bridge loss doesn't help; the gap is in head training, not representation alignment

**Report:** `docs/tasks/T031-reports/CL-Bridge-dual-trunk.md`

---

## Decision Tree After Stages

| Outcome | Next Action |
|---------|------------|
| CL-α FAIL | α is not contrastive-learnable at 50% threshold. Label-free α remains stuck on the shuffle null. Do not proceed to CL-C/Bridge; close contrastive line and return to R11/R12 with a fresh gate design. |
| CL-α PASS, CL-C PASS | Contrastive C learning improves C-head accuracy. Deploy hybrid loss (λ₂ \in [0.1, 0.5]) on production MultiResNet. Measure if this lifts key recovery beyond w=0.5. |
| CL-α PASS, CL-C FAIL | C structure is not amenable to contrastive objectives. Label-free discovery remains via multi-sample second-order (R12). |
| CL-Bridge PASS (regardless of CL-C) | TimesNet's C learning is fixable via trunk alignment. Apply bridge loss + contrastive C head to TimesNet; re-test with w=0.5 weighting. |
| CL-Bridge FAIL | TimesNet's C failure is in head training dynamics (e.g., initialization, gradient flow). This is beyond contrastive scope; architectural redesign needed. |

---

## Implementation Checklist

- [ ] **CL dataset loader** (`cl_alpha_dataset.py`): stratified batching by key byte
- [ ] **InfoNCE loss function** (in `ASCAD_train_models.py` or separate module)
- [ ] **Contrastive trainer loop** (adapter around existing training code)
- [ ] **Unit tests**: batch composition (1k traces), loss computation, gradient flow
- [ ] **CL-α Stage 1:** Contrastive pretraining + linear probe
- [ ] **CL-C Stage 2:** Hybrid loss (if α passes)
- [ ] **CL-Bridge Stage 3:** Dual-trunk trainer (if time permits)
- [ ] **Evaluation:** Per-epoch C accuracy, key recovery z, traces-to-rank-0, GE checks

---

## Rationale & Risks

**Why contrastive learning is worth trying:**
1. C is structurally different from α/β/π (relational, computed from others), not an independent label
2. MultiResNet learned C (empirically true); TimesNet could not (empirically true); contrastive loss might expose why
3. R1–R10 showed label-free α fails at 2% matched accuracy; contrastive learning is a fundamentally different approach (representation alignment vs. direct clustering)
4. If CL-α works, it unblocks the entire non-worst-case pipeline (pseudo-labels → Phase 2)

**Risks:**
1. **Contrastive learning may not be the bottleneck.** C's failure might be initialization, learning rate, or epoch scaling — things CL won't fix
2. **Time cost.** Three stages × 30–45 min GPU = ~2 h, which matches the T-028 budget ceiling. If stages 1–2 fail, no budget left for R11/R12
3. **Hybrid loss tuning.** If CL-C is attempted, λ₂ hyperparameter search could explode cost

**Risk mitigation:**
- Run CL-α as a hard gate. If it fails (< 50% probe accuracy), close the line immediately
- Skip CL-Bridge unless CL-α passes (to preserve budget for R12 if needed)
- Use fixed λ₂ = 0.25 in CL-C (no sweep)

---

## Git Integration & Reporting

- **New scripts:** `experiments/cl_alpha_dataset.py`, `experiments/cl_contrastive_trainer.py`, `experiments/cl_eval.py`
- **Model changes:** `ASCAD_train_models.py` (add `--contrastive-loss` and `--contrastive-alpha-lambda` flags)
- **Reports:** Three markdown files under `docs/tasks/T031-reports/CL-*.md`
- **Ledger:** Add rows to `docs/tasks/T031-reports/LEDGER.md` for each stage (DONE/PASS/FAIL)

---

## Next: User Go-Ahead & Prioritization

Before launching:

1. **Approve Stage 1 (CL-α)?** ~30 min GPU, hard gate at 50% probe accuracy
2. **If Stage 1 fails, fall back to:** R11 label-free π (with new PASS bar) or accept that label-free α is not recoverable and focus on T31-N4c seeded baseline
3. **If Stage 1 passes, proceed to CL-C + CL-Bridge in parallel,** using remaining T-028 budget

---

## Timing Summary

| Stage | Duration | Gate | Budget |
|-------|----------|------|--------|
| Stage 1: CL-α | ~30 min | α probe ≥50% top-1 | 30 min |
| Stage 2: CL-C | ~20 min (if S1 passes) | C accuracy +1.5 pp | 50 min |
| Stage 3: CL-Bridge | ~45 min (parallel, optional) | TimesNet z >+0.4 | 95 min |
| **Total** | — | — | **~95 min** (within 2 h T-028 budget) |

**Uncommitted actions waiting for you:**
- T31-N4c: One seeded w=1 MultiResNet run (~16 min)
- Any contrastive learning stages (CL-α / CL-C / CL-Bridge)
