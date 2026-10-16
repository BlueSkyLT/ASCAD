# MultiResNet: Reference Architecture for ASCADv2

PyTorch implementation of the **MultiResNet** architecture proposed by ANSSI for side-channel analysis of AES on the **ASCADv2** dataset (STM32F303, affine masked with random byte shuffling).

---

## Architecture Overview

MultiResNet is a 1-D Deep Residual Network (Depth 19) designed to defeat higher-order masking and random shuffling through **joint multi-task supervision**.

### Core Specifications
- **Backbone**: Stem Conv1D (16 filters, kernel 11) followed by 9 residual stacks (each with two Conv1D layers and projection shortcuts). Filters increase from 16 up to 256.
- **Pooling**: Global 1D Average Pooling (kernel 4) $\to$ flattening into a dense feature vector of dimension **8,192**.
- **Prediction Head Bank**:
  - `alpha_output`: Multiplicative mask $\alpha \in \{1..255\}$ (256 classes)
  - `beta_output`: Additive mask $\beta \in \{0..255\}$ (256 classes)
  - `sbox_{0..15}_output`: 16 masked S-box outputs $C_i = (\alpha \times S_i) \oplus \beta$ (256 classes each)
  - `permind_{0..15}_output`: 16 shuffle execution slot indices $\pi_i \in \{0..15\}$ (16 classes each)
- **Total Heads**: 34 classification heads (18 heads if trained `--without-permind`).

### Model Size & Parameters
| Variant | Heads | Parameters | Footprint |
|---|---|---|---|
| **MultiResNet (Full)** | 34 heads | **137,396,064 (137.40M)** | ~550 MB |
| **MultiResNet (No Perm)** | 18 heads | **78,397,440 (78.40M)** | ~314 MB |
| **SingleHeadResNet** | 1 head (S-box) | **75,768,064 (75.77M)** | ~303 MB |

---

## Key Recovery Mechanism

Key recovery does not require guessing unmasked $S$ directly from raw power. Instead, MultiResNet solves the components independently:

```text
Trace X (15,000 samples)
  │
  ▼
MultiResNet 1D Backbone
  │
  ├─► Pr(alpha)
  ├─► Pr(beta)
  ├─► Pr(C_i)  for i = 0..15
  └─► Pr(pi_i) for i = 0..15
        │
        ▼
Recombination (multilabel_predict):
  1. De-shuffle traces: Sbox_byte_j = sum_i Pr(pi_j == i) * C_i
  2. Dissect beta:      Pr(alpha * S) = Pr(C) (x) Pr(beta)
  3. Dissect alpha:     Pr(S) = Pr(alpha * S) (x) Pr(alpha)
        │
        ▼
Key Scoring:
  Accumulate log Pr(Sbox(p_j ^ k_candidate)) across N attack traces -> Key Rank
```

---

## Reproduction & Usage

### 1. Training Full MultiResNet (34 Heads)
Train with native AMP and mean-waveform subtraction on 50,000 profiling traces:

```bash
python MultiResNet/train.py \
    --h5 STM32_AES_v2/ASCAD_data/ASCAD_databases/ascadv2-extracted.h5 \
    --out-model STM32_AES_v2/ASCAD_data/ASCAD_trained_models/multiresnet_34h_50k.pth \
    --train-len 50000 \
    --epochs 15 \
    --batch-size 64 \
    --sub-avg
```

### 2. Training Without Permutation Heads (18 Heads)
```bash
python MultiResNet/train.py \
    --h5 STM32_AES_v2/ASCAD_data/ASCAD_databases/ascadv2-extracted.h5 \
    --out-model STM32_AES_v2/ASCAD_data/ASCAD_trained_models/multiresnet_18h_50k.pth \
    --train-len 50000 \
    --epochs 15 \
    --without-permind \
    --sub-avg
```

### 3. Training Single-Head S-box Baseline
```bash
python MultiResNet/train.py \
    --h5 STM32_AES_v2/ASCAD_data/ASCAD_databases/ascadv2-extracted.h5 \
    --out-model STM32_AES_v2/ASCAD_data/ASCAD_trained_models/resnet_single_b0.pth \
    --train-len 20000 \
    --epochs 5 \
    --single-head \
    --target-byte 0 \
    --sub-avg
```

### 4. Evaluating on Attack Traces
```bash
python MultiResNet/evaluate.py \
    --model-file STM32_AES_v2/ASCAD_data/ASCAD_trained_models/multiresnet_34h_50k.pth \
    --num-traces 5000
```
