# T-008: GPU util / memory tuning for faster training

## Status

`in_progress`

## Goal

Use the GPU more effectively (compute **and** memory) so training/eval throughput is higher—especially for graph models where topology is fixed, and more generally across the fixed-key benchmark stack.

## Current Best Configs

Source of truth: `DEFAULT_MODEL_PROFILES` + CLI defaults in `benchmark_full_fixedkey_gpu.py`, plus GNN dense-path settings in `GNN/classifiers.py` / `GNN/data.py` / `ASCAD_train_models.py`.  
**Hardware target:** single CUDA GPU (local RTX 5090-class, high VRAM). OOM path auto-halves train `batch_size`.

### Shared runtime (all models when on CUDA)

| Knob | Current best / default | Notes |
|------|------------------------|--------|
| Device | `cuda` (`--force-device cuda` or auto) | Prefer GPU for all train/eval |
| AMP | **on** for CUDA (`autocast` + `GradScaler`) | In `ASCAD_train_models.train_model` for all network types |
| TF32 | **on** in benchmark driver | `matmul.allow_tf32`, `cudnn.allow_tf32`, `cudnn.benchmark` |
| OOM policy | halve train batch and retry | `train_with_auto_batch` in benchmark |
| Profiling set size | 50 000 traces (ATMEGA fixed-key) | Full profiling split unless overridden |
| Trace length / nodes | **700** | ATMEGA processed ASCAD; GNN = 700 nodes |

### Per-model train/eval (benchmark defaults)

| Model | Train batch | Eval batch | Epochs (profile) | Graph / sequence shape | Arch / structure knobs | Data path (throughput) | Extra efficiency knobs |
|-------|-------------|------------|------------------|------------------------|------------------------|------------------------|------------------------|
| **cnn** | **200** | full attack set (no special micro-batch in benchmark path) | 75 | input `(B, 1, 700)` | classic CNNBest; large FC head | standard `DataLoader` | AMP on; batch matches classic “cnn best” (not yet util-swept) |
| **mlp** | **100** | same as cnn | 75 | flat 700-D | MLPBest-style (node/layers via train params outside profile) | standard `DataLoader` | AMP on; classic small batch (not yet util-swept) |
| **gat** | **512** | **2048** (`--graph-eval-batch-size`) | 75 | **graph: N=700, complete K_700**; feat dim 1→hidden | `hidden=64`, `layers=2`, `dropout=0.2`, **`heads=4`**, dense path, **`attn_chunk=96`** | **`_GPUResidentGraphLoader`** (full X,y on GPU) | AMP; `torch.compile` (reduce-overhead, best-effort); chunked attention for VRAM |
| **gin** | **1024** | **2048** | 75 | N=700, complete K_700 | `hidden=64`, `layers=2`, `dropout=0.2`; dense GIN with **mean-normalized** agg (NaN-safe) | GPU-resident loader | AMP; compile; large batch OK after mean-norm fix |
| **gcn** | **1024** | **2048** | 75 | N=700, complete K_700 | `hidden=64`, `layers=2`, `dropout=0.2`; dense path collapses identical nodes to `(B, F)` after first global agg | GPU-resident loader | AMP; compile; memory-light after collapse |
| **graphsage** | **1024** | **2048** | 75 | N=700, complete K_700 | `hidden=64`, `layers=2`, `dropout=0.2`; dense complete-graph path | GPU-resident loader | AMP; compile |
| **timesnet** | **2048** | **256** CLI default (`--timesnet-eval-batch-size`; profile also lists 1024 but benchmark uses CLI) | 150 | seq `(B, 700, 1)` | `d_model=32`, `d_ff=64`, `e_layers=2`, `top_k=5`, `num_kernels=6`, emb k=3, LR **1e-4** | standard loader (not GPU-resident graph path) | AMP; compute-heavy backbone — train batch already large |
| **timesnet_ascad_head** | **2048** | **256** CLI (same as above) | 150 | seq `(B, 700, 1)` | same backbone; emb k=**11**; head `1024→1024→256`; LR **1e-4** | standard loader | AMP; same train batch as baseline TimesNet |

### Graph-specific “best utilization” stack (GAT/GIN/GCN/GraphSAGE)

These are the knobs that matter most for T-008’s GNN work:

| Setting | Value | Where |
|---------|-------|--------|
| Graph size | **700 nodes**, 1 feature / node | fixed ATMEGA POI window |
| Topology | **complete** `K_700` (identical every sample) | dense kernels; no per-batch `edge_index` |
| Train batch | GAT **512**; others **1024** | `DEFAULT_MODEL_PROFILES` |
| Eval batch | **2048** (try 4k+ for non-GAT if VRAM free) | `--graph-eval-batch-size` |
| Loader | GPU-resident full profiling matrix | `GNN/data.py` when `device=cuda` |
| GAT attention tiling | `attn_chunk = 96` | `GNN/classifiers.py` |
| Precision | AMP fp16/bf16 autocast + GradScaler | `ASCAD_train_models.py` |
| Compile | `torch.compile(..., mode="reduce-overhead")` | graph models on CUDA only |

### Push-further overrides (when util high but memory still low)

```bash
conda run -n py10 python benchmark_full_fixedkey_gpu.py \
  --force-device cuda \
  --models gat,gin,gcn,graphsage \
  --graph-eval-batch-size 4096 \
  --profile-json '{
    "gat": {"batch_size": 1024},
    "gin": {"batch_size": 2048},
    "gcn": {"batch_size": 2048},
    "graphsage": {"batch_size": 2048}
  }'
```

For TimesNet eval throughput (if ranking phase is slow and VRAM allows):

```bash
--timesnet-eval-batch-size 1024   # or higher; 0 = full batch (OOM risk)
```

### Caveats

- **CNN/MLP batches (200/100)** are classic ASCAD “best” reproducibility settings, **not** proven max-utilization points on the 5090 — open work item in Plan.
- **TimesNet train batch 2048** is the current benchmark default for throughput; attack quality recipe (LR/epochs) lives in T-004/T-005, not here.
- Effective batch may be **lower** than requested if OOM halving fires — check `batch_size_effective` in summary JSON.
- Standalone `ASCAD_test_models.py` still defaults `graph_eval_batch_size=8` unless the param file overrides — use the **benchmark** CLI for high-throughput eval.

*Last synced from code: 2026-07-17.*

## Context

- Symptom-driven work from GNN training via `GNN/run_full_gnn_fixedkey_with_delay.sh` / `benchmark_full_fixedkey_gpu.py`.
- Hardware context in project rules: large NVIDIA GPU (high VRAM); under-filled runs waste wall-clock on multi-model sweeps.
- Key invariant leveraged repeatedly: for sample-as-node baselines, **every trace is the same complete graph `K_700`** → dense kernels, no per-sample edge_index, optional full dataset on GPU.
- Related but different: delay scripts that **wait for free GPU** (`GPU_THRESHOLD`) are *scheduling*, not *utilization of a busy job* (see T-003/T-006 runners).

## Plan

- [x] Diagnose low GPU util + high memory (phase 1)
- [x] Dense complete-graph paths (avoid materializing full edge_index)
- [x] Chunked GAT attention to cut peak VRAM
- [x] Enable CUDA AMP (`autocast` + `GradScaler`) on graph train/eval
- [x] Raise default GNN train/eval batch sizes in benchmark profiles
- [x] GCN collapse of identical post-agg node states (stay in `(B, F)`)
- [x] TF32 / `cudnn.benchmark` friendly settings in benchmark path
- [x] Diagnose high util + low memory (phase 2): under-sized batches / host loader overhead
- [x] GPU-resident loader (`_GPUResidentGraphLoader`) — preload X,y once; slice on device
- [x] Further batch increases (GAT ~512; GIN/GCN/SAGE ~1024; eval up to ~2048)
- [x] Optional `torch.compile` on graph models for dense path
- [x] Fix GIN NaNs exposed by large batch + AMP (mean-normalized dense agg) — correctness for efficient configs
- [x] Auto-halve batch size on OOM in benchmark train path
- [ ] Profile TimesNet / CNN / MLP the same way (batch, AMP, dataloader, compile) and document best defaults
- [ ] Measure before/after: samples/sec, epoch time, peak VRAM, sustained util %
- [ ] Optional: configurable GAT `attn_chunk`; explicit CLI batch overrides without full `--profile-json`
- [ ] Optional: multi-GPU / multi-stream if single-GPU headroom is exhausted

## Decisions

- Prefer **throughput** (time-to-epoch / time-to-sweep) over low VRAM for its own sake when headroom exists.
- **Two different failure modes:**
  1. **Low util + high mem** → memory-bound / huge temps / tiny batches (classic early GAT).
  2. **High util + low mem** → GPU busy but under-fed; increase batch + cut host overhead to raise arithmetic intensity and fill SMs/memory subsystem.
- Fixed identical topology → **dense complete-graph kernels + GPU-resident features** are valid optimizations (not general PyG graphs).
- Stability constraint: large batch + AMP can surface numerical bugs (GIN sum-over-N); fix numerics rather than permanently crippling batch size.
- Keep OOM auto-retry so aggressive defaults remain safe across GPUs.

## Working notes / session log

### 2026-07-06/07 — primary session (`019f3896-8203-7520-89d0-e45321fe8d5e`, model `grok-build`)

**User prompts (core):**

1. *When running the GNN script, GPU util is low but memory util is high. Better use of GPU? Graph topo is fixed.*
2. *GPU util is ~98% now, but memory usage is low.*
3. *GIN training fails — all losses NaN* (side effect of efficiency changes).

**Phase 1 — low util / high mem**

Root causes noted:

- `K_700` complete graphs; GAT temps ~`(B, N, N, heads)`
- Small default batches
- fp32-only; repeated host→device / loader overhead
- Topology unused as a structural optimization

Changes:

| Area | What |
|------|------|
| `GNN/classifiers.py` | Chunked GAT attention (`attn_chunk≈96`); GCN early collapse on complete graph |
| `ASCAD_train_models.py` | CUDA AMP + GradScaler; non_blocking transfers |
| `benchmark_full_fixedkey_gpu.py` | Larger graph batch defaults; graph eval batch; TF32/cudnn.benchmark; OOM batch halving |
| `GNN/data.py` | Dense path: feed raw `(B, 700)` instead of building full edge_index batches |

**Phase 2 — ~98% util / low mem**

Interpretation: SMs busy on small work items; not saturating memory bandwidth / occupancy / Tensor Cores as much as possible.

Further changes:

| Area | What |
|------|------|
| Benchmark profiles | GAT batch **512**; GIN/GCN/GraphSAGE **1024**; graph eval batch **2048** |
| `GNN/data.py` | `_GPUResidentGraphLoader` preloads full `X,y` on CUDA; epoch = device slices |
| Train loop | Skip redundant `.to()` when already on device; `torch.compile` (reduce-overhead) for graph models when available |

**Phase 3 — GIN NaNs under new recipe**

- Cause: dense GIN used **sum** over 700 nodes → feature scale explosion under AMP + B=1024.
- Fix: **mean-normalized** dense aggregation in `GINClassifier._forward_dense`.
- Quick check: losses near `ln(256)≈5.545` at start instead of NaN.

### Other sessions (secondary)

| Session | Role |
|---------|------|
| `019f38bb-996f…` | Same code era (GPU-resident loader / dense path present in context); research focus was segments, not util tuning |
| `019f3c93-824a…` | CNN/MLP **delay-until-idle** script — GPU *sharing*, not filling a training job |
| `019f64f3-b502…` | Notes train_seconds / complexity (TimesNet heavy FLOPs); not a util-tuning implementation session |

### Code anchors (current tree)

- `GNN/data.py` — `_GPUResidentGraphLoader`, `make_graph_loader(..., device=...)`
- `GNN/classifiers.py` — dense paths, GAT chunking, GIN mean-norm, GCN collapse
- `ASCAD_train_models.py` — AMP, graph loader device wiring, compile hooks
- `benchmark_full_fixedkey_gpu.py` — per-model batch profiles, OOM retry, eval batch flags
- Launchers: `GNN/run_full_gnn_fixedkey_with_delay.sh`, `run_cnn_mlp_best_with_delay.sh` (queueing)

### Tuning knobs (practical)

```bash
# Push batches further (example)
conda run -n py10 python benchmark_full_fixedkey_gpu.py \
  --models gat,gin,gcn,graphsage \
  --profile-json '{
    "gat": {"batch_size": 1024},
    "gcn": {"batch_size": 2048},
    "gin": {"batch_size": 2048},
    "graphsage": {"batch_size": 2048}
  }' \
  --graph-eval-batch-size 4096
```

Watch `nvidia-smi`: goal is high util **and** healthy memory fill without OOM thrash; use OOM auto-halve as safety net.

## Next actions

1. Log a short **metrics table** (model, batch, peak VRAM, util %, sec/epoch) for GNN defaults vs previous small-batch setup.
2. Apply the same playbook to **TimesNet** (already large batches in some runs—confirm loader/AMP/compile and memory headroom).
3. Apply playbook to **CNN/MLP** baselines if epoch time matters for multi-seed sweeps.
4. Re-run GIN after mean-norm fix with large batch; discard prior NaN/error summary rows (`--rerun-errors` / force as needed).
5. Optional quality-of-life: expose `--graph-batch-size` / GAT chunk size without hand-written profile JSON.

## Related

- Sessions: **`019f3896-8203-7520-89d0-e45321fe8d5e`** (primary); secondary mentions in `019f38bb…`, scheduling in `019f3c93…`
- Tasks: **T-003** (GNN models — research results; GPU work was previously nested there), T-004 (TimesNet cost), T-006 (delay runners)
- Do not confuse with T-005 (ranking metrics) or T-007 (lab log)
