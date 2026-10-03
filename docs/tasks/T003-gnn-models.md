# T-003: Develop and run GNN models on ASCAD

## Status

`active`

## Goal

Implement and benchmark graph neural networks (GCN, GAT, GIN, GraphSAGE) for ASCAD fixed-key profiling attacks, with usable GPU throughput and stable training. Reach training quality good enough for a fair comparison to TimesNet (esp. `timesnet_ascad_head`).

## Context

- Package: `GNN/` (`classifiers.py`, `data.py`, `run_full_gnn_fixedkey_with_delay.sh`, README).
- Default graph: **700 nodes = 700 samples**, **complete graph**, scalar power as node feature; ASCAD-style dense head `4096 → 4096 → 256`.
- Integrated into `benchmark_full_fixedkey_gpu.py` and train/test entry points.
- Commit landmark: `d43d354` (“GNN baseslines”).
- Checkpoints + eval under `.../ASCAD_trained_models/gnn_runs/` (`summary_gnn.json`, `aggregate_gnn.json`). Historical rows were migrated out of mixed `benchmark_runs_lr1e-4/` (see `gnn_runs/MIGRATION.json`).
- Segment helpers (for future graphs): `utils/find_segments.py` (`find_segments` / `find_segments_v2`, `segment_to_fixed_vector`, `trace_to_segment_features*`). See T-001.

## Assessment snapshot

### Legacy sample-node GNN (pre–arch-fix; now in `gnn_runs/summary_gnn.json`)

75e, seeds 0–1, Adam ~1e-3 (folder once mislabeled `lr1e-4`):

| Model | Train CE (random ≈ 5.545) | Attack |
|-------|---------------------------|--------|
| GAT | ~5.543 | rank ~238–239, no ttr0 |
| GCN | ~5.543 | rank ~239–240, no ttr0 |
| GIN | ~5.53 | seed lottery: rank 0 or ~90–238 |
| GraphSAGE | ~5.53 | **best GNN:** desync0 rank ~0–1, **ttr0 ≈ 4.2k–5.7k**; desync100 rank ~9–17 |

TimesNet strong bar: `timesnet_ascad_head` @ 1e-4 / 150e / RAdam — desync0 rank 0, **ttr0 ≈ 150–180**.

### Plan A Phase 1 — GAT/GIN architecture fix → **FAIL** (2026-07-17)

**Intent:** rescue non-learning GAT and seed-fragile GIN without LR grids. Code in `GNN/classifiers.py`.

| Model | What we changed | Why (hypothesis) |
|-------|-----------------|------------------|
| **GAT** | Local temporal attention `attn_window=32` (banded ±w); residual self proj + LayerNorm; no internal attn dropout | Full \(K_{700}\) softmax ≈ uniform 1/N → global mean collapse (same as dead GCN) |
| **GIN** | Keep mean-normalized agg; `train_eps=True`; LayerNorm in MLP; residual around update | Sum-over-N NaN history; brittle multi-seed training |

**Retrain:** `.../gnn_runs/gat_gin_fix/` — gat+gin × desync0/50/100 × seeds 0–1–2 × **75e, Adam @ 1e-3**. All 18 runs `status=ok`.

| Model | New result | vs legacy |
|-------|------------|-----------|
| **GAT** | CE ≈ **5.5425**, rank mean **~239**, **0** ttr0 on all cells | **No change** — still dead |
| **GIN** | CE ≈ **5.5425** almost all; rank mean **~219–239**; **0** ttr0 | **Worse** — lost legacy lottery wins (some rank 0 / CE~5.53) |

**Conclusion:** sample-node GAT/GIN are **not rescued** by local-window + residual/LN. Stop polishing these two on sample-nodes. Keep **GraphSAGE** as the only sample-node GNN with signal. Next research lever: **Plan B (segment-as-node)** or drop GAT/GIN as serious candidates — **not** more LR grids.

Detail: Phase 1 section below; aggregates in `gat_gin_fix/aggregate_full_fixedkey_gpu.json`.

## Plan (high level)

- [x] Implement GCN / GAT / GIN / GraphSAGE classifiers (PyG + dense complete-graph paths)
- [x] Wire into train / test / fixed-key GPU benchmark
- [x] Delay launcher when GPU busy (`GNN/run_full_gnn_fixedkey_with_delay.sh`)
- [x] Exploit fixed topology for denser / faster GPU path → details in **T-008**
- [x] Raise batch sizes + GPU-resident loading → **T-008**
- [x] Fix GIN NaN losses on dense complete-graph path → **T-008**
- [x] Run LR=1e-4 / 75 epoch multi-seed sweep (see T-005; note: GNN may not have used 1e-4 — see Assessment)
- [x] Plan A Phase 0 plumbing (LR/opt wiring + `gnn_runs/` isolation)
- [x] Plan A Phase 1 GAT/GIN arch fix + retrain — **failed** (see Assessment)
- [ ] **Fair TimesNet comparison** — GraphSAGE (only) if still investing in sample-node GNN
- [x] **Explore segment-as-node graphs** — Plan B first cut **done 2026-07-24** (see below + `docs/lab-log/plan-segment-node-gnn.md`)
- [x] **Plan C SNR + weighted product \(A\)** — **done 2026-07-24** (`plan-snr-xor-adjacency.md`); CE 5.498, rank_final=**35** (worse rank than Plan B 7 / old path 8)
- [ ] Plan C follow-ups: multi-\(Z\) / centered product / richer features before more edge ablations
- [ ] Optional: sparse/local edges on sample-nodes if still pursuing sample-node graphs

---

## Detailed plan A — Train GNNs better + fair TimesNet comparison

*Drafted 2026-07-17; revised same day: **architecture/stability first**, not LR grids. Phase 0 wiring kept as plumbing only.*

**Priority (user-aligned):** **fix GAT + GIN so they actually learn**, then re-benchmark under the **basic** recipe. **Do not** treat “force a new LR” as the main lever — Adam @ 1e-3 is fine as the default until models show a learning signal. LR/epoch matching to TimesNet is a **later fairness** step (⏱ / Phase 4).

**Marker:** **`⏱`** = skip on the critical path unless core results need it.

**Default GNN recipe (basic):** Adam @ **1e-3**, profile defaults (`hidden_dim=64`, 2 layers, 75e). No multi-LR / RAdam product grids on the critical path.

### Phase 0 — Plumbing (done, not the research focus)

| Knob | Value |
|------|--------|
| Datasets | desync0 / 50 / 100 |
| Seeds | 0,1,2 (min) |
| Eval | `eval_traces=10000`, `rank_step=10`, `target_byte=2` |
| Metrics | `rank_final`, `traces_to_rank0`, last train CE, train seconds |
| Output dir | `.../ASCAD_trained_models/gnn_runs/` (+ dated subdirs for campaigns) |

- [x] Honor `learning_rate` / `optimizer` in `train_model` + record in JSON (enables fair protocol later; **not** the current bottleneck for GAT/GIN).
- [x] Isolate GNN artifacts under `gnn_runs/`.

### Phase 1 — Fix GAT + GIN — **done (failed)**

Closed by full retrain under basic recipe. Canonical summary is in **Assessment snapshot** above.

**Code (still in tree):** `GNN/classifiers.py` — GAT local window ±32 + residual/LN; GIN mean-agg + train_eps + LN + residual.  
**Artifacts:** `.../gnn_runs/gat_gin_fix/{summary,aggregate}_full_fixedkey_gpu.json` + `*.pth`.  
**Exit taken:** do **not** iterate more sample-node GAT/GIN arch/LR; go Plan B or GraphSAGE-only reference.

#### What the fixes were (kept for the record)

| GAT | GIN |
|-----|-----|
| Local temporal attention instead of full \(K_{700}\) softmax (hypothesis: uniform 1/N collapse) | Mean-normalized dense agg (NaN-safe; from T-008) |
| Residual self projection + LayerNorm | `train_eps=True`, LayerNorm in GIN MLP, residual around update |
| Intended meaning: “look nearby in time + keep self” | Intended meaning: “global context + scaled self + stable MLP” |

**Empirical outcome:** neither produced attack-relevant learning (CE stuck ~5.5425; ranks ~238). Full-\(K_{700}\) was not the *only* blocker for GAT; GIN “stability package” did not preserve or improve legacy lottery wins.

### Phase 2 — GraphSAGE reference + optional longer budget

1. Keep existing GraphSAGE e75 numbers as baseline (`gnn_runs/summary_gnn.json`).
2. ⏱ One fair 150e GraphSAGE matrix later for TimesNet comparison (single LR choice) — only if still investing in sample-node GNN.
3. ⏱ Capacity / schedule knobs — low priority after Phase 1 fail.

### Phase 3 — Topology (preferred after Phase 1 fail)

1. **Segment-as-node** (Plan B) — preferred pivot  
2. ⏱ Other sparse sample-node graphs  
3. Treat sample-node GAT/GIN/GCN as **documented weak baselines**, not active candidates  

### Phase 4 — Fair comparison campaign (later)

Lock one table vs `timesnet_ascad_head` using **GraphSAGE** (and/or segment-node models if Plan B succeeds). ⏱ Matched 150e / LR only if needed for paper fairness.

### Suggested execution order

1. [x] Plumbing: LR/optimizer wiring + `gnn_runs/` isolation  
2. [x] Architecture fix GAT + GIN + retrain → **FAIL**  
3. **Next:** Plan B segment-as-node **or** freeze sample-node story on GraphSAGE-only  
4. ⏱ GraphSAGE 150e fair match + TimesNet table  
5. ⏱ GCN/GAT/GIN sample-node only as negative baselines in writeups

---

## Detailed plan B — Explore segment-as-node graphs

*Drafted 2026-07-17. Leave until scheduled. Depends on T-001 segment tooling; does **not** require Plan A to finish first (can run in parallel once a minimal GraphSAGE path exists).*

### Motivation

Sample-as-node \(K_{700}\) forces:

- Huge graphs, heavy GAT, GCN mean-collapse, weak inductive bias for ordered SCA traces.

Segment-as-node: each **waveform chunk between structure points** (local minima / events) becomes one node with a **feature vector**, not a scalar. Expect:

- **~20–40 nodes** on ATMEGA (T-001: ~27.7 segs/trace @ threshold=10 on desync100) vs 700  
- Topology can be **chain / path / local window** (natural for time order) instead of complete  
- Desync may appear as **segment boundary jitter** rather than total sample misalignment  
- Existing building blocks: `find_segments` / `find_segments_v2`, `segment_to_fixed_vector`, `trace_to_segment_features*` in `utils/find_segments.py`

### Design choices (decide early, keep first prototype minimal)

| Choice | Recommended first cut | Alternatives later |
|--------|----------------------|--------------------|
| Segmenter | **`find_segments_v2`** (prominence + min distance; fewer length-2 tails) | v1; fixed uniform windows as control |
| Threshold | start **10** (matches T-001 stats) | grid {5,10,15,20} |
| Centering | try **with and without** mean-center (v2 count depends on it) | — |
| Node features | resample segment → fixed dim **`out_dim=12`** via `segment_to_fixed_vector` | add stats (mean, std, min, max, length, start/end idx); raw padded crop |
| Variable #nodes | **pad/crop to fixed N** (e.g. N=32) for dense batched path; mask padding | pure PyG batch with variable sizes |
| Edges | **path graph** (i→i+1) + optional self-loops; or banded ±1/±2 | complete on N_seg (still smaller); learned |
| Models | start **GraphSAGE** (only semi-working GNN); then GIN | GCN/GAT once path topology proven |
| Head | small head first (`hidden→256` or modest MLP) | ASCAD 4096×2 if needed |
| Labels / attack | unchanged: 256-way S-box, byte 2, `full_ranks` | — |

**Fixed-N strategy (recommended for v1):**

1. Segment each trace → list of feature vectors.  
2. If count > N: merge adjacent shortest segments or uniformly subsample nodes.  
3. If count < N: pad zero nodes + boolean mask (message passing must ignore pads).  
4. Store `(B, N, F)` tensors; edge index shared for path/banded topology.

### Implementation phases

#### B0 — Segment statistics freeze (T-001 handshake)

- Re-run / record v1 vs v2 stats for **desync0 / 50 / 100** (profiling + attack), with/without centering.  
- Capture: mean/std/min/max segment count, length histogram, % traces with extreme counts.  
- Pick default `(segmenter, threshold, center, N, out_dim)` that covers ≥95% of traces without aggressive merge.  
- Write numbers into T-001 working notes and a one-line pointer here.

#### B1 — Graph builder (no training yet)

New module (suggested): `GNN/segment_graph.py` or extend `GNN/data.py`:

- `trace_to_segment_graph(trace) -> (x[N,F], edge_index, mask)`  
- Batch helpers: stack to `(B,N,F)`, shared `edge_index` for path topology  
- Unit checks: shapes, mask correctness, no HDF5 mutation  
- Optional cache: precompute segment features for 50k+10k to `.npz`/memmap under trained-models or a cache dir (**not** overwriting ASCAD `.h5`)

#### B2 — Minimal train loop smoke test

- desync0, seed 0, GraphSAGE, **20–30 epochs**, small batch  
- Success = train CE clearly &lt; 5.54 (learning signal)  
- Compare one control: **fixed uniform windows** (e.g. 28×25 samples) with same featureizer — isolates “content of segmenter” vs “fewer nodes”

#### B3 — First real benchmark

- Models: GraphSAGE (± GIN if B2 ok)  
- Datasets: desync0,50,100; seeds 0,1  
- Protocol: honor Plan A Phase 0 metrics; 75e first, then 150e if promising  
- Output: `benchmark_runs_segment_node/` (or similar)  
- Compare side-by-side to K_700 GraphSAGE from `benchmark_runs_lr1e-4` and to `timesnet_ascad_head` (same eval_traces)

#### B4 — Ablations (only if B3 shows non-random attack)

1. Topology: path vs banded±2 vs complete-on-N  
2. Features: out_dim {8,12,24}; + length/position channels  
3. Segmenter: v2 vs v1 vs uniform windows  
4. N and threshold grid  
5. Optional: small GAT on N≈32 (cheap vs N=700)

#### B5 — Integration polish

- Wire `network_type` or profile flag e.g. `graph_mode: segment` into train/test/benchmark  
- Document in `GNN/README.md`  
- Update T-001 checkbox “Use segment graphs as GNN nodes” when end-to-end works  
- Do **not** destroy/regenerate ASCAD HDF5; segments are derived views

### Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Variable segment count breaks dense path | Fixed N + mask; or PyG only for prototype |
| Desync changes segment boundaries a lot | Compare desync0 vs 100 segment-count stats; optional alignment features (relative position) |
| Segmenter not leakage-aligned | Uniform-window control; POI-aware later |
| Double-dipping compute (segment every epoch) | Cache features once per dataset |
| Scope creep vs Plan A | Cap: B2 smoke + B3 one GraphSAGE table before more ablations |

### Out of scope (for this exploration)

- Replacing TimesNet  
- Multilabel / multi-byte  
- STM32 long traces (can note feasibility; N_seg ~500 needs different design)  
- Editing or regenerating multi-GB ASCAD `.h5` files  

### Definition of done

- Documented segment→graph pipeline with frozen defaults  
- At least one GraphSAGE (or GIN) segment-node run with valid `rank_final` / `traces_to_rank0` on desync0  
- Written comparison vs sample-node \(K_{700}\) GraphSAGE and vs TimesNet head (even if worse)  
- Clear go/no-go: continue ablations vs abandon segment-nodes  

---

## Decisions

- 2026-07-24: **Plan C = weighted product \(A\)** — \(A_{ij}=\overline{x_i\odot x_j}\), soft \(\tilde A\), product messages. e75: CE **5.498**, rank_final **35** (replaces binary-path Plan C as official definition). Still worse attack than Plan B rank 7.
- 2026-07-24: **Plan B first cut executed** on ASCADv1 desync0. Defaults: `find_segments_v2` th=10, N=32, F=12, path edges, z-score, compact head, GraphSAGE h=128 L=3, 75e Adam 1e-3. Result: CE 5.482, **rank_final=7**, no ttr0. Proceed with feature ablations; hold full ASCADv2 train until better recipe. Code: `GNN/segment_graph.py`, `SegmentGraphSAGEClassifier`, `seg_graphsage` train/test path.
- 2026-06/07: Start with sample-as-node complete graphs for baseline comparability.
- 2026-07-06/07: Because topology is **identical for every trace**, prefer dense batched kernels + large batches over per-graph PyG overhead when possible.
- 2026-07-07: GIN dense path must **not** use raw sum over 700 neighbors; use **mean-normalized** aggregation (`mean_all + (1+eps)*x`) to avoid magnitude explosion / NaNs (esp. with AMP + large batch).
- Result-driven: **GraphSAGE** currently best among GNNs; **GAT/GCN** effectively dead at current LR/hparams; GIN fragile.
- 2026-07-17: Current GNN checkpoints are **not** fairly comparable to TimesNet ASCAD-head (epochs, optimizer/LR wiring, architecture). Plan A/B drafted.
- 2026-07-17: Segment-as-node first cut = `find_segments_v2` + fixed-N path graph + GraphSAGE + resampled features (`out_dim=12`); no HDF5 mutation.
- 2026-07-17 (session `gnn-models-plan-a`): Plan A **Phase 0 done** — GNN `learning_rate` / `optimizer` honored in `train_model` and recorded in benchmark JSON. Default GNN recipe remains Adam @ 1e-3 when profile leaves both `None`.
- 2026-07-17: **GNN outputs isolated** under `.../gnn_runs/` (checkpoints + `summary_gnn.json` / `aggregate_gnn.json`). Mixed `benchmark_runs_lr1e-4/` now TimesNet-only. GNN-only benchmark runs auto-default to `gnn_runs/`.
- 2026-07-17: Adam **base LR still matters** (adaptive per-param scaling multiplies α; not LR-free). See Working notes.
- 2026-07-17: Plan A revised — **keep basic**: no multi-LR grids; **`⏱`** for non-critical work.
- 2026-07-17: **Re-prioritize Plan A:** fix GAT/GIN architecture first (user). LR rewiring is plumbing, not the main lever.
- 2026-07-17: **Ask before launching** long/overnight GPU trainings or full benchmark matrices (user preference). `.grok/rules/04-ask-before-long-runs.md`.
- 2026-07-17: **Phase 1 closed FAIL** — local-window GAT + residual/LN GIN retrain (`gat_gin_fix/`) still random CE/rank. Stop sample-node GAT/GIN tuning; Plan B or GraphSAGE-only next.

## Working notes / session log

### 2026-07-24 — Plan C SNR-peak + XOR/relation edges

- `utils/snr.py`; `segmenter=snr` with frozen profiling peaks; `edge_mode` path/relation/combine.
- Artifacts: `gnn_runs/segment_node_snr/`, summary `summary_seg_graphsage_snr_path_desync0_e75.json`.
- SNR-path rank_final=8 vs Plan B rank 7; not an upgrade. Relation/combine 20e CE~5.54.

### 2026-07-24 — Plan B segment-as-node first cut

- Deep-research report mirrored to `docs/lab-log/research-snr-segment-gnn-report.md`.
- Plan: `docs/lab-log/plan-segment-node-gnn.md`.
- v1 desync0 75e → `gnn_runs/segment_node/seg_graphsage_desync0_e75.pth` + `summary_seg_graphsage_desync0_e75.json`.
- rank_final=7 @10k (meaningful vs random/GCN); not competitive with TimesNet/CNN.
- v2 stats only (`segment-node-v2-feasibility.json`); no v2 GNN train.
- Load fix: strip `_orig_mod.` from torch.compile checkpoints; skip compile for segment models.

### 2026-07-17 — docs: summarize fix + results in T-003 + progress.md

User asked to consolidate fix description and fail outcome into task + progress files (this pass).

### 2026-07-17 — `gat_gin_fix` results: fix failed

- Full matrix ok (gat+gin × 3 desync × seeds 0–2, 75e, Adam 1e-3).
- **GAT:** CE frozen ~5.5425, rank ~238–241 — identical failure mode to pre-fix complete-graph GAT.
- **GIN:** almost all seeds stuck ~5.5425 / rank ~237–240; only gin desync50 seed1 slightly better (CE 5.540, rank 182). **No** rank0; **worse** than legacy GIN lottery.
- GraphSAGE legacy still the only sample-node GNN with real signal (CE~5.53, some rank0).
- Decision: architecture tweak did **not** rescue sample-node GAT/GIN → Plan A Phase 1 exit → topology (Plan B) or abandon these two on sample-nodes.

### 2026-07-17 — GAT/GIN architecture fix + overnight (session `gnn-models-plan-a`)

**User:** forcing LR is *not* most important; fix GIN/GAT first; run overnight.  
**User (follow-up):** next time **ask permission** before launching runs; document what the fixes mean in T-003, progress, meeting notes.

**Agree:** legacy GAT CE flat at ~5.5425 for 75e = architecture (full \(K_{700}\) attention ≈ mean). GIN seed lottery = stability/expressivity, not “wrong folder LR.”

**Code (`GNN/classifiers.py`):**
- GAT: banded local attn `attn_window=32`, residual + LayerNorm, no GATConv internal attn dropout
- GIN: mean-agg retained; `train_eps=True`; MLP LayerNorm; residual

**Smoke (8e, desync0, seed0, Adam 1e-3):** GAT/GIN/SAGE all slowly leave init (GAT drop ~0.005 early — same phase as SAGE; legacy GAT never moved over 75e). Full 75e needed for rank claims.

**Overnight:** `gnn_runs/gat_gin_fix/` — models gat,gin; desync0/50/100; seeds 0,1,2; 75e.

### 2026-07-17 — gnn_runs isolation + Adam LR note (session `gnn-models-plan-a`)

**Adam initial LR claim:** **Incorrect** as stated. Adam adapts *relative* step sizes via moments, but every update is still scaled by the base LR α. α≈1e-2 vs 1e-4 changes trajectory a lot; only true “schedule” would be explicit LR schedulers (cosine, etc.), which we are not using by default.

**Data migration:**
- 26 GNN `.pth` → `ASCAD_trained_models/gnn_runs/`
- 25 eval rows → `gnn_runs/summary_gnn.json` (+ `aggregate_gnn.json`, `MIGRATION.json`, `README.md`)
- Stripped GNN from `benchmark_runs_lr1e-4` summary/aggregate (TimesNet 75e remains)
- Removed empty `benchmark_runs_plan_a_smoke/`
- Launcher / README / benchmark auto-route GNN-only → `gnn_runs/`

### 2026-07-17 — Plan A Phase 0 executed (session `gnn-models-plan-a`)

**Easiest subtask chosen:** Plan A Phase 0 (high leverage, pure code fix, unblocks all later sweeps).

**Code changes:**
- `ASCAD_train_models.py`: `resolve_optimizer_config` + `build_optimizer`; `train_model` now honors `learning_rate` and optional `optimizer_name` for **all** families (GNN was previously stuck on bare `Adam()` with default 1e-3). History records actual `optimizer` + `learning_rate`. Prints `Optimizer: … learning_rate: …` at train start.
- `benchmark_full_fixedkey_gpu.py`: GNN profiles accept `learning_rate` / `optimizer`; pass both into train; summary JSON rows store actual values (no longer null for GNN).
- Profiles: `benchmark_profiles/plan_a_graphsage_lr_smoke.json`, `plan_a_graphsage_fair_lr1e-4.json`, `plan_a_graphsage_fair_radam1e-4.json`.

**Wiring smoke (2 epochs, desync0, seed0, Adam @ 1e-4):**
- Log line: `Optimizer: adam  learning_rate: 0.0001`
- Summary row: `learning_rate=0.0001`, `optimizer=adam`, `train_loss_last≈5.55`, `rank_final=43`, `ttr0=920` (2e only — not a real attack claim)
- Out: `.../ASCAD_trained_models/benchmark_runs_plan_a_smoke/`

**Next experiment (user overnight):** GraphSAGE 150e Adam 1e-4 on desync0/50/100 seeds 0–2 → `benchmark_runs_plan_a_graphsage_adam1e-4`. Optional RAdam twin.

### 2026-07-17 — Plan A simplified (basic GNN, ⏱ markers)

- Removed multi-LR grid `{1e-2, 1e-3, 1e-4, 1e-5}` and Adam×RAdam product sweeps from critical path.  
- Capacity grid / schedulers / AMP / path-parity / sparse edges marked **`⏱`**.  
- Default stance: one basic recipe + short diagnostics, not hparam search.

### 2026-07-17 — training quality Q + plans

- User: are GNN trained well vs TimesNet? → **No** (see Assessment).  
- Wrote **Plan A** / **Plan B** into this file.  
- Key code gap later fixed in Phase 0 (see above).

### 2026-07-06/07 — GPU util & GIN NaNs (session `019f3896-8203-7520-89d0-e45321fe8d5e`, `grok-build`)

**Canonical write-up: T-008** (`T008-gpu-training-efficiency.md`). Summary only:

1. **Low GPU util, high memory** → complete-graph GAT attention tensors + small batches; topology unused.
2. After optimizations: **~98% util but low memory** → under-filling; larger batches + GPU-resident loader.
3. **GIN all losses NaN** → unnormalized sum on K_700 under AMP/large batch; fixed in dense path.

### 2026-07-15 — results analysis (session `019f64f3-b502…`, `grok-4.5`; see T-005)

At `benchmark_runs_lr1e-4` (labeled 1e-4, 75 epochs, seeds 0–1): GraphSAGE best GNN; GAT/GCN dead; GIN fragile. GraphSAGE trains in tens of seconds vs TimesNet tens of minutes.

### Related tooling sessions

- Segment graph idea + stats: `019f38bb-996f…` (T-001)
- Superpowers how-to session `019f3b54…` is tooling-only, not GNN research.

## Next actions

1. Decide: **Plan B (segment-as-node)** vs freeze sample-node story on **GraphSAGE-only** (user/meeting).  
2. Do **not** run further sample-node GAT/GIN LR or arch grids without new topology.  
3. ⏱ GraphSAGE 150e fair TimesNet match — optional.  
4. Metrics always: **rank_final / traces_to_rank0** (+ last train CE).

## Related

- Sessions: `019f3896-8203…`, `019f38bb-996f…`, `019f64f3-b502…`
- Paths: `GNN/`, `utils/find_segments.py`, `benchmark_full_fixedkey_gpu.py`, `.../gnn_runs/`
- Tasks: T-001 (segments), T-004 (TimesNet recipe), T-005 (metrics), T-008 (GPU efficiency)
