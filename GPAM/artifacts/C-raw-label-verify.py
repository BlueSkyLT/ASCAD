#!/usr/bin/env python3
"""Verify GPAM/raw_dataset.py label streaming against independent ground truth.

Checks, in order of how much they would explain a loss pinned at chance:
  1. vectorized compute_labels vs the scalar reference already validated on
     stored ASCADv2 labels
  2. trace<->label alignment through the argsort/unsort path, proven by reading
     traces one index at a time
  3. raw metadata labels vs the extracted file's STORED labels, matched on
     (key, plaintext, masks) triples -- real ground truth for the raw shards
  4. whether the hardcoded min=-128 delta=255 normalization matches the actual
     int8 data, and whether an int8/uint8 wrap is present
"""
import sys
import numpy as np
import h5py

sys.path.insert(0, "GPAM")
from raw_dataset import RawMultiShardLoader, AES_SBOX, MUL_TABLE, perm_indices_vec  # noqa
from run_ascadv2_gpam_baseline import (  # noqa
    masked_sbox_label_for_metadata,
    perm_indices,
)

DB = "STM32_AES_v2/ASCAD_data/ASCAD_databases"
SHARD = f"{DB}/ascadv2-stm32-conso-raw-traces1.h5"
EXTRACTED = f"{DB}/ascadv2-extracted.h5"

loader = RawMultiShardLoader(db_dir=DB, split="train", target_bytes=list(range(16)),
                             chunk_size=512, batch_size=64, shuffle=False)

with h5py.File(SHARD, "r") as f:
    meta = f["metadata"][:512]
FSHARD = h5py.File(SHARD, "r")   # kept open for the trace reads below
traces_ds = FSHARD["traces"]

y = loader.compute_labels(meta)

# ---- 1. vectorized vs scalar reference -------------------------------------
print("== 1. compute_labels vs validated scalar reference ==")
bad = 0
for b in range(16):
    ref_c = np.array([masked_sbox_label_for_metadata(meta[i], b) for i in range(512)], dtype=np.int32)
    ref_p = np.array([perm_indices(b, meta[i]["masks"][0], meta[i]["masks"][1],
                                   meta[i]["masks"][2], meta[i]["masks"][3]) for i in range(512)], dtype=np.int32)
    dc = int((ref_c != y[f"c_{b}"]).sum())
    dp = int((ref_p != y[f"p_{b}"]).sum())
    bad += dc + dp
    if b < 3 or dc or dp:
        print(f"   byte {b:2d}: c mismatches={dc:4d}  p mismatches={dp:4d}")
print(f"   rm[18] ok={bool((y['rm']==np.asarray(meta['masks'])[:,18]).all())} "
      f"rout[17] ok={bool((y['rout']==np.asarray(meta['masks'])[:,17]).all())}")
print(f"   TOTAL label mismatches over 16 bytes x 512 traces: {bad}")

# label marginal sanity (a memorizable-but-flat target would look like this)
for k in ("rm", "rout", "c_0", "p_0"):
    v = y[k]
    print(f"   {k:5s} distinct={len(np.unique(v)):3d} "
          f"top-class-freq={np.bincount(v,minlength=int(v.max())+1).max()/len(v)*100:.2f}%")

# ---- 2. alignment through the sort/unsort path -----------------------------
print("\n== 2. trace<->label alignment through argsort/unsort ==")
rng = np.random.default_rng(0)
idx = rng.choice(20000, size=64, replace=False)
sort_order = np.argsort(idx)
sorted_idx = idx[sort_order]
raw_x = traces_ds[sorted_idx].astype(np.float32)
inv = np.empty_like(sort_order)
inv[sort_order] = np.arange(len(idx))
chunk_x = raw_x[inv]
one_by_one = np.stack([traces_ds[int(i)].astype(np.float32) for i in idx])
print(f"   unsort restores per-index traces exactly: {np.array_equal(chunk_x, one_by_one)}")
print(f"   x[0] matches its own index row: {np.array_equal(chunk_x[0], one_by_one[0])}")

# ---- 3. raw labels vs extracted STORED labels (ground truth) ---------------
print("\n== 3. raw compute_labels vs extracted stored labels, matched on metadata ==")
with h5py.File(EXTRACTED, "r") as f:
    emeta = f["Profiling_traces/metadata"][:40000]
    elab = f["Profiling_traces/labels"][:40000]
    estored = {(bytes(m["key"]) + bytes(m["plaintext"]) + bytes(m["masks"])):
               (int(l["alpha_mask"][0]), int(l["beta_mask"][0]),
                int(l["sbox_masked"][0]), int(l["perm_index"][0]))
               for m, l in zip(emeta, elab)}
print(f"   extracted fingerprints indexed: {len(estored)}")
with h5py.File(SHARD, "r") as f:
    rm_all = f["metadata"][:60000]
yr = loader.compute_labels(rm_all)
n_match = 0
mis = np.zeros(4, dtype=int)
for i in range(len(rm_all)):
    m = rm_all[i]
    fp = bytes(m["key"]) + bytes(m["plaintext"]) + bytes(m["masks"])
    t = estored.get(fp)
    if t is None:
        continue
    n_match += 1
    got = (int(yr["rm"][i]), int(yr["rout"][i]), int(yr["c_0"][i]), int(yr["p_0"][i]))
    mis += np.array([int(g != s) for g, s in zip(got, t)])
print(f"   raw traces matched to an extracted row: {n_match}")
if n_match:
    print(f"   mismatches rm/rout/c_0/p_0: {mis.tolist()} out of {n_match}")
else:
    print("   no overlap found in the sampled ranges (raw shard 1 vs extracted Profiling head)")

# ---- 4. normalization and int8/uint8 wrap ---------------------------------
print("\n== 4. trace value range vs hardcoded min=-128 delta=255 ==")
blk = traces_ds[:200]
print(f"   dtype={blk.dtype} min={int(blk.min())} max={int(blk.max())} "
      f"mean={blk.mean():.2f} std={blk.std():.2f}")
as_u = blk.astype(np.uint8)
print(f"   as uint8: min={int(as_u.min())} max={int(as_u.max())} mean={as_u.mean():.2f}")
tv_s = np.abs(np.diff(blk.astype(np.int32), axis=1)).mean()
tv_u = np.abs(np.diff(as_u.astype(np.int32), axis=1)).mean()
print(f"   mean |neighbor diff|: signed view={tv_s:.2f}  unsigned view={tv_u:.2f}"
      "   (much larger => byte-wrap discontinuity)")
frac_extreme = float((np.abs(blk.astype(np.int32)) > 120).mean())
print(f"   fraction at |x|>120 (saturation/wrap candidates): {frac_extreme*100:.3f}%")

# where is the signal? per-sample across-trace std, top windows
std_per_sample = blk.astype(np.float32).std(axis=0)
top = np.argsort(-std_per_sample)[:2000]
print(f"   across-trace std: overall mean={std_per_sample.mean():.2f}; "
      f"top-2000 samples mean={std_per_sample[top].mean():.2f}")
print(f"   top-signal sample indices: min={int(top.min())} max={int(top.max())} "
      f"(trace length 1,000,000)")
