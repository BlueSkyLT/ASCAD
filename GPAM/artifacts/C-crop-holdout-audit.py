"""Is the crop run's rm 92.95% on genuinely novel traces?

The claim rests on traces 400,000-401,999 never being trained on (train = 0-39,999).
Two ways that can still leak:
  1. duplicate capture rows -- the same (key, plaintext, masks) triple appearing in
     both slices, letting a model that memorised training labels answer the holdout
  2. trace-vector duplication -- identical power traces across slices
Also measures how predictable alpha is from nothing at all: alpha is one byte per
trace, so a model that only outputs the training marginal should sit near chance.
"""
import h5py
import numpy as np

P = "STM32_AES_v2/ASCAD_data/ASCAD_databases/ascadv2-extracted.h5"
TR, VA = (0, 40000), (400000, 402000)

with h5py.File(P, "r") as f:
    mt = f["Profiling_traces/metadata"][TR[0]:TR[1]]
    mv = f["Profiling_traces/metadata"][VA[0]:VA[1]]
    lt = f["Profiling_traces/labels"][TR[0]:TR[1]]
    lv = f["Profiling_traces/labels"][VA[0]:VA[1]]

print("== 1. metadata fingerprint overlap (key|plaintext|masks) ==")
fp_t = {bytes(m["key"]) + bytes(m["plaintext"]) + bytes(m["masks"]) for m in mt}
dups = [bytes(m["key"]) + bytes(m["plaintext"]) + bytes(m["masks"]) for m in mv]
n_hit = sum(1 for d in dups if d in fp_t)
print(f"   train rows {len(mt)} (unique {len(fp_t)}), holdout rows {len(mv)}")
print(f"   holdout rows whose metadata also appears in train: {n_hit}/{len(mv)}")

print("\n== 2. alpha structure: is the target even a per-trace secret? ==")
a_t = np.asarray(lt["alpha_mask"])[:, 0].astype(int)
a_v = np.asarray(lv["alpha_mask"])[:, 0].astype(int)
cnt = np.bincount(a_t, minlength=256)
top = cnt.max() / cnt.sum()
print(f"   train alpha distinct={len(np.unique(a_t))}/256, most frequent class "
      f"= {top*100:.3f}% (uniform = 0.391%)")
print(f"   holdout alpha distinct={len(np.unique(a_v))}, most frequent "
      f"= {np.bincount(a_v, minlength=256).max()/len(a_v)*100:.3f}%")
prior_acc = top * 100
maj = int(cnt.argmax())
print(f"   trace-free prior: always predict class {maj} -> {prior_acc:.3f}% expected")
print(f"   OUR REPORTED RESULT for this head: rm 92.95% holdout / 92.35% attack")

print("\n== 3. do any alpha values repeat across the two slices by coincidence? ==")
# alpha is one of 256 values, so per-row agreement is expected ~1/256 of the time;
# what matters is whether the FULL row (metadata) repeats, already tested in 1.
agree = float((a_v == a_t[:len(a_v)]).mean())
print(f"   positional alpha agreement train[0:2000] vs holdout: {agree*100:.2f}% "
      f"(expect ~0.39% if independent)")

print("\n== 4. identical power traces across slices (sample 300 x 300) ==")
rng = np.random.default_rng(0)
ti = rng.choice(TR[1] - TR[0], 300, replace=False)
vi = rng.choice(VA[1] - VA[0], 300, replace=False)
with h5py.File(P, "r") as f:
    Xt = f["Profiling_traces/traces"][TR[0] + np.sort(ti)]
    Xv = f["Profiling_traces/traces"][VA[0] + np.sort(vi)]
hash_t = {Xt[i].tobytes() for i in range(len(Xt))}
hits = sum(1 for i in range(len(Xv)) if Xv[i].tobytes() in hash_t)
print(f"   exact trace matches between sampled slices: {hits} "
      f"(of {len(Xt)}x{len(Xv)} pairs)")
