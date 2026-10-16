"""T-039: null controls + AOD for MultiResNet / CompactMultiResNet checkpoints.

w3_s3_eval.py does the same checks but its loader only builds TimesNet. This script reuses its
decode/AOD helpers on head posteriors cached by `experiments.w6_nochead_eval --save-npz`, and
runs the constant-trace null through the checkpoint with w6's own loader layout.

Reports, on Attack_traces[:n] with per-trace keys and worst-tie ranks:
  plain decode, shuffled-prediction null, constant (mean) trace null, AOD with the analytic prior,
and a "works" gate: 16/16 non-flat, median z >= 3, |shuffled-null z| < 1, both nulls' median
rank >= 96 (one-sided; w3_s3_eval.py's two-sided [96,160] band is also reported), AOD 16/16.

"works_v2" (T-039 E6 pre-registration, 2026-10-15) replaces the single-permutation nulls, which
E1 showed are too coarse (16 ranks; P(median < 96) ~ 15% under the null): the shuffled null is
pooled over --null-perms permutations and the constant-trace null is scored against
num_traces / const_traces disjoint label blocks; both need pooled median rank >= 96, and the
shuffled one also |median z| < 1.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)

from experiments.w1_timesnet_old_head_accs import load_attack_split  # noqa: E402
from experiments.w3_s3_eval import (  # noqa: E402
    aod_metrics,
    decode_and_measure,
    head_arrays,
    median_final_rank,
)


def constant_trace_heads(model_file, n_attack, n_const, device_str="cuda", batch=64):
    """Head posteriors for n_const copies of the mean (sub_avg-adjusted) attack trace."""
    import torch

    from MultiResNet.model import CompactMultiResNet, MultiResNet

    X, _ = load_attack_split(n_attack)
    mean_path = model_file + ".mean_wf.npy"
    if os.path.exists(mean_path):
        X = X - np.load(mean_path).astype(np.float32)
    Xc = np.tile(X.mean(axis=0), (n_const, 1)).astype(np.float32)
    sidecar = json.load(open(model_file + ".config.json", encoding="utf-8"))
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    if sidecar.get("compact", False):
        model = CompactMultiResNet(input_dim=X.shape[1], depth=19, without_permind=False,
                                   bottleneck_dim=sidecar.get("bottleneck_dim", 512),
                                   head_hidden_dim=sidecar.get("head_hidden_dim", 256))
    else:
        model = MultiResNet(input_dim=X.shape[1], depth=19, without_permind=False)
    model.load_state_dict(torch.load(model_file, map_location="cpu", weights_only=True))
    model.to(device).eval()
    out = {}
    with torch.no_grad():
        for i in range(0, n_const, batch):
            o = model(torch.from_numpy(Xc[i:i + batch]).to(device))
            for k, v in o.items():
                out.setdefault(k, []).append(torch.softmax(v, dim=-1).float().cpu().numpy())
    return {k: np.concatenate(v, axis=0).astype(np.float64) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True)
    ap.add_argument("--model-file", required=True)
    ap.add_argument("--npz", required=True, help="head posteriors from w6_nochead_eval --save-npz")
    ap.add_argument("--num-traces", type=int, default=5000)
    ap.add_argument("--const-traces", type=int, default=500)
    ap.add_argument("--rank-step", type=int, default=10)
    ap.add_argument("--null-perms", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    t0 = time.time()
    n = args.num_traces
    _, lab = load_attack_split(n)
    z = np.load(args.npz)
    heads = head_arrays({k: np.asarray(z[k])[:n] for k in z.files})

    post, plain, _ = decode_and_measure(heads, lab, n, args.rank_step)

    perm = np.random.default_rng(0).permutation(n)
    _, null, null_bytes = decode_and_measure(tuple(a[perm] for a in heads), lab, n,
                                             args.rank_step)

    cp = constant_trace_heads(args.model_file, n, args.const_traces, args.device)
    _, const, const_bytes = decode_and_measure(head_arrays(cp), lab, args.const_traces,
                                               args.rank_step)

    # works_v2 pooled nulls
    pool_rank, pool_z = [], []
    for seed in range(args.null_perms):
        pm = np.random.default_rng(1000 + seed).permutation(n)
        _, _, nb = decode_and_measure(tuple(a[pm] for a in heads), lab, n, args.rank_step)
        pool_rank += [nb[str(b)]["final_rank_worst"] for b in range(16)]
        pool_z += [nb[str(b)]["z_true"] for b in range(16)]
    m = args.const_traces
    cpool_rank = []
    for j in range(n // m):
        lab_j = {k: v[j * m:(j + 1) * m] for k, v in lab.items()}
        _, _, cb = decode_and_measure(head_arrays(cp), lab_j, m, args.rank_step)
        cpool_rank += [cb[str(b)]["final_rank_worst"] for b in range(16)]
    pooled = {"shuffled_perms": args.null_perms, "shuffled_n_byte_ranks": len(pool_rank),
              "shuffled_median_rank": float(np.median(pool_rank)),
              "shuffled_median_z": float(np.median(pool_z)),
              "shuffled_n_rank0": int(sum(r == 0 for r in pool_rank)),
              "const_blocks": n // m, "const_n_byte_ranks": len(cpool_rank),
              "const_median_rank": float(np.median(cpool_rank)),
              "const_n_rank0": int(sum(r == 0 for r in cpool_rank))}

    aod = {str(b): aod_metrics(post[:, b], lab["plaintext"][:, b], lab["key"][:, b], n,
                               args.rank_step) for b in range(16)}
    aod_rank0 = int(sum(aod[str(b)]["final_rank_worst"] == 0 for b in range(16)))
    aod_z = float(np.median([aod[str(b)]["z_true"] for b in range(16)]))

    null_rank = median_final_rank(null_bytes)
    const_rank = median_final_rank(const_bytes)

    def in_band(v):
        return 96 <= v <= 160

    # One-sided: a null only threatens the claim if it ranks the true key high (low rank).
    # The two-sided w3 band is kept for the record; it fails on the accepted 50k seed-1 model
    # (constant-trace median rank 190), which is not a false positive. Fixed 2026-10-13 before
    # any E1/E3 number existed.
    works = bool(plain["n_rank0_and_not_flat"] == 16 and plain["median_z"] >= 3.0
                 and abs(null["median_z"]) < 1.0 and null_rank >= 96
                 and const_rank >= 96 and aod_rank0 == 16)
    works_two_sided_w3 = bool(plain["n_rank0_and_not_flat"] == 16 and plain["median_z"] >= 3.0
                              and abs(null["median_z"]) < 1.0 and in_band(null_rank)
                              and in_band(const_rank) and aod_rank0 == 16)
    works_v2 = bool(plain["n_rank0_and_not_flat"] == 16 and plain["median_z"] >= 3.0
                    and aod_rank0 == 16 and pooled["shuffled_median_rank"] >= 96
                    and abs(pooled["shuffled_median_z"]) < 1.0 and pooled["const_median_rank"] >= 96)
    res = {
        "meta": {"label": args.label, "model_file": args.model_file, "npz": args.npz,
                 "n_traces": n, "const_traces": args.const_traces, "rank_step": args.rank_step,
                 "tie_break": "worst", "aod_prior": "analytic uniform",
                 "wall_s": round(time.time() - t0, 1)},
        "plain": plain,
        "shuffled_null": {**null, "median_final_rank": null_rank},
        "constant_trace_null": {**const, "median_final_rank": const_rank},
        "aod": {"n_rank0": aod_rank0, "median_z": aod_z, "per_byte": aod},
        "pooled_nulls": pooled,
        "works": works,
        "works_v2": works_v2,
        "works_two_sided_w3_band": works_two_sided_w3,
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=float)
    print(f"[{args.label}] plain {plain['n_rank0_and_not_flat']}/16 z {plain['median_z']:+.3f} "
          f"med traces {plain['median_traces_to_rank0']} | null z {null['median_z']:+.3f} "
          f"rank {null_rank} | const rank {const_rank} | AOD {aod_rank0}/16 | works={works} | "
          f"pooled null rank {pooled['shuffled_median_rank']} z {pooled['shuffled_median_z']:+.3f} "
          f"const {pooled['const_median_rank']} | works_v2={works_v2}")


if __name__ == "__main__":
    main()
