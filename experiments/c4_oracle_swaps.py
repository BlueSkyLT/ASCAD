#!/usr/bin/env python3
"""T-028 C4 step 1 — oracle head swaps: which head is actually binding in each model?

The question `docs/tasks/T028-…md` §C4 asks is a localisation question, not a training question:

    "Swap in true C, true beta, true alpha, true pi one at a time and score the key recovery. If true
     C still leaves 0/16 rank-0 on TimesNet, the architecture cannot read C … If true C lifts it
     measurably, then C-loss weight 0.5 should be tried on TimesNet next (GPU, with --seed)."

This script answers it on **cached head posteriors only** — no training, no GPU. It runs the same
decode path T-030 S2 / T-031 W2 used (`slot_C_posteriors` -> `route_with_matrix` -> `byte_metrics`,
tie convention `worst`), so the as-trained arm must reproduce each model's own recorded number before
any swapped arm is believed.

Arms per model
--------------
  A0                     the model's own 34 heads
  B1..B4                 oracle alpha / beta / C / pi, ONE at a time (C4's literal request)
  C1..C3                 oracle pairs (alpha+beta, C+pi, C+beta)
  D0                     all four oracle  (ceiling of the decode path itself)
  E1..E4                 all four oracle EXCEPT one  (what still blocks when everything else is free)
  F1..F3                 uniform C / uniform pi / uniform alpha+beta  (negation controls)
  G1                     shuffled model heads, labels fixed (the no-information null)

Pre-registered reading rules — written before this script produced any number
-----------------------------------------------------------------------------
  R0  SANITY, hard stop. For every model whose number is already on record, A0 must reproduce it:
      |median_z(A0) - recorded| <= 1e-6, identical `n_rank0_and_not_flat`, identical `worst_final_rank`.
      If any model fails R0, NOTHING here is interpreted and the script exits non-zero.
  R1  "True C lifts the model"  <=>  median_z(B3) >= +3.0 AND n_rank0_and_not_flat(B3) >= 1
      (+3.0 is T-030 S3's own "works" z bar, not a new number).
  R2  "C is not the binding constraint"  <=>  median_z(B3) < +1.0 AND n_rank0_and_not_flat(B3) == 0,
      i.e. oracle C is indistinguishable from as-trained.  Between R1 and R2 the answer is PARTIAL and
      is reported as such, with the numbers, not rounded to either side.
  R3  The attribution rule. R1/R2 alone cannot say "the architecture cannot read C": oracle C removes
      the C head entirely, so a lift means "everything except C works", and no lift means "something
      besides C is also broken". The discriminating pair is E3 (all-but-C, i.e. C left to the model)
      against D0 (all four free):
        D0 lifts and E3 does not  ->  the C head is the single binding head;
        D0 lifts and E3 lifts     ->  C is not binding; the model's C head is already good enough;
        D0 does not lift          ->  the failure is upstream of the heads (trunk/routing/decode), and
                                     no per-head loss weight can fix it.
  R4  Any arm claiming 16/16 rank-0 is additionally scored with the label-prior-divided AOD rank and
      with `ASCAD_test_models.guessing_entropy`, and must agree with the tie-safe rank; an arm whose
      bytes are `flat` is never counted as a recovery (`rank()` reports rank 0 on tied vectors —
      T-028 Phase 0).
  R5  Nothing here is a non-worst-case result: every oracle arm reads the true mask labels of the
      ATTACK split. These are diagnostic upper bounds (evaluation-side oracles, as in S2/W2), never
      claimable key recovery. The only question is which head to attack next, and on which model.

Usage
-----
    conda run -n py10 python -u experiments/c4_oracle_swaps.py \
        --out-json experiments/out/c4_oracle_swaps.json
    # --smoke runs n=500 with 2 arms to validate the plumbing first
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
import h5py

from ASCAD_test_models import guessing_entropy
from experiments.w1_timesnet_old_head_accs import head_metrics, load_attack_split
from experiments.w2_decoders import (
    EPS_ORACLE,
    byte_metrics,
    run_variant,
    summarise,
)
from experiments.w3_s3_eval import aod_metrics, head_arrays, median_final_rank
from experiments.w5_c_lut import H5

KEYS_256 = [f"sbox_{i}_output" for i in range(16)]
KEYS_PI = [f"permind_{i}_output" for i in range(16)]
ORACLE = "oracle"
UNIFORM = "uniform"

# name -> (replacement spec, shuffles the model heads?, note)
VARIANTS = {
    "A0_as_trained":        ({}, False, "the model's own 34 heads, repo marginal routing"),
    "B1_oracle_alpha":      ({"alpha": ORACLE}, False, "true alpha replaces the alpha head"),
    "B2_oracle_beta":       ({"beta": ORACLE}, False, "true beta replaces the beta head"),
    "B3_oracle_C":          ({"C": ORACLE}, False, "true masked S-box values replace the 16 C heads"),
    "B4_oracle_pi":         ({"pi": ORACLE}, False, "true permutation replaces the 16 pi heads"),
    "C1_oracle_alpha_beta": ({"alpha": ORACLE, "beta": ORACLE}, False, "true alpha and beta"),
    "C2_oracle_C_pi":       ({"C": ORACLE, "pi": ORACLE}, False, "true C and true routing"),
    "C3_oracle_C_beta":     ({"C": ORACLE, "beta": ORACLE}, False, "true C and true beta"),
    "D0_oracle_all":        ({"alpha": ORACLE, "beta": ORACLE, "C": ORACLE, "pi": ORACLE}, False,
                             "ceiling: every head replaced by its true label"),
    "E1_all_but_alpha":     ({"beta": ORACLE, "C": ORACLE, "pi": ORACLE}, False,
                             "only the alpha head is left to the model"),
    "E2_all_but_beta":      ({"alpha": ORACLE, "C": ORACLE, "pi": ORACLE}, False,
                             "only the beta head is left to the model"),
    "E3_all_but_C":         ({"alpha": ORACLE, "beta": ORACLE, "pi": ORACLE}, False,
                             "only the 16 C heads are left to the model"),
    "E4_all_but_pi":        ({"alpha": ORACLE, "beta": ORACLE, "C": ORACLE}, False,
                             "only the routing is left to the model"),
    "F1_uniform_C":         ({"C": UNIFORM}, False, "negation: C carries nothing"),
    "F2_uniform_pi":        ({"pi": UNIFORM}, False, "negation: routing carries nothing"),
    "F3_uniform_alpha_beta": ({"alpha": UNIFORM, "beta": UNIFORM}, False,
                              "negation: both mask heads carry nothing"),
    "G1_null_shuffled_heads": ({}, True, "no-information null: model heads permuted, labels fixed"),
}

MODELS = [
    {
        "name": "timesnet_e15_last",
        "npz": "experiments/out/w3_head_preds_5k_e15resume_last.npz",
        "role": "C4's named cache (T-030 S3 Arm A, resumed to 15 epochs, last epoch)",
        "expect": {"recorded": "experiments/out/w3_s3_eval_e15resume_last.json",
                   "path": ["summary_plain"], "n": 5000},
    },
    {
        "name": "timesnet_e15_best",
        "npz": "experiments/out/w3_head_preds_5k_e15resume.npz",
        "role": "same campaign, the val-selected 'best' checkpoint (T-030 showed the split is unseeded)",
        "expect": {"recorded": "experiments/out/w3_s3_eval_e15resume.json",
                   "path": ["summary_plain"], "n": 5000},
    },
    {
        "name": "multiresnet_baseline",
        "npz": "experiments/out/head_preds_5k.npz",
        "role": "CONTROL: the working 137M MultiResNet 34-head model; must reproduce S2's headroom "
                "profile (C +6.28, beta +3.42, pi +1.67, alpha 0.00) or the swapped arms mean nothing",
        "expect": {"recorded": "experiments/out/w6_eval_baseline.json",
                   "path": ["decoders", "as_trained", "summary"], "n": 5000},
    },
    {
        "name": "timesnet_const_trace_null",
        "npz": "experiments/out/w3_head_preds_const500_e15resume_last.npz",
        "role": "null: the TimesNet weights read a constant (mean) trace — answers whether its output "
                "is trace-dependent at all",
        "expect": None,
    },
]

GE_VARIANTS = ["A0_as_trained", "B3_oracle_C", "E3_all_but_C", "D0_oracle_all"]


def decode_variant(name, heads, lab, n, rank_step, rng):
    needs, shuffle, note = VARIANTS[name]
    seed_perm = rng.permutation(n) if shuffle else None
    post, notes = run_variant(name, needs, n, heads, lab, seed_perm=seed_perm)
    per_byte = [byte_metrics(post[:, b], lab["plaintext"][:, b], lab["key"][:, b], n, rank_step)
                for b in range(16)]
    rows = {str(b): per_byte[b] for b in range(16)}
    return {"summary": summarise(per_byte),
            "per_byte": rows,
            "median_final_rank": median_final_rank(rows),
            "notes": notes, "replacements": {k: v for k, v in needs.items()},
            "shuffled_heads": bool(shuffle), "description": note}, post


def ge_block(byte_post, meta, n, rank_step, n_attacks, ge_seed):
    rows = {}
    for b in range(16):
        r = guessing_entropy(predictions=byte_post[:, b, :], metadata=meta, num_traces=n,
                             rank_step=rank_step, target_byte=b, n_attacks=n_attacks,
                             rng=ge_seed, simulated_key=1, return_curve=False)
        rows[str(b)] = {k: v for k, v in r.items() if k != "ge_at"}
    return rows


def parse_extra_model(spec):
    """NAME:PATH[:RECORDED_JSON[:DOTTED.PATH]] -- lets the battery score any new cache.

    RECORDED_JSON is an existing eval JSON whose as-trained summary A0 must reproduce (the R0 gate).
    Default recorded path is `summary_plain` (experiments/w3_s3_eval.py output); pass a dotted path
    for other conventions, e.g. `...:experiments/out/w6_eval_baseline.json:decoders.as_trained.summary`.
    """
    parts = spec.split(":")
    if len(parts) < 2:
        raise SystemExit(f"--add-model expects NAME:PATH[:RECORDED[:DOTTED]], got '{spec}'")
    name, npz = parts[0], parts[1]
    rec_file = parts[2] if len(parts) > 2 and parts[2] else ""
    dotted = parts[3].split(".") if len(parts) > 3 and parts[3] else ["summary_plain"]
    expect = {"recorded": rec_file, "path": dotted, "n": 5000} if rec_file else None
    return {"name": name, "npz": npz,
            "role": "ad hoc cache passed with --add-model (no recorded number to reproduce)"
                    if expect is None else
                    f"ad hoc cache passed with --add-model; A0 must reproduce {rec_file}",
            "expect": expect}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-json", default="", required=False)
    ap.add_argument("--num-traces", type=int, default=5000)
    ap.add_argument("--rank-step", type=int, default=10)
    ap.add_argument("--models", default="", help="comma list; default = all in MODELS")
    ap.add_argument("--variants", default="", help="comma list; default = all in VARIANTS")
    ap.add_argument("--ge-attacks", type=int, default=100)
    ap.add_argument("--ge-seed", type=int, default=12345)
    ap.add_argument("--no-ge", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="n=500, three arms, no GE")
    ap.add_argument("--rejudge", default="",
                    help="rewrite gate_readings in an existing JSON from its stored numbers "
                         "(no re-decode); the R1/R2/R3 thresholds are unchanged, only the label logic")
    ap.add_argument("--add-model", action="append", default=[],
                    help="NAME:PATH[:RECORDED_JSON[:DOTTED.PATH]] -- score any extra head-posterior "
                         "cache in this run; repeatable. With RECORDED_JSON the R0 sanity gate also "
                         "applies to it (default dotted path summary_plain).")
    args = ap.parse_args()
    if not args.rejudge and not args.out_json:
        ap.error("--out-json is required unless --rejudge is used")

    if args.rejudge:
        with open(args.rejudge, encoding="utf-8") as f:
            prior = json.load(f)
        prior["gate_readings"] = judge(prior)
        prior["meta"]["rejudge_note"] = (
            "gate_readings regenerated by --rejudge on the same stored variant numbers. The frozen "
            "numeric thresholds (lift = median z >= +3.0 or >= 8/16 non-flat rank-0; R2's z < +1.0) "
            "are unchanged; only the R3 label logic was rewritten, because the first version printed "
            "'C helps but is not sufficient alone' for a table in which oracle C alone reached 16/16.")
        with open(args.rejudge, "w", encoding="utf-8") as f:
            json.dump(prior, f, indent=2)
        print(json.dumps(prior["gate_readings"], indent=2))
        print(f"re-judged {args.rejudge}", flush=True)
        return

    t0 = time.time()
    n_req = 500 if args.smoke else args.num_traces
    for extra in args.add_model:
        MODELS.append(parse_extra_model(extra))
    want_models = [m for m in MODELS
                   if not args.models or m["name"] in args.models.split(",")]
    want_variants = list(VARIANTS) if not args.variants else args.variants.split(",")
    if args.smoke:
        want_variants = ["A0_as_trained", "B3_oracle_C", "G1_null_shuffled_heads"]

    _, lab_full = load_attack_split(max(n_req, 5000))
    with h5py.File(H5, "r") as f:
        meta_full = f["Attack_traces/metadata"][:max(n_req, 5000)]

    out = {"meta": {"step": "T-028 C4 step 1: oracle head swaps on cached posteriors",
                    "reading_rules": {
                        "R0_sanity": "A0 must reproduce each model's recorded median_z to 1e-6 and "
                                     "its rank counts exactly, or nothing is interpreted",
                        "R1_lift": "median_z(B3) >= +3.0 and >= 1 non-flat rank-0",
                        "R2_not_binding": "median_z(B3) < +1.0 and 0 non-flat rank-0",
                        "R3_attribution": "D0 vs E3 separates 'the C head is the single binding head' "
                                          "from 'the failure is upstream of the heads'",
                        "R4_flat_and_aod": "flat bytes never count as recovery; GE and AOD cross-check "
                                           "any arm reaching >= 8/16",
                        "R5_not_a_break": "oracle arms read attack-split labels; diagnostics only, "
                                          "never a claimable break"},
                    "n_traces_requested": int(n_req), "rank_step": args.rank_step,
                    "tie_break": "worst", "eps_oracle": EPS_ORACLE,
                    "ge_attacks": args.ge_attacks, "ge_seed": args.ge_seed,
                    "smoke": bool(args.smoke),
                    "h5": H5, "numpy": np.__version__,
                    "git": os.popen(f"git -C {repo_root} rev-parse --short HEAD").read().strip(),
                    "gpu_used": False},
           "models": {}}

    rng_master = np.random.default_rng(20261008)
    failures = []

    for spec in want_models:
        z = np.load(os.path.join(repo_root, spec["npz"]))
        missing = [k for k in ["alpha_output", "beta_output", *KEYS_256, *KEYS_PI] if k not in z.files]
        if missing:
            raise SystemExit(f"{spec['name']}: cache has no head(s) {missing}")
        n = min(n_req, int(z["alpha_output"].shape[0]))
        heads = head_arrays({k: np.asarray(z[k], dtype=np.float64)[:n] for k in z.files})
        lab = {k: v[:n] for k, v in lab_full.items()}

        rec = None
        if spec["expect"]:
            with open(os.path.join(repo_root, spec["expect"]["recorded"]), encoding="utf-8") as f:
                rec = json.load(f)
            for key in spec["expect"]["path"]:
                rec = rec[key]
        alpha_m = head_metrics(heads[0], lab["alpha"])
        beta_m = head_metrics(heads[1], lab["beta"])
        C_m = [head_metrics(heads[2][:, i], lab["sbox_masked"][:, i]) for i in range(16)]
        P_m = [head_metrics(heads[3][:, i], lab["perm_index"][:, i]) for i in range(16)]
        pi_exact = float(np.mean(np.all(np.stack([heads[3][:, i].argmax(1) for i in range(16)],
                                                 axis=1) == lab["perm_index"][:n], axis=1)))

        table, posts = {}, {}
        for name in want_variants:
            rng = np.random.default_rng(rng_master.integers(1 << 62))
            row, post = decode_variant(name, heads, lab, n, args.rank_step, rng)
            table[name] = row
            posts[name] = post
            s = row["summary"]
            print(f"[{spec['name']}] {name:22s} z {s['median_z']:+7.4f} "
                  f"r0 {s['n_rank0_and_not_flat']:2d}/16 flat {s['n_flat_bytes']:2d} "
                  f"tr->0 {s['median_traces_to_rank0']} medRank {row['median_final_rank']:.1f}",
                  flush=True)

        if spec["expect"] and not args.smoke and n == spec["expect"]["n"]:
            a0 = table["A0_as_trained"]["summary"]
            checks = [("median_z", rec["median_z"], a0["median_z"], 1e-6),
                      ("n_rank0_and_not_flat", rec["n_rank0_and_not_flat"],
                       a0["n_rank0_and_not_flat"], 0),
                      ("worst_final_rank", rec["worst_final_rank"], a0["worst_final_rank"], 0)]
            for field, exp, got, tol in checks:
                ok = abs(float(exp) - float(got)) <= tol + 1e-12
                print(f"[{spec['name']}] R0 sanity {field}: recorded {exp} vs {got} -> "
                      f"{'OK' if ok else 'FAIL'}", flush=True)
                if not ok:
                    failures.append(f"{spec['name']}.{field}: recorded {exp}, got {got}")

        ge = {}
        if not args.no_ge and not args.smoke:
            for name in GE_VARIANTS:
                if name not in table:
                    continue
                rows = ge_block(posts[name], meta_full[:n], n, args.rank_step,
                                args.ge_attacks, args.ge_seed)
                finals = [rows[str(b)]["ge_final"] for b in range(16)]
                mine = [table[name]["per_byte"][str(b)]["final_rank_worst"] for b in range(16)]
                ge[name] = {"per_byte": rows,
                            "median_ge_final": float(np.median(finals)),
                            "max_ge_final": float(np.max(finals)),
                            "cross_check_vs_tie_safe_rank_max_abs_diff":
                                float(np.max(np.abs(np.array(finals) - np.array(mine, dtype=float))))}
                print(f"[{spec['name']}] GE {name}: median {np.median(finals):.2f} "
                      f"cross-check {ge[name]['cross_check_vs_tie_safe_rank_max_abs_diff']:.3e}",
                      flush=True)

        aod = {}
        for name, row in table.items():
            if row["summary"]["n_rank0_and_not_flat"] >= 8:
                rows = {str(b): aod_metrics(posts[name][:, b], lab["plaintext"][:, b],
                                            lab["key"][:, b], n, args.rank_step)
                        for b in range(16)}
                rows["_median_final_rank_worst"] = median_final_rank(rows)
                aod[name] = rows
                print(f"[{spec['name']}] AOD {name}: median rank "
                      f"{aod[name]['_median_final_rank_worst']}", flush=True)

        out["models"][spec["name"]] = {
            "role": spec["role"], "npz": spec["npz"], "n_traces": int(n),
            "heads": {"alpha_top1": alpha_m["top1_acc"], "alpha_gain": alpha_m["gain_over_uniform_nats"],
                      "beta_top1": beta_m["top1_acc"], "beta_gain": beta_m["gain_over_uniform_nats"],
                      "C_mean_top1": float(np.mean([h["top1_acc"] for h in C_m])),
                      "C_mean_gain": float(np.mean([h["gain_over_uniform_nats"] for h in C_m])),
                      "pi_element_top1": float(np.mean([h["top1_acc"] for h in P_m])),
                      "pi_exact_permutation": pi_exact},
            "recorded_A0_expectation": spec["expect"] and {
                "file": spec["expect"]["recorded"],
                "median_z": rec["median_z"], "n_rank0_and_not_flat": rec["n_rank0_and_not_flat"],
                "worst_final_rank": rec["worst_final_rank"]},
            "variants": table, "guessing_entropy": ge, "aod": aod,
        }

    gate = judge(out)
    out["gate_readings"] = gate
    out["meta"]["wall_s"] = round(time.time() - t0, 1)
    out["meta"]["sanity_failures"] = failures

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(gate, indent=2), flush=True)
    print(f"wrote {args.out_json} ({out['meta']['wall_s']}s)", flush=True)
    if failures:
        print("R0 SANITY FAILED — do not interpret any swapped arm:", flush=True)
        for line in failures:
            print("  " + line, flush=True)
        sys.exit(3)


def judge(out):
    """Apply R1/R2/R3 literally, from the numbers in the table, not from prose.

    lift(x) := median_z >= +3.0 (T-030 S3's own "works" z bar) OR >= 8/16 non-flat rank-0.
    The three-way classification is what R3 was written to produce:
      B3 lifts, E3 does not  -> the model's own C heads carry nothing: C is the single binding head.
      B3 lifts, E3 lifts     -> the C heads DO carry the key (given true masks); the limit is elsewhere.
      B3 does not, E3 lifts  -> masks are the binding side, not C-reading.
      neither lifts, D0 does -> no single head is sufficient; the failure is distributed.
      D0 does not lift       -> the failure is upstream of the heads (decode/routing), no loss weight helps.
    """
    def lift(s):
        return bool(s) and (s["median_z"] >= 3.0 or s["n_rank0_and_not_flat"] >= 8)

    res = {}
    for name, m in out["models"].items():
        v = m["variants"]
        if "A0_as_trained" not in v or "B3_oracle_C" not in v:
            continue
        g = lambda k: v.get(k, {}).get("summary")
        a0, b3, d0, e3 = g("A0_as_trained"), g("B3_oracle_C"), g("D0_oracle_all"), g("E3_all_but_C")
        b3_up, e3_up, d0_up = lift(b3), lift(e3), lift(d0)
        if b3_up and not e3_up:
            verdict = ("C IS THE SINGLE BINDING HEAD: giving C away recovers the key, giving away "
                       "everything except C recovers nothing")
        elif b3_up and e3_up:
            verdict = ("the model's C heads already carry the key (E3 lifts); C-reading is not what "
                       "limits this model — the mask/routing heads are")
        elif (not b3_up) and e3_up:
            verdict = "C-reading is not the blocker; the mask side is (B3 alone cannot lift it)"
        elif d0_up:
            verdict = ("no single swap lifts it but the full oracle set does: the failure is "
                       "distributed over heads, so no per-head lever can fix it alone")
        else:
            verdict = "even the full oracle set does not lift it — the decode path itself is the limit"
        res[name] = {"A0_z": a0["median_z"], "A0_rank0": a0["n_rank0_and_not_flat"],
                     "B3_z": b3["median_z"], "B3_rank0": b3["n_rank0_and_not_flat"],
                     "D0_z": d0["median_z"] if d0 else None,
                     "D0_rank0": d0["n_rank0_and_not_flat"] if d0 else None,
                     "E3_z": e3["median_z"] if e3 else None,
                     "E3_rank0": e3["n_rank0_and_not_flat"] if e3 else None,
                     "R1_lift_by_oracle_C": b3_up, "R3_attribution": verdict,
                     "headroom_single_swaps": {
                         k.replace("B", "swap_"): (round(g(k)["median_z"] - a0["median_z"], 4)
                                                   if g(k) else None)
                         for k in ("B1_oracle_alpha", "B2_oracle_beta", "B3_oracle_C",
                                   "B4_oracle_pi") if k in v},
                     "leave_one_out_loss_from_D0": {
                         k.replace("E", "loo_"): (round(d0["median_z"] - g(k)["median_z"], 4)
                                                  if d0 and g(k) else None)
                         for k in ("E1_all_but_alpha", "E2_all_but_beta", "E3_all_but_C",
                                   "E4_all_but_pi") if k in v}}
    return res


if __name__ == "__main__":
    main()
