"""T-039 held-out check scoring (attack traces 5000-9999), against the rules frozen in the T-039 task file
("HOLD-OUT CHECK FROZEN 2026-11-01"). Reads experiments/out/t039_e6/holdout/<label>_<which>_<kind>_controls.json.

Comparison of two copies a, b of one checkpoint ("a better than b"):
  a passes works_v2 and b does not -> a better; both fail -> tie (not better); both pass -> lower MTD1 better, equal = tie.
A second reading ("gate = plain 16/16") is printed as a sensitivity check; the frozen rule does not say which gate
"failed" refers to, so the stricter works_v2 is primary.
Usage: python experiments/t039_holdout_score.py [> holdout/summary.txt]
"""
import json
import os

import numpy as np

H = "/home/lantian/ASCAD/experiments/out/t039_e6/holdout"
U = [f"{m}_s{s}" for m in ["RCpiT300k", "RCpiUT300k", "3headT300k", "3headAT300k"] for s in (1, 2)] + ["3headWT300k_s1"]
L = ["e1_t300k_s1", "e1_t300k_s2", "R1T300k_s2", "R2T300k_s1"]
OTHER = ["R1T300k_s1", "3headBitT300k_s1", "3headABT300k_s1"]
KINDS = ["orig", "tempmatchbit", "bit", "hw"]


def load(lw, kind):
    p = f"{H}/{lw}_{kind}_controls.json"
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    pl = d["plain"]
    return {"n16": pl["n_rank0_and_not_flat"], "mtd": pl["median_traces_to_rank0"], "z": pl["median_z"],
            "v2": d["works_v2"], "plain16": pl["n_rank0_and_not_flat"] == 16,
            "nulls_ok": d["pooled_nulls"]["shuffled_median_rank"] >= 96
            and abs(d["pooled_nulls"]["shuffled_median_z"]) < 1 and d["pooled_nulls"]["const_median_rank"] >= 96}


def better(a, b, gate):
    ga, gb = a[gate], b[gate]
    if ga and not gb:
        return 1
    if gb and not ga:
        return -1
    if not ga:
        return 0
    return int(np.sign(b["mtd"] - a["mtd"]))


def cell(r):
    if r is None:
        return f"{'-':>21}"
    m = f"{r['mtd']:.0f}" if r["mtd"] is not None else "-"
    return f"{r['n16']:>3}/16 {m:>5} z{r['z']:+5.1f} {'P' if r['v2'] else 'f'}"


rows = {}
print("held-out attack traces 5000-9999; cell = plain n/16, MTD1, median z, works_v2 (P/f)")
print(f"{'checkpoint':<26}" + "".join(f"{k:>22}" for k in KINDS))
for grp, labels in [("U", U), ("L", L), ("other", OTHER)]:
    print(f"--- family {grp}")
    for lab in labels:
        for which in ("best", "last"):
            lw = f"{lab}_{which}"
            rows[lw] = {k: load(lw, k) for k in KINDS}
            print(f"{lw:<26}" + "".join(f" {cell(rows[lw][k])}" for k in KINDS))

uw = [f"{x}_{w}" for x in U for w in ("best", "last")]
lw_ = [f"{x}_{w}" for x in L for w in ("best", "last")]
print()
for gate in ("v2", "plain16"):
    print(f"=== gate = {gate}")
    for a, b, h in [("bit", "orig", "H1"), ("bit", "tempmatchbit", "H2"), ("hw", "orig", "HW vs orig"),
                    ("tempmatchbit", "orig", "temp vs orig")]:
        c = [better(rows[x][a], rows[x][b], gate) for x in uw]
        w, l_, t = c.count(1), c.count(-1), c.count(0)
        rule = f"  -> {'PASS' if w >= 13 else 'FAIL'} (need >= 13/18)" if h in ("H1", "H2") else ""
        print(f"{h:<14} {a} better / worse / tie vs {b} over U (18): {w} / {l_} / {t}{rule}")
    fails = [x for x in uw if not rows[x]["orig"][gate]]
    rep = [x for x in fails if rows[x]["bit"][gate]]
    rept = [x for x in fails if rows[x]["tempmatchbit"][gate]]
    print(f"H3 original fails in U: {len(fails)}; bit repairs {len(rep)}"
          f" -> {'PASS' if fails and 2 * len(rep) >= len(fails) else 'FAIL'}; temperature repairs {len(rept)}")
    print(f"   failing: {', '.join(fails)}")

ratios = [rows[x]["bit"]["mtd"] / rows[x]["orig"]["mtd"] for x in lw_
          if rows[x]["bit"]["mtd"] and rows[x]["orig"]["mtd"]]
gate_l = sum(rows[x]["bit"]["v2"] for x in lw_), sum(rows[x]["orig"]["v2"] for x in lw_)
print(f"H4 L bit/orig MTD1 ratios: {', '.join(f'{r:.2f}' for r in ratios)}; median {np.median(ratios):.2f}"
      f" -> {'PASS' if np.median(ratios) <= 1.3 else 'FAIL'}; works_v2 bit {gate_l[0]}/8, orig {gate_l[1]}/8")
ab = [rows[f"3headABT300k_s1_{w}"][k] for w in ("best", "last") for k in KINDS if rows[f"3headABT300k_s1_{w}"][k]]
proj = [r for lw in rows for k in ("bit", "hw", "tempmatchbit") for r in [rows[lw][k]] if r and r["plain16"]
        and r["z"] >= 3]
bad = [lw + "_" + k for lw in rows for k in ("bit", "hw", "tempmatchbit")
       if rows[lw][k] and rows[lw][k]["plain16"] and rows[lw][k]["z"] >= 3 and not rows[lw][k]["nulls_ok"]]
print(f"H5 3headAB any pass: {any(r['v2'] for r in ab)} (max plain {max(r['n16'] for r in ab)}/16); "
      f"projected copies passing plain decode: {len(proj)}, of which failing a null: {len(bad)} {bad}"
      f" -> {'PASS' if not any(r['v2'] for r in ab) and not bad else 'FAIL'}")
