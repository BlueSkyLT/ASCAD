"""Attack eval: p_S → existing full_ranks / guessing_entropy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ASCAD_test_models import full_ranks, guessing_entropy

from .data_v1 import infer_simulated_key, load_ascadv1
from .model import build_poi_searcher


@torch.no_grad()
def predict_p_s(model, x, batch_size: int = 256, device: str = "cpu") -> np.ndarray:
    model.eval()
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(np.asarray(x, dtype=np.float32))
    x = x.float()
    outs = []
    for start in range(0, x.shape[0], batch_size):
        xb = x[start : start + batch_size].to(device)
        out = model(xb)
        outs.append(out["p_s"].detach().cpu().numpy())
    return np.concatenate(outs, axis=0)


def evaluate_sbox_ge(
    p_s,
    metadata,
    num_traces: int,
    rank_step: int = 10,
    target_byte: int = 2,
    n_attacks: int = 100,
    ge_seed: int = 12345,
    simulated_key: int = 0,
    x_attack=None,
) -> dict:
    p_s = np.asarray(p_s[:num_traces], dtype=np.float64)
    dummy = x_attack if x_attack is not None else p_s
    ranks = full_ranks(
        p_s, dummy, metadata, 0, num_traces, rank_step, target_byte, simulated_key
    )
    rank_final = int(ranks[-1][1]) if len(ranks) else None
    traces_to_rank0 = None
    for n_tr, rv in ranks:
        if int(rv) == 0:
            traces_to_rank0 = int(n_tr)
            break
    ge = guessing_entropy(
        predictions=p_s,
        metadata=metadata,
        num_traces=num_traces,
        rank_step=rank_step,
        target_byte=target_byte,
        n_attacks=n_attacks,
        rng=int(ge_seed),
        simulated_key=simulated_key,
    )
    return {
        "rank_final": rank_final,
        "traces_to_rank0": traces_to_rank0,
        "ge_final": ge["ge_final"],
        "ge_at": ge["ge_at"],
        "ge_traces_to_0": ge["ge_traces_to_0"],
        "mean_traces_to_rank0": ge["mean_traces_to_rank0"],
        "median_traces_to_rank0": ge["median_traces_to_rank0"],
        "frac_reached_rank0": ge["frac_reached_rank0"],
        "success_rate_final": ge["success_rate_final"],
        "ge_n_attacks": ge["n_attacks"],
        "target_byte": target_byte,
        "simulated_key": int(simulated_key),
    }


def load_trained(model_file: str, device: str = "cpu"):
    sidecar_path = model_file + ".config.json"
    sidecar = {}
    if Path(sidecar_path).exists():
        with open(sidecar_path, "r", encoding="utf-8") as f:
            sidecar = json.load(f)
    trace_len = int(sidecar.get("input_dim") or sidecar.get("poi_config", {}).get("trace_len") or 700)
    cfg = dict(sidecar.get("poi_config") or {})
    model = build_poi_searcher(trace_len, cfg)
    state = torch.load(model_file, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, sidecar


def eval_from_params(params: dict, device: Optional[str] = None) -> dict:
    model_file = params["model_file"]
    h5 = params["ascad_database"]
    num_traces = int(params.get("num_traces", 10000))
    rank_step = int(params.get("rank_step", 10))
    target_byte = int(params.get("target_byte", 2))
    n_attacks = int(params.get("ge_attacks", params.get("n_attacks", 100)))
    ge_seed = int(params.get("ge_seed", 12345))
    dev = device or params.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    model, sidecar = load_trained(model_file, device=dev)
    mask_idx = int(sidecar.get("mask_idx", (sidecar.get("poi_config") or {}).get("mask_idx", 0)))
    att = load_ascadv1(h5, split="attack", max_traces=num_traces, mask_idx=mask_idx, zscore=True)
    n_use = min(num_traces, att["n"])
    p_s = predict_p_s(model, att["x"][:n_use], batch_size=int(params.get("batch_size", 256)), device=dev)
    simulated_key = infer_simulated_key(h5, params.get("simulated_key", sidecar.get("simulated_key")))
    result = evaluate_sbox_ge(
        p_s,
        att["metadata"],
        num_traces=n_use,
        rank_step=rank_step,
        target_byte=target_byte,
        n_attacks=n_attacks,
        ge_seed=ge_seed,
        simulated_key=simulated_key,
        x_attack=att["x"][:n_use].numpy(),
    )
    out_json = params.get("save_json")
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Wrote {out_json}")
    print(
        f"rank_final={result['rank_final']} ge_final={result['ge_final']} "
        f"ttr0={result['traces_to_rank0']} mean_mtd={result['mean_traces_to_rank0']}"
    )
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description="Evaluate ASCADv1 POI searcher with GE")
    p.add_argument("param_file")
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)
    with open(args.param_file, "r", encoding="utf-8") as f:
        params = eval(f.read())
    if args.device:
        params["device"] = args.device
    eval_from_params(params)
    return 0


if __name__ == "__main__":
    sys.exit(main())
