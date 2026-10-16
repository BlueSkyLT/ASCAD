"""Train the ASCADv1 POI searcher (param-file CLI, same style as ASCAD_train_models)."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data_v1 import infer_simulated_key, load_ascadv1, snr_init_mus
from .losses import total_loss
from .model import build_poi_searcher
from .synthetic import make_synthetic
from .windows import default_bands, sigma_max_at_epoch


def read_param_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return eval(f.read())


def _device(name: Optional[str] = None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _poi_config(params: dict, trace_len: int) -> dict:
    cfg = dict(params.get("poi_config") or params.get("poi_searcher_config") or {})
    cfg.setdefault("variant", "boolean")
    cfg.setdefault("mask_idx", 0)
    cfg.setdefault("sigma_min", 1.0)
    r_band, c_band, mu_r, mu_c = default_bands(trace_len)
    cfg.setdefault("r_band", list(r_band))
    cfg.setdefault("c_band", list(c_band))
    cfg.setdefault("init_mu_r", mu_r)
    cfg.setdefault("init_mu_c", mu_c)
    cfg.setdefault("init_sigma", 15.0)
    cfg.setdefault("lambda_ind", 1.0)
    cfg.setdefault("lambda_cons", 1.0)
    cfg.setdefault("lambda_ent", 1e-3)
    cfg.setdefault("lambda_rep", 0.1)
    cfg.setdefault("tau", 40.0)
    cfg.setdefault("lr_psi", 1e-3)
    cfg.setdefault("lr_phi", 1e-3)
    cfg.setdefault("warmup_epochs", 0)
    # SNR peaks are the evaluation target, not h_ψ's starting output.
    cfg.setdefault("snr_init", False)
    cfg.setdefault("lambda_hw", 0.0)
    cfg.setdefault("proposer", "attention")
    return cfg


def _build_optimizer(model, cfg: dict):
    psi, phi = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("proposer") or name.startswith("raw_"):
            psi.append(p)
        else:
            phi.append(p)
    groups = []
    if psi:
        groups.append({"params": psi, "lr": float(cfg["lr_psi"])})
    if phi:
        groups.append({"params": phi, "lr": float(cfg["lr_phi"])})
    if not groups:
        groups = [{"params": list(model.parameters()), "lr": float(cfg["lr_phi"])}]
    return torch.optim.Adam(groups)


def _set_requires_grad(model, train_proposer: bool, train_heads: bool):
    for name, p in model.named_parameters():
        if name.startswith("proposer") or name.startswith("raw_"):
            p.requires_grad = train_proposer
        else:
            p.requires_grad = train_heads


def run_loader_epoch(model, loader, device, optimizer, cfg, train: bool):
    if train:
        model.train()
    else:
        model.eval()
    total = 0.0
    n = 0
    parts_acc = {}
    mu_r_sum = 0.0
    mu_c_sum = 0.0
    sig_r_sum = 0.0
    sig_c_sum = 0.0
    for batch in loader:
        xb, sb, rb, cb = [t.to(device) for t in batch]
        if train:
            optimizer.zero_grad(set_to_none=True)
        out = model(xb)
        if "mu_r" in out:
            mu_r_sum += float(out["mu_r"].detach().mean()) * xb.shape[0]
            mu_c_sum += float(out["mu_c"].detach().mean()) * xb.shape[0]
        if "sigma_r" in out:
            sig_r_sum += float(out["sigma_r"].detach().mean()) * xb.shape[0]
            sig_c_sum += float(out["sigma_c"].detach().mean()) * xb.shape[0]
        loss, parts = total_loss(
            out,
            sb,
            r=rb,
            c=cb,
            lambda_ind=float(cfg["lambda_ind"]),
            lambda_cons=float(cfg["lambda_cons"]),
            lambda_ent=float(cfg["lambda_ent"]),
            lambda_rep=float(cfg["lambda_rep"]),
            lambda_hw=float(cfg.get("lambda_hw", 1.0)),
            tau=float(cfg["tau"]),
            x=xb,
        )
        if train:
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue
            loss.backward()
            optimizer.step()
        bs = xb.shape[0]
        total += float(loss.detach()) * bs
        n += bs
        for k, v in parts.items():
            if torch.is_tensor(v):
                parts_acc[k] = parts_acc.get(k, 0.0) + float(v.detach()) * bs
    if n == 0:
        return float("nan"), {}
    stats = {k: v / n for k, v in parts_acc.items()}
    if mu_r_sum:
        stats["mu_r"] = mu_r_sum / n
        stats["mu_c"] = mu_c_sum / n
    if sig_r_sum:
        stats["sigma_r"] = sig_r_sum / n
        stats["sigma_c"] = sig_c_sum / n
    return total / n, stats


def fit_arrays(
    model,
    x,
    s,
    r,
    c,
    cfg: dict,
    epochs: int,
    batch_size: int,
    device: torch.device,
    validation_split: float = 0.0,
):
    n = x.shape[0]
    idx = np.arange(n)
    if validation_split and validation_split > 0:
        rng = np.random.default_rng(0)
        rng.shuffle(idx)
        n_val = int(n * validation_split)
        val_idx, tr_idx = idx[:n_val], idx[n_val:]
    else:
        tr_idx, val_idx = idx, np.array([], dtype=int)

    def _ds(ii):
        return TensorDataset(x[ii], s[ii], r[ii], c[ii])

    train_loader = DataLoader(_ds(tr_idx), batch_size=batch_size, shuffle=True)
    val_loader = (
        DataLoader(_ds(val_idx), batch_size=batch_size, shuffle=False)
        if len(val_idx)
        else None
    )
    optimizer = _build_optimizer(model, cfg)
    warmup = int(cfg.get("warmup_epochs", 0) or 0)
    history = {"loss": [], "val_loss": []}
    last_metric = float("inf")
    for epoch in range(epochs):
        cap = sigma_max_at_epoch(
            epoch, epochs, float(cfg["init_sigma"]), float(cfg["sigma_min"])
        )
        if hasattr(model, "set_sigma_max"):
            model.set_sigma_max(cap)
        if hasattr(getattr(model, "proposer", None), "temperature") and epochs > 1:
            # Sharpen attention over time so mass can collapse onto the peak.
            t0, t1 = 1.0, 0.25
            model.proposer.temperature = t0 + (t1 - t0) * (float(epoch) / float(epochs - 1))
        if cfg.get("freeze_windows"):
            _set_requires_grad(model, False, True)
        elif warmup and epoch < warmup:
            _set_requires_grad(model, True, False)
        else:
            _set_requires_grad(model, True, True)
        tr, tr_parts = run_loader_epoch(model, train_loader, device, optimizer, cfg, True)
        history["loss"].append(tr)
        mu_msg = f"  sigma_max={cap:.2f}"
        if "mu_r" in tr_parts:
            mu_msg += f"  mu_r={tr_parts['mu_r']:.1f} mu_c={tr_parts['mu_c']:.1f}"
            if "sigma_r" in tr_parts:
                mu_msg += f"  sig_r={tr_parts['sigma_r']:.2f} sig_c={tr_parts['sigma_c']:.2f}"
            if "ce_r" in tr_parts:
                mu_msg += (
                    f"  ce_r={tr_parts['ce_r']:.3f} ce_c={tr_parts['ce_c']:.3f}"
                    f"  ce_s={tr_parts['ce_s']:.3f}"
                )
            if "hw" in tr_parts:
                mu_msg += f"  hw={tr_parts['hw']:.3f}"
        if val_loader is not None:
            with torch.no_grad():
                va, _ = run_loader_epoch(model, val_loader, device, optimizer, cfg, False)
            history["val_loss"].append(va)
            metric = va
            print(f"Epoch {epoch + 1}/{epochs} - loss: {tr:.6f} - val_loss: {va:.6f}{mu_msg}")
        else:
            metric = tr
            print(f"Epoch {epoch + 1}/{epochs} - loss: {tr:.6f}{mu_msg}")
        last_metric = metric
    return history, last_metric


def fit_synthetic(
    n: int = 2048,
    epochs: int = 80,
    device: str = "cpu",
    T: int = 400,
    mu_r: float = 80.0,
    mu_c: float = 280.0,
    jitter: float = 40.0,
    seed: int = 0,
    variant: str = "boolean",
):
    data = make_synthetic(n=n, T=T, mu_r=mu_r, mu_c=mu_c, jitter=jitter, seed=seed)
    cfg = _poi_config(
        {
            "poi_config": {
                "variant": variant,
                "snr_init": False,
                "r_band": [20.0, 160.0],
                "c_band": [200.0, 360.0],
                "init_mu_r": mu_r,
                "init_mu_c": mu_c,
                "sigma_min": 8.0,
                "init_sigma": 30.0,
                "lr_psi": 2e-2,
                "lr_phi": 1e-3,
            }
        },
        T,
    )
    dev = _device(device)
    model = build_poi_searcher(T, cfg).to(dev)
    hist, _ = fit_arrays(
        model, data["x"], data["s"], data["r"], data["c"], cfg, epochs, 256, dev, 0.0
    )
    model.eval()
    with torch.no_grad():
        out = model(data["x"].to(dev))
        pred_r = out["mu_r"].cpu()
        pred_c = out["mu_c"].cpu()
        mae_r = float((pred_r - data["mu_r"]).abs().mean())
        mae_c = float((pred_c - data["mu_c"]).abs().mean())
        ce_s = float(-(out["p_s"].cpu().clamp_min(1e-12).log()[torch.arange(n), data["s"]]).mean())
    print(f"S0 mae_mu_r={mae_r:.2f} mae_mu_c={mae_c:.2f} ce_s={ce_s:.3f} last_loss={hist['loss'][-1]:.3f}")
    return {
        "model": model,
        "history": hist,
        "mae_r": mae_r,
        "mae_c": mae_c,
        "ce_s": ce_s,
        "data": data,
    }


def train_from_params(params: dict, device: Optional[str] = None):
    h5 = params["ascad_database"]
    save_path = params.get("training_model") or params.get("model_file")
    if not save_path:
        raise ValueError("params need training_model (or model_file)")
    epochs = int(params.get("epochs", 15))
    batch_size = int(params.get("batch_size", 256))
    train_len = int(params.get("train_len", 0) or 0)
    validation_split = float(params.get("validation_split", 0.1) or 0.0)
    max_traces = train_len if train_len > 0 else None

    probe = load_ascadv1(h5, split="profiling", max_traces=8, zscore=False)
    trace_len = probe["trace_len"]
    cfg = _poi_config(params, trace_len)
    mask_idx = int(cfg.get("mask_idx", 0))

    prof = load_ascadv1(
        h5, split="profiling", max_traces=max_traces, mask_idx=mask_idx, zscore=True
    )
    if cfg.get("snr_init") and prof["n"] >= 256:
        n_search = min(10000, prof["n"])
        mu_r, mu_c, sr, sc = snr_init_mus(prof["x"][:n_search], prof["r"][:n_search], prof["c"][:n_search])
        print(f"SNR *targets* (not used as h_ψ init): mu_r={mu_r} (snr={sr:.3f}) mu_c={mu_c} (snr={sc:.3f})")
        # Do not copy these into init_mu: the searcher starts at the band front.

    dev = _device(device or params.get("device"))
    model = build_poi_searcher(trace_len, cfg).to(dev)
    print(
        f"poi_searcher variant={cfg.get('variant')} T={trace_len} N={prof['n']} "
        f"device={dev} bands r={cfg['r_band']} C={cfg['c_band']}"
    )
    hist, best = fit_arrays(
        model,
        prof["x"],
        prof["s"],
        prof["r"],
        prof["c"],
        cfg,
        epochs,
        batch_size,
        dev,
        validation_split,
    )
    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    model.eval()
    with torch.no_grad():
        n_diag = min(512, prof["n"])
        dout = model(prof["x"][:n_diag].to(dev))
        peak = {
            "mu_r_mean": float(dout["mu_r"].mean()),
            "mu_r_std": float(dout["mu_r"].std()),
            "mu_c_mean": float(dout["mu_c"].mean()),
            "mu_c_std": float(dout["mu_c"].std()),
            "sigma_r_mean": float(dout["sigma_r"].mean()),
            "sigma_c_mean": float(dout["sigma_c"].mean()),
            "target_mu_r": 156.0 * (trace_len / 700.0),
            "target_mu_c": 517.0 * (trace_len / 700.0),
        }
        peak["err_r"] = abs(peak["mu_r_mean"] - peak["target_mu_r"])
        peak["err_c"] = abs(peak["mu_c_mean"] - peak["target_mu_c"])
    print(
        f"Window diagnostic: mu_r={peak['mu_r_mean']:.1f}±{peak['mu_r_std']:.1f} "
        f"(target {peak['target_mu_r']:.1f}, |err|={peak['err_r']:.1f})  "
        f"mu_c={peak['mu_c_mean']:.1f}±{peak['mu_c_std']:.1f} "
        f"(target {peak['target_mu_c']:.1f}, |err|={peak['err_c']:.1f})  "
        f"sig_r={peak['sigma_r_mean']:.2f} sig_c={peak['sigma_c_mean']:.2f}"
    )
    sidecar = {
        "kind": "best",
        "network_type": "poi_searcher",
        "ascad_database": h5,
        "input_dim": trace_len,
        "poi_config": cfg,
        "mask_idx": mask_idx,
        "metric": best if math.isfinite(best) else None,
        "epochs": epochs,
        "batch_size": batch_size,
        "train_len": train_len,
        "history": {k: [float(v) for v in vs] for k, vs in hist.items()},
        "simulated_key": infer_simulated_key(h5, params.get("simulated_key")),
        "window_diagnostic": peak,
    }
    with open(save_path + ".config.json", "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)
    print(f"Saved {save_path}")
    return sidecar


def main(argv=None):
    p = argparse.ArgumentParser(description="Train ASCADv1 POI searcher")
    p.add_argument("param_file", nargs="?", help="eval()-style param file")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--n", type=int, default=2048)
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)
    if args.synthetic:
        fit_synthetic(n=args.n, epochs=args.epochs or 80, device=args.device or "cpu")
        return 0
    if not args.param_file:
        p.error("param_file is required unless --synthetic")
    params = read_param_file(args.param_file)
    if args.epochs is not None:
        params["epochs"] = args.epochs
    if args.device:
        params["device"] = args.device
    train_from_params(params)
    return 0


if __name__ == "__main__":
    sys.exit(main())
