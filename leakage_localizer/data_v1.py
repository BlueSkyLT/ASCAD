"""Read-only ASCADv1 loader (fixed or variable key). Never writes HDF5."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np
import torch

MASK_IDX = 0


def sbox_labels(plaintext: np.ndarray, key: np.ndarray, target_byte: int) -> np.ndarray:
    """S = Sbox(p[i] ⊕ k[i]) for one AES byte. Does not use HDF5 `labels`."""
    from ASCAD_generate import AES_Sbox

    i = int(target_byte)
    p = np.asarray(plaintext)[:, i].astype(np.uint8)
    k = np.asarray(key)[:, i].astype(np.uint8)
    return np.asarray(AES_Sbox[np.bitwise_xor(p, k)], dtype=np.int64)


def infer_simulated_key(h5_path: Union[str, Path], simulated_key: Optional[int] = None) -> int:
    if simulated_key is not None:
        return int(simulated_key)
    p = str(h5_path).replace("\\", "/")
    return 1 if "variable_key" in p else 0


def zscore_per_trace(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    return (x - mean) / (std + eps)


def share_labels(s: np.ndarray, masks: np.ndarray, mask_idx: int = MASK_IDX):
    s = np.asarray(s).ravel().astype(np.int64)
    r = np.asarray(masks)[:, int(mask_idx)].astype(np.int64)
    c = np.bitwise_xor(s.astype(np.uint8), r.astype(np.uint8)).astype(np.int64)
    return s, r, c


def load_ascadv1(
    path: Union[str, Path],
    split: str = "profiling",
    max_traces: Optional[int] = None,
    mask_idx: int = MASK_IDX,
    zscore: bool = True,
    target_byte: int = 2,
) -> dict:
    """Load one split. split in {profiling, attack}.

    HDF5 `labels` are S-box of **byte 2**. Other bytes are rebuilt from
    plaintext and key (same formula as `ASCAD_generate.labelize`).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    key = "Profiling_traces" if split == "profiling" else "Attack_traces"
    with h5py.File(path, "r") as f:
        n_all = f[f"{key}/traces"].shape[0]
        n = n_all if max_traces is None else min(int(max_traces), n_all)
        traces = np.array(f[f"{key}/traces"][:n], dtype=np.float32)
        s = np.array(f[f"{key}/labels"][:n]).ravel()
        meta = f[f"{key}/metadata"][:n]
        masks = np.array(meta["masks"])
        plaintext = np.array(meta["plaintext"])
        key_bytes = np.array(meta["key"])
        if "desync" in meta.dtype.names:
            desync = np.array(meta["desync"]).ravel().astype(np.int64)
        else:
            desync = np.zeros(n, dtype=np.int64)
    target_byte = int(target_byte)
    s = sbox_labels(plaintext, key_bytes, target_byte)
    # r/C via MASK_IDX=0 is only meaningful for the official byte-2 share.
    _, r, c = share_labels(s, masks, mask_idx=mask_idx)
    x = zscore_per_trace(traces) if zscore else traces
    return {
        "x": torch.from_numpy(np.ascontiguousarray(x)),
        "s": torch.from_numpy(s),
        "r": torch.from_numpy(r),
        "c": torch.from_numpy(c),
        "plaintext": plaintext,
        "key": key_bytes,
        "desync": desync,
        "metadata": meta,
        "trace_len": int(x.shape[1]),
        "n": int(x.shape[0]),
        "path": str(path),
        "split": split,
        "mask_idx": int(mask_idx),
        "target_byte": int(target_byte),
    }


def with_target_byte(data: dict, target_byte: int) -> dict:
    """Reuse loaded traces; replace S* (and dummy r/C) for another AES byte."""
    out = dict(data)
    s = sbox_labels(data["plaintext"], data["key"], target_byte)
    masks = np.array(data["metadata"]["masks"])
    _, r, c = share_labels(s, masks, mask_idx=int(data.get("mask_idx", MASK_IDX)))
    out["s"] = torch.from_numpy(s)
    out["r"] = torch.from_numpy(r)
    out["c"] = torch.from_numpy(c)
    out["target_byte"] = int(target_byte)
    return out


def snr_init_mus(x: torch.Tensor, r: torch.Tensor, c: torch.Tensor):
    """Peak times of SNR(r) and SNR(C) on a profiling-search split."""
    from utils.snr import snr_vector

    xa = np.asarray(x.detach().cpu().numpy(), dtype=np.float32)
    ra = np.asarray(r.detach().cpu().numpy()).ravel()
    ca = np.asarray(c.detach().cpu().numpy()).ravel()
    snr_r = snr_vector(xa, ra)
    snr_c = snr_vector(xa, ca)
    return int(snr_r.argmax()), int(snr_c.argmax()), float(snr_r.max()), float(snr_c.max())
