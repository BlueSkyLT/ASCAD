# T-021: NVIDIA driver recovery after kernel upgrade

## Status

`done`

## Goal

Restore GPU access on this host so `nvidia-smi` and `torch.cuda.is_available()` work in conda env `py10`. Done when CUDA training can run on the RTX 5090.

## Context

- Hardware: NVIDIA GeForce RTX 5090, 32607 MiB VRAM, PCI `02:00.0` (device `2b85`)
- Driver package: `nvidia-driver-580-open` **580.142** (Ubuntu 24.04)
- CUDA reported by driver: **13.0**
- Python: conda env `py10`; PyTorch must see `cuda`
- Related tasks: unblocks T-008, T-016, T-017, T-020 GPU runs
- Constraints: never touch `*.h5` / `ASCAD_data`; long trainings still need explicit go-ahead (`.grok/rules/04-ask-before-long-runs.md`)

## Plan

- [x] (I:H U:H S:H) Confirm GPU is down: `nvidia-smi` + `torch.cuda.is_available()`
- [x] (I:H U:H S:H) Diagnose: PCI device present, kernel modules missing, DKMS built only for old kernel
- [x] (I:H U:H S:H) Rebuild/install NVIDIA DKMS for the running kernel (`dkms install nvidia/580.142 -k $(uname -r)`) — user ran with sudo
- [x] (I:H U:H S:H) Load modules (`modprobe nvidia` and related) — user ran
- [x] (I:H U:H S:H) Verify `nvidia-smi` and PyTorch CUDA in `py10`

## Decisions

- 2026-09-22: Rebuild DKMS for the **running** kernel (`7.0.0-30-generic`) rather than reboot into the old kernel (`6.17.0-23-generic`).
- 2026-09-22: Do not fall back to CPU for reprogram training / vocab-report while the driver is down (user request).

## Working notes / session log

### 2026-09-22 — driver restore

- **Symptom:** `nvidia-smi` failed with “couldn't communicate with the NVIDIA driver”; `conda run -n py10` → `torch.cuda.is_available() == False`.
- **Hardware was fine:** `lspci` showed the 5090 at `02:00.0`. `/proc/driver/nvidia` missing; `lsmod` had no `nvidia*`; `modinfo nvidia` → module not found.
- **Root cause:** kernel upgraded to **7.0.0-30-generic**. DKMS had NVIDIA **580.142** built only for **6.17.0-23-generic** (`built`, not installed on the new kernel).
- **Fix (user sudo):**
  1. `sudo dkms install nvidia/580.142 -k $(uname -r)` → status became `nvidia/580.142, 7.0.0-30-generic, x86_64: installed`
  2. `sudo modprobe nvidia` (and related modules as needed) — install alone does not load modules until reboot
- **Verified:**
  - `nvidia-smi`: Driver 580.142, CUDA 13.0, RTX 5090 idle (~28°C, 2 MiB / 32607 MiB)
  - `py10` PyTorch: `CUDA available: True`, device count 1, name `NVIDIA GeForce RTX 5090`
- Agent could not run `sudo` (no password TTY); user executed the privileged steps.

## Next actions

| Action | I | U | S | Notes |
|--------|---|---|---|-------|
| After a future kernel upgrade: `dkms status` then `sudo dkms install nvidia/580.142 -k $(uname -r)` and `sudo modprobe nvidia` (or reboot) | H | L | H | Same failure mode if DKMS autoinstall is skipped |
| Resume GPU work that was blocked (reprogram training / vocab-report) only with explicit go-ahead | H | M | L | T-016 / T-017 / T-020; ask before long runs |

### Priority snapshot

- **Do now:** none for this task (driver restored)
- **Before next meeting:** none
- **Deep work (park):** none

## Related

- Task ids: T-008 (GPU util/memory; assumes CUDA works), T-016, T-017, T-020
- `docs/lab-log/todo.md` items: none open; restore logged under Done (recent)
- External: Ubuntu DKMS + `nvidia-driver-580-open`
