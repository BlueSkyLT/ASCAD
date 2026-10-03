# T-015: ASCADv2 3-window leakage localizer

## Status

`planned`

## Goal

Implement the ASCADv2 affine three-window localizer plan using the repo’s existing `leakage_localizer/` conventions, keeping the ASCADv1 2O path untouched and adding v2-specific affine moment logic, model, loader, and validation.

## Scope

- Add `m3_table` and affine 3-way moment helpers
- Add `loss_3o` / `loss_head` while preserving `loss_2o`
- Add `ThreeOrderWindows` and v2 loader
- Keep v1 model and v2 work isolated from existing T-014 code paths
- Register the plan in `docs/superpowers/plans/2026-08-21-ascadv2-3o-localizer.md`

## Current progress

- [x] Plan file created and staged under `docs/superpowers/plans/2026-08-21-ascadv2-3o-localizer.md`
- [x] Affine moment computation implemented in `leakage_localizer/affine_moments.py`
- [x] v2 three-window model added in `leakage_localizer/model.py`
- [x] 3O loss added in `leakage_localizer/losses.py`
- [x] v2 loader scaffold added in `leakage_localizer/data_v2.py`
- [x] Regression checks for the affine table and shape conditions pass in local Python verification

## Next actions

| Action | I | U | S | Notes |
|--------|---|---|---|-------|
| Oracle 3O on extracted 15k | H | M | H | Do not train windows until the physics gate is checked |
| Known-π / loss-3 smoke | H | M | M | Requires explicit GPU go-ahead |
| Finalized T-015 task log | M | L | H | Keep short and evidence-based |

## Work log

### 2026-08-26

- Implemented the v2 affine moment table using `ASCAD_generate.multGF256` to match the repo’s GF(256) arithmetic.
- Confirmed the expected values: `m3[32] ≈ 0.031373` and the table matches the documented weak affine-moment physics.
- Added a three-window proposer with alpha/beta first-band and C second-band constraints.
- Kept the legacy v1 `loss_2o` and `TwoOrderWindows` interfaces intact for T-014 compatibility.
- Registered the final plan file and task memory entry as required by the project instructions.
