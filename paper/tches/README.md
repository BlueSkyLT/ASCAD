# CompactMultiResNet — TCHES draft (T-039)

- Class: `iacrtrans.cls` v0.94 (CC0, github.com/Cryptosaurus/iacrtrans), `journal=tches,submission`.
  TCHES's LaTeX page still names iacrtrans; CTAN's newer `iacrj` claims to replace it — check the
  call for papers before submitting.
- Tables are generated, never typed: `scripts/make_tables.py` reads `GPAM/artifacts/crop34/*.json`
  (compact_verify decodes, raw π routing) and `ge_ms23_style.json` (500-attack GE).
- Build (Tectonic in its own conda env `tex`; no system TeX on this machine):

```bash
/home/lantian/miniconda3/envs/py10/bin/python paper/tches/scripts/make_tables.py
cd paper/tches && /home/lantian/miniconda3/envs/tex/bin/tectonic -X compile main.tex
```

- `\todo{}` marks (red) are open items; claim set and evidence: `docs/tasks/T039-reports/tches-claim-review.md`.

## Out of scope (user, 2026-11-03)

`GPAM-c3/` and `aes_fpga/` are separate lab-mate projects. Do not cite them, import
them, or add their datasets, numbers, or figures to this draft. Rule:
`.grok/rules/06-side-projects.md`.
