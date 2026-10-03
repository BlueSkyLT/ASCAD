# Task memory

This directory stores **per-task** plans, decisions, and session summaries so work survives abandoned chats and model changes.

| File | Role |
|---|---|
| `INDEX.md` | Registry of all tasks; start here |
| `_template.md` | Template for new tasks |
| `TNNN-short-slug.md` | One file per task (e.g. `T001-dataset-visualization.md`) |

Grok project rules (`AGENTS.md` and `.grok/rules/02-task-memory.md`) require agents to read and update these files during work.

## Major research tasks (recovered 2026-07-17)

| ID | Topic | Status |
|----|--------|--------|
| T-001 | Dataset visualization & understanding | in_progress |
| T-002 | TensorFlow → PyTorch migration | **done** |
| T-003 | GNN models | in_progress |
| T-004 | TimesNet models | in_progress |
| T-005 | SCA metrics & result analysis | in_progress |
| T-006 | CNN/MLP baselines | in_progress |
| T-007 | Lab log — progress & meetings | in_progress |
| T-008 | GPU training efficiency (util/memory) | in_progress |

Stable topic references (datasets, metrics, notation, paper lists): **`docs/notes/`** (catalog: [`../notes/README.md`](../notes/README.md)).

Human-facing progress / meeting notes (not technical experiment dumps): **`docs/lab-log/`**

| Lab-log file | Role |
|--------------|------|
| [`todo.md`](../lab-log/todo.md) | **Prioritized todos** (Importance / Urgency / Simpleness) |
| `progress.md` | Narrative progress |
| `notes.md` | Stable meeting background |
| `meetings/` | Per-meeting briefs |

Triage rule: `.grok/rules/03-priority-triage.md`. Task **Next actions** should tag I/U/S; meeting-facing items also go in `todo.md`.

Old Grok session transcripts live under `~/.grok/sessions/` (not always openable in the UI when models retire). Durable conclusions should be copied here.

## Quick start for humans

1. Skim `docs/lab-log/todo.md` for what to do next (I/U/S).
2. Skim `INDEX.md` for the active or relevant technical task.
3. Open that task file for plan / next actions / session log.
4. Or ask Grok: “register a new task …” / “continue T-00x” / “retriage todo.md”.
