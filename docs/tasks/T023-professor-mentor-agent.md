# T-023: Professor Mentor Agent Definition & Setup

## Status

`done`

## Goal

Define and configure a specialized research professor/mentor agent (`ascad_professor`) who explains side-channel analysis, AES cryptography, math, and papers to a high schooler. The agent cannot write executable code, but can search online, explore the codebase, and write/edit human-readable `.md` reports, and is registered for tool invocation by other agents.

## Context

- Related files:
  - Subagent name: `ascad_professor` (registered via `define_subagent`)
  - Config: [.grok/agents/ascad-professor.md](file:///home/lantian/ASCAD/.grok/agents/ascad-professor.md)
  - Notes catalog: [docs/notes/README.md](file:///home/lantian/ASCAD/docs/notes/README.md)
- Constraints:
  - Strictly no code writing (no `.py`, `.sh`, `.c`, etc.).
  - Reads/explores codebase freely.
  - Searches web and consults papers/math.
  - Can write/edit `.md` reports and notes.
  - High-school pedagogical style: intuitive analogies, first-principles explanations, clear math demystification.
  - Accessible via `invoke_subagent` for multi-agent workflows.

## Plan

- [x] (I:H U:H S:H) Define `ascad_professor` runtime subagent via `define_subagent`.
- [x] (I:H U:H S:H) Persist configuration in `.grok/agents/ascad-professor.md`.
- [x] (I:M U:M S:H) Record task registration in `docs/tasks/INDEX.md` and `docs/tasks/T023-professor-mentor-agent.md`.

## Decisions

- 2026-09-24: Registered `ascad_professor` with `enable_write_tools=true` so it can draft/edit Markdown reports, while setting hard prompt boundaries prohibiting executable code generation or modification.

## Working notes / session log

### 2026-09-24 — Initial agent creation

- Defined `ascad_professor` subagent with comprehensive pedagogical system prompt: high school audience level, everyday analogies, math explained from first principles, strict prohibition on writing executable code, authorization to write/edit `.md` reports, full read access for code/traces, and web search access.
- Registered via `define_subagent` so other agent tools can invoke it using `invoke_subagent`.
- Updated disk definition in `.grok/agents/ascad-professor.md`.

## Next actions

| Action | I | U | S | Notes |
|--------|---|---|---|-------|
| Test invocation via `invoke_subagent` if requested | M | L | H | Can be summoned whenever explanations/reports are needed |
