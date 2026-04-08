# reality — Architecture

*Session 25 auto-generated baseline (2026-04-08). To be enriched over time as architectural decisions are made.*

## Purpose

Universal truth encoded in code. Pure math, physics, constants. Zero dependencies. MIT open source.

## Language + build

| Field | Value |
|---|---|
| **Primary language** | Go |
| **Build tool** | go modules |
| **Dependencies** | go.mod |
| **Layer** | foundation service |

## Top-level structure

```
acoustics
calculus
chaos
color
combinatorics
compression
conduit
constants
control
crypto
em
fluids
gametheory
geometry
graph
linalg
optim
orbital
physics
prob
```

## Ecosystem role

As a foundation service, reality is a Tier 0 primitive that flagships and other services depend on — reality, knowledge, or aicore. These are the load-bearing layers per David's 'reality > knowledge > lore' priority.

See `architecture/FLAGSHIP_REGISTRY.md` in the LimitlessGodfather repo for the canonical role description (if this repo is registered).

## Standards compliance checklist

Per the Wave 8.1 Synthesis (`reviews/session_24_adversarial/WAVE_8_1_SYNTHESIS.md`) §4 — universal findings across all 16 delve embeds:

- [ ] **Five-noun skeleton** (Query · Corpus · dig/walk · Result · EscapeReason)
- [ ] **FNV-1a 64-bit situation hash** over sorted dimensions (canonical vectors from `architecture/fnv1a_canonical_vectors.json`)
- [ ] **Three-way verdict** (Dominated/Uncertain/Refuses or equivalent)
- [ ] **Closed escape-reason enum** (7-10 variants typical)
- [ ] **Jeffreys (0.5, 0.5) quality-weighted dominance**
- [ ] **Fail-silent fire-and-forget Conduit emit** with 100ms timeout
- [ ] **Deterministic ordering** via sorting at every collection-return seam
- [ ] **Constructor-invariant refusal** before work begins
- [ ] **Structured escape** (no exceptions, escape as return value)

Session 25 baseline: most repos have not been audited against this checklist yet. The checklist is tracked here so each repo can verify compliance over time.

## Cross-references

- **CONTEXT.md** — current runtime state (file counts, git info, structure)
- **CLAUDE.md** — session handover notes (if present)
- **docs/** — planning, design, personality, dreams (if present)
- **LimitlessGodfather repo** — canonical ecosystem architecture, cross-pollination plans, session history

## Session 25 audit metadata

- Auto-generated: 2026-04-08
- Generator: godfather session 25 template-driven repo introspection
- This file is a BASELINE. It should be replaced with a substantial architecture document once the repo's architecture matures.
