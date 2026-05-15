# Overnight 400-Agent Reality Review

**Started:** 2026-05-06
**Goal:** Identify the most advanced uses of math possible, math we're missing, math we can deliver better, and math that could combine in interesting ways.

## Files

- `MASTER_PLAN.md` — 400 distinct review angles, numbered 001–400
- `PROGRESS.md` — append-only completion log (one line per finished agent)
- `agents/NNN-slug.md` — full review output from agent NNN
- `SYNTHESIS.md` — running synthesis (updated every ~25 agents)

## Blocks

- A (001–150): Per-package depth, 5 angles × 30 packages
- B (151–200): Cross-package synergies
- C (201–300): Cutting-edge math not yet present
- D (301–350): Specific deep dives
- E (351–380): Internet research themes
- F (381–400): Meta / cross-cutting

## Running

Agents fire one at a time. Each agent reads the relevant package code, may search the web, writes to `agents/NNN-slug.md`, and appends a line to `PROGRESS.md`.

## Output schema for each agent file

```
# NNN — <topic>

## Headline
<one-sentence finding>

## Findings
- <bullet>
- <bullet>

## Concrete recommendations
1. <recommendation>
2. <recommendation>

## Sources
- <web sources or repo files>
```
