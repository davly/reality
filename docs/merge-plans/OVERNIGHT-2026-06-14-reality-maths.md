# Merge Plan — reality-maths session (foundation/reality)

**For:** the live session building causal-identification + conformal + robust-stats primitives
into `foundation/reality` (8 branches dated 2026-06-14, last commit ~3 min before this scan).
**Status when written:** LIVE — do not assume this is the final shape; re-derive before merging.
**Verified (read-only, against `master` = `eaed6c8`):** all 7 distinct topic branches merge with
**ZERO conflict markers** (`git merge-tree`). Stale-base note below.

## Branch topology (re-derived from disk)

The four **graph** branches are a **linear stack**, not siblings — each contains the prior's
commits plus its own:

```
master (eaed6c8)
  └─ id-algorithm   (+1)  graph/idalgorithm.go            — causal-effect identification (ID)
       └─ idc        (+2)  + graph/idc.go                  — conditional ID (IDC)
            └─ hedge  (+3)  + graph/hedge_test.go           — hedge-witness (non-identifiability)
                 └─ scm (+4) + graph/scm.go                 — discrete-SCM evaluator + caps
                     ≡ idfuzz-2026-06-14 (same SHA 110ea9a)
```

Three **independent** new-package branches (no overlap with graph or each other):

```
aci          (+1)  prob/conformal/aci.go     — Adaptive Conformal Inference (Gibbs-Candès)
edetector    (+1)  changepoint/edetector.go  — anytime-valid e-detectors (Shin-Ramdas)
robust-mean  (+1)  prob/robust.go            — sub-Gaussian robust location (median-of-means etc.)
```

## ⚠️ The one trap: do NOT merge the graph branches separately

`id-algorithm`, `idc`, `hedge-witness`, `scm-validate` **all modify `graph/idalgorithm.go`** from
the same old base. Merging more than one of them as independent PRs will 3-way-conflict on that
file after the first lands.

**Recommended graph merge = merge `scm-validate` (≡ `idfuzz`) ONLY.** It is the cumulative tip and
already contains id + idc + hedge + scm. The other three graph branches are then redundant ancestors —
delete them after, do not merge them.

## Recommended merge order (all conflict-free vs current master)

1. `claude/scm-validate-2026-06-14`  → master   (brings the entire graph ID suite in one merge)
2. `claude/aci-2026-06-14`           → master   (independent)
3. `claude/edetector-2026-06-14`     → master   (independent)
4. `claude/robust-mean-2026-06-14`   → master   (independent)
5. delete redundant: `id-algorithm`, `idc`, `hedge-witness`, `idfuzz` (== scm)

After each merge, run the **full** suite (`go test ./...` + `-race`) — these are new pkgs with
golden/mutation tests; full-package green is the receipt.

## Stale-base note (why a 2-dot diff looks alarming but isn't)

Every branch forked **before** `moments/` landed on master (master added it at `4422c06`/`eaed6c8`).
A `git diff master..<branch>` therefore shows `moments/moments.go` as *deleted* — an artifact of
the 2-dot diff, **not** a deletion the branch makes. A real 3-way `git merge` keeps master's
`moments/` (merge-base lacks it, only master added it). Confirmed: merge-tree shows 0 conflicts.
If you'd rather remove the artifact entirely, rebase each branch onto current master first.

## If you prefer PR-per-primitive over the cumulative-tip merge

Then merge the graph stack **in strict order with a rebase between each**
(id → rebase idc onto new master → merge → rebase hedge → merge → rebase scm → merge). More work,
same result. The cumulative-tip merge (above) is strictly simpler and is recommended.
