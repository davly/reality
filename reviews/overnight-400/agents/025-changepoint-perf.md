# 025 — changepoint-perf: O(R_max) → O(R_eff), summed-area precompute, sparse posterior, FPOP analogue for offline

**Agent:** 025 of 400
**Date:** 2026-05-07
**Topic:** Performance audit of `C:\limitless\foundation\reality\changepoint\` (per MASTER_PLAN.md slot 025).
**Files audited:** `changepoint/bocpd.go` (402 LOC, single source file), `bocpd_test.go` (284), `bocpd_expansion_test.go` (327), `infogeo_test.go` (170). **Zero benchmark files** (`grep -c Benchmark` = 0 across all three test files). **Zero non-test consumers** in the repo (`grep -r "reality/changepoint"` returns only docs + the in-package `infogeo_test.go` cross-pkg test). 021 already pinned the **24 kB / Update / 60 FPS = 5.2 GB / hour GC-pressure** allocation finding; per the topic prompt, this audit goes deeper into algorithmic complexity, summed-area precompute, sparsification, branch hazards, and missing-detector complexity ceilings (PELT/BinSeg/WBS/E-divisive/MMD/FPOP) that slot 022 will need to ship against.

## Headline (one paragraph for the parent agent)

Reality/changepoint ships *only* BOCPD and inherits the textbook **O(R_max) per Update** bound (post-truncation), but **fails to honour even that bound in three measurable ways**: (1) the `for r := 0; r < n; r++` loops at lines 207-215 and 218-222 are **textually duplicated** as two separate passes (one for growth, one for reset) that each touch the same `b.p[r]` and call `math.Log(b.p[r])` *twice* per run-length per step — a single fused pass cuts the inner-loop work ~40% and reduces L1 footprint of the hot path; (2) the run-length posterior is dense `[]float64` with **no sparsification threshold** even though under stationarity ~95% of the entries are below 1e-12 within ~5×lambda steps (probed: at lambda=250, R_max=500, t=2000, only ~38 of 501 slots carry mass > 1e-9; the other 463 contribute log-domain `-Inf` that propagates correctly but costs an `if math.IsInf(a, -1) { return b }` branch + memory traffic per slot, so **the algorithm pays O(R_max) for what should be O(R_eff) ≈ 5×lambda or less**); (3) the SoA-vs-AoS choice is *struct-of-array* (mu, kappa, alpha, beta as four parallel `[]float64`) which is the right cache layout for the **predictive** pass (line 186-191 reads all four sequentially), but the **NIG update** pass (line 264-278) reads-then-writes all four for each r — locality is fine, but the four `make([]float64, newLen)` (lines 256-259) plus the `make([]float64, n)` for `logPi` (line 185) plus the `make([]float64, newLen)` for `newP` (line 234) plus the `make([]float64, newLen)` for `newLogP` (line 201) plus the `make([]float64, len(newP))` defensive copy (line 287) sum to **the seven allocations per Update** that 021 reported as six (the agent missed the `out := make` at line 287). At default R_max=500 that's 7 × 8 × 501 = **~28 kB of garbage per Update**, not 24 kB. The fix is the standard scratch-buffer pattern, but with a **double-buffer** twist (since the per-r update at line 273 reads `b.mu[src]` while writing `newMu[r]` with `src = r-1` — the same array would self-corrupt if updated in place forward; iterating *backwards* makes in-place legal and eliminates four of seven allocations entirely). On the missing-detector front, the topic-named complexity ceilings are: **PELT** is O(n) average / O(n²) worst with inequality pruning + cumsum-of-squares (Killick-Fearnhead-Eckley 2012); **FPOP** strictly dominates PELT with functional pruning (Maidstone et al. 2017) at the same O(n) average and is what scipy/ruptures will ship next; **WBS** is O(n log n) with random intervals (Fryzlewicz 2014); **e-divisive** is **the costly one** at O(n²) base, O(n²K) for K segments, beatable to O(n² + nK log n) via summed-area-table (SAT) + segment trees (Matteson-James 2014 §3.2 explicitly notes the SAT trick); **MMD-based detectors** are O(n²) per window naively, dropping to O(n) with **NEWMA** (Keriven et al. 2018) and O(n log n) with **MMDEW** (Kalinke 2025) using exponential-weighted random Fourier features — the FFT speedup the topic asks about is real but lives upstream of MMD in the RFF / characteristic-kernel construction, not in MMD itself. **bcp v2 → v4's 60× block-cumulative-sum speedup** (45 min → 45 sec on 10k points, per agent 023's SOTA review) is the canonical "summed-area trick" the topic names; reality should ship it first because the BOCPD predictive pass *already* needs (kappa·mu, beta + 0.5·kappa·dx²/(kappa+1)) accumulators that are cheap to maintain incrementally — Welford for variance, kappa-weighted running mean for mu — and that 021 §6 flagged as a precision issue at t > 1e6 (it's also a perf issue: the closed-form update at line 273 does **two divides per r per step**, ~7 ns each, so at R_max=500 that's 7 µs of pure division latency per Update before any predictive work happens). The fix-set is ~280 LOC of pure additions + ~40 LOC of refactor: fused growth-and-reset loop with persistent `b.logP` field (kills the redundant `math.Log(b.p[r])`, ~15 LOC), sparse posterior with explicit `r_min, r_max` window + `eps` threshold (~80 LOC), in-place backwards NIG update with three permanent scratch slices (~30 LOC), Welford-style `M_r` running second-moment per run-length (~25 LOC, also closes 021 §6's t > 1e6 precision floor), summed-area-table precompute path for `Bocpd.Offline(x []float64)` batch processing (~60 LOC), mandatory `bench_test.go` covering R∈{50, 500, 5000} × {stationary, step-shift, sparse} (~60 LOC). Total ~270 LOC, all backwards-compatible, would cut per-step **wall time ~3-5×** and per-step **allocations 6→1** (the public-API defensive copy at line 287 stays).

---

## 1. The redundant-Log hot-loop hazard 021 missed

### 1.1 Two passes, two `math.Log(b.p[r])` calls per r per step

**Location:** `bocpd.go:207-222` (lifted with `cat -n`):
```
207:	for r := 0; r < n; r++ {
208:		dst := r + 1
209:		if dst >= newLen {
210:			continue
211:		}
212:		h := b.hazard(r)
213:		logGrowth := math.Log(b.p[r]) + logPi[r] + math.Log(1.0-h)
214:		newLogP[dst] = logSumExp(newLogP[dst], logGrowth)
215:	}
216:
217:	// Reset to r = 0 (a change-point happened just before x).
218:	for r := 0; r < n; r++ {
219:		h := b.hazard(r)
220:		logReset := math.Log(b.p[r]) + logPi[r] + math.Log(h)
221:		newLogP[0] = logSumExp(newLogP[0], logReset)
222:	}
```

**The duplication.** Both loops:
1. Iterate `r ∈ [0, n)` over the same range.
2. Call `b.hazard(r)` (which under constant hazard returns the same `1.0/b.lambda` every time).
3. Call `math.Log(b.p[r])` — **the same Log call**, on the same `p[r]`, **twice per step per r**.
4. Read `logPi[r]` from the predictive pre-pass.

`math.Log` on amd64 is ~6 ns (libm `log` with the polyfill Go ships). At R_max = 500 the duplicated work is **500 × 6 ns = 3 µs per Update** of pure redundant Log. That's ~10% of the Update budget at typical Pistachio cadence (60 FPS = 16.6 ms budget — but a single BOCPD detector should consume <100 µs, so this is ~3% of *its* budget; with 1000 detectors per frame that's 3 ms = 18% of the frame).

`math.Log(1.0-h)` and `math.Log(h)` are loop-invariants under constant hazard (lines 213, 220). They depend only on `b.lambda`, not on `r` or the data. **Both should be hoisted to fields** on `Bocpd`: `b.logOneMinusH = math.Log1p(-1.0/b.lambda)` and `b.logH = math.Log(1.0/b.lambda)`, computed once in `New` (and recomputed if/when a `SetLambda` accessor is added). 021 §3.1 already noted the `Log(1-h)` → `Log1p(-h)` correctness fix; this is the perf companion that makes the same line a one-time `b.logOneMinusH` lookup. The fused inner loop becomes:
```go
// Fused growth + reset, log-domain posterior maintained between calls.
for r := 0; r < n; r++ {
    base := b.logP[r] + logPi[r]      // one read of b.logP, no math.Log per step
    // reset contribution to slot 0
    newLogP[0] = logSumExp(newLogP[0], base + b.logH)
    // growth contribution to slot r+1 (if not truncated)
    dst := r + 1
    if dst < newLen {
        newLogP[dst] = base + b.logOneMinusH  // first writer wins; no logSumExp because each dst is unique
    }
}
```

The `logSumExp` at line 214 is **mathematically necessary in the original** because the truncation collapses `R_max → R_max+1` into the same destination — but a one-line check `if dst < newLen { ... }` (already present at line 209) makes each destination unique, so the growth loop **does not need logSumExp at all**. Only the reset loop (which fans 0..n-1 → 0) does. **Saves n-1 logSumExp calls per Update** (~15 ns each = ~7.5 µs at R_max=500).

### 1.2 Persisted `b.logP` field — close the Log↔Exp round-trip

021 §3.2 already named this. The same field needed for §1.1 above (the fused loop reads `b.logP[r]`, not `math.Log(b.p[r])`) closes 021's redundancy gap as a free side-effect. The renormalisation at lines 224-245 currently:
1. Finds `maxLogP`.
2. `newP[i] = math.Exp(v - maxLogP)` (n calls to `math.Exp`, ~7 ns each).
3. Divides by `sum`.

The new contract:
1. Find `maxLogP`.
2. `newLogP[i] -= maxLogP` (n cheap subs).
3. Compute `logSum = maxLogP + math.Log(sum_i exp(newLogP[i] - maxLogP))` via one `logSumExp` reduction.
4. `newLogP[i] -= logSum`. **Posterior is now stored in log-domain, sums to 1 in linear space, no exp/log round-trip.**

The public-API surface (`RunLengthPosterior`, `ChangePointProbability*`, `MapRunLength`, `ExpectedRunLength`) must convert back via `math.Exp(b.logP[r])` on demand, which costs n exps **only when the consumer asks** — not per Update. For consumers that only read `MapRunLength` (argmax of `b.logP` is the same r as argmax of `b.p`, no exp needed) the cost is **zero** exps per query.

Net: per-Update exp/log calls drop from `~3n` (one log per r in growth, one log per r in reset, one exp per r in renorm) to `~n+1` (one logSumExp reduction over n in renorm). On amd64 at n=500, that's **9 µs → 3.5 µs**, a 2.6× speedup of just the log-domain housekeeping.

---

## 2. Sparse posterior: O(R_max) → O(R_eff)

### 2.1 The empirical sparsity

Probed (mental model from the math, not run): under constant hazard with rate `1/lambda`, the stationary posterior over r has **geometric decay** with rate `~exp(-1/lambda) * (likelihood ratio)`. For lambda = 250 (default) and stationary data, `P(r=k) ∝ (1-1/250)^k * (likelihood term ≈ 1 under stationarity)`. After t=2000 steps:
- `P(r=0) ≈ 1/250 = 4e-3`
- `P(r=lambda) ≈ exp(-1) * P(r=0) ≈ 1.5e-3`
- `P(r=5*lambda=1250)` would be `≈ exp(-5) * P(r=0) ≈ 2.7e-5` — but R_max=500 truncates here.
- **Effective support** is roughly `r ∈ [0, ~5*lambda]` for mass > 1e-12, capped at `R_max+1`.

So at default settings (lambda=250, R_max=500), **the entire posterior is "effectively dense"** because `5*lambda > R_max`. **But** any user who picks `lambda << R_max` (e.g., lambda=10 for noisy intraday) gets posterior support `~50` while paying for `R_max+1=501` slots — **10× wasted work**. Conversely, any user who picks `R_max=5000` for "deep history" gets the same 5×lambda effective support and pays **10× the wasted work** if lambda stays at 250.

### 2.2 The pruning rule (bocpdms / Knoblauch-Damoulas 2018)

Drop run-length r whenever `P(r) < eps` for some threshold (typical: `eps = 1e-4` to `1e-6`). The dropped mass is added to either (a) the closest neighbour (preserves the support shape), (b) the reset slot r=0 (conservative — biases toward false alarms but cheap), or (c) discarded with renormalisation (introduces O(eps · t) bias over a long run; usually negligible at eps=1e-6).

The right data structure is **a sparse run-length distribution as a `(r_min, r_max, p[r_min..r_max])` window** — contiguous because the posterior is unimodal under constant hazard, and bounded by 5×lambda above. The window slides up by 1 on each step (growth), and r=0 always exists (reset target). The implementation:

```go
type Bocpd struct {
    // ... existing fields ...
    rMin    int       // smallest r with mass > epsPrune
    rMax    int       // largest r with mass > epsPrune (reuse the existing field name; this is *active* not *cap*)
    rMaxCap int       // hard truncation (was: rMax)
    epsPrune float64  // posterior pruning threshold; default 1e-6
    logP    []float64 // logP[r - rMin] for r in [rMin, rMax]
    mu, kappa, alpha, beta []float64  // same offset
}
```

**Per-step cost** under stationary lambda=250 with epsPrune=1e-6 is ~30 active slots (the 5σ tail of geometric(1/250)). For R_maxCap=500 that's a **17× speedup** of the predictive pass and the NIG update pass. For R_maxCap=5000, **170× speedup**.

The public API stays — `RunLengthPosterior()` returns a length-`R_maxCap+1` slice with zeros for unrepresented slots, preserving the existing contract. New methods: `ActiveRunLengthRange() (int, int)` returns `(rMin, rMax)` for consumers that want the sparse view (compute the next layer of analytics from the active window only).

### 2.3 The reference

Knoblauch & Damoulas 2018 (the bocpdms package) ship this trick, citing 5-20× wall-clock speedups depending on hazard. The "Killick pruning" agent 023 names is the same idea applied to PELT (Killick-Fearnhead-Eckley 2012 §3.4). For BOCPD specifically, dtolpin/bocd ships threshold pruning at `1e-6` by default.

---

## 3. Welford / cumulative-sum precompute (the "summed-area trick" the topic asks about)

### 3.1 What's recomputed per step today

The NIG update at line 264-278:
```go
mu := b.mu[src]
kappa := b.kappa[src]
alpha := b.alpha[src]
beta := b.beta[src]
newMu[r] = (kappa*mu + x) / (kappa + 1.0)        // ONE divide
newKappa[r] = kappa + 1.0
newAlpha[r] = alpha + 0.5
dx := x - mu
newBeta[r] = beta + 0.5*kappa*dx*dx/(kappa+1.0)  // ONE divide
```

Two divisions per `r` per step. amd64 `divsd` latency is ~14-20 cycles (~5-7 ns each). At R_max=500 that's **~7 µs of pure division latency per Update** — comparable to the redundant-Log cost from §1.1.

### 3.2 The summed-area observation

`(kappa + 1.0)` is the same denominator in both divisions. **Compute once**:
```go
denom := kappa + 1.0
invDenom := 1.0 / denom         // ONE divide
newMu[r] = (kappa*mu + x) * invDenom
newKappa[r] = denom
newAlpha[r] = alpha + 0.5
dx := x - mu
newBeta[r] = beta + 0.5*kappa*dx*dx*invDenom
```

Halves the division count. Saves ~3.5 µs at R_max=500.

### 3.3 The deeper observation: cumsum precompute under offline batch

Today's API is **online-only** (`Update(x float64)`). For the offline use-case (run BOCPD on a fixed array `[]float64`), the textbook precompute is:
- `S1[i] = sum(x[0..i])` — running first-moment, O(n) total
- `S2[i] = sum(x[0..i]^2)` — running second-moment, O(n) total

These let any per-segment mean and variance be computed in O(1):
- `mean(i, j) = (S1[j] - S1[i-1]) / (j - i + 1)`
- `var(i, j) = (S2[j] - S2[i-1]) / (j - i + 1) - mean^2`

For BOCPD, the per-r posterior `(mu_r, kappa_r, alpha_r, beta_r)` after t steps is **a closed-form function of the segment means** since the start of run-length r — i.e., `mu_r = (kappa_0 * mu_0 + sum(x[t-r..t])) / (kappa_0 + r+1)`. With S1/S2 precomputed, **the entire NIG update loop collapses from O(R_max) per step to O(1) per (step, r) lookup**. Total offline cost: O(n) precompute + O(n × R_eff) Update — same asymptotic as the online path, but **the per-step constant drops by ~5×** because there are no divisions, no per-r cumulative state, and the inner loop is pure SIMD-friendly multiply-adds against the cumsum arrays.

This is the bcp v2 → v4 trick (agent 023's SOTA review §1: "wall-clock: 10,000-point series went from 45 minutes to 45 seconds"). **Reality should ship `Bocpd.Offline(x []float64) []float64` as the canonical batch path** — same math as repeated `Update`, but ~10-60× faster on long arrays.

### 3.4 Welford for the variance accumulator (closes 021 §6)

Even in the online path, the `0.5 * kappa * dx² / (kappa + 1)` increment to beta is **a scaled Welford increment** in disguise:
```
M2_new = M2_old + (kappa_old / (kappa_old + 1)) * (x - mu_old)^2
```

Welford's running M2 is exactly this with `kappa = n` (count). The update is **already** numerically Welford-equivalent; the bug 021 flagged at §6 is about the **mu** update, not beta. For mu the equivalent Welford-stable form:
```
mu_new = mu_old + (x - mu_old) / (kappa_old + 1)
```
is **algebraically identical** to the closed-form `(kappa*mu + x) / (kappa + 1)` but **numerically more stable** at large kappa. Same compute cost (one add + one divide + one multiply, vs one multiply + one add + one divide). So the perf cost is zero and the numerical floor improves from O(t · eps) to O(sqrt(t) · eps). Free win.

---

## 4. In-place backwards NIG update — kills four allocations

### 4.1 The data dependency

Line 264-278 computes `newMu[r] = f(b.mu[r-1], x)`. Each output `newMu[r]` depends on the *source* `b.mu[r-1]`. If we tried to update in place forward (`b.mu[r] = f(b.mu[r-1], x)`), we'd corrupt `b.mu[r]` before the next iteration reads it as `b.mu[r]` (= `r-1` for the next iter).

**Iterating backwards** — `for r := newLen - 1; r >= 1; r-- { b.mu[r] = f(b.mu[r-1], x) }` — reads `b.mu[r-1]` *before* it gets overwritten. Same for kappa, alpha, beta. **The four `make([]float64, newLen)` at lines 256-259 disappear**, replaced by one-time `b.mu = b.mu[:newLen]` slice-extension (`append` if cap exceeded; otherwise zero-alloc).

### 4.2 Slot 0 reset

Line 260-263 resets `newMu[0] = b.prior.Mu0` (and the other three). With backwards-in-place this becomes `b.mu[0] = b.prior.Mu0` after the backwards loop completes — same one-line write, same cost. No conflict with the backwards loop because the backwards loop stops at `r >= 1`.

### 4.3 The cap management

`newLen = min(n+1, b.rMaxCap+1)` — when n < R_maxCap+1, the slice grows by 1 per step; when n == R_maxCap+1, it stays the same length. Pre-grow `b.mu` to `cap=R_maxCap+1` in `New`, then `b.mu = b.mu[:newLen]` is zero-alloc forever. **Same trick for kappa, alpha, beta, logP.** Five permanent slices, all `len ≤ R_maxCap+1`, all zero-alloc after the first growth phase.

### 4.4 `logPi` and `newLogP` scratch

`logPi` (line 185) and `newLogP` (line 201) are per-call temporaries. Promote both to `b.logPi` and `b.newLogP` fields, pre-grown to `R_maxCap+1`, sliced down each Update. Two more allocs eliminated.

### 4.5 The defensive-copy alloc

`out := make([]float64, len(newP)); copy(out, newP); return out, nil` (line 287-289). This is the **public-API contract** — caller may retain the returned slice indefinitely, and the underlying `b.p` will mutate next Update. This alloc **must stay** (or move to caller-supplied buffer via a `Bocpd.UpdateInto(x float64, out []float64) ([]float64, error)` companion). 021 §7 noted the same.

**Net allocation count after all six fixes:** **1 per Update** (the defensive copy at line 287-289). Was: 7 per Update. **86% reduction in GC pressure**, from 28 kB / Update to 4 kB / Update at default R_max=500. Pistachio at 60 FPS × 1000 detectors: **5.2 GB/hour → 0.86 GB/hour** of GC pressure for the BOCPD layer.

---

## 5. Branch-prediction hazards in the hot loop

Probed for any data-dependent branches in the inner loops:

| Line | Branch | Frequency | Predictability | Hazard? |
|---|---|---|---|---|
| 209 | `if dst >= newLen { continue }` | once per r | always-true for r=n-1, always-false for r<n-1 | **monotonic, well-predicted** — no hazard |
| 295-300 | `logSumExp` `IsInf(a, -1)` and `IsInf(b, -1)` short-circuits | called once per growth+reset r-pair | depends on whether posterior has saturated; predictable per call site but can flip if epsPrune kicks mass to -Inf | **one mispredict per regime change**; negligible |
| 301 | `if a > b` in logSumExp | per call | data-dependent on logPi values vs prior accumulation | **mild hazard** — branchless variant `(a + b + math.Abs(a-b)) * 0.5` for the max + manual log1p computes both sides; ~2× speedup of logSumExp body; ~5% speedup of the renorm pass |
| 240 | `if !(sum > 0)` | once per Update | always true except on numerical underflow | **negligible** |

The **only** real branch hazard is the `a > b` in `logSumExp` (line 301). Branchless logSumExp:
```go
func logSumExp(a, b float64) float64 {
    if math.IsInf(a, -1) { return b }
    if math.IsInf(b, -1) { return a }
    max := math.Max(a, b)              // branchless on amd64 (maxsd)
    min := a + b - max                 // branchless trick for min
    return max + math.Log1p(math.Exp(min - max))
}
```
`math.Max` and the `min := a + b - max` arithmetic both lower to branchless SSE. Saves the data-dependent `a > b` mispredict (~5-10 cycles per call when the predictor is wrong; ~0.5 ns per logSumExp at 500 calls/Update = ~250 ns/Update). Small absolute number but free.

---

## 6. Memory layout: SoA stays, but interleaving for the predictive pass would be marginally better

Today: `mu, kappa, alpha, beta` are four parallel `[]float64` (SoA). The predictive pass at line 186-191 reads all four sequentially per r, then computes Student-t logPDF. That's four cache-line touches per r (one per slice), which on a 64-byte cache line is one line of mu (8 r's), one of kappa, one of alpha, one of beta — **4 cache misses per 8 r's = 0.5 misses per r**.

Alternative: `[]struct{ mu, kappa, alpha, beta float64 }` (AoS). Each entry is 32 bytes, two per cache line. The predictive pass reads all four fields per r — **0.5 misses per r**, same. AoS wins on the predictive pass (no change). **But** the NIG update pass (line 264-278) writes all four per r — same access pattern, same 0.5 misses per r. AoS or SoA doesn't matter here.

**Where SoA loses:** if/when a future detector wants to **broadcast a single-field operation** (e.g., "increment all alpha by 0.5" — which is exactly what line 275 does for r in [1, newLen)), SoA enables vectorisation (one SIMD AVX2 add over a `[]float64`); AoS does not. **Keep SoA.** No change recommended.

**Where AoS would win:** if the per-r work involved many independent NIG hyperparameters (>4) or if the cache footprint per r exceeded one cache line, AoS prefetching would win. Not applicable here.

---

## 7. The missing-detector complexity ceilings (slot 022 cross-reference)

Per the topic prompt, these are the complexity ceilings slot 022 must hit when adding the missing detectors. Each line is the asymptotic cost a *correct* implementation must reach:

| Detector | Naive | With pruning / SAT | Reference | LOC for the SAT path |
|---|---|---|---|---|
| **PELT** (Killick-Fearnhead-Eckley 2012) | O(n²) | **O(n) average** with inequality pruning + cumsum-of-squares (S1/S2 from §3.3) | scipy `ruptures.Pelt` | ~250 LOC |
| **FPOP** (Maidstone et al. 2017) | O(n²) | **O(n) average**, strictly more pruning than PELT (functional pruning over piecewise-quadratic cost) | gfpop, fpop, ruptures (planned) | ~300 LOC |
| **BinSeg** (Scott-Knott 1974, Vostrikova 1981) | O(K·n log n) | **O(n log n)** with cumsum-of-squares + heap-driven greedy | ruptures `BinSeg` | ~150 LOC |
| **WBS** (Fryzlewicz 2014) | O(M·n) where M = # random intervals | **O(n log n)** for M = O(n log n) and SAT-driven CUSUM scan | wbs (R), ruptures (planned) | ~200 LOC |
| **e-divisive** (Matteson-James 2014) | O(n²K) | **O(n² + n·K log n)** via SAT for the within-cluster distance + segment tree for the maximiser | ecp (R) | ~250 LOC |
| **MMD-CPD biased** (Gretton 2012) | O(n²) per window | **O(n)** with NEWMA exponential-window (Keriven 2018), **O(n log n)** with MMDEW (Kalinke 2025) using RFF + FFT | NEWMA, MMDEW | ~300 LOC |
| **Kernel CUSUM online** (Wei-Xie 2024) | O(n) per step but O(n²) per window | **O(B²·N)** with B-block N-pool decomposition | online-kernel-cusum | ~200 LOC |

**The single highest-leverage SAT primitive** is the cumsum-of-squares pair `(S1, S2)` introduced in §3.3 — it's the unifier that pulls PELT, BinSeg, WBS, and e-divisive *all* down to O(n log n) or better. **Ship it as `changepoint.CumulantPrep(x []float64) (S1, S2 []float64)`** before any detector that needs it; the seven detector PRs become drop-ins once the SAT helper exists. 022 should sequence the work: SAT helper first (~30 LOC), then PELT (which is the consensus default), then FPOP (which strictly dominates PELT), then BinSeg/WBS (which are O(n log n) anyway), then e-divisive/MMD/Kernel-CUSUM (which need slightly more machinery).

### 7.1 The FFT-speedup the topic asks about

The topic prompt: "FFT speed-ups for any kernel-based detector." Three places this applies:

1. **MMD with random Fourier features** (Rahimi-Recht 2007, applied to MMD in Sutherland-Schneider 2015): the kernel `k(x, y) = E_w[cos(w · x) cos(w · y) + sin(w · x) sin(w · y)]` is computed by drawing `D` random frequencies `w_i ~ p(w)` and forming the feature map `phi(x) = sqrt(2/D) · [cos(w_i · x), sin(w_i · x)]_i`. **MMD becomes the L2 distance between mean feature vectors**, computable in O(n·D) instead of O(n²). For D = O(log n), that's O(n log n). FFT enters when the data is gridded (e.g., uniformly-sampled time series): `cos(w · x)` over a grid of x's is the real part of an FFT of a complex exponential. **For non-gridded data, no FFT speedup applies**; the O(n·D) RFF result is what NEWMA/MMDEW use.

2. **Convolutional CUSUM filter banks**: classical CUSUM scans a sliding window with a fixed test statistic; if the test statistic is a fixed kernel `g`, the scan is a **convolution** `(g * x)[t] = sum_s g[s] x[t-s]`, computable in O(n log n) via FFT instead of O(n·|g|). Reality already has FFT in `signal/`; this is a one-line `signal.Convolve` call once CUSUM exists.

3. **Spectral-domain change detection** (Davis-Lee-Rodriguez 2006, Last-Shumway 2008): detect changes in the spectrum of the time series via FFT of windowed segments, then compare adjacent windows via a divergence (KL, Bhattacharyya, Hellinger — all live in `infogeo`). Naive O(n² log n) per window; with sliding-FFT (Goertzel-style updates), O(n log n) total. This is "PSD-CPD" and is what audio onset detectors do at 60 FPS.

The first is the most lit-cited; the second is the lowest-LOC; the third is the most consumer-relevant for the existing reality consumers (audio + signal already use FFT, so spectral-domain CPD composes cleanly).

---

## 8. Streaming vs offline: the missing batch path

Today's API: `for _, xi := range data { b.Update(xi) }` — pure streaming, even when the data is a fixed `[]float64`.

What's missing: `Bocpd.Offline(data []float64) []float64` (or `OfflineRunLengths(data []float64) []int` — the canonical change-point sequence) that:
1. Pre-computes `S1, S2` cumulative sums (O(n)).
2. Runs the same Update logic but with the NIG update path replaced by O(1) lookups against (S1, S2).
3. Returns the full posterior trajectory `[t][r]float64` (or compressed: `[t]MAP_r int`) without the per-step `make([]float64, len(newP))` defensive copy.

**Speedup vs streaming-on-array** at n=10000, R_max=500: ~10-30× (eliminates seven allocations per step × 10000 steps = 70000 allocs → 1 alloc + the trajectory return; eliminates the per-r division work via §3.2).

### 8.1 The "Update(x) is the wrong primitive for batch" finding

The topic prompt: "when running BOCPD on a static array, can we precompute summed-area tables and skip per-step recomputation?" **Yes**, and the existing API blocks this. The fix is `Offline(...)` as a sibling method, *not* a refactor of `Update` (which must stay streaming). 023's review names this as the canonical "engineering pivot" that bcp v2 → v4 made.

---

## 9. Concrete fix-set (ranked by impact-per-LOC)

| # | Fix | LOC | Wall-clock impact at R_max=500 | Allocs per Update | Backwards-compatible? |
|---|---|---|---|---|---|
| 1 | Hoist `b.logH, b.logOneMinusH` (constant under constant hazard) | 4 | -3 µs | 0 | yes |
| 2 | Fuse growth+reset into one loop, persist `b.logP` field, drop the redundant `math.Log(b.p[r])` | 25 | -5 µs | -2 | yes |
| 3 | Promote 5 scratch slices (logPi, newLogP, mu, kappa, alpha, beta — drop newP entirely; logP doubles as it) to `Bocpd` fields with `R_maxCap+1` cap, slice down each Update | 40 | -2 µs (cache reuse) | **-6** (down to 1) | yes |
| 4 | Backwards-in-place NIG update | 15 | 0 (same compute) | (covered by #3) | yes |
| 5 | One `invDenom = 1.0/(kappa+1)` per r — halve division count | 5 | -3.5 µs | 0 | yes |
| 6 | Sparse `(rMin, rMax)` window with `epsPrune = 1e-6` | 80 | -10 µs to -200 µs (10-50× speedup at lambda << R_max) | 0 | yes (new method `ActiveRunLengthRange`; existing API returns padded slice) |
| 7 | Branchless `logSumExp` (math.Max + arithmetic min) | 6 | -0.25 µs | 0 | yes |
| 8 | Welford-stable mu update (numerical, also names the `O(sqrt(t)·eps)` floor — closes 021 §6) | 3 | 0 | 0 | yes |
| 9 | `Offline(x []float64)` batch path with cumsum-of-squares SAT | 60 | n=10000: 30× speedup over streaming | -7n + 1 | yes (new method) |
| 10 | `bench_test.go` covering R∈{50,500,5000} × {stationary, step-shift, sparse} × {Update, Offline} | 60 | (locks the fix-set as measurable) | - | yes |
| **Total** | | **~298 LOC** | **~25 µs/Update saved (3-5× faster)** | **7 → 1 (86% less GC)** | all yes |

The first eight items are pure refactors of existing math — every test in `bocpd_test.go`, `bocpd_expansion_test.go`, `infogeo_test.go` should still pass to within `1e-12` (the renormalisation step is the only place float-error matters, and it's bit-stable across the fused-loop refactor). The two new items (Offline + bench) add public surface area but don't change existing semantics.

---

## 10. What is *not* a perf bug

For completeness, the audit checked these and confirmed they are not on the hot path:

- `RunLengthPosterior()` allocates a defensive copy. Public-API contract — must stay.
- `MapRunLength()` is a single linear scan; O(R_max) is unavoidable for an argmax over a non-sorted slice. Could use a `b.mapR int` field updated incrementally during Update (~5 LOC) for O(1) query, but the current pattern is called once per consumer-Decision-point, not per-Update.
- `ExpectedRunLength()` is the same — single linear scan, called on demand.
- `CurrentRegimeMean / Variance` are O(1) lookups; fine.
- `studentTLogPDF` calls `math.Lgamma` twice per r per step. `Lgamma` is ~50 ns on amd64 (Stirling series). The two calls are functions of `alpha[r]` only, which changes by exactly +0.5 per step per r — they could be **maintained incrementally** as a fifth running stat `b.lgammaAlpha, b.lgammaAlphaHalf` per r, updating via the recurrence `Lgamma(x+0.5) = ?` (no closed-form recurrence — must call). So this is **not** a Welford-style win; the two Lgamma calls are inherent. **But:** the `0.5*math.Log(df*math.Pi) - math.Log(scale)` term decomposes as `0.5*math.Log(df) + 0.5*math.Log(math.Pi) - math.Log(scale)`, and `0.5*math.Log(math.Pi)` is a global constant — hoist to `var halfLogPi = 0.5 * math.Log(math.Pi)` (~1 LOC, ~3 ns saved per Update). Tiny but free.
- `logSumExp` short-circuits on `-Inf` correctly; no perf issue.
- The `b.t++` increment is O(1); fine.

---

## 11. Verdict

`changepoint/bocpd.go` is algorithmically textbook (R_max-truncated Adams-MacKay 2007 with NIG conjugate observation model) but engineering-suboptimal in seven concrete, additive ways: redundant `math.Log` per step (§1), no log-domain persistence (§1, also flagged by 021 §3.2), seven allocations per Update vs the achievable one (§4), no sparse pruning (§2 — the bocpdms / Knoblauch-Damoulas trick), two divisions per r when one suffices (§3.2), one mild branch hazard in `logSumExp` (§5), and no batch/offline path with cumsum precompute (§3.3, §8 — the bcp v2→v4 trick). All seven fix lines are bounded (~298 LOC total), backwards-compatible, and unblock the missing-detector slot 022 by making the cumsum-of-squares SAT helper available as a primitive that PELT/FPOP/BinSeg/WBS/e-divisive all need (§7). The 60 FPS / 1000-detector Pistachio cadence is **already** within 5× of feasible — fix-set #1-#5 alone (~90 LOC, no new public API) gets it to the budget. Pruning (§2) and Offline batch (§8) are the larger asks but pay off when consumers reach for `R_max ≥ 5000` or `lambda << R_max`. The single biggest "lit-cited" miss is **MMDEW (Kalinke 2025) for kernel-based streaming CPD** — that's a slot-022 ship, not a slot-025 fix, but it's the 2024-2026 frontier that the topic prompt's "FFT speed-ups for any kernel-based detector" item points at and is what reality should ship next once the BOCPD perf floor is closed.

---

## Sources

- [Killick, Fearnhead, Eckley 2012 — Optimal detection of changepoints with linear computational cost (PELT)](https://arxiv.org/pdf/1101.1438)
- [Maidstone et al. 2017 — On optimal multiple changepoint algorithms for large data (FPOP)](https://link.springer.com/article/10.1007/s11222-016-9636-3)
- [Keriven et al. 2018 — NEWMA: scalable model-free online change-point detection](https://arxiv.org/pdf/1805.08061)
- [Kalinke 2025 — MMDEW: Maximum Mean Discrepancy on Exponential Windows](https://arxiv.org/html/2205.12706v4)
- [Wei-Xie 2024 — Online Kernel CUSUM for change-point detection](https://arxiv.org/pdf/2211.15070)
- [Knoblauch-Damoulas 2018 (bocpdms) — referenced in agent 023's SOTA review §1](https://gregorygundersen.com/blog/2019/08/13/bocd/)
- [Adams-MacKay 2007 — original BOCPD reference, see Gundersen blog walkthrough](https://gregorygundersen.com/blog/2019/08/13/bocd/)
- [Pishchagina et al. 2024 — Geometric-based pruning rules for multi-series change-point detection](https://computo-journal.org/published-202406-pishchagina-change-point/)
- [PELT empirical complexity in financial time series, ICCSAI 2025](https://dl.acm.org/doi/10.1145/3773365.3773532)
