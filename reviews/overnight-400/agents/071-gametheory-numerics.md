# 071 | gametheory-numerics

**Scope:** numerical-correctness audit of `C:\limitless\foundation\reality\gametheory\`
**Files:** `nash.go`, `voting.go`, `matching.go`, `bandit.go`, `kelly.go`, `gametheory_test.go`, `testdata/gametheory/{nash_2x2,gale_shapley}.json`
**Test status:** all gametheory tests pass (`go test ./gametheory/` green).

## TL;DR (headline)

The package implements the *2x2-only* Nash slice, an *iteration-bounded* fictitious play minimax, and exact Banzhaf/Shapley enumeration. **It contains no Lemke-Howson, no support enumeration, no replicator dynamics, no iterated-game convergence detector, and no LP-based minimax.** The MASTER_PLAN topic (Lemke-Howson edge cases, replicator stability, LP minimax, iterated-game convergence) is therefore largely a *missing-functionality* finding, not a bug audit. Of what *is* implemented, the most material numerical issues are: (a) Minimax has a hard-coded 100,000-iteration cap with no convergence test and no value-bound output, (b) `NashEquilibrium2x2` uses a fixed `1e-15` indifference-denominator tolerance that is below double-precision noise for moderately-scaled payoffs, and (c) `ShapleyValue`'s exact branch caps at n=12 and the Monte-Carlo branch above that uses an LCG with a hard-coded seed (=42) — fully deterministic but silently lossy and never validated by golden vectors.

---

## What is implemented vs. what the topic asks for

| Topic bullet | Status in code | File:line |
|---|---|---|
| Pure-strategy NE iteration | 2x2 only, exhaustive over 4 cells | `nash.go:83-96` |
| Mixed NE via Lemke-Howson | **absent** | — |
| Mixed NE via support enumeration | **absent** (closed-form only for 2x2) | `nash.go:55-74` |
| Indifference tolerance | hard-coded `1e-15` on denominator | `nash.go:62` |
| Lemke-Howson pivot / cycling / lexicographic perturbation | **absent** | — |
| Best-response correspondence ties at boundary | uses `>=` (returns first cell) | `nash.go:87-92` |
| Shapley exact for n ≤ 20 | **caps at n ≤ 12**, then Monte-Carlo | `voting.go:124-128` |
| Replicator dynamics | **absent** | — |
| Minimax via LP | **fictitious play instead** (Brown 1951) | `nash.go:170-231` |
| Iterated game / convergence detection | **absent** (no `Iterate*`/`Repeated*` funcs) | — |

So the topic is roughly 50% audit, 50% missing-feature flag. The audit below is for what exists.

---

## Findings, ranked by severity

### F1 — `Minimax` has no convergence test and no value bracket. Severity: **high**.

`nash.go:170-231`. Fictitious play runs **exactly** `maxIter = 100000` iterations regardless of game size or rate of convergence. Robinson (1951) proves convergence at rate `O(t^{-1/(m+n-2)})` for the *empirical-frequency* value, which for a 5×5 zero-sum game means worst-case ~O(t^{-1/8}) — extremely slow. Specific issues:

1. No tracking of `maxRowVal` (Brown's lower bound `v_t = max_i rowSums[i]/t`) and `minColVal` (upper bound `V_t = min_j colSums[j]/t`). The game value is squeezed inside `[v_t/t, V_t/t]`; the true value is unknown to the caller and *not returned*. The function returns only `p^T A q`, which can be on either side of the true minimax value during the transient.
2. Doc claims "~1e-6 for fictitious play". For Rock-Paper-Scissors the test tolerates **0.05** on the value (`gametheory_test.go:133`) and **0.05** on each strategy probability (`gametheory_test.go:137,143`). The docstring's 1e-6 is aspirational, not measured — actual achievable precision after 100k iterations on a 3×3 zero-sum game is ~1e-3 to 1e-2.
3. `bestRowVal` initialized to `rowSums[0]` not `-Inf`. Iteration 0 has `rowSums[0]=0`, ties broken by lowest index, then `payoff[i][bestCol]` is added. This is fine for correctness but the first iteration is wasted on an arbitrary initial pick.
4. No early-out. A 2×2 with a saddle point still runs 100k iterations.
5. `value = sum_i sum_j p_i q_j A_ij` (`nash.go:225-229`) is the *empirical* value of the *averaged* strategies, not the iterate value. For a non-converged trajectory this can differ noticeably from the actual minimax value of the empirical pair.

**Recommendation:** return `(rowStrat, colStrat, value, gap)` where `gap = V_t - v_t` is the bracket width, and add an `epsilon` tolerance argument; converge when `gap < epsilon` or `iter >= maxIter`.

### F2 — `NashEquilibrium2x2` indifference tolerance is dimensionally wrong. Severity: **medium**.

`nash.go:62`: `if math.Abs(denomA) > 1e-15 && math.Abs(denomB) > 1e-15`.

`denomA = A[0][0] - A[0][1] - A[1][0] + A[1][1]` — a *signed sum of four payoffs*. If the payoffs have magnitude ~1e6 (e.g. dollar payoffs in cents, or rescaled utility), `1e-15` is **20 orders of magnitude below** the floating-point noise of the addition. Conversely, if payoffs are ~1e-10, then a real degeneracy looks larger than `1e-15` and the code computes `q,p` from a near-singular system.

A correct test is *relative*: `math.Abs(denomA) > eps * (|A[0][0]| + |A[0][1]| + |A[1][0]| + |A[1][1]|)` with `eps ≈ 4 * math.Nextafter(1,2) - 1 ≈ 1e-15` *as a relative scale*. Current code is fine for the unit-magnitude golden cases but will misclassify games with payoffs outside ~[1e-10, 1e10].

### F3 — Best-response ties use first-cell preference, no boundary-tie reporting. Severity: **medium**.

`nash.go:87,90`: `aBestResponse := payoffA[r][c] >= payoffA[otherR][c]`. With strict equality the code marks *both* cells as best responses, both pure-NE candidates pass through, and `candidates[0]` (lowest `(r,c)`) is returned. Symptoms:

- "all-equal payoffs" test (`nash.go:71-77`, golden `nash_2x2.json:67-75`) returns `(1,0,1,0)` (cell (0,0)) — but cell (1,1) is equally a NE, as is the entire mixed-strategy continuum. The function silently picks one. The docstring (`nash.go:36-37`) says "returns the first pure NE found" but does not warn the user that the equilibrium **set** is degenerate.
- Coordination games with identical payoffs on the diagonal pick the first diagonal entry. This is consistent with comments but is a *selection*, not a fact about the game. For Pistachio's 60-FPS use case (RubberDuck auctions), this is a stable choice; for general game-theoretic analysis it is silently lossy.
- When `denomA == 0` exactly (e.g. payoff matrix `[[1,1],[1,1]]`), the mixed branch is skipped on the strict `> 1e-15` test, which is the right call for these cases.

**Recommendation:** at minimum, return a `multiplePureNE bool` or count, so the caller knows the equilibrium is non-unique. Long-term, expose support enumeration.

### F4 — Shapley exact-branch threshold is n ≤ 12 (topic asked n ≤ 20). Severity: **medium-low**.

`voting.go:124-128`: exact only for `n <= 12`, else 100,000-sample Monte-Carlo. At n=20, exact would be 2^20 = 1M coalitions × 20 marginal queries = 20M `charFunc` calls — feasible but slow (~seconds). At n=12 exact runs in <1ms. The 12 vs. 20 cut-off is conservative; bumping to 16 or 18 is cheap and matches the topic's "n ≤ ~20" expectation.

The Monte-Carlo path:
- Uses a fixed LCG seed of 42 (`voting.go:191`), so it's *deterministic* but every call produces the same 100,000 random permutations. Two calls with the same `n` and `charFunc` give bitwise-identical output, which is good for golden-file testing — but **there are no Shapley golden files**. The MC branch is reachable from `ShapleyValueWeightedVoting` for n>12 but no test exercises it.
- LCG modulo bias: `int(lcgNext() % uint64(i+1))` (`voting.go:204`). For i+1 not dividing 2^64, lower-index swaps are slightly more likely. Negligible at n=20 (~10^-18 bias) but worth a comment.
- `iterations := 100000` is hard-coded; doc says "~1e-3 relative". For a Shapley value of ~1, 100k samples on n=20 has standard error ~σ/sqrt(N) where σ is the per-permutation marginal std-dev — easily 1e-2 in pathological games (e.g. veto-player games where most marginals are 0 except one). Doc's 1e-3 is optimistic by ~10×.

### F5 — Banzhaf normalization choice is non-standard. Severity: **low**.

`voting.go:73-91`. Returns the **normalized** Banzhaf index β'_i = β_i / Σ β_j (sums to 1). The "raw" Banzhaf β_i = swings_i / 2^(n-1) is not exposed. Both are used in the literature; Penrose's original (1946) version is the raw form. Docstring `voting.go:24-25` says "normalized so that all indices sum to 1" — accurate, but a `BanzhafIndexRaw` companion would be useful for power-comparison across different voting bodies.

The degenerate-game fallback (`voting.go:80-86`) returns uniform 1/n if no voter ever swings — debatable. The literature usually reports **0** for all voters in this case (no power exists). The current choice prevents NaN propagation and is documented.

### F6 — `ShapleyValueWeightedVoting` double-normalizes. Severity: **low**.

`voting.go:271-291`: calls `ShapleyValue` (which already returns values summing to v(N)=1 for a 0/1 simple game, by the efficiency axiom), then *re-normalizes* defensively. Fine for robustness but the re-normalization absorbs/hides any numerical violation of efficiency that would otherwise surface as a useful test signal. Comment at line 273 acknowledges this.

### F7 — `KellyGrowthRate` ruin handling is silent. Severity: **low**.

`kelly.go:122-136`. For `f >= 1` on a non-certain bet, `loseTerm = 1 - f <= 0` triggers `NaN`. Test (`gametheory_test.go:808-814`) accepts either `NaN` or `-Inf`. The code path always produces NaN (early `<= 0` return), never `-Inf`. Doc-vs-test alignment: doc says "returns NaN", test accepts both. Tighten the test or the doc, not both.

### F8 — `KellyFractionMultiple` independent-bet assumption silently invalid. Severity: **low**.

`kelly.go:78-107`. Computes per-bet Kelly independently then scales the *positive* sum down to ≤1. Doc explicitly says this is exact only for independent bets. No test for correlated bets (none would be expected since the formula is wrong for them); fine. Worth noting that real portfolio Kelly requires solving a QP — this is the next-step extension if a consumer needs it.

### F9 — Bandit `EpsilonGreedy` random-arm draw can theoretically return n. Severity: **negligible**.

`bandit.go:203`: `int(rng.Float64() * float64(n))`. `rand.Float64()` returns `[0,1)`, so `int(... * n)` is `[0, n-1]`. Safe with the standard library RNG; fragile if a custom `Float64()` ever returns a value just below 1.0 + ULP.

### F10 — UCB1 `log(0)` when `totalPulls = 0` after the unexplored-arm guard. Severity: **negligible**.

`bandit.go:51`. Unreachable in practice because the unexplored-arm guard at lines 44-48 returns early whenever any arm has count 0, and if all counts are positive then `totalPulls >= n > 0`. Worth a one-line assert or doc note.

---

## Golden-file coverage

Two files exist:
- `testdata/gametheory/nash_2x2.json` — 10 cases, tolerances 1e-12 to 1e-10. Covers matching pennies, prisoner's dilemma, battle of sexes, dominant strategy, coordination, hawk-dove, saddle point, all-equal, asymmetric mixed, stag hunt. Solid coverage of the **2x2 case**.
- `testdata/gametheory/gale_shapley.json` — 8 cases (1x1, 2x2 aligned, 2x2 competing, 3x3 classic, 3x3 same-receiver, 4x4 identity, 4x4 reverse, 3x3 cyclic). Good.

**Missing golden vectors:**
- `Minimax` — no golden file at all. The 5 unit tests use 0.05-tolerance assertions or ad-hoc bounds. Cross-language validation is impossible.
- `BanzhafIndex` — no golden file. Tests are inline.
- `ShapleyValue` and `ShapleyValueWeightedVoting` — no golden file. Inline tests cover symmetric/dictator/unanimity/efficiency but not the Monte-Carlo path (n>12).
- `UCB1`, `ThompsonSampling`, `EpsilonGreedy` — no golden file (RNG-dependent, harder).
- `KellyFraction`, `KellyFractionMultiple`, `KellyGrowthRate` — no golden file. Per CLAUDE.md target of 30 vectors per function, this is a gap of 5+ functions × ~25 vectors each.

**Per CLAUDE.md "minimum 20 vectors per function, target 30":** Nash2x2 has 10 (under), Gale-Shapley 8 (under), the rest 0. The package is *substantially* below the design bar on golden coverage.

---

## Cross-cutting numerical observations

1. **No `math/big` reference path.** Per CLAUDE.md the canonical way to generate goldens is `math/big` at 256-bit precision. Nash2x2 expected values (e.g. `0.6666666666666666`, `0.333333333333333` — note the truncated digits in `nash_2x2.json:46,82,91`) appear to be hand-typed double-precision representations, not big-rational round-trips. The tolerance of 1e-10 hides this, but a true 1e-15 round-trip would require regenerating from `math/big`.

2. **No IEEE-754 edge-case vectors.** CLAUDE.md mandates "+Inf, -Inf, NaN, -0.0, subnormals" — none of the golden files contain such inputs. `KellyFraction(NaN, 1.0)` for example: at `kelly.go:37` the comparison `prob <= 0` is false when prob is NaN, so falls through to `prob >= 1` (also false), then computes `q=1-NaN=NaN` and returns NaN. Untested.

3. **No degeneracy/cycling tests.** Lemke-Howson cycling is the canonical numerical pitfall in mixed-NE computation; it's irrelevant here only because Lemke-Howson is absent. If/when added, lexicographic perturbation (Lemke 1965) is the standard fix, and golden files should include known-degenerate games (e.g. games on the boundary of the polytope).

4. **Minimax does not assert zero-sum.** Doc says "two-player zero-sum game" but the function takes only `payoff` (the row player's matrix); the column player's payoff is implicitly `-payoff`. If a caller passes a general-sum matrix expecting bimatrix Nash, results are wrong silently. Consider an assertion or a separate `BimatrixNash` function.

---

## Recommendations (ordered)

1. **Add a Minimax convergence-bracket return** (`v_t`, `V_t`) and an `epsilon` argument; use simplex/LP for exact value when m, n are modest.
2. **Replace `1e-15` indifference test in `NashEquilibrium2x2` with a relative tolerance** scaled by payoff magnitude.
3. **Generate `math/big` golden files** for Minimax (5 small zero-sum games), BanzhafIndex (10 weighted-voting cases), ShapleyValue (10 cases including 1 unanimity, 1 veto, 1 superadditive, 1 cost-sharing), and Kelly functions (20+ each).
4. **Add IEEE-754 edge-case rows** to all golden files (NaN/Inf/-0).
5. **Bump Shapley exact threshold from 12 to 18** (still <1s).
6. **Document Banzhaf normalization** and add `BanzhafIndexRaw`.
7. **Add support enumeration for general bimatrix games** (no Lemke-Howson needed up to ~5×5; iterate over support pairs and solve the linear system).
8. **Add replicator dynamics** with RK4 integrator and a long-horizon-stability test (e.g. RPS limit cycle preserves H = -ln(p1·p2·p3) to 1e-9 over 10^6 steps).
9. **Add iterated/repeated game convergence detector** (Aumann folk-theorem playground).

---

## Files referenced (absolute paths)

- `C:\limitless\foundation\reality\gametheory\nash.go`
- `C:\limitless\foundation\reality\gametheory\voting.go`
- `C:\limitless\foundation\reality\gametheory\matching.go`
- `C:\limitless\foundation\reality\gametheory\bandit.go`
- `C:\limitless\foundation\reality\gametheory\kelly.go`
- `C:\limitless\foundation\reality\gametheory\gametheory_test.go`
- `C:\limitless\foundation\reality\testdata\gametheory\nash_2x2.json`
- `C:\limitless\foundation\reality\testdata\gametheory\gale_shapley.json`
