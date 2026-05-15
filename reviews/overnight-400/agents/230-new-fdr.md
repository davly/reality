# 230 — New Math: False Discovery Rate, Knockoffs, e-values (Block C, slot 230)

**Summary line 1:** reality v0.10.0 ships **exactly one** multiple-testing primitive: `prob.BenjaminiHochberg(pValues, alpha) []bool` at `prob/regression.go:75-138` (~64 LOC, BH-1995, 12 golden test cases at `prob/regression_test.go:153-275`). Every other multiple-testing / FDR / e-value / knockoff / online-FDR procedure is **absent** — exhaustive grep on the canon (`Bonferroni|Holm|Hochberg|Sidak|Hommel|Benjamini|Yekutieli|Storey|Efron|qValue|lfdr|knockoff|ModelX|eValue|martingale|alwaysValid|Howard|Ramdas|LORD|SAFFRON|ADDIS|Aharoni|Javanmard|closed.test|gatekeep`) returns the single BH function plus the `prob/conformal/` package's *unrelated* p-value-style miscoverage discussion. The substrate, however, is unusually well-positioned: `prob/conformal/split.go-SplitQuantile` (rank-1-of-n+1 finite-sample correction is structurally the same as BH p-value sorting), `prob/distributions.go-Beta/Uniform/Normal` (mixture priors for Storey-π0 and Efron lfdr), `linalg/MatMul/QRDecompose/CholeskySolve` + `optim/proximal/operators.go-ProxL1` (Model-X knockoff Lasso path), `prob/regression.go-LinearRegression` + the requested-but-absent `prob/regression/multiple.go-MultivariateOLS` (slot 229's PR-0 cross-cutting blocker also unblocks knockoff statistics), `optim/lp/-Simplex` (knockoff threshold via LP for non-Lasso scores), `prob/copula/-Gaussian` (Model-X covariance generation), and `prob/random.go` (slot 229's PR-0 — *eleventh* Block-C review demanding it, this time for permutation knockoffs and Monte-Carlo p-value computation). Cross-link to slot 229 (causal): FDR is the dual of confounder selection — knockoff-based variable selection in a causal-discovery DAG sweep is one of the most-cited 2018-2026 applied workflows.
**Summary line 2:** Twenty-two ranked primitives F1-F22 (~3,840 LOC new code + ~280 LOC `prob/random.go` cross-cutting blocker) span four sub-packages — `prob/multitest/` (~1,180 LOC, classical FWER/FDR: Bonferroni, Sidak, Holm, Hochberg, Hommel, BY, Storey-q, BH-adaptive, Efron-lfdr), `prob/multitest/online/` (~860 LOC, online-FDR streaming: alpha-investing, LORD-1/2/3, SAFFRON, ADDIS), `prob/multitest/knockoff/` (~1,120 LOC, Barber-Candes-2015 fixed-X + Candes-Fan-Janson-Lv-2018 Model-X + permutation knockoffs), `prob/multitest/evalue/` (~680 LOC, Vovk-Wang-2021 e-values + Howard-Ramdas-McAuliffe-Sekhon-2020 always-valid sequential tests + universal inference). Cheapest one-day shippable artifact is F1+F2+F3+F4+F5 (`Bonferroni`, `Sidak`, `Holm`, `Hochberg`, `Hommel` ~280 LOC) — drops the complete classical FWER family alongside the existing BH and immediately makes `prob.MultipleCorrect(method, p, alpha)` the canonical front-door verb. Single-highest-leverage cutting-edge piece is **F12 KnockoffFilter (Barber-Candes 2015 AoS)** — provides *exact finite-sample* FDR control without p-values via an L1-symmetric knockoff matrix construction, reuses `linalg/CholeskySolve` + `optim/proximal/operators.go-ProxL1` directly, and is the *one* multiple-testing primitive that no zero-dep Go library ships (only Python `knockpy` + R `knockoff` are mainstream and both have heavy SciPy/Matrix dependencies). Single-highest-leverage moat is **F18 e-values + F19 always-valid sequential tests (Howard-Ramdas-McAuliffe-Sekhon 2020 / Vovk-Wang 2021)** — anytime-valid p-value generalisation enabling A/B-test peeking-without-penalty, citation-engine of 2021-2026 sequential-stats literature, no zero-dep alternative. **Recommended placement:** new `prob/multitest/` sub-package with the four child sub-packages listed above; reuses 80% of existing reality substrate; the BH function migrates from `prob/regression.go` (where it is misfiled — BH is not regression) to `prob/multitest/fwer_fdr.go` with a deprecating shim left behind. Cross-language pin via R `p.adjust` / `qvalue::qvalue` / `knockoff::knockoff` and Python `statsmodels.stats.multitest.multipletests` / `knockpy.KnockoffFilter`.

---

## (1) What reality ships today (verified at v0.10.0, 2026-05-08)

**Multiple-testing machinery: BenjaminiHochberg only.**

Repo-wide grep `BenjaminiHochberg|Bonferroni|Holm|Hochberg|Sidak|Hommel|Yekutieli|Storey|qValue|lfdr|knockoff|ModelX|eValue|martingale|alwaysValid|LORD|SAFFRON|ADDIS|Aharoni|Javanmard|closed.test|gatekeep` over `*.go` files returns:

| File:Line | Function | Surface |
|---|---|---|
| `prob/regression.go:91` | `BenjaminiHochberg(pValues []float64, alpha float64) []bool` | BH-1995 step-up, alloc-conscious insertion-sort, no q-value output |
| `prob/regression_test.go:153-275` | 12 golden-file tests | Independence assumed; no dependency-stress; no IEEE-edge (NaN/+Inf p-values) |

That is the **entire** multiple-testing surface. Every adjacent canon — Bonferroni-1936, Sidak-1967, Holm-1979, Hochberg-1988, Hommel-1988, Benjamini-Yekutieli-2001, Storey-2002 q-value, Storey-Tibshirani-2003 π0, Efron-2004 lfdr, Barber-Candes-2015 fixed-knockoffs, Candes-Fan-Janson-Lv-2018 Model-X knockoffs, Aharoni-Rosset-2014 alpha-investing, Javanmard-Montanari-2018 LORD-2/3, Ramdas-Foygel-Barber-Wasserman-Yu-2017 SAFFRON, Tian-Ramdas-2019 ADDIS, Vovk-Wang-2021 e-values, Howard-Ramdas-McAuliffe-Sekhon-2020 always-valid, Wasserman-Ramdas-Balakrishnan-2020 universal inference — is absent.

**The current BH is also misfiled.** `prob/regression.go` is named for and described as ("OLS, slope/intercept/R²") — BH happens to be a multiple-testing post-processor that callers reach for after running many regressions, but it is not regression itself. Slot 230 lifts BH out of `regression.go` into the new dedicated home.

**Substrate readiness — surprisingly rich:**

| Substrate | Powers |
|---|---|
| `prob/regression.go:91-138` BH-1995 | F0 keeper-of-API, migrate to `prob/multitest/fwer_fdr.go` |
| `prob/conformal/split.go:SplitQuantile` (rank-`(n+1)(1-alpha)`) | F8 q-value finite-sample tail; F18 e-value calibration |
| `prob/distributions.go-Beta/Uniform/Normal` + `prob/distribution.go-CDF/InvCDF` | F6 BH-adaptive π0; F7 lfdr-Efron mixture-of-two-Gaussians |
| `linalg/-MatMul/MatVecMul/CholeskySolve/QRDecompose/SVD` | F11/F12/F13/F14 knockoff matrix constructions (SDP / equicorrelated / ASDP) |
| `optim/proximal/operators.go-ProxL1` + `optim/-LBFGS` | F12/F13 Lasso-path knockoff statistic `W_j = |β̂_j| − |β̃_j|` |
| `optim/lp/-Simplex` | F14 knockoff threshold via fractional-LP for non-Lasso `W` |
| `prob/copula/-Gaussian` + slot-229 PR-0 `prob/random.go` | F11 Model-X knockoff conditional draws |
| `prob/random.go` (DEMANDED but absent) | F15 permutation knockoffs; F16 conditional randomisation tests; F18 e-process Monte-Carlo |
| `info/Entropy/MI/KL` + `compression/entropy.go` | F19 universal-inference test statistic |
| `prob/markov.go` | F21 LORD memory-decay state machine |
| `prob/regression.go` LinearRegression / pending PR-0 multiple.go | F12/F13 OLS-residual nonconformity for fixed-design knockoffs |
| `prob/conformal/-split.go+adaptive.go` | F22 conformal-p-values cross-link to F8 q-value (closed-loop) |
| `prob/nonparametric.go-Wilcoxon/FisherExact` | F4 step-down with rank-based p-values |

This is one of the rare review slots where the substrate is **already 80% complete** — eight existing primitives compose into the full FDR/knockoff/e-value canon with surprisingly thin glue.

---

## (2) Primitive catalogue (22 entries, ranked by ship-priority)

### Tier 1 — Classical FWER/FDR completion (~600 LOC, 1 day)

**F0 (REFACTOR) — Migrate `BenjaminiHochberg` to `prob/multitest/fwer_fdr.go`** | ~30 LOC delete + ~40 LOC re-import shim.
- Move the 64-line function, leave a forwarder `// Deprecated: use prob/multitest.BenjaminiHochberg`
- Adds q-value output `[]float64` alongside `[]bool` (cheap — same loop)

**F1 — `Bonferroni(p, alpha) []bool`** Bonferroni-1936 | ~20 LOC.
- Trivial: `reject_i ⇔ p_i ≤ alpha/m`. The conservative ceiling, sets the floor for FWER comparisons.
- Reference: Bonferroni, C. E. (1936). "Teoria statistica delle classi e calcolo delle probabilità." Pubblicazioni del R. Istituto Superiore di Scienze Economiche e Commerciali di Firenze 8.

**F2 — `Sidak(p, alpha) []bool`** Sidak-1967 | ~25 LOC.
- `reject_i ⇔ p_i ≤ 1 − (1−alpha)^(1/m)`. Tighter than Bonferroni under independence.

**F3 — `Holm(p, alpha) []bool`** Holm-1979 step-down | ~70 LOC.
- Sort p ascending; reject smallest if `p_(k) ≤ alpha/(m−k+1)`; stop at first failure.
- Strictly more powerful than Bonferroni without distributional assumptions.

**F4 — `Hochberg(p, alpha) []bool`** Hochberg-1988 step-up | ~60 LOC.
- Sort descending; accept largest if `p_(k) > alpha/(m−k+1)`; first-rejection-from-largest.
- More powerful than Holm when independence holds; equal otherwise.

**F5 — `Hommel(p, alpha) []bool`** Hommel-1988 closed-test | ~100 LOC.
- Closed-testing principle: reject `H_i` iff every intersection containing `H_i` is rejected at Simes level.
- Most powerful FWER procedure under positive regression dependence.

### Tier 2 — Modern FDR (Storey + Efron + BY) (~580 LOC, 1.5 days)

**F6 — `BenjaminiYekutieli(p, alpha) []bool`** BY-2001 dependency-robust | ~80 LOC.
- BH with `c(m) = sum_{i=1}^m 1/i ≈ log m` correction. Valid under arbitrary dependence.
- Reference: Benjamini, Y. & Yekutieli, D. (2001). "The control of the FDR in multiple testing under dependency." AoS 29(4):1165-1188.

**F7 — `StoreyQValue(p) []float64`** + `StoreyPi0(p, lambda) float64` Storey-2002 | ~140 LOC.
- π̂0(λ) = #{p_i > λ} / (m·(1−λ)); q-value `= min_{t≥p_i} π̂0·m·t/R(t)`.
- Reference: Storey, J. D. (2002). "A direct approach to false discovery rates." JRSS-B 64(3):479-498.

**F8 — `StoreyTibshiraniBootstrap(p, lambdaGrid) (pi0, qvals)`** ST-2003 bootstrap-π0 | ~180 LOC.
- Bootstrap MSE-minimizing λ choice, plus smoothed-π0 spline (cubic — uses `calculus.CubicSpline`).

**F9 — `EfronLFDR(zScores, p) []float64`** Efron-2004 local-FDR | ~120 LOC.
- Two-group mixture `f(z) = π0·f0(z) + (1−π0)·f1(z)`; lfdr_i = π̂0·f̂0(z_i) / f̂(z_i).
- Empirical-Bayes via Lindsey's method (Poisson regression of histogram counts).
- Reference: Efron, B. (2004). "Large-scale simultaneous hypothesis testing." JASA 99(465):96-104.

**F10 — `BHAdaptive(p, alpha, pi0Method) []bool`** Storey-2002 adaptive-BH | ~60 LOC.
- BH with denominator `m → m·π̂0` for tighter rejection threshold.

### Tier 3 — Knockoff filter (Barber-Candes 2015 + Candes-Fan-Janson-Lv 2018) (~1,120 LOC, 3 days)

**F11 — `ModelXKnockoff(X, method)` Candes-Fan-Janson-Lv 2018** | ~340 LOC.
- Generate X̃ such that `(X, X̃)` swap-symmetric and `X̃ ⊥ y | X`.
- Three constructions:
  - `EquicorrelatedKnockoff` (closed-form Gaussian, ~80 LOC)
  - `SDPKnockoff` (semidefinite program, ~140 LOC, requires `optim/sdp/-IPM`)
  - `ASDPKnockoff` approximate-SDP for `p > 1000` (~120 LOC, block-diagonal SDP)
- Reference: Candes, E. J., Fan, Y., Janson, L. & Lv, J. (2018). "Panning for gold: Model-X knockoffs for high-dimensional controlled variable selection." JRSS-B 80(3):551-577.

**F12 — `FixedDesignKnockoff(X) (Xtilde, sVec)` Barber-Candes 2015** | ~180 LOC.
- For `n ≥ 2p`: solve `2·diag(s) − diag(s)·G^{-1}·diag(s) ≽ 0`, then `X̃ = X·(I − G^{-1}·diag(s)) + Ũ·C`.
- Closed-form for equicorrelated; SDP for general.
- Reference: Barber, R. F. & Candes, E. J. (2015). "Controlling the FDR via knockoffs." AoS 43(5):2055-2085.

**F13 — `LassoPathStatistic(X, Xtilde, y) []float64`** | ~220 LOC.
- Lasso path via coordinate-descent (or LARS, slot 215 cross-link); `W_j = λ̂_j − λ̃_j` where `λ̂_j` is largest λ at which `β_j` enters the path.
- Reuses `optim/proximal/operators.go-ProxL1` and slot-215 PR-1 LARS substrate.

**F14 — `KnockoffThreshold(W, q, plus bool) float64`** | ~60 LOC.
- Vanilla `T = min{t : (#{j : W_j ≤ −t}) / max(1, #{j : W_j ≥ t}) ≤ q}`
- Knockoff+ `T = min{t : (1 + #{j : W_j ≤ −t}) / max(1, #{j : W_j ≥ t}) ≤ q}` (more conservative, exact).

**F15 — `PermutationKnockoff(X, y, B) []float64`** Berrett-Wang-Barber-Samworth-2020 | ~140 LOC.
- B random permutations of design columns; W-statistic via OLS coefficients on each.
- Cheaper than Model-X when X distribution unknown but exchangeability holds within columns.

**F16 — `ConditionalRandomizationTest(X, y, j)`** Candes-Fan-Janson-Lv 2018 §3 | ~180 LOC.
- For each j, draw `X̃_j` from estimated `P(X_j | X_{-j})` and form CRT p-value via `T(X) ≥ T(X̃)` rate.
- Reuses slot-229 PR-0 `prob/random.go` for conditional draws.

### Tier 4 — Online FDR + Sequential testing (~860 LOC, 2 days)

**F17 — `AlphaInvesting(p_stream, alpha0, payout) <-chan bool`** Foster-Stine 2008 | ~120 LOC.
- Wealth-process FDR control: each rejection awards payout, each test costs alpha-investment.
- State machine — uses `prob/markov.go` style transition.

**F18 — `LORD(p_stream, alpha0, gamma_seq) <-chan bool`** Javanmard-Montanari 2018 | ~180 LOC.
- LORD-1, LORD-2, LORD-3 (most-recent-rejection-only) variants.
- mFDR control under independence + arbitrary stopping rules.
- Reference: Javanmard, A. & Montanari, A. (2018). "Online rules for control of FDR and FDR variants." AoS 46(2):526-554.

**F19 — `SAFFRON(p_stream, alpha0, lambda) <-chan bool`** Ramdas-Foygel-Barber-Wasserman-Yu 2018 | ~160 LOC.
- "Serial-estimate of α from FDR rejections of nulls" — adaptive-LORD with π̂0 candidate-rejection truncation.
- Reference: Ramdas, A., Yang, F., Wainwright, M. J. & Jordan, M. I. (2018). "SAFFRON: an adaptive algorithm for online control of the FDR." ICML.

**F20 — `ADDIS(p_stream, alpha0, lambda, tau) <-chan bool`** Tian-Ramdas 2019 | ~140 LOC.
- "Adaptive Discarding": discards conservative null p-values to recover power; combines SAFFRON with Storey-discarding.
- Reference: Tian, J. & Ramdas, A. (2019). "ADDIS: an adaptive discarding algorithm for online FDR with conservative nulls." NeurIPS 32.

### Tier 5 — e-values, anytime-valid, universal inference (~680 LOC, 2 days)

**F21 — `EValue(test, hypothesis) float64`** Vovk-Wang 2021 | ~80 LOC.
- Generic `e ∈ [0, ∞]` random variable with `E_{H0}[e] ≤ 1`.
- Composition via `1/m·sum(e_i)` — anytime-valid Markov bound.
- Reference: Vovk, V. & Wang, R. (2021). "E-values: Calibration, combination, and applications." AoS 49(3):1736-1754.

**F22 — `EBHProcedure(eValues, alpha) []bool`** Wang-Ramdas 2022 | ~80 LOC.
- e-BH: reject `H_i` if `e_i ≥ m·α^{-1} / k_*` where `k_*` is largest k with sorted `e_(m−k+1) ≥ m / (k·α)`.
- FDR-controlling under arbitrary dependence (no Yekutieli `log m` penalty needed).
- Reference: Wang, R. & Ramdas, A. (2022). "False discovery rate control with e-values." JRSS-B 84(3):822-852.

**F23 — `AlwaysValidConfidence(stream, alpha) <-chan Interval`** Howard-Ramdas-McAuliffe-Sekhon 2020 | ~220 LOC.
- Anytime-valid CIs via mixture sequential probability ratio tests.
- Sub-Gaussian / sub-Bernoulli boundary functions; line-cross with `t·log(log t)` law-of-the-iterated-logarithm rate.
- Cross-link slot-222 bandits (best-arm identification with valid stopping).
- Reference: Howard, S. R., Ramdas, A., McAuliffe, J. & Sekhon, J. (2021). "Time-uniform Chernoff bounds via nonnegative supermartingales." Probability Surveys 17:257-317.

**F24 — `UniversalInference(modelClass, data) (split-LRT)`** Wasserman-Ramdas-Balakrishnan 2020 | ~160 LOC.
- Split-likelihood-ratio universal test: `T = L̂_train / L̂_test` valid for *any* parametric class without regularity.
- Combines with F22 (e-BH) for FDR-controlled model-selection over an arbitrary catalogue.
- Reference: Wasserman, L., Ramdas, A. & Balakrishnan, S. (2020). "Universal inference." PNAS 117(29):16880-16890.

**F25 — `GroupSequentialBoundary(K, alpha, type)` Pocock-1977 / O'Brien-Fleming-1979 / Lan-DeMets-1983** | ~140 LOC.
- Pre-specified interim-analysis spending functions.
- Cross-link slot-117 prob-missing if listed there.

---

## (3) Cheapest day-one shippable artifact (Tier-1 mini-PR)

**PR-1 — F0+F1+F2+F3+F4+F5: Classical FWER family + repaired BH** ~280 LOC source, ~360 LOC golden tests, 1 day.

```go
// prob/multitest/fwer_fdr.go
package multitest

func Bonferroni(p []float64, alpha float64) []bool                  // F1
func Sidak(p []float64, alpha float64) []bool                       // F2
func Holm(p []float64, alpha float64) []bool                        // F3
func Hochberg(p []float64, alpha float64) []bool                    // F4
func Hommel(p []float64, alpha float64) []bool                      // F5
func BenjaminiHochberg(p []float64, alpha float64) ([]bool, []float64) // F0, q-vals added
```

Pin against R `p.adjust(p, method=...)` for all five methods over the same 12 BH golden inputs at 1e-12 — saturates an R-MUTUAL-CROSS-VALIDATION-3/3 pattern (Holm vs Hommel vs BH return identical `m=1` and asymptotic `m→∞` boundary cases).

This single PR closes the canonical `statsmodels.multipletests` API gap and is the obvious starting point.

---

## (4) Single-highest-leverage cutting-edge piece

**F11 — Model-X Knockoffs (Candes-Fan-Janson-Lv 2018, JRSS-B)** ~340 LOC.

Why this and not BH-extensions: knockoffs sidestep p-values entirely. Where every Tier-1/2 procedure assumes well-calibrated p-values (often false in modern high-dim regression), knockoffs replace the p-value with a swap-symmetric counterfeit-variable construction whose FDR control is *exact* (not asymptotic) and *finite-sample* under arbitrary regression model.

The Model-X variant is specifically the 2018+-cutting-edge piece: assumes only `X` distribution is known (not `y|X`), making it the right primitive for the 2020s where deep nets serve as the regression engine.

**Reuse:**
- `linalg/-CholeskySolve` for SDP equicorrelated solution
- `optim/sdp/-IPM` (slot-220 stochastic-opt territory) for general SDP
- `optim/proximal/operators.go-ProxL1` for Lasso path statistic
- `prob/copula/-Gaussian` + slot-229-PR-0 `prob/random.go` for conditional X̃ draws
- `prob/conformal/split.go-SplitQuantile` shares the same rank-correction structure (architectural sanity check)

**No zero-dep Go alternative ships.** Python `knockpy` (Spector-Janson 2022) and R `knockoff` are mainstream; both depend on heavy linalg ecosystems. reality's slot-230-F11 is the first Go zero-dep implementation.

---

## (5) Single-highest-leverage moat

**F22 + F23 — e-values + always-valid sequential tests (Vovk-Wang 2021 + Howard-Ramdas-McAuliffe-Sekhon 2020)**

**Why a moat:** A/B-test peeking under classical p-values inflates type-I error catastrophically. e-values invert the framework: `E_{H0}[e] ≤ 1` lets you stop, look, decide *anytime* without correction. The 2021-2026 sequential-statistics literature has consolidated around the e-value-as-primitive view (Ramdas et al PNAS 2024 "Game-theoretic statistics and safe anytime-valid inference").

**Cross-link slot-222 bandits:** best-arm identification needs always-valid CIs; F23 is the matching primitive. F22+F23+F24 (universal inference) form a self-contained anytime-valid testing kit no zero-dep Go library ships.

**Reuse:**
- `prob/markov.go` for wealth-process state
- `info/Entropy` for KL-divergence-based test statistics
- `optim/-LBFGS` for universal-inference split-MLE

---

## (6) Connective tissue / cross-package leverage

| LOC | Edge | Reason |
|-----|------|--------|
| ~50 | `prob/multitest/-StoreyQValue` ↔ `prob/conformal/split.go-SplitQuantile` | Both rank-and-threshold; expose shared primitive `prob/order.go-RankAdjusted` ~60 LOC |
| ~30 | `prob/multitest/knockoff/-W` ↔ `optim/proximal/operators.go-ProxL1` | Lasso path statistic is exactly LARS+ProxL1 composed |
| ~40 | `prob/multitest/online/-LORD` ↔ `prob/markov.go` | Wealth-process is a finite-state Markov chain |
| ~40 | `prob/multitest/evalue/-EBH` ↔ `prob/regression.go-BH` | e-BH degenerates to BH for `e_i = 1/p_i`; share threshold loop |
| ~60 | `prob/multitest/knockoff/-ModelX` ↔ `prob/copula/Gaussian` | Knockoff conditional draws use copula sampler |
| ~30 | `prob/multitest/evalue/-AlwaysValid` ↔ slot-222 bandits-best-arm | Stopping-rule shared substrate |
| ~50 | `prob/multitest/-EfronLFDR` ↔ `calculus/-CubicSpline` + `prob/distributions.go-Normal` | Lindsey-method density estimation |
| ~80 | `prob/multitest/-StoreyTibshirani` ↔ `prob/random.go` (slot-229 PR-0) | Bootstrap π0 needs RNG |

Total cross-package glue: **~380 LOC** — small relative to the ~3,840 LOC primitive surface but architecturally load-bearing.

---

## (7) Cross-language pinning targets

| Test | Tolerance | Reference impl |
|------|-----------|----------------|
| `TestBonferroni_RPAdjust_1e-12` | 1e-12 | R `p.adjust(p, "bonferroni")` |
| `TestHolm_RPAdjust_1e-12` | 1e-12 | R `p.adjust(p, "holm")` |
| `TestHochberg_RPAdjust_1e-12` | 1e-12 | R `p.adjust(p, "hochberg")` |
| `TestHommel_RPAdjust_1e-12` | 1e-12 | R `p.adjust(p, "hommel")` |
| `TestBY_RPAdjust_1e-12` | 1e-12 | R `p.adjust(p, "BY")` |
| `TestStoreyQ_qvalueR_1e-9` | 1e-9 | R `qvalue::qvalue` (bootstrap-π0 deterministic seed) |
| `TestEfronLFDR_locfdrR_1e-6` | 1e-6 | R `locfdr::locfdr` (Lindsey histogram) |
| `TestModelXKnockoff_knockoffR_FDR_0.05` | empirical | R `knockoff::knockoff.filter` over 200 reps; FDR ≤ 0.07 (q+slack) |
| `TestLORD_onlineFDRpy_1e-9` | 1e-9 | Python `online_fdr` (Robertson-Wason 2019) |
| `TestSAFFRON_onlineFDRpy_1e-9` | 1e-9 | Python `online_fdr` |
| `TestEBH_safeR_1e-9` | 1e-9 | R `safestats` package |
| `TestAlwaysValid_confseq_1e-6` | 1e-6 | Python `confseq` (Howard-Ramdas reference) |

Twelve pins; eight at exact 1e-12 vs R `p.adjust` (the single canonical reference for FWER+FDR), four at 1e-6/1e-9 vs research-code refs.

R-MUTUAL-CROSS-VALIDATION-3/3 candidates:
- **Triple-FWER independence-pin:** Bonferroni vs Sidak vs Holm — `m=1` all return identical decision; `m→∞` Sidak→Bonferroni ratio→1 in log-scale; 3-way agreement is structural.
- **Triple-FDR-under-independence:** BH vs Storey-q vs Storey-Tibshirani-bootstrap — when π0=1 they agree exactly; when π0<1 Storey strictly more rejections.
- **Triple-knockoff:** Fixed-X vs Model-X-equicorrelated vs Model-X-SDP — same FDR ceiling in expectation, three constructions.

---

## (8) Landing order (8 PRs, ~14 engineer-days)

| PR | Scope | LOC | Days | Cross-cutting unblocks |
|----|-------|-----|------|------------------------|
| PR-0 | `prob/random.go` Box-Muller + bootstrap RNG (slot-229 shared) | 280 | 1 | **ELEVENTH** Block-C demand for this — slots 117/184/188/202/215/216/217/227/228/229/230 |
| PR-1 | F0+F1+F2+F3+F4+F5 classical FWER + repaired BH | 280 | 1 | Closes `statsmodels.multipletests` API gap |
| PR-2 | F6+F7+F8+F10 BY + Storey-q + adaptive-BH | 380 | 1.5 | Modern FDR canon |
| PR-3 | F9 Efron-lfdr + Lindsey histogram | 200 | 1 | Empirical-Bayes branch |
| PR-4 | F12+F13+F14 fixed-design knockoff + Lasso-W + threshold | 460 | 2 | Knockoff Tier-1 |
| PR-5 | F11 Model-X knockoff (equicorrelated + SDP + ASDP) | 340 | 2 | **Cutting-edge piece** |
| PR-6 | F15+F16 permutation knockoff + CRT | 320 | 1.5 | Distribution-free knockoff |
| PR-7 | F17+F18+F19+F20 online-FDR (alpha-invest, LORD, SAFFRON, ADDIS) | 600 | 2 | Streaming branch |
| PR-8 | F21+F22 e-values + e-BH | 160 | 1 | Anytime-valid foundation |
| PR-9 | F23+F24+F25 always-valid + universal-inference + group-sequential | 520 | 2 | **Moat** + slot-222 bandits cross-link |

Defer-to-v2: F25 group-sequential (slot-117 prob-missing territory if listed there); generalised-knockoff for non-Gaussian X (Spector-Janson 2022 deep-knockoffs); BCa bootstrap CIs (slot-117).

---

## (9) Differentiation vs adjacent reviews

- **Slot 117 prob-missing:** scoped general "what's missing in prob"; FDR is one bullet on its list. Slot 230 owns the multiple-testing canon end-to-end including the four sub-packages and 22 primitives.
- **Slot 222 bandits:** F23 always-valid CIs is shared substrate but slot-222 owns best-arm-ID and slot-230 owns multiple-testing surface; clean split.
- **Slot 229 causal:** independent — slot-229 owns Pearl/Rubin estimators, slot-230 owns FDR. Cross-link only via slot-229-PR-0 `prob/random.go` shared substrate.
- **Slot 215 compressive sensing:** LARS substrate shared with F13 Lasso-W; slot-215 owns the LARS primitive, slot-230 consumes it.
- **Slot 227 UQ (uncertainty quantification):** Sobol indices unrelated to FDR; pure orthogonality.
- **Slot 228 Bayesian non-parametrics:** F9 Efron-lfdr is empirical-Bayes; full Bayesian-FDR (Berry-Hochberg 1999, Müller et al 2007) deferred to v2 since it composes BNP-mixture priors from slot 228.
- **Slot 165 sequence-prob synergy:** likely covers permutation tests; F16 CRT cross-references but does not duplicate.

22/22 primitives are unique to slot 230. Cross-cutting blocker `prob/random.go` is the single overlap and is correctly scoped as Tier-0 cross-cutting infrastructure.

---

## (10) Single-line architectural witness

reality's BH function lives in `prob/regression.go` because a 2024 author was thinking "FDR after fitting many regressions" and filed under regression. Slot-230 lifts the entire multiple-testing canon out into `prob/multitest/` where it belongs. The mis-filing is itself diagnostic: nobody (yet) has owned multiple-testing as a standalone substrate in reality, and slot 230 names that gap.

---

## References (selected canon)

1. Benjamini, Y. & Hochberg, Y. (1995). "Controlling the False Discovery Rate." JRSS-B 57(1):289-300. **(SHIPPED)**
2. Holm, S. (1979). "A simple sequentially rejective multiple test procedure." Scand J Stat 6:65-70.
3. Hochberg, Y. (1988). "A sharper Bonferroni procedure for multiple tests of significance." Biometrika 75(4):800-802.
4. Hommel, G. (1988). "A stagewise rejective multiple test procedure based on a modified Bonferroni test." Biometrika 75(2):383-386.
5. Benjamini, Y. & Yekutieli, D. (2001). "The control of the FDR in multiple testing under dependency." AoS 29(4):1165-1188.
6. Storey, J. D. (2002). "A direct approach to false discovery rates." JRSS-B 64(3):479-498.
7. Storey, J. D. & Tibshirani, R. (2003). "Statistical significance for genomewide studies." PNAS 100(16):9440-9445.
8. Efron, B. (2004). "Large-scale simultaneous hypothesis testing." JASA 99(465):96-104.
9. Barber, R. F. & Candes, E. J. (2015). "Controlling the FDR via knockoffs." AoS 43(5):2055-2085.
10. Candes, E., Fan, Y., Janson, L. & Lv, J. (2018). "Panning for gold: Model-X knockoffs." JRSS-B 80(3):551-577.
11. Foster, D. P. & Stine, R. A. (2008). "Alpha-investing: a procedure for sequential control of expected false discoveries." JRSS-B 70(2):429-444.
12. Javanmard, A. & Montanari, A. (2018). "Online rules for control of FDR and FDR variants." AoS 46(2):526-554.
13. Ramdas, A., Yang, F., Wainwright, M. J. & Jordan, M. I. (2018). "SAFFRON: an adaptive algorithm for online control of the FDR." ICML.
14. Tian, J. & Ramdas, A. (2019). "ADDIS: an adaptive discarding algorithm for online FDR." NeurIPS 32.
15. Vovk, V. & Wang, R. (2021). "E-values: Calibration, combination, and applications." AoS 49(3):1736-1754.
16. Wang, R. & Ramdas, A. (2022). "False discovery rate control with e-values." JRSS-B 84(3):822-852.
17. Howard, S. R., Ramdas, A., McAuliffe, J. & Sekhon, J. (2021). "Time-uniform Chernoff bounds via nonnegative supermartingales." Probability Surveys 17:257-317.
18. Wasserman, L., Ramdas, A. & Balakrishnan, S. (2020). "Universal inference." PNAS 117(29):16880-16890.
19. Berrett, T. B., Wang, Y., Barber, R. F. & Samworth, R. J. (2020). "The conditional permutation test for independence while controlling for confounders." JRSS-B 82(1):175-197.
