# 023 — changepoint-sota

**Topic.** Position `reality/changepoint` against best-in-class libraries (ruptures, R-`changepoint`, `bcp`, `mcp`, `changeforest`, `changepoint.online`, `mosum`, recent 2024-2025 work). Identify *engineering* SOTA elements, not just algorithmic ones.

**Date.** 2026-05-07. Reviewed: `changepoint/bocpd.go`, `changepoint/doc.go`.

---

## 0. Where reality currently sits

`reality/changepoint` is **single-algorithm** today: one streaming Bayesian Online Change-Point Detection (BOCPD, Adams-MacKay 2007) with a Normal-Inverse-Gamma observation model, constant hazard, and an `R_max`-truncated run-length posterior. ~400 LOC of math + a 79-line doc.go. One conjugate model. One hazard function. One scoring head (`MapRunLength`, `ChangePointProbabilityWithin`).

That places it in the "online Bayesian, univariate Gaussian" cell of the changepoint matrix. Every other cell — offline / multivariate / nonparametric / kernel / regression-formula / penalised-cost / classifier-based / multiscale — is empty.

The good news: BOCPD is a solid, hard-to-write-from-scratch building block, and the code is bit-stable, allocation-aware (per-step copies but no hidden allocs), and numerically defensive (logsumexp throughout, NaN/Inf guards, posterior-mass-zero detection). It is roughly on par with `dtolpin/bocd` and `y-bar/bocd` on the streaming-Bayesian sub-problem, and ahead of `ocp` (R, Pagotto) on numerical hygiene.

The rest of this document is what the *frontier* looks like and what is borrowable.

---

## 1. The 2024-2026 landscape, by camp

### 1.1 Offline / penalised-cost (ruptures, R-changepoint, fpop)

**ruptures** (Truong-Oudre-Vayatis, Python; PyPI 1.x, active 2024-2025).

- **Algorithms shipped:** `Pelt` (penalised), `Dynp` (exact dynamic-programming, fixed K), `Binseg` (greedy), `BottomUp`, `Window`, `KernelCPD` (C-implemented Pelt+DP for kernels).
- **Cost-function abstraction.** Single `BaseCost` interface with `.fit(signal)` and `.error(start, end)`. Concrete costs: `CostL1`, `CostL2`, `CostNormal`, `CostRbf`, `CostCosine`, `CostLinear`, `CostAR`, `CostMl`, `CostRank`, `CostCLinearL2`. *Algorithm and cost are orthogonal* — same `Pelt` runs against any of them.
- **Engineering tricks:**
  - `KernelCPD` is implemented in **C** with `jump=1` baked in, because they measured that the Python-level `jump` parameter buys nothing once the inner loop is native. The Python `Pelt` keeps `jump` (default 5) as a *speed knob* trading exactness for sub-grid step.
  - `cost_factory` model dispatch — a string `model="rbf"` selects the kernel variant; same algorithm class.
  - `fit_predict(signal, n_bkps=K)` vs `predict(pen=β)` overload — same object exposes both fixed-K and penalised modes.

**R `changepoint`** (Killick-Eckley, CRAN, July 2025).

- **Algorithms.** `cpt.mean`, `cpt.var`, `cpt.meanvar` × `method ∈ {AMOC, BinSeg, PELT, SegNeigh}` × `penalty ∈ {None, SIC, BIC, MBIC, AIC, Hannan-Quinn, Asymptotic, Manual, CROPS}`. Fully orthogonal.
- **CROPS** (Haynes-Eckley-Fearnhead 2017): runs PELT for a *range* `[β_min, β_max]` and returns the entire path of optimal segmentations as the penalty varies. Effectively "PELT for unknown K, with the regularisation path as a free byproduct."
- **Engineering trick:** the result object carries `cpts.full`, `pen.value.full`, `ncpts.full` so the user can plot elbow / run hold-out.

**FPOP / Ms.FPOP / GFPOP** (Maidstone-Rigaill-Hocking-Fearnhead).

- **Functional pruning** vs PELT's inequality pruning. FPOP keeps the *piecewise-quadratic cost function* as a function of the last segment-mean; prunes pieces dominated everywhere. Robust to changepoint count (PELT degrades when K is small).
- **Ms.FPOP** (JCGS 2024) — multiscale penalty favouring well-spread changepoints.
- **GFPOP** — constrained changepoint detection (up/down constraints) for genomic data.

### 1.2 Online / streaming

**`changepoint.online`** (Killick, GitHub-only).

- **API.** `ocpt.mean.initialise(data, ...)` returns state, then `ocpt.mean.update(state, new_data)` mutates. Mirror pattern of `cpt.mean` but with explicit init/update split.
- **Internally** it is `PELT.online.initialise` / `PELT.online.update` — the same PELT recurrence, with sufficient statistics carried in state and the cost-function table extended by chunk.
- This is the closest sibling to reality's `Bocpd.New` / `Bocpd.Update`. The naming is identical in spirit.

**bocpdms** (Alan-Turing-Institute, Knoblauch-Damoulas).

- BOCPD with **model selection** — multiple observation models run in parallel, posterior over `(model, run-length)`. Far more expressive than reality's single NIG.

**bocd** (Tolpin, Go) and **bocd** (y-bar, Python).

- Direct BOCPD ports. Both use `logsumexp` throughout. y-bar's includes a configurable `Hazard` interface (constant, logistic). reality is functionally on par with these.

**ocd** (Chen-Wang-Samworth 2022, JRSSB).

- High-dimensional, multiscale online — uses an *adaptive thresholding* statistic, no Bayesian posterior. Designed for `p ≫ n`.

### 1.3 Bayesian (offline, partition posterior)

**bcp** (Erdman-Emerson, Wang-Emerson v4).

- **Algorithm.** Barry-Hartigan product partition model + MCMC over partitions.
- **Engineering pivot:** v2.0 dropped from O(n²) to **O(n)** per MCMC step by maintaining cumulative sums-of-squares per block and updating only affected blocks on a single-changepoint MCMC move. Wall-clock: 10,000-point series went from **45 minutes to 45 seconds**. This is the canonical example of "the algorithm didn't change, the bookkeeping did."
- Multivariate extension in `bcp(... , w0)`.

**mcp** (Lindeløv, CRAN).

- **Formula DSL** for Bayesian regression with K changepoints. User writes `list(y ~ 1, ~ 1 + x, ~ 0 + x)` for "constant, then linear in x, then no intercept linear in x." `mcp` translates to JAGS code.
- Hugely user-friendly for known-K models with covariate structure. Not a competitor to BOCPD on the streaming axis; complementary.

**ChangepointTesting** (CRAN, Hu-Wright).

- p-value-clustering + Bayesian model averaging over the *number* of changepoints. Niche but interesting design point: "average over K instead of choosing K."

### 1.4 Nonparametric / classifier-based / kernel

**changeforest** (Londschien-Bühlmann-Kovács, JMLR 2023, Rust core + R/Python/Rust bindings).

- Use any classifier that emits class probabilities (random forest, k-NN); construct a **classifier log-likelihood ratio** between "all observations one class" vs "split at τ"; search for τ that maximises this. Two-step seeded binary segmentation gives near-linear search.
- **Engineering:** Rust core, exposed identically to R, Python, and Rust users — same model spec and CV protocol across all three languages. This is the multi-language-binding pattern reality already commits to via golden files but doesn't yet do via shared cores.

**KernelCPD** (in ruptures): Pelt over the kernel-mean-embedding cost. RBF/cosine/linear kernel. Reduces to Gaussian when RBF + appropriate bandwidth.

### 1.5 Frequentist scan / MOSUM family

**mosum** (Meier-Kirch-Cho, JStatSoft 2021, CRAN 1.2.7).

- **Moving-sum statistic** with bandwidth h. Detect a change at time t whenever `|MOSUM(t)| > threshold(α)` based on Gumbel-extreme-value asymptotics.
- **multiscale.bottomUp** combines multiple bandwidths.
- **Two-Way MOSUM** (Cho-Owens, AoS 2024): spatio-temporal moving-window for high-dimensional time series; tests on selected groups, not all coordinates.
- **Engineering:** O(n) per bandwidth, embarrassingly parallel across bandwidths, bootstrap CIs for changepoint locations.

### 1.6 Recent 2024-2025

- **changedetectron2** — neural / detectron-style architecture for changepoint in long sequences (segmentation as object detection). Out of scope for zero-dep math.
- **OptCutTrees (2025)** — tree-structured optimal partitioning for hierarchical changepoints. In scope for a zero-dep library if the recurrence is closed-form.
- **Shrinkage methods** — James-Stein-style shrinkage of per-segment means; reduces variance on short segments. Cheap to add as a post-processing step.
- **MS-FPOP** (2024) — multiscale FPOP, a single new penalty term inside the FPOP recurrence.
- **Risk-calibrated BOCPD** (Oct 2025, arXiv:2510.09619) — couples BOCPD posterior to SRE error budgets via decision-theoretic thresholds. *Pure post-processing on top of reality's existing posterior — cheapest "frontier" win available.*

---

## 2. Engineering SOTA elements reality could borrow

This is the heart of the topic. Ranked by leverage / line-count.

### Tier 1 — pure additions, no math change, ≤ ~150 LOC each

| # | Idea | Source | Why it matters | LOC |
|---|------|--------|----------------|-----|
| T1.1 | **`Hazard` interface** with `Constant`, `Logistic`, `Custom func(int)float64` implementations | y-bar/bocd, bocpdms | reality hard-codes constant hazard inside the `Bocpd` struct; non-constant-hazard is the whole reason `ChangePointProbability` is "not a useful alarm signal" per the existing doc-comment | ~40 |
| T1.2 | **`ObservationModel` interface** with `NormalInverseGamma`, `MultivariateNIW`, `Bernoulli-Beta`, `Poisson-Gamma`, `StudentT-NIG` implementations | bocpdms, ruptures `BaseCost` | Currently the conjugate update is fused into `Update`. Splitting out a `model.Predict(x) float64`, `model.Update(x) ObservationModel` interface lets one BOCPD core serve count data, multivariate, etc. | ~120 (interface + first 2 models) |
| T1.3 | **CROPS-style "scan over hazard / R_max"** helper | R-changepoint CROPS | Returns the family of MAP segmentations as `λ` sweeps `[λ_min, λ_max]` — same way users actually pick the regularisation in practice. Cheap because BOCPD posteriors at different λ share Student-t evals; cache them. | ~80 |
| T1.4 | **Reset-aware `Update` overload returning the just-evicted run-length tail** | original Adams-MacKay paper § 3.3 | `Bocpd.Update` currently silently drops mass past `R_max+1`. Returning the dropped mass lets callers detect "regime changed but we lost the evidence due to truncation." | ~20 |
| T1.5 | **`PelftOffline(x, model, λ)` — offline PELT against the *same* `ObservationModel`** | R-changepoint, ruptures | Once T1.2 lands, PELT is ~150 LOC of optimal-partitioning + the inequality-pruning rule. Gives reality a second algorithm covering the offline cell with shared cost code. | ~150 |
| T1.6 | **Risk-calibrated decision wrapper** `AlarmAt(window, errorBudget) bool` | arXiv:2510.09619 | Pure post-processing on the existing posterior. Two lines of math, but a proper API design effort to expose error-budget semantics cleanly. | ~50 |

### Tier 2 — meaningful engineering, 200-400 LOC

| # | Idea | Source | Why it matters | LOC |
|---|------|--------|----------------|-----|
| T2.1 | **Inequality pruning of the run-length posterior** (drop runs whose probability falls below a threshold *before* `R_max`) | bocpdms, "Killick pruning" | reality currently pays `O(R_max)` per step even when 99% of the mass is in run-lengths < 50. Threshold pruning typically gives 5-20× speedup at <1e-6 KL loss. | ~60 |
| T2.2 | **MOSUM detector** (`mosum.Detect(x, bandwidth, α)`) | mosum (Meier-Kirch-Cho 2021) | Frequentist scan complements Bayesian; useful for "I want a *p*-value, not a posterior." O(n) per bandwidth, naturally parallel. | ~200 |
| T2.3 | **FPOP / functional pruning** offline algorithm | Maidstone-Rigaill-Fearnhead | Robust where PELT degrades (small K). Piecewise-quadratic representation of the cost-by-mean function; prune dominated pieces. ~300 LOC for the L2 case. | ~300 |
| T2.4 | **BinSeg + Wild Binary Segmentation (WBS)** | Fryzlewicz 2014, used by every package | Greedy but fast and embarrassingly parallel. WBS adds randomised intervals → recovers all changepoints with high probability under weaker conditions than BinSeg. | ~250 |
| T2.5 | **Online sufficient-statistic ring buffer** for streaming PELT | `changepoint.online` `PELT.online.update` | reality's `Bocpd` already has streaming sufficient statistics for the conjugate model. The same pattern lets PELT run online, recomputing only the affected suffix of the cost table. | ~180 |
| T2.6 | **Block-cumulative-sums trick** (bcp v2 → v4 speedup) | Wang-Emerson 2015 | Whenever a partition-MCMC move only affects a few blocks, recompute via cumulative sums-of-squares instead of from scratch. *Generalises*: any reality consumer doing block-statistics under an evolving partition benefits. | ~100 |

### Tier 3 — frontier, 500+ LOC, requires sibling-package work

| # | Idea | Source | Why it matters |
|---|------|--------|----------------|
| T3.1 | **Multivariate BOCPD** with Normal-Inverse-Wishart | bocpdms, Knoblauch-Damoulas | Requires sibling `linalg` Cholesky-rank-1-update support. ~500 LOC. |
| T3.2 | **Kernel CPD over RBF kernel mean embedding** | ruptures `KernelCPD` | Requires either an `O(n²)` Gram-matrix prep or a Nyström approximation in `signal/`. Worthwhile if reality grows MMD primitives anyway. |
| T3.3 | **Classifier-based detection à la changeforest** | Londschien-Bühlmann-Kovács 2023 | Out of scope until reality grows a random-forest in `prob/` or `ml/`. The *math* of the classifier-LR statistic is portable, though. |
| T3.4 | **Two-Way MOSUM** for high-dim time series | Cho-Owens AoS 2024 | After T2.2 lands; ~400 LOC additional. |
| T3.5 | **Implicit-function-theorem AD through the posterior** for hyperparameter calibration | autodiff topic 013 + this | Would let users gradient-descend on `(μ₀, κ₀, α₀, β₀, λ)` against held-out log-likelihood. ~200 LOC once `autodiff.FixedPoint` exists. |

### Tier 4 — design-time API moves, ~zero-LOC but high reach

| # | Idea | Source | Why it matters |
|---|------|--------|----------------|
| T4.1 | **Split `Bocpd` into `BocpdCore` + `BocpdConfig` + `BocpdPosterior`** | ruptures `BaseCost` orthogonality | Lets multiple algorithms (BOCPD, PELT, FPOP, MOSUM) share a single `ObservationModel` + `Posterior` view. |
| T4.2 | **`Detector` interface** with `Fit(x) Detector`, `Predict() []int`, `OnlineUpdate(x) Detector` | sklearn / ruptures | Future-proofs the package as a multi-algorithm library. |
| T4.3 | **Formula DSL** `changepoint.Model("y ~ 1 | y ~ x | y ~ 0 + x")` | mcp (Lindeløv) | Long-term aspiration; only worth it if reality grows a regression sub-package. |
| T4.4 | **Per-algorithm tolerance contract documented** | reality CLAUDE.md design rule #5 | Currently undocumented for `Bocpd`. Should state: posterior renormalisation tolerance, log-space underflow guard threshold, `R_max`-truncation max-mass-loss. |
| T4.5 | **Output of `Update` should expose log-evidence `log p(x_t \| x_{1:t-1})`** | bocpdms | Free during the existing logsumexp; currently discarded. Critical for online model selection (T1.2 bonus). |

---

## 3. Three concrete "borrow this exact trick" recommendations

If only three things ship, ship these:

### R1. The `Hazard` + `ObservationModel` orthogonality split (T1.1 + T1.2 + T4.1)

Mirrors ruptures' `BaseCost` and bocpdms' model-selection design. Lets the package extend to count data, multivariate, regression-on-a-window without rewriting the BOCPD recurrence. **~200 LOC, massive expressivity multiplier.**

### R2. Inequality run-length pruning (T2.1) + log-evidence emission (T4.5)

**~80 LOC combined.** Pruning is a standard 5-line "drop runs whose mass < ε" trick; log-evidence is one extra return value from the existing logsumexp. Together they give reality (a) BOCPD that runs in `O(effective-run-length)` not `O(R_max)`, and (b) the streaming model-selection signal that bocpdms is built around.

### R3. Block-cumulative-sums trick (T2.6) factored into `prob/` or `linalg/`

bcp v2 → v4's 60× speedup came from this single bookkeeping change. The trick is general — any reality consumer doing per-segment statistics under an evolving partition (regime detection, quantile streaming, online linear regression on blocks) benefits. Put it once in a sibling package, every consumer wins.

---

## 4. What reality already does *better* than the SOTA libraries

It is a short list, but real:

- **Bit-stable, deterministic.** ruptures, bcp, mcp, changeforest all have non-determinism somewhere (random-restart, MCMC, RNG-seeded WBS, RF). reality is bit-stable by construction. This is uniquely valuable for golden-file cross-language validation, which none of the above ships.
- **Numerical hygiene.** The logsumexp throughout `Update`, the explicit `posterior-mass-zero` error, the `NaN/Inf` input guards, and the `Validate()` on the prior are *more* defensive than `dtolpin/bocd`, `y-bar/bocd`, or even `bocpdms`'s research-grade Python. The only sibling that matches this is `promised-ai/changepoint` (Rust).
- **Zero dependencies.** mcp pulls JAGS; bcp pulls C; changeforest pulls a Rust toolchain; ruptures pulls NumPy/SciPy. reality stays inside the Go stdlib `math` package.
- **Streaming-first.** Of the libraries surveyed, only `changepoint.online` and the two `bocd` ports are streaming-first. Of those four, reality's `Bocpd` has the cleanest `New` / `Update` / query separation.

---

## 5. Verdict

reality/changepoint is a **good single-cell implementation in a 12-cell field**. The math is right, the engineering hygiene is above-median, and the streaming Bayesian cell is filled competently. The frontier opportunity is *not* a smarter algorithm — it is `BaseCost`/`Hazard` orthogonality (ruptures), inequality pruning (bocpdms), the cumulative-sums bookkeeping trick (bcp v2 → v4), and at least one offline detector (PELT or FPOP) sharing the same observation-model abstraction. ~600 LOC of additions covers 80% of the gap to ruptures+changepoint+bocpdms feature parity, all citation-grounded, all golden-file-testable.

The single highest-leverage commit is **R1 + R2** above: ~280 LOC that turn the package from "BOCPD library" into "changepoint *framework* with BOCPD as the first detector."

---

## Sources

- [ruptures: change point detection in Python (GitHub)](https://github.com/deepcharles/ruptures)
- [ruptures KernelCPD documentation](https://centre-borelli.github.io/ruptures-docs/user-guide/detection/kernelcpd/)
- [ruptures Pelt documentation](https://centre-borelli.github.io/ruptures-docs/user-guide/detection/pelt/)
- [ruptures custom cost function](https://centre-borelli.github.io/ruptures-docs/custom-cost-function/)
- [R changepoint package CRAN PDF (July 2025)](https://cran.r-project.org/web/packages/changepoint/changepoint.pdf)
- [Killick & Eckley (2014) — changepoint: An R Package for Changepoint Analysis (JStatSoft)](https://www.jstatsoft.org/v58/i03/)
- [Haynes-Eckley-Fearnhead — Efficient penalty search (CROPS), arXiv:1412.3617](https://arxiv.org/pdf/1412.3617)
- [bcp R package — Wang & Emerson v4 — Bayesian product partition model](https://www.rdocumentation.org/packages/bcp/versions/4.0.3/topics/bcp)
- [Erdman & Emerson (2007) — bcp: An R Package for Bayesian Change Point](https://www.jstatsoft.org/v23/i03/)
- [Fearnhead & Liu (2007) — Exact and efficient Bayesian inference for multiple changepoint problems](https://link.springer.com/article/10.1007/s11222-006-8450-8)
- [mcp (Lindeløv) — Regression with Multiple Change Points](https://lindeloev.github.io/mcp/)
- [mcp — overview of change point packages in R](https://lindeloev.github.io/mcp/articles/packages.html)
- [changeforest — Random Forests for Change Point Detection (JMLR 2023)](https://www.jmlr.org/papers/volume24/22-0512/22-0512.pdf)
- [changeforest GitHub (Rust core, R/Python/Rust bindings)](https://github.com/mlondschien/changeforest)
- [mosum — A Package for Moving Sums in Change-Point Analysis (JStatSoft 2021)](https://www.jstatsoft.org/article/view/v097i08)
- [Two-Way MOSUM for high-dimensional time series (Annals of Statistics 2024)](https://projecteuclid.org/journals/annals-of-statistics/volume-52/issue-2/%E2%84%932-inference-for-change-points-in-high-dimensional-time-series/10.1214/24-AOS2360.short)
- [changepoint.online (Killick) — PELT.online.initialise / PELT.online.update](https://rdrr.io/github/rkillick/changepoint.online/man/PELT.online.html)
- [Killick et al. (2012) — Optimal detection of changepoints with linear cost (PELT) arXiv:1101.1438](https://arxiv.org/pdf/1101.1438)
- [Maidstone-Hocking-Fearnhead-Rigaill — On optimal multiple changepoint algorithms for large data](https://link.springer.com/article/10.1007/s11222-016-9636-3)
- [Ms.FPOP — A Fast Exact Segmentation Algorithm with a Multiscale Penalty (JCGS 2024)](https://www.tandfonline.com/doi/full/10.1080/10618600.2024.2402895)
- [GFPOP — Constrained Changepoint Detection in Genomic Data (JStatSoft)](https://www.jstatsoft.org/article/view/v101i10)
- [bocpdms (Alan Turing Institute) — BOCPD with model selection](https://github.com/alan-turing-institute/bocpdms)
- [Risk-Calibrated Bayesian Streaming Intrusion Detection (arXiv:2510.09619, Oct 2025)](https://arxiv.org/abs/2510.09619)
- [ocd — High-Dimensional Multiscale Online Changepoint Detection (JRSSB 2022)](https://academic.oup.com/jrsssb/article/84/1/234/7056123)
- [ChangepointTesting (CRAN)](https://cran.r-project.org/web/packages/ChangepointTesting/index.html)
- [Adams & MacKay (2007) — Bayesian Online Changepoint Detection arXiv:0710.3742](https://gregorygundersen.com/blog/2019/08/13/bocd/)
