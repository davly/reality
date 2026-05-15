# 022 — changepoint: missing algorithms

**Agent:** 022 / 400
**Topic:** changepoint-missing — `changepoint` package ships only BOCPD; enumerate canonical algorithms that should land.
**Date:** 2026-05-07
**Confirmed by 021:** package contents are `bocpd.go`, `bocpd_expansion_test.go`, `bocpd_test.go`, `doc.go`, `infogeo_test.go` — BOCPD-only.

## Verified gap

`C:/limitless/foundation/reality/changepoint/` ships exactly one detector
(Adams-MacKay 2007 BOCPD, Normal-Inverse-Gamma conjugate, constant-hazard
truncated to `R_max=500`). No CUSUM, no PELT, no e-divisive, no kernel CPD,
no Binary Segmentation family, no MOSUM, no high-dimensional / multivariate
detector, no offline exact dynamic-programming detector, no Bayesian
alternative parameterisation. The `doc.go` "consumers" list contains a
single intra-repo cross-test (`infogeo_test.go`); flagship consumers remain
hunt-citations not import-citations as of 2026-05-05.

This is one detector against the ruptures-library standard of six
(Pelt, Dynp, BinSeg, BottomUp, Window, KernelCPD), against the changepoint::cpt
R-package standard of ten+, and against the published 2024-2025 frontier
(FOCuS, OCD, changeforest, MOSUM-multiscale, mixFOCuS, disMOSUM, CHAD,
tidychangepoint).

## Tier 1 — must ship (canonical, decades-stable, single-author can implement, golden-file ready)

These are textbook detectors with closed-form or O(n log n) reference
implementations, stable since the 20th century or early 2010s, no
unsettled hyperparameter debate, and minimax-optimal or provably-consistent
under standard assumptions.

| # | Detector | Reference | Mode | Cost | Why Tier 1 |
|---|----------|-----------|------|------|------------|
| T1.1 | **CUSUM** (Page test) | Page 1954 | Online single-CP | O(1)/step | Textbook baseline. ARL₀/EDD trade-off has closed form. Industry-standard SPC. Reality has zero online detector other than BOCPD; CUSUM is the floor. |
| T1.2 | **GLR (generalised likelihood ratio)** | Lorden 1971; Lai 1995 | Online single-CP | O(t)/step | Optimal-EDD complement to CUSUM under composite alternative. Window-limited variant gives O(W). |
| T1.3 | **PELT** (Pruned Exact Linear Time) | Killick-Fearnhead-Eckley 2012 | Offline multi-CP | O(n) avg, O(n²) worst | The modal offline detector. Exact (not heuristic). Pruning rule is one inequality. Cost-function-pluggable (mean, variance, mean+var, regression, NIG, exponential). Proven linear in n when CPs grow linearly. |
| T1.4 | **OP** (Optimal Partitioning, dynamic programming) | Jackson et al. 2005 | Offline multi-CP | O(n²) | The substrate PELT prunes. Ship both — OP is the correctness witness for PELT and the only choice when the pruning inequality fails (non-additive costs). |
| T1.5 | **Binary Segmentation** | Vostrikova 1981; Scott-Knott 1974 | Offline multi-CP | O(n log n) | Greedy single-CP recursion. Universal baseline: every survey reports it. Fast but inconsistent for short segments — ship as the speed-floor, not the accuracy-floor. |
| T1.6 | **Wild Binary Segmentation** (WBS) | Fryzlewicz 2014 | Offline multi-CP | O(Mn) | Fixes BinSeg's short-segment failure by drawing M random sub-intervals. Consistent. Threshold-based. WBS2 (2020) auto-tunes M; ship the 2014 version first, WBS2 in Tier 2. |
| T1.7 | **e-divisive** | Matteson-James 2014 | Offline multi-CP, multivariate, distribution-free | O(kn²) | Energy-distance-based, no parametric assumption, multivariate-native. Only Tier-1 detector that handles arbitrary distributional changes (not just mean/var). The non-Gaussian backstop. |
| T1.8 | **MOSUM** (moving sum) | Eichinger-Kirch 2018; mosum R-pkg | Offline multi-CP, also online | O(n) | Bandwidth-window scan. Asymptotically Gumbel under H₀ → analytic threshold. Multiscale extension handles mixed-magnitude jumps. Strong 2024 momentum (Two-Way MOSUM, disMOSUM). |
| T1.9 | **Two-sample test statistics under CPD framing** — KS, Cramér-von Mises, Anderson-Darling, Wilcoxon rank-sum, Mann-Whitney U | various 1940s-1950s | Offline pairwise / scan substrate | O(n log n) per pair | Substrate for nonparametric segmentation (sliding-window KS is a complete detector). Several already belong in `prob`; this agent flags them as a CPD-substrate dependency to be coordinated, not duplicated. |

**Tier-1 rationale:** every entry is (a) cited by a textbook or named survey,
(b) implemented in `ruptures`, `changepoint::cpt`, or `mosum` (all three are
the reference implementations), (c) has a closed-form or trivially-tested
golden-file output, (d) has fewer than ten hyperparameters, (e) is
language-agnostic — Python/C++/C# can validate against the same JSON test
vectors per the `testutil` design.

## Tier 2 — should ship (modern, validated, fills a real gap)

| # | Detector | Reference | Why Tier 2 |
|---|----------|-----------|------------|
| T2.1 | **WBS2** | Fryzlewicz 2020 | Auto-tuning of WBS interval count via SDLL. Ship after WBS gold-file lands. |
| T2.2 | **Narrowest-Over-Threshold (NOT)** | Baranowski-Chen-Fryzlewicz 2019 | Generalisation handling change-in-slope, change-in-poly, AR-coefficient changes. Same family as WBS. |
| T2.3 | **Kernel CPD** (KernelCPD; KCpA) | Harchaoui-Cappé 2007; Arlot-Celisse-Harchaoui 2019 | RBF/linear kernel, MMD-style, multivariate. `ruptures.KernelCPD` is the reference. Bridges to `signal` kernel work. |
| T2.4 | **FOCuS** (Functional-pruning CUSUM) | Romano-Eckley-Fearnhead-Rigaill 2023 (JMLR 24) | The modern online detector. Pruned CUSUM functional, O(log n) per step amortised, optimal EDD up to constants. The 2023-2025 frontier replacement for vanilla CUSUM. mixFOCuS (2024) is a distributed variant. |
| T2.5 | **OCD (Online Changepoint Detection)** | Chen-Wang-Samworth 2022; CRAN `ocd` v2025-07 | High-dimensional online: storage and per-step cost independent of t. Aggregates LR tests across scales × coordinates. Only high-dim online detector with proven ARL/EDD bounds. |
| T2.6 | **changeforest** | Londschien-Bühlmann-Kovács 2023 (JMLR 24, mlondschien/changeforest) | Random-forest classifier-based, distribution-free, high-dim. Beats every other detector in the paper's simulation grid (avg ≥0.9 vs ≤0.75 for the next-best). Note: 2023 not 2025 — original prompt year is off. |
| T2.7 | **Inspect** (high-dim sparse mean change) | Wang-Samworth 2018 | Sparse-projection-based detector for the p ≫ n regime. Multivariate complement to e-divisive when sparsity is the prior. |
| T2.8 | **CROPS** (changepoints over a range of penalties) | Haynes-Eckley-Fearnhead 2017 | Penalty-path traversal for PELT. Lets the user pick K from a Pareto curve instead of fixing β. Ship paired with PELT. |
| T2.9 | **BottomUp segmentation** | classical (Keogh et al. 2001) | The recursive-merging dual of BinSeg. Fast, suboptimal, ubiquitous as a baseline in the time-series-mining literature. ruptures ships it. |
| T2.10 | **Window sliding (Window detector)** | survey-standard | Two-sample test on sliding pre/post windows. The substrate for KS-CPD, MMD-CPD, AD-CPD. ruptures ships it. |
| T2.11 | **product-of-experts BOCPD** / **BOCPDMS** | Knoblauch-Damoulas 2018 | Multivariate-multimodel BOCPD. Direct extension of the existing detector. Lowest marginal cost given BOCPD already lives here. |
| T2.12 | **Group Fused Lasso (GFL)** | Bleakley-Vert 2011 | Convex-optimisation framing of multi-CP detection. Multivariate-native. Couples cleanly to the existing `optim` package. |

## Tier 3 — nice to have (specialist, domain-narrow, or active research)

| # | Detector | Notes |
|---|----------|-------|
| T3.1 | **Sparsified Binary Segmentation (SBS)** Cho-Fryzlewicz 2015 | High-dim mean change via CUSUM thresholding. |
| T3.2 | **Subspace identification CPD** Truong et al. 2020 | Spectral-method baseline; ruptures cites. |
| T3.3 | **TV-denoised / fused-lasso single-series** Tibshirani et al. 2005 | 1-D specialisation of GFL; near-trivial given `optim`. |
| T3.4 | **DLM-based Bayesian segmentation** West-Harrison 1997 | State-space alternative to BOCPD. |
| T3.5 | **e-cp3o / ED-PELT** Zhang-James 2017; Haynes 2017 | E-statistic ⊕ exact-search hybrids. ED-PELT especially is a high-value addition once PELT lands. |
| T3.6 | **CHAD** (Moen 2024) | Online unified detector framework, R impl. Watch, do not yet pin. |
| T3.7 | **disMOSUM / mixFOCuS** (2024) | Distributed-system online detectors. Out-of-scope for a math library; flagged for architectural awareness only. |
| T3.8 | **TUNE** (Carrington et al. 2024) | Algorithm-agnostic post-detection inference (CIs on detected CPs). Ship as a wrapper, not a detector. |
| T3.9 | **prophet-style segmented Bayesian regression** | Domain-narrow (forecasting). Reality is not a forecasting library. Skip unless explicit consumer demand. |

## Cross-package coupling notes

- `prob` already owns Student-t, Gamma, KS, AD, CvM. PELT cost functions and
  the two-sample-test substrate (T1.9) should import `prob`, not duplicate.
- `optim` already owns L-BFGS / simplex. GFL (T2.12) and TV-denoised (T3.3)
  should compose, not re-derive.
- `linalg` PCA / sparse: Inspect (T2.7) and SBS (T3.1) are projection-based
  and should reuse.
- `signal` kernels (RBF, etc.): KernelCPD (T2.3) and MMD-CPD should reuse
  whatever kernel substrate `signal` exposes.
- `infogeo` is already a cross-test consumer of `changepoint`; the new
  detectors should preserve the same pattern (KL/Hellinger/TV witnesses on
  posterior trajectories where applicable — only BOCPD-family detectors
  expose a posterior, but T2.11 BOCPDMS and T3.4 DLM-Bayesian do).

## Ordering recommendation

**Sprint 1 (next 2 weeks):** T1.1 CUSUM, T1.5 BinSeg, T1.4 OP — three
small, decoupled, high-leverage detectors. CUSUM unblocks the streaming-SPC
consumers; BinSeg + OP unblock the entire offline-multi-CP family by giving
a correctness witness pair.

**Sprint 2:** T1.3 PELT, T1.6 WBS, T1.8 MOSUM. After this point reality is
at parity with `changepoint::cpt`'s offline core.

**Sprint 3:** T1.7 e-divisive, T1.2 GLR, T2.4 FOCuS. After this reality
covers nonparametric, online-optimal, and high-dim-online.

**Sprint 4+:** Tier 2 remainder, then Tier 3 by demand-pull.

## Sources (verified 2026-05-07)

- [GitHub: mlondschien/changeforest — Random Forests for Change Point Detection](https://github.com/mlondschien/changeforest)
- [arXiv 2205.04997 — Random Forests for Change Point Detection (Londschien-Bühlmann-Kovács)](https://arxiv.org/abs/2205.04997)
- [JMLR vol 24 22-0512 — Random Forests for Change Point Detection](https://www.jmlr.org/papers/volume24/22-0512/22-0512.pdf)
- [GitHub: gtromano/FOCuS — Fast Online Changepoint Detection via Functional Pruning CUSUM](https://github.com/gtromano/FOCuS)
- [JMLR vol 24 21-1230 — FOCuS paper PDF](https://www.jmlr.org/papers/volume24/21-1230/21-1230.pdf)
- [arXiv 2504.09573 — A general methodology for fast online changepoint detection (2025)](https://arxiv.org/pdf/2504.09573)
- [CRAN: ocd — High-Dimensional Multiscale Online Changepoint Detection (2025-07)](https://cran.r-project.org/web/packages/ocd/index.html)
- [GitHub: wangtengyao/ocd — online changepoint detection](https://github.com/wangtengyao/ocd)
- [Wiley JTSA — mixFOCuS distributed online CPD (2024)](https://onlinelibrary.wiley.com/doi/10.1111/jtsa.12834)
- [arXiv 2409.15676 — TUNE: Algorithm-Agnostic Inference after Changepoint Detection (2024)](https://arxiv.org/html/2409.15676)
- [arXiv 2407.14369 — tidychangepoint: a unified framework (2024)](https://arxiv.org/pdf/2407.14369)
- [Semantic Scholar — mosum: A Package for Moving Sums in Change-Point Analysis (Meier-Kirch)](https://www.semanticscholar.org/paper/mosum:-A-Package-for-Moving-Sums-in-Change-Point-Meier-Kirch/a4e132ede2780a7d2dc2d95b9958022d84df8d42)
- [arXiv 1101.1438 — PELT: Optimal detection of changepoints with a linear computational cost (Killick et al. 2012)](https://arxiv.org/pdf/1101.1438)
- [ruptures — Pelt user guide](https://centre-borelli.github.io/ruptures-docs/user-guide/detection/pelt/)
- [ruptures — Kernel CPD performance comparison](https://centre-borelli.github.io/ruptures-docs/examples/kernel-cpd-performance-comparison/)
- [GitHub: deepcharles/ruptures — change point detection in Python](https://github.com/deepcharles/ruptures)
- [Lancaster MATH337 — PELT, WBS and Penalty choices (2024-25 course notes)](https://www.lancaster.ac.uk/~romano/teaching/2425MATH337/4_algos_and_penalties.html)
- [ScienceDirect — Selective review of offline change point detection (Truong et al.)](https://www.sciencedirect.com/science/article/abs/pii/S0165168419303494)
- [Springer — A computationally efficient nonparametric approach for changepoint detection (ED-PELT)](https://link.springer.com/article/10.1007/s11222-016-9687-5)
- [GitHub: STOR-i/Changepoints.jl — Julia changepoint package](https://github.com/STOR-i/Changepoints.jl)
