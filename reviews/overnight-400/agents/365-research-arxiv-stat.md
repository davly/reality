# 365 — research-arxiv-stat (recent stat.ME / stat.ML methodology)

## Headline
Conformal/extreme + e-value/anytime-valid testing + grid-online changepoint + knockoff/synthetic FDR + sliced/streaming OT are the methodologically dense 2025–2026 advances most directly applicable to reality's `prob`, `signal`, and a future `timeseries` package.

## Top papers

1. **Extreme Conformal Prediction (Pasche et al., arXiv:2505.08578, May 2025 / rev. Mar 2026)**
   Bridges extreme value theory and split conformal prediction. Classical CP fails when the desired coverage exceeds n_cal/(n_cal+1) (intervals become infinite). They wrap any black-box extreme quantile regressor with EVT-tail extrapolation to produce finite, calibrated intervals at e.g. 99.9% with hundreds of points. Methodology is distribution-free for the bulk and EVT-parametric in the tail; provides finite-sample coverage bounds. Directly relevant to a `prob/conformal` slot — CP is currently a documented gap (slot 156). Keys: GPD tail, Pickands, conditional coverage diagnostics. Pure methodology, no DL prerequisites; ~150 LOC reachable in Go.

2. **A Gentle Introduction to Conformal Time Series Forecasting (arXiv:2511.13608, Nov 2025)**
   Survey of CP under non-exchangeability: weighted CP, ACI (adaptive conformal inference), AgACI, NexCP. Focuses on three families — calibration reweighting, online residual updating, and adaptive target-coverage tuning. Provides decision tree for choosing method by stationarity assumptions. Pairs naturally with reality's planned timeseries module; the algorithms are pure (residual quantile + decay), no learning loops needed. Cites Tibshirani 2019, Gibbs/Candès 2021, Zaffran 2022, Angelopoulos 2024.

3. **Conformal Prediction for Time-series Forecasting with Change Points (arXiv:2509.02844, Sep 2025)**
   Couples online CP with online changepoint detection — when a CP is declared, calibration window is reset/shortened. Provides regret bounds on miscoverage relative to oracle that knows the changepoints. Closes the loop between two reality slots: changepoint detection and CP intervals. Good integration target for a `prob/conformal/cpd_aware` function once a CPD primitive exists.

4. **Conformal Prediction Assessment / CVI (arXiv:2603.27189, Mar 2026)**
   Reframes conditional-coverage evaluation as a supervised problem: train a reliability estimator predicting per-instance coverage, then decompose into "safety" (P[under-coverage]) and "efficiency" (E[over-coverage cost]). Yields the Conditional Validity Index. Useful as a CP **diagnostic** in golden tests — far more informative than marginal coverage averages.

5. **Stable Localized Conformal Prediction via Transduction (arXiv:2605.01452, May 2026)**
   StCP: transductive transfer-style CP that stabilizes localized CP (k-NN-conformal) when calibration is small. Uses unlabeled target features without target labels. Methodology is a weighted quantile reweighting trick — small, golden-file friendly.

6. **A Grid-Based Methodology for Fast Online Changepoint Detection (Romano et al., arXiv:2504.09573, Apr 2025 / rev. Mar 2026)**
   Generic online wrapper that lets *any* offline CPD test (CUSUM, BinSeg, PELT-style) run sequentially via a dynamically maintained grid of candidate locations. O(log n) update, O(log n) memory amortized. This is the right shape for reality: pure, deterministic, allocation-free, and golden-file testable. Cites Page 1954, Killick 2012, Truong 2020. Strong candidate for a `signal/cpd/online_grid.go` primitive.

7. **Changepoint Detection As Model Selection (Levine, arXiv:2601.22481, Jan 2026)**
   Unifying L0-penalty framework. Introduces Iteratively Reweighted Fused Lasso (IRFL): adaptively reweights generalized-lasso penalties, BIC-minimizing. Handles seasonality, trend, AR(p) noise simultaneously. Better support recovery than fused lasso / SaRa. Good reference for golden vectors but algorithm is heavier (iterative, IRLS-style).

8. **Synthetic-Powered Multiple Testing with FDR Control / SynthBH (arXiv:2602.16690, Feb 2026)**
   Combines real test statistics with **synthetic** (e.g. generative-model) draws under mild PRDS positive-dependence; provides finite-sample, distribution-free FDR control without requiring the synthetic null p-values to be valid. Practical mechanism: a calibrated mirror statistic on real ∪ synthetic. Methodology is small (BH on a transformed statistic), but the **proof requires PRDS**. Good complement to a future reality `prob/multipletest` extension beyond plain BH/BY.

9. **FDR Control via Knockoffs + Debiased p-values (arXiv:2505.16124, May 2025)**
   First combination of model-X knockoffs with debiased lasso p-values for high-dim linear regression with **diverging** parameter dimension. Provides finite-sample FDR under arbitrary Σ. Reality has linear-model primitives in `linalg`; this is the natural FDR add-on for that domain.

10. **Bayes, E-values and Testing (arXiv:2602.04146, Feb 2026)**
    Reconciles Bayesian posterior/Bayes-factor inference with frequentist e-value/e-process anytime-valid testing via Ville's inequality. Shows when Bayes factors are valid e-values (proper prior, no data-dependent reweighting). E-values are the "right" object for sequential, never-stop monitoring.

11. **Anytime Validity is Free: Inducing Sequential Tests (Pérez-Ortiz et al., arXiv:2501.03982, Jan 2025; JRSSB)**
    Constructive: any fixed-n test with type-I error α → an anytime-valid sequential test that matches power at n=N. The mechanism is a stopped-process supermartingale construction. This is a methodologically clean, code-able primitive (~50 LOC). Good fit for a `prob/sequential` slot — golden tests are easy because the construction is explicit.

12. **Sequential Randomization Tests Using e-values (arXiv:2512.04366, Dec 2025)**
    Permutation/randomization tests promoted to anytime-valid via e-values. Important because randomization tests are otherwise the "gold standard" but only at fixed n. Useful as a methodological reference for a future `prob/permutation` extension.

13. **Streaming Sliced Optimal Transport / Stream-SW (arXiv:2505.06835, May 2025; rev. Jan 2026)**
    First estimator for Sliced-Wasserstein (SW) from a sample stream. Uses online-mean updates of the per-projection 1D Wasserstein distance with deterministic projection schedules. O(d) memory per projection, O(L d) total. SW already enjoys √n statistical rates (vs nᵅ curse for full OT). For reality this is a clear drop-in: reuse `signal` quantile/sort routines + per-direction integration. Fits a `prob/wasserstein/sliced.go` slot.

14. **Slicing Wasserstein over Wasserstein via Functional OT (arXiv:2509.22138, Sep 2025)**
    Double-sliced Wasserstein DSW for "meta-measures" (distributions of distributions). Avoids the Wₚ-on-Wₚ instability. Shows DSW-min ≡ WoW-min on discretized meta-measures. Heavier; useful as reference, not as immediate target.

15. **Smoothed Estimation of Wasserstein Barycenters (arXiv:2605.03300, May 2026)**
    Sample complexity for Wasserstein barycenters from point clouds with kernel/Gaussian smoothing — closes a gap left by entropic-OT-based barycenter estimators (which converge to a *biased* limit). Methodology is a fixed-point iteration on Sinkhorn-regularized OT; pairs with reality's `linalg`/`prob`.

## Reality slot recommendations

- **`prob/conformal` (slot 156 + extensions).** Ship in this order, all pure-math/golden-friendly:
  1. Split-CP (baseline, ~30 LOC)
  2. Weighted/CV+/jackknife+ (Barber 2021 — already older, citeable as foundation)
  3. **Adaptive Conformal Inference (ACI)** — online residual quantile, exponentially small algorithm
  4. **Extreme CP** (paper 1) for high-coverage tail intervals
  5. **CP-CPD-aware** (paper 3) once `signal/cpd` lands
  6. **CVI / CPA diagnostic** (paper 4) for golden-file coverage validation

- **`signal/cpd` or `timeseries/cpd`.** Start with offline (CUSUM, BinSeg, PELT) for golden parity with `ruptures`, then add **grid-based online wrapper** (paper 6) — pure, allocation-free, suits 60-FPS hot path. IRFL (paper 7) is a stretch goal.

- **`prob/multipletest`.** Reality likely has BH/BY already; extend with:
  - Storey's q-value (well-established; cite Storey 2002 + recent reviews)
  - Knockoff filter (Barber-Candès 2015 + paper 9 extension)
  - SynthBH (paper 8) — short and conceptually clean

- **`prob/sequential` (new slot).** Anytime-valid testing is a coherent pure-math sub-package:
  - e-values, e-processes, Ville's inequality
  - Construction from paper 11 (anytime-valid wrapper around any fixed-n test)
  - Sequential probability ratio test (Wald 1947 — classical) as anchor

- **`prob/wasserstein` (new slot).** OT belongs in reality:
  - 1D Wasserstein-p (closed form via sorted quantiles — trivial, golden-file ready)
  - Sliced-Wasserstein (Bonneel 2015) — projection + 1D OT integration
  - **Streaming Sliced-Wasserstein** (paper 13) — natural for reality's no-allocation ethos
  - Sinkhorn (entropic regularization) for general OT — iterative but well-bounded
  - Wasserstein barycenters (smoothed, paper 15) as advanced

- **`prob/causal` (new slot, optional).** Causal inference is generally out of reality's pure-math scope (it's about modeling assumptions), but specific primitives are pure: backdoor-set algorithm on a DAG (graph package already has DAG primitives), do-calculus rule application, Pearl's identifiability checker.

- **Variational inference**: too application-coupled (gradient-flow, NN-coupled) for reality's zero-dependency stance. Skip.

- **Survival analysis**: classical estimators are pure (Kaplan-Meier, Nelson-Aalen, Cox partial likelihood, log-rank). Recent methodology (paper 4 = History-Aware CP for censored events; CARE) is too ML-coupled. Recommend a small `prob/survival` slot with KM + NA + log-rank only.

## Sources
- [arXiv stat.ME recent listing](https://arxiv.org/list/stat.ME/recent)
- [arXiv stat.ML recent listing](https://arxiv.org/list/stat.ML/recent)
- [Extreme Conformal Prediction (2505.08578)](https://arxiv.org/abs/2505.08578)
- [Conformal Prediction Assessment (2603.27189)](https://arxiv.org/abs/2603.27189)
- [Stable Localized Conformal Prediction (2605.01452)](https://arxiv.org/abs/2605.01452)
- [Conformal Time Series with Change Points (2509.02844)](https://arxiv.org/abs/2509.02844)
- [Gentle Introduction to Conformal TS Forecasting (2511.13608)](https://arxiv.org/abs/2511.13608)
- [Grid-Based Online Changepoint Detection (2504.09573)](https://arxiv.org/abs/2504.09573)
- [Changepoint Detection as Model Selection / IRFL (2601.22481)](https://arxiv.org/abs/2601.22481)
- [Synthetic-Powered Multiple Testing / SynthBH (2602.16690)](https://arxiv.org/abs/2602.16690)
- [Knockoffs + debiased p-values FDR (2505.16124)](https://arxiv.org/abs/2505.16124)
- [Bayes, E-values and Testing (2602.04146)](https://arxiv.org/abs/2602.04146)
- [Anytime Validity is Free (2501.03982)](https://arxiv.org/abs/2501.03982)
- [Sequential Randomization Tests via e-values (2512.04366)](https://arxiv.org/abs/2512.04366)
- [Streaming Sliced OT / Stream-SW (2505.06835)](https://arxiv.org/abs/2505.06835)
- [Slicing Wasserstein over Wasserstein (2509.22138)](https://arxiv.org/abs/2509.22138)
- [Smoothed Wasserstein Barycenters (2605.03300)](https://arxiv.org/abs/2605.03300)
- [DoWhy library (py-why/dowhi)](https://github.com/py-why/dowhy)
- [False Discovery Control review (2411.10647)](https://arxiv.org/html/2411.10647v1)
