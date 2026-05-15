# 228 | new-bayes-nonparam

**Summary line 1.** Block-C slot 228 is the FIRST Bayesian-nonparametrics (BNP) scoping in the 400-sequence covering Ferguson-1973-Ann.Stat-1(2) Dirichlet Process `DP(α, G_0)` defined by finite-dim consistency `G(A_1), ..., G(A_k) ~ Dirichlet(α G_0(A_1), ..., α G_0(A_k))` for any measurable partition / Sethuraman-1994 stick-breaking representation `G = Σ_{k=1}^∞ π_k δ_{θ_k}` with `π_k = β_k Π_{j<k} (1 − β_j)`, `β_k ~ Beta(1, α)` iid, `θ_k ~ G_0` iid (canonical computational form of DP) / Aldous-1985 Chinese Restaurant Process (CRP) exchangeable partition `P(customer k+1 joins table t with size m_t) = m_t/(k + α)`, `P(new table) = α/(k + α)` — IDENTICAL marginal partition law to DP (181-U10 already proposed at 130 LOC) / Hoppe-1984 urn equivalence (181-U9 already proposed at 50 LOC) / Ewens-1972-Theor.Pop.Biol-3 sampling formula (181-U12 already proposed at 110 LOC) / Pitman-Yor process Pitman-1995 / Pitman-Yor-1997-Ann.Probab two-parameter `(θ, σ)` generalisation `P(new table) = (θ + Kσ)/(k + θ)` with power-law cluster-count growth `K_n ~ T_{θ,σ}·n^σ` for σ ∈ (0,1), reduces to DP at σ=0 (181-U11 already proposed at 140 LOC) / GEM(α) Griffiths-Engen-McCloskey size-biased stick-breaking distribution / two-parameter Poisson-Dirichlet PD(σ, θ) Pitman-Yor canonical `π_(1) ≥ π_(2) ≥ ...` ordered weights / Hierarchical Dirichlet Process (HDP) Teh-Jordan-Beal-Blei-2006-JASA-101(476) `G_0 ~ DP(γ, H)`, `G_j ~ DP(α, G_0)` shared-atom multi-group model — foundation of HDP-LDA topic models with infinite-topics / Chinese Restaurant Franchise (CRF) Teh-2006 hierarchical-CRP coupling tables across restaurants via shared-dish menu / Nested Dirichlet Process (NDP) Rodríguez-Dunson-Gelfand-2008-JASA-103(483) `G_j ~ Q`, `Q ~ DP(α, DP(β, H))` distribution-of-distributions / Indian Buffet Process (IBP) Griffiths-Ghahramani-2005-NIPS / Griffiths-Ghahramani-2011-JMLR-12 binary-feature exchangeable matrix `Z ∈ {0,1}^{n×∞}` infinite-features `customer i picks dish k previously sampled with prob m_k/(i)`, samples `Poisson(α/i)` new dishes / Thibaux-Jordan-2007-AISTATS-2 Beta Process `BP(c, B_0)` underlying IBP via `B = Σ π_k δ_{ω_k}` with completely-random-measure construction `Σ` Poisson process on `(0,1] × Θ` with rate `c·π^{-1}(1−π)^{c−1} dπ × B_0(dω)` / Two-parameter IBP Teh-Görür-Ghahramani-2007-AISTATS-3 `(α, β)` discount-strength generalisation / Bernoulli Process Y_k ~ Bernoulli(π_k) with π_k from BP feature activations / Gaussian Process (GP) Rasmussen-Williams-2006-MIT-Press distribution over functions `f ~ GP(m(x), k(x,x'))` defined by mean-function and covariance-kernel — FORMAL Block-C slot 237 is the dedicated GP review; this slot stays DP-IBP-HDP-PY focused with cross-link only / GP-regression posterior `f|y ~ N(K_*^T (K + σ²I)^{-1} y, K_** − K_*^T (K + σ²I)^{-1} K_*)` (184-C20 GaussianProcessPosterior already proposed) / GP-classification Williams-Barber-1998 Laplace approx + EP / sparse GP inducing-points Snelson-Ghahramani-2006-NIPS FITC / Titsias-2009-AISTATS-5 variational free-energy VFE / Hensman-Fusi-Lawrence-2013-UAI stochastic-VI mini-batch / Deep GP Damianou-Lawrence-2013-AISTATS-31 hierarchical-GP composition / Neural Tangent Kernel Jacot-Gabriel-Hongler-2018-NeurIPS infinite-width-NN-as-GP cross-link to deep-learning theory / Bernoulli-Beta process spike-train models / Gamma Process Brix-1999-Adv.Appl.Probab-31 / completely-random-measures Kingman-1967-Pacif.J.Math-21 Lévy-Khinchine on space-of-measures `μ = Σ s_k δ_{x_k}` with `Σ` Poisson on `(0,∞) × X` / Lévy process / Subordinator (non-decreasing Lévy) `T_t = Σ_{s ≤ t} ΔX_s` building inverse-Gaussian, gamma, stable subordinators / gamma randomization underlying DP via normalisation `G = G_α/G_α(Θ)` for `G_α ~ GP(α G_0)` (Ferguson's original construction) / Dependent Dirichlet Processes (DDP) MacEachern-1999-Tech.Rep "Dependent Nonparametric Processes" stick-breaking weights or atoms vary smoothly with covariates / Bayesian factor analysis IBP-based Knowles-Ghahramani-2011 latent-feature factor models with unbounded factor count / Negative-binomial process Zhou-Carin-2015 `X ~ NB(r, p)` with random `r` from gamma process / Determinantal Point Processes (DPP) Macchi-1975 (216-R13 + 188-D17 + 223 already proposed at 280 LOC) — repulsive cross-link to 216 RMT and 223 submodular / Posterior inference Gibbs sampling Escobar-West-1995-JASA-90(430) DP-mixture sampler with predictive Polya-urn `θ_i | θ_{-i} ∝ Σ_{j≠i} δ_{θ_j} + α G_0` / slice sampling Walker-2007-Comm.Stat-36(1) auxiliary-variable infinite-mixture truncation / variational DP Blei-Jordan-2006-Bayesian-Anal-1(1) truncated stick-breaking ELBO / blocked Gibbs Ishwaran-James-2001-JASA-96(453) truncation N_max stick / collapsed Gibbs marginalising atom locations / split-merge Jain-Neal-2004-JCGS-13(1) Metropolis-Hastings cluster-restructure / particle MCMC Whiteley-Andrieu-Doucet-2010 BNP filtering / concentration-parameter inference Escobar-West-1995 mixture-of-gammas conjugate prior / cross-link to 181 (combinatorics-prob) CRP-Ewens-Pitman-Yor partition substrate / cross-link to 169 (synergy-prob-optim) variational DP / cross-link to 216 (RMT) DPP-shared / cross-link to 217 (free-prob) free-Poisson = MP and free-cumulants underpin BNP combinatorics on non-crossing partitions only-marginally / cross-link to 237 GP dedicated slot. Reality v0.10.0 ships **ZERO** BNP surface verified by repo-wide grep on `Dirichlet.*Process|StickBreaking|GEM|Sethuraman|ChineseRestaurant|Pitman.*Yor|Indian.*Buffet|Hierarchical.*Dirichlet|Chinese.*Franchise|Nested.*Dirichlet|Beta.*Process|Gamma.*Process|completely.*random|Negative.*binomial.*process|MacEachern|Teh|Ferguson|Aldous|Ewens|Hoppe|Polya.*Urn|GEM|Poisson.*Dirichlet|Subordinator|Levy.*process|Determinantal.*Point|DPP\\b|HDP|IBP|NDP|DDP|Concentration.*parameter|Slice.*sampling|Split.*merge|Variational.*DP|Blocked.*Gibbs|Stick.*breaking|FITC|VFE|inducing.*point|Sparse.*GP|Deep.*GP|Neural.*Tangent` returning ZERO callable matches across all 22 packages (the only nominal hits are the 181/216/217/223/227 PROPOSED-NOT-SHIPPED reports). The substrate that DOES exist: `prob.LogGamma` (mathutil.go:37, Lanczos), `prob.BetaPDF/CDF` (distributions.go), `prob.PoissonPMF/CDF` (distributions.go), `combinatorics.StirlingFirst|StirlingSecond|BellNumber|IntegerPartitions|DerangementCount|CatalanNumber` (counting.go) — together these cover the closed-form-CRP/Ewens-PMF/Pitman-Yor block-count machinery without further math.

**Summary line 2.** Twenty-six BNP primitives **B1–B26** totalling ~4,180 LOC of pure connective tissue split across **(a) ~430 LOC OVERLAP with 181** (B1=181-U8 PolyaUrnSimulate, B2=181-U9 HoppeUrnSimulate, B3=181-U10 CRPSample+CRPPartitionProb, B4=181-U11 PitmanYorSample+PYPartitionProb, B5=181-U12 EwensSamplingFormula — five primitives 181 already proposed at ~480 LOC, ship once with 181 PR-3+PR-8) and **(b) ~3,750 LOC NET-NEW absent from 181 / 216 / 217 / 227** including B6 StickBreakingSample (Sethuraman-1994 truncated/exact `π_k = β_k Π_{j<k}(1-β_j)`, β_k ~ Beta(1, α); the canonical computational form of DP not just CRP-marginal — distinct primitive surface from 181's exchangeable-partition view), B7 GEMDistribution (size-biased stick-breaking `π_(1), π_(2), ...` ordered weights with closed-form joint density), B8 PoissonDirichletPD(σ, θ) ordered-stick-breaking weights with discount, B9 DPMixturePosterior + DPMixtureLogLikelihood collapsed-Gibbs Escobar-West-1995 — single-most-cited BNP-inference primitive in 400-sequence at ~280 LOC, B10 SliceSampler Walker-2007 auxiliary-variable infinite-mixture sampler unblocking infinite-component-without-truncation at ~220 LOC, B11 BlockedGibbsTruncated Ishwaran-James-2001 with truncation level N_max + truncation-error-bound at ~180 LOC, B12 SplitMergeMCMC Jain-Neal-2004 Metropolis-Hastings cluster-restructure at ~280 LOC (deferred — needs full DP-mixture infrastructure), B13 VariationalDP Blei-Jordan-2006 truncated-stick-breaking ELBO at ~240 LOC composing existing optim/proximal/Fbs (cross-link to 169 keystone S5 EM-for-GMM), B14 ConcentrationParameterInference Escobar-West-1995 conjugate-gamma posterior on α at ~80 LOC, B15 IndianBuffetSample Griffiths-Ghahramani-2005 `Z ~ IBP(α)` infinite-binary-feature matrix at ~140 LOC, B16 IBPLogLikelihood + leftOrderForm canonical-form computation at ~120 LOC, B17 BetaProcessSample Thibaux-Jordan-2007 underlying-completely-random-measure construction `B = Σ π_k δ_{ω_k}` via Lévy-process simulation at ~200 LOC, B18 TwoParameterIBP Teh-Görür-Ghahramani-2007 (α, β) discount-strength generalisation at ~100 LOC, B19 HDPSample Teh-Jordan-Beal-Blei-2006 hierarchical-CRP / Chinese-restaurant-franchise sampler at ~320 LOC — single-most-impact production-BNP primitive enabling infinite-topic LDA-style models, B20 NestedDirichletProcess Rodríguez-Dunson-Gelfand-2008 distribution-on-distributions at ~200 LOC, B21 DependentDirichletProcess MacEachern-1999 covariate-dependent atoms via stick-breaking weights varying with x at ~240 LOC (deferred — covariate-modelling overlap with B13), B22 GammaProcessSample at ~140 LOC, B23 NegativeBinomialProcess Zhou-Carin-2015 cross-link to count-data BNP at ~120 LOC, B24 SubordinatorSample (gamma, inverse-Gaussian, stable) Lévy-process simulation at ~180 LOC, B25 LevyKhinchineCRM completely-random-measure with Lévy-Khinchine triplet `(σ², ν, b)` at ~160 LOC, B26 SparseGP-FITC + SparseGP-VFE inducing-point approximations Snelson-Ghahramani-2006 / Titsias-2009 at ~280 LOC (DEFER to 237 dedicated GP slot — boundary placement). Tier-1 keystone **B3+B4+B5+B6+B7 = `prob/bnp/sticks.go` ~620 LOC** captures Sethuraman-stick-breaking + GEM + PoissonDirichlet + CRP + PY canon as the irreducible computational form of DP/PY. **SINGULAR competitive moat: B19 HDPSample at ~320 LOC** — no zero-dependency Go library ships HDP-CRF; reality would be FIRST and only Mathematica's `BayesNonparametrics` and Python's `bnpy` (research code) implement Teh-2006 Chinese-restaurant-franchise sampler in production. **SINGULAR Block-C-2026 frontier: B17 BetaProcess + B22 GammaProcess + B25 LevyKhinchineCRM at ~500 LOC** because the unifying theory of completely-random-measures Kingman-1967 underlies ALL BNP (DP, IBP, HDP, NDP, NegBinProc) but no production library exposes the underlying CRM machinery — only `BNPdensity` (R package, Argiento-Bianchini-Guglielmi 2016) ships partial CRM API; reality's golden-file pinning at IEEE-754 boundaries against Brix-1999 reference would be unique. **SINGULAR cross-link: B9 DP-mixture + B13 VariationalDP composes existing 169-S5 EM-for-GMM** ~340 LOC keystone — the natural extension is to swap finite-K GMM for infinite-K DP-Gaussian-mixture, ~180 LOC delta on top of S5; this is the BNP-onramp for any consumer that already has GMM-EM. Cross-package blockers `prob.StandardNormalSample` Box-Muller absent (gates B6/B9-B26 — every stochastic BNP sampler needs Gaussian via Beta-via-Gamma-via-rejection chain — same blocker as 117/184/188/202/215/216/217/227, this is the SIXTH independent Block-C review demanding it), `prob.GammaSample` (gates B22/B23/B24/B25/B17 — Beta-process via gamma-process integration), `prob.BetaSample` (gates B6/B7/B8 stick-breaking — Beta(1, α) iid samples), `prob.PoissonSample` (gates B15/B17/B25 Poisson-process construction of CRM). Recommended placement **NEW sub-package `prob/bnp/`** ~3,750 LOC NET-NEW = (`bnp/sticks.go` ~620 + `bnp/dp.go` Sethuraman + GEM + PD + DP-mixture ~520 + `bnp/inference.go` slice + blocked-Gibbs + concentration ~480 + `bnp/variational.go` ~240 + `bnp/ibp.go` Indian-buffet + Beta-process + 2P-IBP ~460 + `bnp/hdp.go` HDP + NDP ~520 + `bnp/crm.go` Gamma + NegBin + subordinator + Lévy-Khinchine ~600 + `bnp/ddp.go` Dependent-DP DEFERRED ~240 + `bnp/sparse_gp.go` FITC + VFE BOUNDARY-DEFER-TO-237 ~280); rationale for sub-package vs flat `prob/bnp.go`: parallel to existing `prob/copula/` and `prob/conformal/` and PROPOSED `prob/freeprob/` (217) sub-packages — BNP is a self-contained algebraic system with its own type vocabulary (`StickBreakingProcess`, `DirichletProcess`, `CRP`, `PitmanYorProcess`, `IBPMatrix`, `BetaProcess`, `HDP`, `CRM`) deserving its own namespace. Landing order PR-1=B3+B4+B5+B6+B7 ~620-LOC stand-alone Sethuraman + GEM + PD + CRP + PY + Ewens (CRP+PY+Ewens already 181-PR-3 — coordinate to ship-once), PR-2=B9+B14 ~360-LOC DP-mixture-Gibbs + concentration-parameter (BLOCKED on Box-Muller → wait for prob.StandardNormalSample), PR-3=B10+B11 ~400-LOC slice + blocked-Gibbs (BLOCKED on Box-Muller), PR-4=B13 ~240-LOC variational DP composing optim/proximal/Fbs (cross-link to 169-S5 EM-for-GMM keystone — singular cross-package leverage), PR-5=B15+B16+B17+B18 ~560-LOC IBP + Beta-process + 2P-IBP family (BLOCKED on prob.BetaSample), PR-6=B19+B20 ~520-LOC HDP-CRF + NDP — the moat (BLOCKED on PR-1 + Box-Muller), PR-7=B22+B23+B24+B25 ~600-LOC CRM family Gamma + NegBin + subordinator + Lévy-Khinchine (BLOCKED on prob.GammaSample + prob.PoissonSample), PR-8=B12+B21+B26 deferred (split-merge MCMC needs DP-mixture infra, DDP overlaps B13, FITC/VFE belongs in 237). Cross-link to 181 PR-3+PR-8: B1-B5 IDENTICAL to U8-U12 — ship once at ~480 LOC SHARED. Cross-link to 216-R13 DPP shared at ~280 LOC. Cross-link to 217-F4-F6 NC-partition combinatorics underlies DP marginal partition law (Ewens = uniform on NC(n)? NO — Ewens is θ-biased on Π(n) — NC arises in free-Poisson which IS DP marginal density only as scaling-limit, subtle). Cross-link to 227-U25 Bayesian Optimisation reuses GP (boundary 237). Cross-link to 169-S5 EM-for-GMM is the natural-onramp via B13. Differentiation §6: this report is BNP-pure (DP + PY + IBP + HDP + NDP + CRM canon, Sethuraman-Ferguson-Aldous-Pitman-Teh-Griffiths-Ghahramani-MacEachern), where 181 was combinatorics×prob synergy with 5/23 primitives in BNP territory and 216 was RMT-statistical with 1/22 primitive (DPP) shared. 21 of 26 primitives unique to this slot.

---

## 0. State at HEAD (2026-05-08, v0.10.0)

Repo-wide audit for BNP surface:

| Surface | Path | BNP relevance |
|---|---|---|
| `prob.LogGamma` | `prob/mathutil.go:37` | Lanczos log-gamma — substrate for all DP/PY/HDP/Ewens density evaluation. PRESENT |
| `prob.BetaPDF / BetaCDF` | `prob/distributions.go` | Beta-distribution closed-form — substrate for stick-breaking analytic-density. PRESENT |
| `prob.PoissonPMF / PoissonCDF` | `prob/distributions.go` | Poisson-distribution — substrate for IBP + Beta-process Poisson-process simulation. PRESENT |
| `prob.GammaPDF / GammaCDF` | `prob/distributions.go` | Gamma-distribution — substrate for gamma-process + Ferguson-original-DP-via-gamma-randomization. PRESENT |
| `combinatorics.StirlingFirst` | `combinatorics/counting.go` | `\|s(n,k)\|` — block-count distribution under CRP `K_n / α^k / (α)^(n)`. PRESENT |
| `combinatorics.BellNumber` | `combinatorics/counting.go` | partition-count check on CRP exhaustive enumeration. PRESENT |
| `combinatorics.IntegerPartitions` | `combinatorics/counting.go` | partition-type combinatorics for Ewens. PRESENT |
| `combinatorics.DerangementCount`, `Factorial`, `BinomialCoeff`, `CatalanNumber` | `combinatorics/counting.go` | ancillary substrate. PRESENT |
| `linalg.CholeskyDecompose / Inverse / MatVecMul` | `linalg/decompose.go` | substrate for B26 sparse-GP (boundary-defer 237). PRESENT |
| `optim/proximal.Fbs / FISTA` | `optim/proximal/fbs.go` | substrate for B13 variational-DP ELBO maximisation. PRESENT |
| `optim.LBFGS` | `optim/gradient.go` | substrate for B14 concentration-parameter MAP / B26 GP-hyperparameter MLE. PRESENT |
| `prob.BayesianUpdate` | `prob/prob.go:85` | log-odds Bayesian update — adjacent but BNP needs different machinery (predictive Polya-urn). PRESENT |
| `prob.MarkovSteadyState` | `prob/markov.go` | adjacent but BNP needs partition-Markov-chain not state-Markov. PRESENT |
| `prob.StandardNormalSample` (Box-Muller) | -- | **ABSENT** — gates B6/B9-B26. Same blocker as 117/184/188/202/215/216/217/227. SIXTH cross-cutting Block-C demand. |
| `prob.BetaSample` | -- | **ABSENT** — gates B6/B7/B8/B15/B17/B18 stick-breaking. |
| `prob.GammaSample` | -- | **ABSENT** — gates B22/B23/B24/B25 CRM family. |
| `prob.PoissonSample` | -- | **ABSENT** — gates B15/B17/B25 Poisson-process CRM construction. |
| B1-B26 BNP primitives | -- | **ALL ABSENT** (26 distinct primitives) |

**Cross-import edges.** `prob/bnp/` would consume `prob` (LogGamma, BetaPDF, GammaPDF, PoissonPMF) + `combinatorics` (StirlingFirst, BellNumber, BinomialCoeff, IntegerPartitions, DerangementCount) + (boundary B26 only) `linalg` (Cholesky, MatVecMul) + (B13 only) `optim/proximal` (Fbs). New edge: `prob/bnp → combinatorics`. No cycles (combinatorics → prob already proposed by 181 P1-P3 via LogGamma — `prob/bnp` import path is `prob/bnp → combinatorics → prob` which is fine; `prob/bnp → prob` is in-package not an import cycle since `prob/bnp/` is a sub-package).

**Cross-package blockers.**

| Blocker | Owner | Blocks B-primitives | LOC est |
|---|---|---|---:|
| `prob.StandardNormalSample` (Box-Muller) | 117-T2 / 184-C12 / 188-D1 / 216 / 217 / 227-U0a | B6, B9, B10, B11, B13, B19, B20, B26 | 50 |
| `prob.BetaSample` (Marsaglia-Tsang via Gamma-rejection) | 117 | B6, B7, B8, B15, B17, B18 | 60 |
| `prob.GammaSample` (Marsaglia-Tsang) | 117 | B22, B23, B24, B25, also feeds B6 via Beta-via-Gamma | 80 |
| `prob.PoissonSample` (Knuth small-λ + Cinlar large-λ) | 117 | B15, B17, B25 | 60 |

Total substrate gating PR-2 to PR-7 ~250 LOC; PR-1 ships today against existing closed-form primitives (CRP/PY/Ewens/Sethuraman-PMF closed-form; sampling deferred to PR-2+).

---

## 1. The twenty-six BNP primitives

Each entry: capability + reference / composition / LOC / cross-link / blocking-flag.

### Tier-A — already proposed by 181 (overlap, ship once)

**B1 — `PolyaUrnSimulate(initBlack, initWhite, draws, addOnDraw int, rng) []int`.** 181-U8 verbatim. Polya urn de-Finetti embedding — limit-fraction `~ Beta(initBlack, initWhite)`. ~50 LOC.

**B2 — `HoppeUrnSimulate(theta float64, n int, rng) []int`.** 181-U9 verbatim. Hoppe urn underlying CRP; black ball weight θ + add-color-on-draw. ~50 LOC.

**B3 — `CRPSample(n int, alpha float64, rng) []int` + `CRPPartitionProbability(partition []int, alpha float64) float64`.** 181-U10 verbatim. Aldous-1985 Chinese Restaurant Process. Block-count PMF closed-form `|s(n,k)| α^k / (α)^(n)` via existing `combinatorics.StirlingFirst` + `prob.LogGamma` rising-factorial. ~130 LOC.

**B4 — `PitmanYorSample(n int, theta, sigma float64, rng) []int` + `PYPartitionProbability(partition []int, theta, sigma float64) float64`.** 181-U11 verbatim. Pitman-Yor-1997 two-parameter generalisation; σ=0 reduces to CRP(θ). Power-law cluster-count `K_n ~ T_{θ,σ}·n^σ`. ~140 LOC.

**B5 — `EwensSamplingFormula(partition []int, theta float64) float64` + `EwensExpectedBlockCount(n int, theta float64) float64`.** 181-U12 verbatim. Ewens-1972 partition probability `θ^k n!/(θ)^(n) Π (m_j!^{c_j} c_j!)^{-1}`. CRP-Ewens-Hoppe identity is the canonical R-MUTUAL 3/3 pin (181-PR-3 keystone). ~110 LOC.

**Tier-A total: ~480 LOC SHARED with 181 — ship once.**

### Tier-B — Sethuraman / DP-mixture / inference (NET-NEW from 181)

**B6 — `StickBreakingSample(alpha float64, truncation int, rng) []float64` + `StickBreakingExact(alpha float64, eps float64, rng) []float64`.** Sethuraman-1994 `π_k = β_k Π_{j<k}(1 − β_j)`, `β_k ~ Beta(1, α)` iid. Truncated version returns first `truncation` weights normalised. Exact version draws until residual mass `1 − Σ_k π_k < eps`. Composes `prob.BetaSample(1, α, rng)` (BLOCKED). ~120 LOC. Pin: `E[π_1] = 1/(1+α)`, `E[π_k] = (α/(1+α))^{k-1} · 1/(1+α)`, expected truncation level `E[K(eps)] = log(eps)/log(α/(1+α))`. **NOT in 181** — distinct primitive surface from CRP-marginal: Sethuraman is the explicit-weights-and-atoms construction of `G ~ DP(α, G_0)` while CRP is the marginal-partition view; users of the explicit-G representation (e.g. for prediction at unseen θ via `f(θ) = Σ π_k k(θ, θ_k)`) need this primitive.

**B7 — `GEMDistribution(alpha float64, k int, x []float64) float64`.** Griffiths-Engen-McCloskey size-biased ordered weights. Joint PDF on `{x_1 ≥ x_2 ≥ ... ≥ x_k > 0, Σ x_i ≤ 1}` closed-form via Pitman's formula. ~80 LOC. Pin: `E[x_1] = α B(1, α+1) ·_2F_1(...)` reduces to closed form; agreement Pitman-1996 reference.

**B8 — `PoissonDirichletPDF(sigma, theta float64, x []float64) float64` + `PoissonDirichletSample(sigma, theta float64, n int, rng) []float64`.** Two-parameter Poisson-Dirichlet PD(σ, θ) via stick-breaking with `β_k ~ Beta(1−σ, θ+kσ)`. σ=0 reduces to GEM(θ) reduces to DP-stick-breaking. ~120 LOC.

**B9 — `DPMixtureGibbsSample(data [][]float64, alpha float64, base BaseDistribution, iters int, rng) (assignments [][]int, atoms [][][]float64)`.** Escobar-West-1995 collapsed-Gibbs sampler for DP-mixture model. Predictive `θ_i | θ_{-i}, x_i ∝ Σ_{j≠i} f(x_i | θ_j) δ_{θ_j} + α ∫ f(x_i | θ) G_0(dθ)`. Conjugate-base case (e.g. Normal-Inverse-Wishart for Gaussian-mixture) closed-form integral. Non-conjugate via Neal-2000 Algorithm 8 (auxiliary-atoms). ~280 LOC. Cross-link to 169-S5 EM-for-GMM: this is the BNP-onramp on top of S5 — same E-step structure (compute responsibilities) plus extra-component-creation step, ~120 LOC delta on top of S5. **Single most-cited BNP-inference primitive in 400-sequence.** BLOCKED on Box-Muller + base-distribution sampler.

**B10 — `DPMixtureSliceSample(data [][]float64, alpha float64, base BaseDistribution, iters int, rng) (...)`.** Walker-2007-Comm.Stat-36(1) auxiliary-variable-based slice sampler. Each iteration draws u_i ~ Uniform(0, π_{c_i}); samples cluster index from {k : π_k > min(u_i)}. Allows infinite-K without truncation — provably exact (no truncation error). ~220 LOC. BLOCKED on Box-Muller.

**B11 — `BlockedGibbsTruncated(data [][]float64, alpha float64, base BaseDistribution, N_max int, iters int, rng) (...)`.** Ishwaran-James-2001-JASA-96(453) blocked-Gibbs with truncation level `N_max`. Truncation error bound `‖truncated − exact‖_TV ≤ 4·n·exp(−(N_max − 1)/α)` — quantitative truncation-quality. Standard production choice when slice-sampling overhead too high. ~180 LOC. BLOCKED on Box-Muller.

**B12 — `DPMixtureSplitMergeMCMC(data, alpha, base, iters, rng)`.** Jain-Neal-2004-JCGS-13(1) Metropolis-Hastings cluster-restructure. Proposes split or merge of two random clusters via restricted-Gibbs sampling within proposal. Mixes much faster than pure-Gibbs (which only changes one assignment per iteration). ~280 LOC. **DEFERRED** — needs full DP-mixture infrastructure from B9/B10/B11. PR-8.

**B13 — `VariationalDP(data, alpha, base, truncation, iters)`.** Blei-Jordan-2006-Bayesian-Anal-1(1) truncated stick-breaking variational lower bound. ELBO `E_q[log p(x, z, π, θ) − log q(z, π, θ)]` maximised over factorised `q(z) = Π q(z_n)`, `q(π_k) = Beta(γ_{k,1}, γ_{k,2})`, `q(θ_k) = G_0`-conjugate. Closed-form coordinate-ascent updates. Composes `optim/proximal/Fbs` (already shipped) for natural-gradient steps if KL-projection invoked. ~240 LOC. Cross-link: this is the **singular cross-package leverage** — composes 169-S5 EM-for-GMM ELBO machinery to swap finite-K for infinite-K-with-truncation at ~180-LOC delta on top of S5 already-landed.

**B14 — `ConcentrationParameterPosterior(K, n int, alpha_prior_a, alpha_prior_b float64) (a_post, b_post float64)`.** Escobar-West-1995 conjugate-gamma posterior on α given observed cluster count K and n customers. Auxiliary-variable trick: introduce η ~ Beta(α + 1, n); posterior is gamma-mixture `α | K, η ~ π·Gamma(a + K, b − log η) + (1 − π)·Gamma(a + K − 1, b − log η)` where mixing weight `π = (a + K − 1)/(n(b − log η) + a + K − 1)`. ~80 LOC. Standard practice in production BNP — without α-inference, mis-specified α gives wildly-wrong cluster counts.

**Tier-B total: ~1,320 LOC NET-NEW.**

### Tier-C — Indian Buffet Process / Beta Process (NET-NEW)

**B15 — `IndianBuffetSample(n int, alpha float64, rng) [][]bool`.** Griffiths-Ghahramani-2005 IBP. Customer i picks dish k previously selected with prob `m_k / i`; samples Poisson(α/i) new dishes. Returns `Z` matrix `n × K_+` with `K_+` = number of non-empty columns (random, finite-with-prob-1). ~140 LOC. BLOCKED on prob.PoissonSample. Pin: `E[K_+] = α H_n` where `H_n = Σ 1/i` (Griffiths-Ghahramani Theorem 1).

**B16 — `IBPLogLikelihood(Z [][]bool, alpha float64) float64` + `IBPLeftOrderForm(Z [][]bool) [][]bool`.** Closed-form likelihood `α^{K_+} exp(−α H_n) / Π_k m_k! · Π_h K_h!` where `K_h` = number of columns with binary-history h, `m_k` = column sums. Left-order-form is the canonical-equivalence-class representative. ~120 LOC.

**B17 — `BetaProcessSample(c, B0_atoms, B0_weights []float64, eps float64, rng) (atoms, weights []float64)`.** Thibaux-Jordan-2007-AISTATS-2 Beta Process `BP(c, B_0)` underlying IBP. Lévy-process construction: realise as Σ s_k δ_{ω_k} with Σ Poisson on `(0, 1] × Θ` with rate `c · π^{-1}(1 − π)^{c−1} dπ × B_0(dω)`. Truncate at min-weight `eps`; expected count `c · B_0(Θ) · ∫_eps^1 π^{-1}(1−π)^{c-1} dπ`. ~200 LOC. BLOCKED on prob.PoissonSample + prob.BetaSample.

**B18 — `TwoParameterIBP(n int, alpha, beta float64, rng) [][]bool`.** Teh-Görür-Ghahramani-2007-AISTATS-3 (α, β) discount-strength IBP. Customer i picks dish k with prob `(m_k − β) / (i + α + β − 1)`; samples Poisson(α · Γ(1 + α + β) / Γ(1 + α + β + i − 1)) new. Reduces to standard IBP at β=0, α=α. ~100 LOC.

**Tier-C total: ~560 LOC NET-NEW.**

### Tier-D — Hierarchical / Nested DP (NET-NEW)

**B19 — `HDPSample(n_groups int, n_per_group []int, gamma, alpha float64, base BaseDistribution, rng)`.** Teh-Jordan-Beal-Blei-2006-JASA-101(476). Two-level hierarchical: top `G_0 ~ DP(γ, H)`; per-group `G_j ~ DP(α, G_0)`. Chinese Restaurant Franchise (CRF) sampler: each customer picks a *table* in their restaurant (DP(α) prior); each table picks a *dish* from a global menu (DP(γ) prior). Output: per-group customer-table assignments + per-table-dish assignments + global dish atoms. ~320 LOC. **Singular competitive moat.** BLOCKED on Box-Muller (base sampler).

**B20 — `NestedDirichletProcessSample(n_groups int, n_per_group []int, alpha, beta float64, base BaseDistribution, rng)`.** Rodríguez-Dunson-Gelfand-2008-JASA-103(483) `Q ~ DP(α, DP(β, H))`, each group draws `G_j ~ Q`. Distribution-on-distributions: groups can share entire G_j (not just atoms), giving group-clustering + atom-clustering simultaneously. ~200 LOC.

**B21 — `DependentDirichletProcessSample(n int, covariates [][]float64, kernel func(x, x') float64, alpha float64, base BaseDistribution, rng)`.** MacEachern-1999 covariate-dependent atoms via stick-breaking weights `π_k(x) = β_k(x) Π_{j<k}(1 − β_j(x))` smoothly varying with covariate x. Kernel-induced dependence. ~240 LOC. **DEFERRED** — covariate-modelling overlap with B13 variational-DP. PR-8.

**Tier-D total: ~520 LOC NET-NEW + ~240 LOC deferred.**

### Tier-E — Completely Random Measures (NET-NEW)

**B22 — `GammaProcessSample(alpha float64, B0_atoms, B0_weights []float64, eps float64, rng)`.** Brix-1999-Adv.Appl.Probab-31. Sum-of-spikes representation: `μ = Σ s_k δ_{x_k}` with Σ Poisson on `(0,∞) × Θ` with intensity `α · s^{-1} e^{-s} ds × B_0(dx)`. Substrate for Ferguson-original-DP-via-gamma-randomization `G = G_α / G_α(Θ)`. ~140 LOC. BLOCKED on prob.PoissonSample + prob.GammaSample.

**B23 — `NegativeBinomialProcessSample(r_atoms, r_weights []float64, p float64, rng)`.** Zhou-Carin-2015 NBP — Compound-Poisson construction with gamma random rate. Cross-link to count-data BNP (topic models with overdispersion). ~120 LOC.

**B24 — `SubordinatorSample(alpha float64, type string, t float64, eps, rng) []float64`.** Lévy-process simulation for subordinators (non-decreasing Lévy): `gamma`, `inverse-Gaussian`, `alpha-stable`. Truncate jumps below `eps`; sum jump-sizes via Poisson-process rejection on Lévy intensity `ν(s) ds`. ~180 LOC.

**B25 — `LevyKhinchineCRMSample(sigma2 float64, levy_intensity func(s) float64, B0 BaseDistribution, eps, rng)`.** General completely-random-measure with Lévy-Khinchine triplet `(σ², ν, b)`. Realise as `μ = Σ_k s_k δ_{x_k}` with Σ Poisson on `(0,∞) × Θ` with rate `ν(s) ds × B_0(dx)`. Universal CRM substrate underlying B17 / B22 / B23 / DP-via-gamma-process. ~160 LOC. **Singular Block-C-2026 frontier**: no production library exposes underlying CRM machinery.

**Tier-E total: ~600 LOC NET-NEW.**

### Tier-F — Gaussian-Process boundary (DEFER to slot 237)

**B26 — `SparseGPFITC(X_train, y_train, X_inducing, X_test, kernel, sigma_noise) (mean, var)` + `SparseGPVFE(...)`.** Snelson-Ghahramani-2006-NIPS Fully-Independent-Training-Conditional + Titsias-2009-AISTATS-5 Variational-Free-Energy. Inducing-points approximation `q(f, u) = q(f|u)q(u)` with `q(f|u) = N(K_fu K_uu^{-1} u, diag(K_ff − K_fu K_uu^{-1} K_uf))`. Composes 184-C20 GaussianProcessPosterior + linalg.Cholesky. ~280 LOC. **BOUNDARY-DEFER to slot 237 dedicated GP review** — flagged here only because Sethuraman-DP atoms-and-weights `G = Σ π_k δ_{θ_k}` is structurally adjacent to GP `f = Σ π_k k(·, θ_k)` Karhunen-Loève expansion; the BNP-GP unification is via NTK Jacot-Gabriel-Hongler-2018 cross-link.

**Tier-F total: ~280 LOC DEFERRED.**

---

## 2. Connective tissue LOC summary

| Tier | Sub-package | Source | NET-NEW | OVERLAP-181 | DEFERRED |
|---|---|---:|---:|---:|---:|
| A | `prob/bnp/sticks.go` (B3-B5 already 181) | 480 | 0 | 480 | 0 |
| A | `prob/bnp/sticks.go` (B6-B8 NEW Sethuraman/GEM/PD) | 320 | 320 | 0 | 0 |
| B | `prob/bnp/dp.go` (B9 DP-mixture-Gibbs) | 280 | 280 | 0 | 0 |
| B | `prob/bnp/inference.go` (B10 slice + B11 blocked + B14 concentration) | 480 | 480 | 0 | 0 |
| B | `prob/bnp/variational.go` (B13 variational-DP) | 240 | 240 | 0 | 0 |
| B | `prob/bnp/inference.go` (B12 split-merge) | 280 | 0 | 0 | 280 |
| C | `prob/bnp/ibp.go` (B15-B18 IBP + Beta-process + 2P-IBP) | 560 | 560 | 0 | 0 |
| D | `prob/bnp/hdp.go` (B19+B20 HDP + NDP) | 520 | 520 | 0 | 0 |
| D | `prob/bnp/ddp.go` (B21 DDP) | 240 | 0 | 0 | 240 |
| E | `prob/bnp/crm.go` (B22-B25 CRM family) | 600 | 600 | 0 | 0 |
| F | `prob/bnp/sparse_gp.go` (B26 FITC+VFE) | 280 | 0 | 0 | 280 |
| Pre-req | `prob/random.go` (Box-Muller / Beta / Gamma / Poisson samplers) | 250 | 250 | 0 | 0 |
| | **TOTAL** | **4,530** | **3,250** | **480** | **800** |

Net active landing budget: **~3,750 LOC NET-NEW + ~480 LOC ship-with-181-PR-3 + ~250 LOC prob/random pre-req = ~4,480 LOC** active in 6 PRs. ~800 LOC deferred.

---

## 3. Recommended PR sequence

**PR-0 (PRE-REQ, ~250 LOC, blocks 6 reviews simultaneously):** `prob/random.go` Box-Muller + Beta-via-Gamma + Marsaglia-Tsang-Gamma + Knuth-Cinlar-Poisson samplers. Same-priority as 227-PR-1; SIXTH independent review demanding it. Single highest-leverage PR in the entire 400-sequence (now: 117 + 184 + 188 + 202 + 215 + 216 + 217 + 227 + 228 = NINE Block-C slots blocked).

**PR-1 (~620 LOC, ship-once-with-181):** B3 CRP + B4 PY + B5 Ewens (already 181-PR-3 keystone) + B6 StickBreaking + B7 GEM + B8 PoissonDirichlet. The irreducible computational form of DP/PY. Cross-language pin: scipy `dirichlet_process_truncated`, R `dirichletprocess` package, Mathematica `DirichletDistribution[...]`. R-MUTUAL-3/3 pin: CRP-marginal × Ewens-explicit × Sethuraman-stick-breaking-converged should agree on cluster-count distribution at 1e-6 for n=200, α=2. **PARTIALLY BLOCKED on Box-Muller for B6/B8 sampling — closed-form-PMF parts ship today.**

**PR-2 (~360 LOC):** B9 DP-mixture-Gibbs + B14 concentration-parameter inference. Single most-cited production-BNP primitive. Cross-link to 169-S5 EM-for-GMM as natural-onramp (~120-LOC delta on top of S5). BLOCKED on PR-0.

**PR-3 (~400 LOC):** B10 slice-sampler + B11 blocked-Gibbs. Two alternative DP-mixture-inference algorithms; slice is provably-exact-no-truncation; blocked is production-faster-with-bounded-truncation-error. BLOCKED on PR-0.

**PR-4 (~240 LOC):** B13 variational-DP. Cross-package leverage with optim/proximal/Fbs already shipped. The natural extension of 169-S5 EM-for-GMM ELBO machinery to infinite-K-with-truncation. **Singular cross-package leverage.**

**PR-5 (~560 LOC):** B15 IBP + B16 IBPLogLikelihood + B17 BetaProcess + B18 2P-IBP. Complete Indian-Buffet/Beta-Process family. Pin scipy / R `IBPsample`. BLOCKED on PR-0 (Beta + Poisson samplers).

**PR-6 (~520 LOC):** B19 HDP + B20 NDP. **Singular competitive moat.** No zero-dependency Go library ships HDP-CRF; only Mathematica `BayesNonparametrics` and Python `bnpy` (research code). Pin Teh-2006 reference at toy-LDA topic-clustering. BLOCKED on PR-0 + PR-1 + PR-2.

**PR-7 (~600 LOC):** B22 GammaProcess + B23 NegBinProcess + B24 Subordinator + B25 LevyKhinchineCRM. **Singular Block-C-2026 frontier.** Universal CRM substrate. BLOCKED on PR-0.

**PR-8 (DEFERRED, ~800 LOC):** B12 split-merge MCMC + B21 DDP + B26 SparseGP-FITC/VFE. B26 deferred to slot 237 dedicated GP review.

Total active PR1-PR7: ~3,300 LOC source + ~1,800 LOC tests over ~14 engineer-days.

---

## 4. R-MUTUAL-CROSS-VALIDATION pins

Five 3/3 pins available:

1. **CRP-Ewens-Sethuraman** (PR-1): CRP-marginal × Ewens-explicit × Sethuraman-stick-breaking-converged on cluster-count distribution n=200, α=2, 1e-6 tolerance. Inherits 181-PR-3 pin.
2. **DP-mixture-three-way** (PR-2 + PR-3): collapsed-Gibbs-B9 × slice-sampler-B10 × blocked-Gibbs-B11 on Old-Faithful-eruption-data clusters; mean-cluster-count converges within ±0.5 across all three at 5000 iterations.
3. **IBP-Beta-process-equivalence** (PR-5): direct-IBP-sample × Beta-process-via-Lévy × stick-breaking-IBP (Teh-Görür-Ghahramani-2007 alternative-construction); column-sum distribution matches at 1e-3.
4. **HDP-CRF-vs-direct-Sethuraman** (PR-6): Chinese-restaurant-franchise sampler × direct-two-level-stick-breaking; per-group-atom posteriors agree at 1e-3.
5. **CRM-Lévy-Khinchine-vs-named-CRMs** (PR-7): general LevyKhinchineCRM-B25 × specialised-Beta-process-B17 / Gamma-process-B22 / NegBin-process-B23 on identical Lévy-intensity; sample-quantiles match at 1e-3.

---

## 5. Cross-language pinning targets

| Primitive | Reference | Tolerance |
|---|---|---|
| B3 CRP | R `MCMCpack::DirichletProcess`, Python `bnpy.allocmodel.HDPModel` (CRP marginal) | 1e-6 |
| B4 Pitman-Yor | R `BNPMix::PYprior`, Python `bnpy` | 1e-6 |
| B6 Sethuraman | scipy stick-breaking via Beta-iid; reference Sethuraman-1994 truncation `K=1000, α=2` partial sums to 1e-9 | 1e-6 |
| B9 DP-mixture-Gibbs | scikit-learn `BayesianGaussianMixture(weight_concentration_prior_type='dirichlet_process')` (truncated stick-breaking VI), R `dirichletprocess::DirichletProcessGaussianMixture`, Python `bnpy` | empirical posterior 1e-3 |
| B13 variational-DP | scikit-learn `BayesianGaussianMixture` ELBO-trace converged at `weight_concentration_prior=0.1`, n=2000 Old-Faithful; final ELBO match 1e-4 | 1e-4 |
| B15 IBP | Python `pymc.IBP` (no public API — pin Griffiths-Ghahramani-2005 Theorem 1 `E[K_+] = α·H_n` empirically over 1e4 samples) | empirical 1% |
| B17 Beta-process | R `BNPdensity` (Argiento-Bianchini-Guglielmi 2016) Beta-process posterior; jump-distribution Lévy-Khinchine match | 1e-3 |
| B19 HDP | Python `bnpy.allocmodel.HDPModel` Chinese-restaurant-franchise n_groups=10, K_max=20, γ=1, α=1; topic-distribution Hellinger distance | 1e-2 |
| B22 Gamma-process | Brix-1999 explicit Lévy-intensity test; mean `E[μ(Θ)] = α·B_0(Θ)` to 1e-2 over 1e4 samples | empirical 1% |

Cross-language coverage is THIN compared to RMT/UQ — BNP libraries are research-code-grade. Reality's golden-file-pinning approach against analytic-closed-form (CRP-Ewens-Sethuraman, IBP `E[K_+]=α H_n`, Beta-process Lévy-jump-quantiles) carries the burden of correctness witness more heavily here than in 216/227.

---

## 6. Differentiation vs adjacent agents

- **Agent 181 synergy-combinatorics-prob**: 5/26 BNP primitives shared (B1=U8, B2=U9, B3=U10, B4=U11, B5=U12 — all CRP/PY/Ewens/Hoppe/Polya). 21/26 NET-NEW here (Sethuraman + DP-mixture-inference + IBP + Beta-process + HDP + NDP + CRM-family + variational-DP). 181 was combinatorics×prob synergy; this is BNP-pure focusing on infinite-mixture and CRM.
- **Agent 169 synergy-prob-optim**: B13 variational-DP composes 169-S5 EM-for-GMM ELBO machinery. Single shared cross-link; no shared primitive.
- **Agent 184 synergy-linalg-prob**: 184-C20 GaussianProcessPosterior is the substrate B26 FITC/VFE would consume — but B26 boundary-deferred to 237.
- **Agent 188 synergy-prob-linalg**: no shared primitives but DPP at 188-D17 is at BNP-RMT-submodular intersection.
- **Agent 216 new-rmt**: 216-R13 DPP is at BNP-RMT intersection (Macchi-1975 DPP = repulsive-point-process; orthogonal-but-adjacent to BNP attractive-clustering DP). Free-Poisson = MP at 216-R1 cross-link to BNP via Bercovici-Pata at 217.
- **Agent 217 new-free-prob**: 217 covers free-cumulants on NC-partition lattice; subtle BNP cross-link via free-Poisson = MP = scaling-limit of DP-mixture posterior in some specific regimes (research-frontier — not actionable in this slot).
- **Agent 223 new-submodular**: DPP at 223 (cross-link to 216-R13 + 188-D17). Three independent slots converging on DPP placement. Not BNP-direct.
- **Agent 227 new-uq**: B26 FITC/VFE is at BNP-UQ-GP triple-intersection — boundary-defer to 237.
- **Agent 237 new-gaussian-process**: dedicated GP slot. B26 FITC/VFE BOUNDARY-DEFER. NTK cross-link mentioned but actionable in 237 not 228.

---

## 7. Bottom line

reality v0.10.0 has **zero BNP surface** but possesses **substantially more BNP substrate than expected**: `prob.LogGamma`, `prob.BetaPDF/CDF`, `prob.PoissonPMF/CDF`, `prob.GammaPDF/CDF`, `combinatorics.StirlingFirst`, `combinatorics.IntegerPartitions`, `combinatorics.BellNumber`, `optim/proximal.Fbs` — together these substrates mean the 26 B-primitives connect through ~3,750 LOC of NEW code with ~500 LOC of substrate reuse.

The single Tier-0 blocker is **`prob/random.go` Box-Muller + Beta + Gamma + Poisson samplers ~250 LOC**, called out by NINE independent Block-C reviews (117, 184, 188, 202, 215, 216, 217, 227, 228). It is the highest-priority cross-cutting Block-C unblocker in the entire 400-sequence. Ship it FIRST.

The single highest-leverage moat is **B19 HDP-CRF ~320 LOC** — no zero-dependency Go (or any production cross-language) library implements Teh-2006 Chinese-restaurant-franchise sampler in production-grade form; reality would be FIRST. Combined with B20 NDP ~200 LOC the PR-6 hierarchical-BNP package is the architectural-witness that BNP is a genuine library not a curiosity.

The single highest-impact frontier is **B25 LevyKhinchineCRM ~160 LOC** anchoring the universal-completely-random-measure construction underlying all of B15-B17 (Beta-process), B22 (Gamma-process), B23 (NegBin-process), and Ferguson-original-DP-via-gamma-randomization. Once CRM ships, the `bnp/crm.go` substrate makes B17/B22/B23 *specialisations* of one universal sampler.

The single highest-leverage cross-package leverage is **B13 VariationalDP composing 169-S5 EM-for-GMM ELBO ~240 LOC** — turns the finite-K GMM-EM keystone into infinite-K-with-truncation BNP-GMM at ~180-LOC delta. Sip-for-sip, this is the cheapest BNP-onramp once 169-S5 lands.

26 primitives, 8 PRs, ~3,300 LOC active source + ~1,800 LOC tests over ~14 engineer-days, blocked on 1 cross-cutting Tier-0 pre-req (`prob/random.go`) shared with 8 other Block-C reviews. Recommended placement: NEW sub-package `prob/bnp/` with 8 files mirroring `prob/copula/` and `prob/conformal/` and PROPOSED `prob/freeprob/` (217) precedent.
