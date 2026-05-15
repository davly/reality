# 234 — new-copulas

**Summary L1:** `prob/copula/` ships a credible v1 (Sklar n=2,3 elliptical CDFs + bivariate Clayton/Gumbel CDF/PDF/h-fn + minimal D-vine LogPDF + KendallTau + Kruskal link), pinned to a Solvency II Article 104 use-case and an autodiff-gradient saturation, totalling ~3,984 LoC.
**Summary L2:** Major gaps versus the ~30 cutting-edge bullets are: full Archimedean family (Frank/Joe/AMH/Plackett/BB1/BB7/BB8) including negative-tau Clayton + 90/180/270 rotations, Gaussian/t copula PDFs (only CDFs ship), tail-dependence λ_U/λ_L generic accessors, survival/conditional/empirical copulas, GoF (Cramér-von Mises / Anderson-Darling / White), MPL/IFM/CML estimation, R-vine matrix structure + Dißmann sequential selection, NAC, mixed/time-varying copulas, Vuong/Clarke tests, and Spearman ρ + Blomqvist β rank stats. Estimated connective-tissue addition ~2,500 LoC across 8 new files; reality is ~30% of the way to a full McNeil-Frey-Embrechts/QRMQuantitative-Risk-Management-grade copula substrate.

## What ships today (as of v0.10.0)

Files (`C:\limitless\foundation\reality\prob\copula\`, 11 source files + 9 test files, 3,984 LoC):

| File | Purpose | LoC |
|------|---------|-----|
| `doc.go` | Package doc, Sklar statement, Solvency II citation, references | 192 |
| `sklar.go` | `SklarJointFromMarginals`, `CopulaCDF`/`MarginalCDF`/`JointCDF` types, `GaussianCopulaCDFFn`, `StudentTCopulaCDFFn`, `GaussianCopulaCorrelationFromTau` (Kruskal sin·π·τ/2) | 142 |
| `gaussian.go` | `GaussianCopulaCDF` for n∈{2,3} (Drezner-Wesolowsky 1990 10-pt GL bivariate, Plackett 1954 reduction 16-pt GL trivariate), `BivariateNormalCDF`, `TrivariateNormalCDF` | 357 |
| `studentt.go` | `StudentTCDF`, `StudentTQuantile`, `BivariateTCDF`, `TrivariateTCDF` | 240 |
| `t.go` | `StudentTCopulaCDF` n∈{2,3} via Genz-Bretz 2009 §4.2 | 255 |
| `archimedean.go` | `ClaytonCopulaCDFFn`, `GumbelCopulaCDFFn`, `ThetaFromKendallTau` (Genest-MacKay 1986), `ClaytonLowerTailDependence`, `GumbelUpperTailDependence`, `ArchimedeanFamily` enum | 180 |
| `pdf.go` | `ClaytonPDFFn`, `ClaytonLogPDFFn`, `GumbelPDFFn`, `GumbelLogPDFFn`, `LogPDFFnForFamily` | 169 |
| `hfunctions.go` | `ClaytonHFn`, `GumbelHFn`, `HFnForFamily` (conditional CDF ∂C/∂v) | 131 |
| `vine.go` | `VineEdge`, `DVine` struct, `NewDVine`, `HFunctionPass`, `LogPDF` | 240 |
| `kendall_tau.go` | `KendallTau` (O(n²) tau-a), `EmpiricalCdf` (rank/(n+1) PIT) | 105 |
| `errors.go` | Sentinels: `ErrEmptyU`, `ErrUOutOfRange`, `ErrSigmaDimensionMismatch`, `ErrSigmaNotPSD`, `ErrLengthMismatch` | 43 |

Tests: `archimedean_test.go` (307), `autodiff_test.go` (156 — `TestClaytonLogPDF_AutodiffGradientMatchesAnalytic` saturating R-CLOSED-FORM-PINNED-TO-AUTODIFF 3/3), `gaussian_test.go` (299, includes `TestCrossSubstratePrecision_RubberDuck_*` R80b parity), `hfunctions_test.go` (160), `kendall_tau_test.go` (191), `pdf_test.go` (181), `sklar_test.go` (182), `t_test.go` (223), `vine_test.go` (231).

Internal deps: `linalg.CholeskyDecompose` (Σ PSD validation), `prob.NormalCDF` / `prob.NormalQuantile` (probit). Zero non-stdlib external deps.

Provenance: package was promoted via Reality Math Hunt §12 (composite C×T×N=5×4×5=100, "no other ecosystem flagship has the regulatory standing to land it first"), L13 REVOLUTIONARY in Ecosystem Revolution Hunt 2026-04-28, with named consumers (relic-insurance Solvency II, rampart cyber-loss, Sentinel-AV, odds-engine, oracle-vote, underwrite, insights, crucible R-stats, RubberDuck.Core.Analysis.CopulaModels keystone).

## Coverage map vs cutting-edge bullets

| # | Topic | Reality status | What to add |
|---|-------|---------------|-------------|
| 1 | Sklar's theorem F(x)=C(F_i(x_i)) | **SHIPS** `SklarJointFromMarginals` (sklar.go:81) | — |
| 2 | Copula = joint w/ uniform margins | **SHIPS** (definitionally encoded by `CopulaCDF` type) | — |
| 3 | Independence copula Π(u,v)=uv | **MISSING** explicit; falls out of Clayton θ→0 / Gumbel θ=1 | 5 LoC `IndependenceCopulaCDFFn` for completeness + n-variate variant |
| 4 | Comonotone M(u,v)=min(u,v) (Fréchet upper) | **MISSING** | 8 LoC `FrechetUpperBoundCopulaCDFFn` |
| 5 | Countermonotone W(u,v)=max(u+v−1,0) (Fréchet lower; only valid for n=2) | **MISSING** | 8 LoC `FrechetLowerBoundCopulaCDFFn` (with n=2 guard) |
| 6 | Gaussian copula | **SHIPS CDF n={2,3}**; **MISSING PDF, sampler, n≥4 via Genz QMC** | 60 LoC `GaussianCopulaPDF`+`GaussianCopulaLogPDF` (closed-form: ratio of n-variate normal density to product of Φ′(probit u_i)); 200 LoC Genz QMC for n≥4 |
| 7 | Student-t copula (heavy tails) | **SHIPS CDF n={2,3}**; **MISSING PDF, sampler, n≥4 QMC** | 80 LoC `StudentTCopulaPDF` (closed-form via t_n-density / product of t_1-densities); 250 LoC Genz QRSVN for n≥4; 30 LoC `StudentTCopulaLowerTailDependence`/`UpperTailDependence` (= 2 t_{ν+1}(−√((ν+1)(1−ρ)/(1+ρ))), symmetric) |
| 8 | Archimedean family | **PARTIAL** Clayton + Gumbel only (BB1 omitted) | See row 11 |
| 9 | Generator φ | **IMPLICIT** in closed-form CDFs; **NO** explicit generator interface | 120 LoC `ArchimedeanGenerator` interface (`Phi(t)`, `PhiInv(s)`, `PhiPrime`, `PhiInvPrime`) + concrete impls; enables generic `c(u,v)`, `C(u,v)`, h-fn, sampler factories |
| 10 | 2D Archimedean → multidim via nested (Joe-Hu 1996) | **MISSING** | 250 LoC `NestedArchimedean` tree struct + LogPDF + sampler (Marshall-Olkin 1988 frailty representation; sufficient nesting condition Hofert 2010) |
| 11 | Frank, Joe, AMH, BB1, BB7, BB8, Plackett, FGM, Cuadras-Augé, Galambos, Hüsler-Reiss, Tawn | **MISSING** | ~700 LoC, one file each: Frank (negative-τ admissible, nice symmetry, eq. (10) Joe 1997), Joe (upper-tail, no lower, complement of Clayton-rotated-180), AMH (limited τ∈(−0.181,1/3), Genest 1987), BB1 (2-param Clayton-Gumbel hybrid, both tails), BB7 (Joe-Clayton, both tails), BB8 (Joe-Frank), Plackett (no analytic τ↔θ but log-odds-ratio interpretation), FGM (perturbation of independence, |τ|≤2/9), Galambos/HR/Tawn (extreme-value family, link to slot 233 EVT) |
| 12 | Negative-θ Clayton + Frank symmetry | **MISSING**; current `ClaytonCopulaCDFFn` rejects θ≤0 | 20 LoC: extend Clayton to θ∈[−1,∞)\{0}; document Frank handles all θ∈ℝ\{0} |
| 13 | Rotations 90°/180°/270° (survival, etc.) | **MISSING** | 40 LoC generic `RotateCopulaCDFFn(C, deg)` and `RotateCopulaPDFFn`; rotation 180 = survival copula |
| 14 | Nested Archimedean (NAC, Hofert 2010) | **MISSING** | See row 10. Add `NACSufficientCondition` validator (outer θ_outer ≤ inner θ_inner for compatibility) |
| 15 | Vine copulas (Bedford-Cooke 2002, Aas-Czado 2009) | **PARTIAL** D-vine LogPDF only, Clayton+Gumbel only | 350 LoC: C-vine struct + LogPDF, R-vine matrix (Dißmann 2013 §3) sequential structure with selection by max-tau spanning-tree (Kruskal MST); generic per-edge family selection |
| 16 | C-vine | **MISSING** (vine.go ships D-vine only, comments call out C-vine deferred to demand-side pull) | 180 LoC `CVine` mirror of `DVine` |
| 17 | D-vine | **SHIPS** (`DVine.LogPDF`) | Add D-vine sampler (Aas-Czado 2009 Algorithm 5) ~120 LoC |
| 18 | R-vine | **MISSING** | 600 LoC: `RVineMatrix` (Dißmann 8×8 lower-triangular family/parameter encoding), `RVineMatrix.LogPDF`, `RVineMatrix.Sample`, `RVineMatrix.Validate` (proximity condition Bedford-Cooke 2002) |
| 19 | Pair-copula construction (PCC) | **PARTIAL** D-vine only | Row 18 generalises |
| 20 | 16 standard bivariate building blocks | **PARTIAL** 2 of 16 (Clayton, Gumbel); BiCopSelect (VineCopula R-pkg) lists 16 | Row 11 covers 6, row 13 covers 4 rotations × 6 = 24 → 40+ identifiable blocks |
| 21 | Copula density (mixed partial ∂²C/∂u∂v) | **SHIPS** Clayton + Gumbel via closed-form (`pdf.go`), not via differentiation | Add Gaussian, t, Frank, Joe, AMH, BB1, BB7 closed forms (~250 LoC) |
| 22 | Tail dependence λ_U, λ_L | **PARTIAL** `ClaytonLowerTailDependence`, `GumbelUpperTailDependence`; **NO** generic accessor, **NO** Gaussian (always 0)/t (formula above) | 60 LoC: `func TailDependence(c CopulaSpec) (lower, upper float64)`; cross-link to slot 233 EVT extreme-value bivariate copulas |
| 23 | Conditional copula | **PARTIAL** h-functions = ∂C/∂v (Clayton + Gumbel) | Generalise to `ConditionalCopula(C, condIdx)` for any C; ~80 LoC |
| 24 | Survival copula Ĉ(u,v)=u+v−1+C(1−u,1−v) | **MISSING** | 40 LoC `SurvivalCopulaCDFFn(C)`; same as 180° rotation |
| 25 | Empirical copula Ĉ_n(u,v)=(1/n)Σ 1{F̂_n(X_i)≤u, Ĝ_n(Y_i)≤v} | **PARTIAL** `EmpiricalCdf` returns ranks, no joint stepfunction | 80 LoC `EmpiricalCopula(x,y)` returning closure |
| 26 | GoF: Cramér-von-Mises, Anderson-Darling | **MISSING** | 200 LoC `CramerVonMisesCopula(samples, C)`, `AndersonDarlingCopula`, parametric-bootstrap p-value (Genest-Rémillard-Beaudoin 2009) |
| 27 | Estimation: ML, semi-parametric, IFM | **PARTIAL** `ThetaFromKendallTau` (closed-form moment match for Clayton/Gumbel only) | 400 LoC: `MaximumPseudoLikelihood(samples, family) → θ̂` (Genest-Ghoudi-Rivest 1995, profile-likelihood w/ Brent root-find using existing `optim.Brent`), `IFM` (Joe-Xu 1996) for parametric margins, generic `MaximumLikelihood` Newton with autodiff-supplied Hessian |
| 28 | Maximum pseudo-likelihood | Row 27 covers | — |
| 29 | Kendall's τ | **SHIPS** `KendallTau` O(n²) | Add Knight 1966 O(n log n) variant for large n; ~80 LoC |
| 30 | Spearman's ρ | **MISSING** | 40 LoC `SpearmanRho` (rank Pearson) + closed-form Gaussian-copula link ρ_S = (6/π)·arcsin(ρ/2) (Pearson 1907) |
| 31 | Blomqvist's β (medial correlation) | **MISSING** | 30 LoC `BlomqvistBeta` = 4·C(½,½) − 1 |
| 32 | Concordance τ = 4·∫∫C dC − 1 | **PARTIAL** sample form via `KendallTau`; **MISSING** population form `ConcordanceFromCopula(C)` numeric integral | 60 LoC quadrature + closed-form per-family overrides |
| 33 | Vine structure selection | **MISSING** | 250 LoC: max-tau Kruskal MST per Dißmann 2013, plus AIC/BIC family pick per edge |
| 34 | Vuong test for copula model selection | **MISSING** | 100 LoC `VuongTest(logL_A, logL_B) → z, pvalue` |
| 35 | Clarke test (sign test variant) | **MISSING** | 60 LoC |
| 36 | Bayesian copula | **MISSING** | Out-of-scope without MCMC; cross-link to slot for Bayesian/MCMC. Could ship MAP estimator (90 LoC) using existing `optim` + Gaussian prior on θ |
| 37 | Mixed copulas (convex combo of multiple) | **MISSING** | 80 LoC `MixedCopulaCDF(weights, copulas)` + EM-fit (~150 LoC) |
| 38 | Stochastic copula simulation (sampling) | **MISSING** entirely | 400 LoC: Gaussian sampler (chol·Z then Φ), t sampler (chol·Z·√(ν/χ²) then T_ν), Marshall-Olkin frailty for Archimedean (Clayton: Gamma; Gumbel: positive-stable), conditional inversion for D/C/R-vines |
| 39 | Time-varying copula (DCC analog: DCC-copula, Patton 2006) | **MISSING** | 300 LoC: `DCCCopulaParams` ARMA(1,1)-on-Fisher-z evolution of θ_t, MLE; cross-link to slot for GARCH/DCC |
| 40 | High-dimensional copula challenges (sparse vines, regularisation) | **MISSING** | Lasso-vine + truncation `RVineTruncate(level k)` ~150 LoC |

## Specific surface-level recommendations

1. **Promote `ArchimedeanFamily` enum** from a 2-value enum into a 14-value enum + `ArchimedeanSpec` struct (theta1, theta2 for 2-param families).
2. **Add `CopulaSpec` interface** with methods `CDF(u []float64) (float64,error)`, `LogPDF`, `Sample(rng) []float64`, `TailDep() (l,u float64)`, `KendallTau() float64`, `SpearmanRho() float64` so vines, mixed copulas, and GoF can dispatch generically.
3. **Wire `optim` for MLE.** Currently the package estimates θ only via the closed-form `ThetaFromKendallTau` (Clayton/Gumbel). For families without analytic τ↔θ inversion (Frank, Plackett, BB1, BB7), MPL needs a 1D root-find in Brent or a 1D optim. `optim.Brent` already exists per CLAUDE.md.
4. **Sampler is the most-load-bearing missing piece.** Without samplers there's no Monte-Carlo VaR/ES under the copula — yet that's the entire actuarial use-case (Solvency II Article 122 internal models). Marshall-Olkin frailty representation is one paragraph of code per Archimedean family.
5. **R-vine matrix encoding.** The Dißmann lower-triangular matrix is the standard data structure; once `RVineMatrix` exists, vine LogPDF/sample/select all become passes over the matrix. Currently `DVine` ships the simplest topology only.
6. **Cross-link to slot 233 EVT.** Bivariate extreme-value copulas (Galambos, Hüsler-Reiss, Tawn, t-EV) are the natural co-tenants of slot 233's GPD/GEV. The Pickands dependence function A(t) is the EV-copula equivalent of the Archimedean generator and deserves a `PickandsDependenceFn` type.
7. **Cross-link to slot 117 prob.** `prob/copula/EmpiricalCdf` and `prob/copula/KendallTau` could promote to top-level `prob` (they're general rank statistics, not copula-specific); doc.go already says "the prob package keeps its t-CDF internal" — moving rank stats up is the symmetric refactor.

## Connective-tissue total

| Bucket | LoC |
|--------|-----|
| Independence/M/W explicit closures | 25 |
| Generator interface + 6 concrete | 120 |
| 8 missing Archimedean families (Frank/Joe/AMH/BB1/BB7/BB8/Plackett/FGM) | 700 |
| Negative-θ Clayton + 4 rotations | 60 |
| EV copulas (Galambos/HR/Tawn/t-EV) | 250 |
| Gaussian + t PDFs / log-PDFs | 140 |
| Sampler (Gaussian, t, Marshall-Olkin Archimedean, vine cond-inv) | 400 |
| Genz QMC for n≥4 elliptical | 250 |
| C-vine struct + LogPDF + sampler | 300 |
| R-vine matrix + LogPDF + sampler + Dißmann selection | 600 |
| NAC (nested Archimedean) | 250 |
| Survival/conditional/empirical copulas | 200 |
| GoF: CvM/AD + bootstrap | 200 |
| MPL/IFM/CML estimation framework | 400 |
| Vuong/Clarke tests | 160 |
| Spearman/Blomqvist/concordance integrals | 130 |
| Mixed copulas + EM | 230 |
| Time-varying (Patton DCC-copula) | 300 |
| Tail-dependence accessor + Gaussian/t formulas | 90 |
| Knight 1966 O(n log n) τ | 80 |
| **TOTAL ESTIMATE** | **~4,885 LoC** |

i.e. the full programme would more-than-double the package (3,984 → ~8,800 LoC), but the natural minimal-viable subset for the relic-insurance Solvency II R98 envelope is: full Archimedean family + all PDFs + sampler + R-vine matrix + MPL fitting + GoF, ~2,500 LoC.

## Priority ranking by consumer pull

| P | Item | Consumer |
|---|------|----------|
| P0 | Sampler (Gaussian + t + Clayton/Gumbel Marshall-Olkin) | All consumers — without sampling, the package is CDF-only and can't do MC-VaR/ES |
| P0 | Frank copula (handles negative τ, the only Archimedean that does naturally) | rampart cyber-loss aggregation (negative-correlation regimes between threat tiers) |
| P0 | Gaussian + t copula PDF + log-PDF | MLE/IFM estimation, autodiff-gradient pinning extension to elliptical |
| P1 | R-vine matrix + Dißmann selection | flagships/folio high-dim portfolio (15+ Solvency II sub-modules) |
| P1 | Survival copula + 180° rotation | Symmetry-breaking lower-tail-only Joe variants |
| P1 | MaximumPseudoLikelihood (generic Brent over family) | Replaces the closed-form-only `ThetaFromKendallTau` for non-{Clayton,Gumbel} |
| P2 | Cramér-von-Mises GoF + parametric bootstrap | Solvency II Article 122 internal-model "statistical quality test" — the regulator literally requires GoF |
| P2 | Spearman ρ + Blomqvist β | Convergence checks vs Kendall τ; cheap |
| P2 | NAC + Joe-Hu sufficient condition | Higher-dim Archimedean alternative to vines |
| P3 | Time-varying copula (Patton 2006) | Derivatives-pricing consumers |
| P3 | Vuong/Clarke tests | Model-selection between competing fits |
| P3 | Bayesian (MAP via optim) | Cross-link to a Bayesian slot |

## File-level proposed additions

| New file | Contents | LoC |
|----------|----------|-----|
| `prob/copula/independence.go` | Π, M (Fréchet upper), W (lower) closures | 50 |
| `prob/copula/frank.go` | `FrankCopulaCDFFn`, `FrankPDFFn`, `FrankLogPDFFn`, `FrankHFn`, `FrankThetaFromTau` (Debye-1 numeric), `FrankSample` | 220 |
| `prob/copula/joe.go` | Joe family parallel to frank.go | 200 |
| `prob/copula/amh.go` | Ali-Mikhail-Haq | 180 |
| `prob/copula/bb.go` | BB1/BB7/BB8 two-parameter families | 350 |
| `prob/copula/extreme_value.go` | Galambos, Hüsler-Reiss, Tawn, t-EV; `PickandsDependenceFn` interface | 280 |
| `prob/copula/sampler.go` | `GaussianCopulaSample`, `StudentTCopulaSample`, Marshall-Olkin frailty for Archimedean, conditional-inversion for vines | 450 |
| `prob/copula/rvine.go` | `RVineMatrix` struct, LogPDF, sample, Dißmann sequential selection | 600 |
| `prob/copula/cvine.go` | `CVine` mirror of DVine | 200 |
| `prob/copula/nac.go` | Nested Archimedean tree | 280 |
| `prob/copula/gof.go` | Cramér-von-Mises, Anderson-Darling, parametric bootstrap | 220 |
| `prob/copula/estimation.go` | `MaximumPseudoLikelihood`, `IFM`, AIC/BIC | 350 |
| `prob/copula/concordance.go` | `SpearmanRho`, `BlomqvistBeta`, `Concordance`, `TailDependence` generic | 180 |
| `prob/copula/rotation.go` | `RotateCopulaCDFFn`, `SurvivalCopulaCDFFn` | 80 |
| `prob/copula/empirical.go` | `EmpiricalCopula(x, y) → CopulaCDF` | 100 |
| `prob/copula/mixed.go` | `MixedCopulaCDF` + EM fit | 230 |
| `prob/copula/time_varying.go` | Patton DCC-copula | 300 |
| `prob/copula/tests.go` | Vuong, Clarke | 140 |

## Cross-references confirmed

- 117-prob-missing.md exists (slot 117 = prob general).
- 233-new-extreme-value.md exists — bivariate EV copulas (Tawn, Galambos, Hüsler-Reiss, t-EV) are the explicit cross-link; Pickands dependence A(t) deserves shared infrastructure.
- 222 (causal/bandits — referenced by neighbouring agents) does **not** intersect copulas.
- `flagships/rubberduck/RubberDuck.Core/Analysis/CopulaModels.cs` (213 LoC + 143 LoC tests, zero production wires) is the C# parity reference for `KendallTau` + `GaussianCopulaCorrelation` per R80b.
- Reality Math Hunt §12 op 01 (composite N=5) is the strategic lift for §12 risk-measure programme — copula primitive is the keystone, with VaR/ES/spectral risk and Panjer recursion riding on top.

## Key file paths

- `C:\limitless\foundation\reality\prob\copula\doc.go` — package overview, statutory citations, deferred-to-v2 list
- `C:\limitless\foundation\reality\prob\copula\sklar.go` — Sklar reconstruction, GaussianCopulaCorrelationFromTau (Kruskal link)
- `C:\limitless\foundation\reality\prob\copula\archimedean.go` — Clayton + Gumbel CDF, ThetaFromKendallTau, tail-dependence
- `C:\limitless\foundation\reality\prob\copula\pdf.go` — Clayton + Gumbel PDF/LogPDF
- `C:\limitless\foundation\reality\prob\copula\hfunctions.go` — Clayton + Gumbel ∂C/∂v
- `C:\limitless\foundation\reality\prob\copula\vine.go` — D-vine (no C-vine, no R-vine)
- `C:\limitless\foundation\reality\prob\copula\autodiff_test.go` — saturates R-CLOSED-FORM-PINNED-TO-AUTODIFF 3/3 (per S62)
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\233-new-extreme-value.md` — EV cross-link
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\117-prob-missing.md` — broader prob coverage
