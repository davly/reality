// Package copula implements the Gaussian + Student-t copula CDFs and
// the Sklar 1959 reconstruction theorem for joint distributions.
//
// A copula isolates the dependence structure of a multivariate
// distribution from its marginals.  Sklar's theorem (1959) states
// that for any joint CDF F on R^n with continuous marginals F_1, ...,
// F_n there is a unique copula C such that
//
//	F(x_1, ..., x_n) = C(F_1(x_1), ..., F_n(x_n))
//
// and conversely any (marginals, copula) pair composes into a valid
// joint distribution.  Copulas thus answer the actuarial question
// "given separate sub-distributions, what is their joint risk?" —
// the question Solvency II Article 104 + EIOPA Delegated Regulation
// 2015/35 Annex IV explicitly *prescribes* a Gaussian-copula answer
// to for the aggregation of sub-module SCRs.
//
// # Why this exists in Reality
//
// Per the Ecosystem Revolution Hunt 2026-04-28 entry L13 (lens
// `copulas-solvency-ii`, verdict REVOLUTIONARY) and the Reality Math
// Hunt §12 op 01 (composite C×T×N = 5×4×5 = 100 — the *only* N=5
// in the 22-agent hunt with the explicit "no other ecosystem flagship
// has the regulatory standing to land it first" rationale), copula
// aggregation is the cheapest path to a regulator-grade Solvency II
// SCR.  L13 was the third REVOLUTIONARY in the hunt because it maps
// uniquely to a NEW REGULATED REVENUE LINE under prescriptive
// statute (rather than an audit defence or a recommendatory standard).
//
// Day-one consumer surface verified at file:line in the L13 entry
// (5-substrate, 8-inverse-consumer):
//
//   - relic-insurance R98 envelope (Python, divergences.py:152-178) —
//     the FIRST insurance-domain R98 entry in the ecosystem; cites
//     `Solvency_II_Art35 + FCA_ICOBS_8.1 + FCA_SYSC_9 +
//     Insurance_Act_2015_s13A` as statutory_basis but ships zero
//     load-bearing aggregation math.  This package is the math the
//     citation was always pointing at.
//   - rampart cyber-loss aggregation (Go, internal/threat/check.go) —
//     linear sum of threat tiers; cyber-CAT has documented zero-day
//     cross-excitation that is canonical t-copula territory.
//   - Sentinel-AV ordered-tier bps (Rust, src/ops.rs:50-52) —
//     dominance / confidence / escape_score_bps with no joint
//     distribution.
//   - odds-engine line-movement (Rust template) — Reality Math Hunt
//     §19 named this as a copula-tail-dependence consumer.
//   - oracle-vote SCR-attribution-by-peril (Go) — Shapley aggregation
//     wants copula tail-dependence decomposition.
//   - underwrite Trinity-engine fraud-cluster (Go, internal/insurance/
//     screen.go) — linear evidence screen; copy-bet detection wants
//     Hawkes branching ratio + copula tail.
//   - insights observable-math risk stub (Go, internal/innovations/
//     observable_math.go).
//   - crucible R-stats sister (R) — Reality FFI consumer; first
//     non-Python copula consumer cross-substrate.
//
// Plus 1 active-but-orphaned consumer (the keystone): RubberDuck
// `RubberDuck.Core.Analysis.CopulaModels` (213 LoC + 143 LoC tests at
// `flagships/rubberduck/RubberDuck.Core/Analysis/CopulaModels.cs`,
// zero production wires).  This package is the cross-substrate Go
// translation; the FW C# original is the R80b output-parity
// reference.
//
// Consumers (verified):
//   - prob/copula/autodiff_test.go:TestClaytonLogPDF_AutodiffGradientMatchesAnalytic —
//     pins the analytic gradient of Clayton's log-density w.r.t. θ
//     against reverse-mode autodiff at 1e-9 tolerance across 5
//     (u, v, θ) cases. Third consumer of the
//     R-CLOSED-FORM-PINNED-TO-AUTODIFF pattern (after autodiff × garch
//     and infogeo × autodiff); saturates the rule to 3/3 — promotable
//     to ECOSYSTEM_QUALITY_STANDARD (S62 overnight, 2026-05-06).
//
// Flagship first-consumer push (relic-insurance Solvency II SCR
// engine wiring) remains queued; see
// LimitlessGodfather/reviews/SESSION_62_PROGRESS.md.
//
// # API surface
//
//   - GaussianCopulaCDF(u, sigma)           — n-variate (n in {2, 3})
//                                            Gaussian copula CDF.
//   - StudentTCopulaCDF(u, sigma, df)       — n-variate (n in {2, 3})
//                                            Student-t copula CDF
//                                            with tail dependence
//                                            for cat-cluster perils.
//   - SklarJointFromMarginals(margs, cop)   — Sklar reconstruction:
//                                            wires marginals + copula
//                                            into a joint CDF on R^n.
//   - GaussianCopulaCDFFn(sigma)            — closure form for use
//                                            with Sklar.
//   - StudentTCopulaCDFFn(sigma, df)        — t-copula closure form.
//   - KendallTau(x, y)                      — concordance-counter rank
//                                            correlation; canonical
//                                            fitting statistic.
//   - GaussianCopulaCorrelationFromTau(tau) — Kruskal 1958 link:
//                                            rho = sin(pi * tau / 2).
//   - EmpiricalCdf(data)                    — rank-based PIT helper
//                                            for transforming raw
//                                            samples into uniform
//                                            margins.
//   - BivariateNormalCDF / TrivariateNormalCDF /
//     BivariateTCDF / TrivariateTCDF        — exported standard-CDF
//                                            building blocks for
//                                            advanced consumers (e.g.
//                                            arbitrary-dimensionality
//                                            consumers that combine
//                                            with their own QMC).
//   - StudentTCDF / StudentTQuantile        — univariate Student-t
//                                            CDF + inverse, exposed
//                                            because the prob package
//                                            keeps its t-CDF internal.
//
// # R80b cross-substrate output parity
//
// The TestCrossSubstratePrecision_RubberDuck_* tests in
// kendall_tau_test.go + gaussian_test.go assert that the same input
// produces the same output in this Go package and in
// `RubberDuck.Core.Analysis.CopulaModels` (the C# reference impl).
// The byte-for-byte equivalence point is KendallTau (whose
// concordance-counter loop is line-by-line identical between the
// substrates) and GaussianCopulaCorrelationFromTau ↔ the FW
// `GaussianCopulaCorrelation` after the Kruskal substitution.  R80b
// (output-parity, not strict-byte) is the appropriate parity tier
// because the substrate differs (Go float64 vs C# double — both
// IEEE-754 binary64 but with differing intermediate roundings in
// transcendental functions).  Tolerance: ≤1e-9 absolute.
//
// # Dimensionality + numerical methods
//
// v1 supports n in {2, 3} only.  The bivariate normal CDF uses
// Drezner-Wesolowsky 1990 Gauss-Legendre 10-node quadrature on the
// rho-parameterised integral form.  The trivariate normal CDF uses
// the Plackett 1954 conditional reduction integrating Phi_2 over the
// marginal of the first axis with 16-node Gauss-Legendre.  The
// bivariate Student-t CDF uses the Genz-Bretz 2009 §4.2 1D integral
// form after the u = T_df(z) substitution.  The trivariate t-CDF uses
// the conditional reduction with df+1 shifted conditional law.
//
// Higher-dimensional CDFs (Genz QMC, n >= 4) are deferred to v2.
// Solvency II SCR is structurally n=15 (number of sub-modules) but
// consumers typically aggregate pairwise (n=2) and trivariate (n=3
// cat-cluster) — so v1 covers >80% of consumer call sites.
//
// Pure float64, deterministic, allocation-free in the hot path.
// Zero non-stdlib deps (only `math` + `sort` + `errors`); the only
// internal Reality dep is `linalg.CholeskyDecompose` for sigma PSD
// validation and `prob.NormalCDF` / `prob.NormalQuantile` for the
// probit-and-back transforms.
//
// # Deferred to v2
//
// Archimedean copulas (Clayton, Gumbel, Frank — the FW C# reference
// already ships closed-form theta-from-tau fits and tail-dependence
// coefficients, port is straightforward).  Vine copulas (D-vine /
// C-vine / R-vine) for high-dimensional dependence under sparse
// pairwise structure.  Genz QMC for general n.  Compound-Poisson
// Panjer recursion + GPD/POT extreme-value tail-fit + VaR / ES /
// spectral-risk-measure quartet (the full Reality Math Hunt §12
// risk-measure programme — the L13 keystone is the copula primitive,
// the rest of the §12 programme rides on top).
//
// # References
//
//   - Sklar, A. (1959).  Fonctions de répartition à n dimensions et
//     leurs marges.  Publ. Inst. Statist. Univ. Paris 8: 229-231.
//   - Kendall, M. G. (1938).  A New Measure of Rank Correlation.
//     Biometrika 30: 81-93.
//   - Kruskal, W. H. (1958).  Ordinal Measures of Association.  JASA
//     53: 814-861.
//   - Plackett, R. L. (1954).  A Reduction Formula for Normal
//     Multivariate Integrals.  Biometrika 41: 351-360.
//   - Drezner, Z. & Wesolowsky, G. O. (1990).  On the Computation of
//     the Bivariate Normal Integral.  J. Stat. Comp. Sim. 35: 101-107.
//   - Embrechts, P., Lindskog, F. & McNeil, A. (2003).  Modelling
//     Dependence with Copulas and Applications to Risk Management.
//     In Handbook of Heavy Tailed Distributions in Finance.
//   - Demarta, S. & McNeil, A. (2005).  The t Copula and Related
//     Copulas.  Int. Stat. Review 73: 111-129.
//   - Genz, A. (2004).  Numerical Computation of Rectangular
//     Bivariate and Trivariate Normal and t-Probabilities.  Statistics
//     and Computing 14: 251-260.
//   - Genz, A. & Bretz, F. (2009).  Computation of Multivariate
//     Normal and t Probabilities.  Lecture Notes in Statistics 195.
//     Springer.
//   - Nelsen, R. B. (2006).  An Introduction to Copulas, 2nd ed.
//     Springer.
//   - European Insurance and Occupational Pensions Authority (EIOPA).
//     Commission Delegated Regulation (EU) 2015/35 — Solvency II
//     supplementing Directive 2009/138/EC, Annex IV (correlation
//     matrices for SCR aggregation).  Article 104 of Directive
//     2009/138/EC prescribes Gaussian-copula aggregation of sub-
//     module SCRs.
package copula
