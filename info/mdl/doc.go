// Package mdl implements Minimum Description Length / codelength
// primitives for parametric model selection: NML (Normalised
// Maximum Likelihood) parametric-complexity for Bernoulli /
// multinomial families, BIC and AIC adapter scores, universal
// integer codelengths, and the SelectMDL argmin-on-codelengths
// helper.
//
// # Why this exists in Reality
//
// Per the Ecosystem Revolution Hunt 2026-04-28 entry L12 (lens
// `minimum-description-length`, verdict NOTABLE), the ecosystem
// today picks model orders / lag counts / bin counts via:
//
//   - Hand-rolled BIC = T*log(RSS/T) + numParams*log(T)
//     (RubberDuck Granger lag selection).
//   - AIC = 2k - 2 logL (RubberDuck GEV tail; Simulacra octave
//     calibration).
//   - "Simplified BIC" comments admitting non-rigour (RubberDuck
//     Cointegration tests).
//   - Hard-coded scalars: 50 bins (Sensorhub), 10 bins (Nexus
//     oracle), 40 schema cards (FleetWorks BM25).
//
// These are asymptotic Laplace approximations that the regularity
// conditions (smooth log-likelihood interior, finite Fisher
// information) often fail in the actual production regime — e.g.
// the GEV xi parameter sits at a boundary (xi = 0 separates
// Gumbel/Frechet/Weibull regimes) where AIC's shape assumption
// breaks (Grünwald 2007 §11).
//
// NML supersedes AIC/BIC by computing the *exact* parametric
// complexity of the model class instead of its asymptotic
// approximation.  For finite-alphabet (multinomial) models the
// Kontkanen-Myllymäki 2007 linear-time recursion makes the
// computation tractable; for arbitrary parametric classes the
// universal-integer codelength + two-part MDL formulation gives
// a constructive recipe.
//
// This package is the codelength side of the L12 deliverable;
// `reality/info/lz` (sibling package) ships the LZ76 model-free
// side.  Both ship under `reality/info/` alongside the L10
// capacity-bound packages (when those land); the three subpackages
// are co-located but disjoint in scope.
//
// # API surface
//
//   - NMLMultinomial(counts) — Kontkanen-Myllymäki 2007 linear-time
//     parametric-complexity regret in nats.
//   - NMLBernoulli(successes, trials) — special case k = 2.
//   - BernoulliCodeLength(successes, trials) — full MDL codelength
//     (NLL + NML regret).
//   - GaussianCodeLength(samples, mu, sigma) — fixed-Gaussian
//     codelength of a sample vector.
//   - ModelCodeLength(numParams, sampleSize) — BIC-shape
//     (numParams/2) * log(n).
//   - BICShape(negLogLikelihood, numParams, sampleSize) — full BIC
//     adapter for legacy consumer 1-to-1 swap.
//   - AICShape(negLogLikelihood, numParams) — full AIC adapter for
//     legacy consumer 1-to-1 swap.
//   - UniversalIntegerCodeLength(n) — Rissanen 1983 log*(n) in
//     nats.
//   - UniversalIntegerCodeLengthBits(n) — same, in bits.
//   - SelectMDL(codeLengths) — argmin index over model codelengths.
//   - SelectMDLWithMargin(codeLengths) — argmin + gap-to-second-best
//     diagnostic.
//
// # Inverse-consumers
//
// Same seven cross-substrate sites the L12 hunt identified — see
// `reality/info/lz/doc.go` for the full list (RubberDuck Granger
// + GEV + Cointegration; Simulacra octave AIC+BIC; Causal-engine
// GES local_score_BIC; Sensorhub histogram; Nexus oracle bins).
// The recommended retrofit pattern is:
//
//  1. Identify the asymptotic Laplace BIC term in the consumer's
//     code:  `bic = -2*ll + k*log(n)`.
//  2. Replace with `mdl.BICShape(-ll, k, n)` for a 1-to-1 numeric
//     swap (validates that the existing implementation matches the
//     standard formula).
//  3. For multinomial / Bernoulli / GEV-tail consumers, swap further
//     to `mdl.NMLMultinomial(counts)` or `mdl.NMLBernoulli(s, t)`
//     once a consumer-side test corpus shows BICShape and NML
//     diverge (the boundary case).
//
// # Phase A scope and v2 deferrals
//
// v1 ships the cheapest cut of the MDL surface identified by L12:
//
//   - NML for Bernoulli + multinomial.
//   - BIC + AIC adapters.
//   - Universal integer code (Rissanen 1983).
//   - Gaussian + multinomial codelength helpers.
//   - SelectMDL argmin helper.
//
// Deferred to v2 per Pre-Mortem 007 founder-time discipline:
//
//   - Solomonoff prior M (incomputable; approximate via CTW or
//     Block Decomposition Method).
//   - Wallace MML formulation (Wallace-Boulton 1968 / Wallace 2005)
//     — the Bayesian-prior alternative to NML; subtly-different
//     answers in boundary regimes.
//   - BIC-as-MDL formal proof (Schwarz 1978 -> Rissanen 1996
//     stochastic-complexity equivalence under regularity).
//   - fNML for graphical models (Roos-Silander-Kontkanen-Myllymäki
//     2008) — decomposable per-context NML for Bayesian-network
//     structure learning.  The Causal-engine GES retrofit lives
//     here.
//   - Luckiness-NML for unbounded parameter spaces (Grünwald 2007
//     §11.4) — the GEV-xi-boundary case proper handling.
//   - Context Tree Weighting (Willems-Shtarkov-Tjalkens 1995) —
//     the universal predictor on tree-shaped sources.  See
//     `reality/info/lz` for the model-free codelength side.
//
// # Determinism and allocations
//
// Pure float64 + int, deterministic given a fixed iteration order.
// Zero non-stdlib deps (only `math` + `errors`).  Allocations
// bounded by O(n) for the NMLMultinomial Bernoulli-mass log-sum-
// exp accumulator; all other functions are O(1) allocation.
//
// # Numerical stability notes
//
// NMLMultinomial computes C(n, 2) via direct log-space accumulation
// of the Bernoulli-mass sum, then applies the Kontkanen-Myllymäki
// recurrence.  For very large n (n > 10^6) the binomial coefficients
// in the C(n, 2) sum become numerically unwieldy; the v2 path
// would route through the Mononen-Myllymäki 2008 sub-linear FFT-
// evaluable variant.  The v1 implementation is well-conditioned for
// the L12 consumer-scale (n < 10^5) verified by the test corpus.
//
// UniversalIntegerCodeLength terminates the iterated log when the
// inner log goes non-positive (log(log(...)) < 0); for n >= 2 the
// recursion runs at most a handful of iterations (log* converges
// extremely fast).
//
// SelectMDL guards against non-finite codelengths surfacing
// upstream numerical bugs.  Callers should treat
// ErrNonFiniteCodeLength as a routing signal toward the upstream
// model-fit diagnostic.
//
// # References
//
//   - Rissanen, J. (1978).  Modeling by shortest data description.
//     Automatica 14(5): 465-471.
//   - Rissanen, J. (1983).  A universal prior for integers and
//     estimation by minimum description length.  Annals of
//     Statistics 11(2): 416-431.
//   - Shtarkov, Y. M. (1987).  Universal sequential coding of single
//     messages.  Problems Inform. Transmission 23(3): 3-17.
//   - Kontkanen, P. & Myllymäki, P. (2007).  A linear-time algorithm
//     for computing the multinomial stochastic complexity.
//     Information Processing Letters 103(6): 227-233.
//   - Grünwald, P. D. (2007).  The Minimum Description Length
//     Principle.  MIT Press.
//   - Schwarz, G. (1978).  Estimating the dimension of a model.
//     Annals of Statistics 6(2): 461-464.
//   - Akaike, H. (1974).  A new look at the statistical model
//     identification.  IEEE Trans. Automatic Control 19(6): 716-723.
//
// Package layout:
//
//   - errors.go        — sentinel errors.
//   - universal_int.go — UniversalIntegerCodeLength + Bits variant.
//   - bernoulli.go     — NMLBernoulli + BernoulliCodeLength.
//   - nml.go           — NMLMultinomial Kontkanen-Myllymäki 2007
//                        linear-time recursion + computeCn2.
//   - codelength.go    — GaussianCodeLength + ModelCodeLength +
//                        BICShape + AICShape adapters.
//   - select.go        — SelectMDL + SelectMDLWithMargin.
//   - doc.go           — this file.
package mdl
