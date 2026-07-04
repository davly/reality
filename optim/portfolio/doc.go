// Package portfolio implements the composed Black-Litterman posterior and
// its neighbouring mean-variance / continuous-Kelly weight maps over a
// covariance matrix — the money-proximal step that turns a covariance and a
// set of views into actual portfolio WEIGHTS.
//
// # Why this exists in Reality
//
// Reality already ships every linear-algebra primitive Black-Litterman needs
// (linalg.MatMul, linalg.Inverse, linalg.CholeskyDecompose) and the single-
// asset / multi-asset continuous Kelly maps (gametheory.KellyContinuous,
// gametheory.KellyContinuousMulti). What it did NOT have is the *composed*
// Black-Litterman posterior — and the composition is exactly the missing
// math, because Black-Litterman's real-world failure mode is not a wrong
// matrix multiply, it is silent divergence in the tau / Omega convention used
// to blend the CAPM prior with the views. Two shops both "run Black-Litterman"
// and get different weights because one uses Omega = diag(P (tau*Sigma) P')
// and the other rolls a bespoke per-view uncertainty and an abs-sum weight
// normalisation. Encoding the canonical He-Litterman (1999) parameterisation
// and pinning it to that paper's published worked example turns "which
// Black-Litterman are we running" into a testable fact.
//
// RubberDuck ships a live IBacktestStrategy
// (RubberDuck.Brain/Services/Strategies/BlackLittermanTacticalStrategy.cs)
// that calls a hand-rolled BlackLitterman.Compute(marketWeights, cov,
// riskAversion, views, tau) once per rebalance cycle
// (RubberDuck.Core/Analysis/BlackLitterman.cs), composing pi = delta*Sigma*w,
// the view matrix P, the diagonal uncertainty Omega and the posterior by ~20
// MatMul + 2 Inverse calls — and its tests assert only directional / property
// facts (sums-to-one, in-range), with zero published-fixture grounding. This
// package is the substrate-neutral reference those C# functions are pinned
// against by cross-language golden vectors, so a refactor that flips the
// tau/Omega convention breaks a golden test instead of silently re-weighting
// live capital.
//
// # The tau / Omega convention (the load-bearing choice)
//
// The posterior MEAN under this package's canonical He-Litterman convention
//
//	Omega = diag( P (tau*Sigma) P' )    // HeLittermanOmega
//
// is INVARIANT to the value of tau: substituting Omega = tau*P*Sigma*P' into
// the master formula lets tau factor out of both the precision sum and the
// right-hand side, so
//
//	mu = [ (tau*Sigma)^-1 + P'Omega^-1 P ]^-1 [ (tau*Sigma)^-1 pi + P'Omega^-1 Q ]
//	   = [ Sigma^-1 + P'(P Sigma P')^-1 P ]^-1 [ Sigma^-1 pi + P'(P Sigma P')^-1 Q ]
//
// This is a real robustness property, not an accident, and this package's
// tests pin it (the posterior mean is identical at tau = 0.01, 0.05, 0.5).
// The posterior COVARIANCE M = [ (tau*Sigma)^-1 + P'Omega^-1 P ]^-1 does still
// scale with tau. Callers who want a different view-uncertainty model simply
// pass their own Omega to BlackLittermanPosterior; HeLittermanOmega is the
// convention this package pins to the published fixture.
//
// # API surface
//
//   - ImpliedEquilibriumReturns(w, Sigma, delta)      — reverse optimisation
//     pi = delta * Sigma * w (the CAPM-implied equilibrium excess returns).
//   - HeLittermanOmega(P, Sigma, tau)                 — the canonical diagonal
//     view-uncertainty Omega = diag(P (tau*Sigma) P').
//   - BlackLittermanPosterior(pi, Sigma, P, Q, Omega, tau) — the composed
//     posterior mean (He-Litterman / Satchell-Scowcroft master formula).
//   - BlackLittermanPosteriorCovariance(Sigma, P, Omega, tau) — the posterior
//     parameter covariance M = [ (tau*Sigma)^-1 + P'Omega^-1 P ]^-1.
//   - MeanVarianceWeights(mu, Sigma, delta)           — unconstrained Markowitz
//     optimum w = (1/delta) Sigma^-1 mu.
//   - MeanVarianceWeightsLongOnly(mu, Sigma, delta)   — the same, Euclidean-
//     projected onto the fully-invested long-only simplex.
//   - ContinuousKellyWeights(mu, Sigma, fraction)     — fractional continuous
//     Kelly weights = fraction * Sigma^-1 mu (thin wrapper over the landed
//     gametheory.KellyContinuousMulti; NOT a re-implementation).
//   - ProjectSimplex(v)                               — Euclidean projection
//     onto the probability simplex { w : sum w = 1, w >= 0 }.
//
// # Precision & honesty
//
//   - All matrices are dense [][]float64 in row-major (outer index = row),
//     matching the sibling gametheory.KellyContinuousMulti covariance form.
//   - The composed posterior is ~15 significant digits (float64) for a well-
//     conditioned Sigma / Omega; the underlying solves are LU with partial
//     pivoting (linalg.Inverse) and Gaussian elimination (gametheory).
//   - Every entry point returns nil on a dimension mismatch, a non-finite
//     input (NaN / +-Inf), a non-positive delta where one is required, or a
//     singular / near-singular Sigma or Omega. Nothing is silently clamped or
//     re-normalised; the caller decides what to do with a nil (ill-posed)
//     result. This is deliberately unlike RubberDuck.Core's abs-sum weight
//     normalisation, which is a convention this package does NOT adopt.
//
// # Cross-substrate golden grounding
//
// testdata/he_litterman_1999.json encodes the He-Litterman (1999) seven-
// country worked example verbatim (Table 1 volatilities / market-cap weights
// / equilibrium returns, Table 2 correlations, delta = 2.5, tau = 0.05, and
// the "Germany outperforms the rest of Europe by 5%" view). The tests assert
// (a) ImpliedEquilibriumReturns reproduces the paper's published equilibrium
// returns (3.9, 6.9, 8.4, 9.0, 4.3, 6.8, 7.6 %) to rounding, (b) the reverse
// map recovers the published market-cap weights, (c) BlackLittermanPosterior
// reproduces the published combined returns, and (d) the paper's central
// theorem — the optimal deviation from equilibrium is EXACTLY proportional to
// the view portfolio (nonzero only in Germany / France / UK, in the market-cap
// ratio -5.2/17.6 : -12.4/17.6). Ports (Python / C# / C++) validate against
// the same JSON.
//
// References:
//   - He, G. & Litterman, R. (1999) "The Intuition Behind Black-Litterman
//     Model Portfolios", Goldman Sachs Investment Management Research (the
//     source of every golden vector here; Appendix A Tables 1-2, Appendix B
//     master formula item 4, theorem item 5).
//   - Black, F. & Litterman, R. (1992) "Global Portfolio Optimization",
//     Financial Analysts Journal 48(5):28-43.
//   - Satchell, S. & Scowcroft, A. (2000) "A demystification of the Black-
//     Litterman model", Journal of Asset Management 1(2):138-150.
//   - Markowitz, H. (1952) "Portfolio Selection", Journal of Finance 7(1):77-91.
//   - Duchi, J., Shalev-Shwartz, S., Singer, Y. & Chandra, T. (2008)
//     "Efficient Projections onto the l1-Ball for Learning in High
//     Dimensions", ICML (the ProjectSimplex algorithm; see also Held, Wolfe &
//     Crowder 1974).
package portfolio
