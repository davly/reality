package portfolio

import (
	"math"
	"sort"

	"github.com/davly/reality/gametheory"
	"github.com/davly/reality/linalg"
)

// ---------------------------------------------------------------------------
// Internal helpers: dense [][]float64 <-> flat row-major, finite validation.
// linalg operates on flat row-major []float64 with an explicit dimension; we
// keep the public API in the ergonomic [][]float64 form that matches
// gametheory.KellyContinuousMulti and RubberDuck's covariance callers.
// ---------------------------------------------------------------------------

// flattenSquare validates that m is a finite n x n matrix and returns its
// row-major flattening. ok is false on nil, ragged, empty, or non-finite input.
func flattenSquare(m [][]float64) (flat []float64, n int, ok bool) {
	n = len(m)
	if n == 0 {
		return nil, 0, false
	}
	flat = make([]float64, n*n)
	for i := 0; i < n; i++ {
		if len(m[i]) != n {
			return nil, 0, false
		}
		for j := 0; j < n; j++ {
			v := m[i][j]
			if math.IsNaN(v) || math.IsInf(v, 0) {
				return nil, 0, false
			}
			flat[i*n+j] = v
		}
	}
	return flat, n, true
}

// flattenRect validates that m is a finite rows x cols matrix (rows = len(m),
// cols = the required column count) and returns its row-major flattening.
func flattenRect(m [][]float64, cols int) (flat []float64, rows int, ok bool) {
	rows = len(m)
	if rows == 0 || cols == 0 {
		return nil, 0, false
	}
	flat = make([]float64, rows*cols)
	for i := 0; i < rows; i++ {
		if len(m[i]) != cols {
			return nil, 0, false
		}
		for j := 0; j < cols; j++ {
			v := m[i][j]
			if math.IsNaN(v) || math.IsInf(v, 0) {
				return nil, 0, false
			}
			flat[i*cols+j] = v
		}
	}
	return flat, rows, true
}

// finiteVec reports whether every element of v is finite.
func finiteVec(v []float64) bool {
	for _, x := range v {
		if math.IsNaN(x) || math.IsInf(x, 0) {
			return false
		}
	}
	return true
}

// unflatten converts a row-major n x n flat slice back to [][]float64.
func unflatten(flat []float64, rows, cols int) [][]float64 {
	out := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		out[i] = make([]float64, cols)
		copy(out[i], flat[i*cols:(i+1)*cols])
	}
	return out
}

// ---------------------------------------------------------------------------
// ImpliedEquilibriumReturns — CAPM reverse optimisation.
// ---------------------------------------------------------------------------

// ImpliedEquilibriumReturns computes the CAPM-implied equilibrium excess-
// return vector (the Black-Litterman prior mean) from a market/benchmark
// weight vector w, the return covariance Sigma, and the world-average risk-
// aversion coefficient delta:
//
//	pi = delta * Sigma * w
//
// This is the "reverse optimisation" step: rather than forecasting returns and
// optimising to weights, it takes the observed market-capitalisation weights
// as the neutral optimal portfolio and backs out the returns that would make
// them optimal under mean-variance utility. It is the exact inverse of
// MeanVarianceWeights: MeanVarianceWeights(ImpliedEquilibriumReturns(w, Sigma,
// delta), Sigma, delta) == w for any nonsingular Sigma.
//
// Returns nil if w is empty, Sigma is not len(w) x len(w), any input is non-
// finite, or delta is non-finite. delta is permitted to be any finite value
// (a negative delta simply flips the sign of pi); the caller supplies the
// world risk-aversion (He-Litterman use delta = 2.5).
//
// Precision: ~15 significant digits (float64); a single matrix-vector product.
// Reference: He & Litterman (1999) Appendix B item 2 (Pi = delta * Sigma * w_eq).
func ImpliedEquilibriumReturns(w []float64, Sigma [][]float64, delta float64) []float64 {
	n := len(w)
	if n == 0 || !finiteVec(w) || math.IsNaN(delta) || math.IsInf(delta, 0) {
		return nil
	}
	sig, sn, ok := flattenSquare(Sigma)
	if !ok || sn != n {
		return nil
	}
	sw := make([]float64, n)
	linalg.MatVecMul(sig, n, n, w, sw)
	pi := make([]float64, n)
	for i := range sw {
		pi[i] = delta * sw[i]
	}
	return pi
}

// ---------------------------------------------------------------------------
// HeLittermanOmega — the canonical diagonal view-uncertainty matrix.
// ---------------------------------------------------------------------------

// HeLittermanOmega returns the canonical He-Litterman view-uncertainty matrix
//
//	Omega = diag( P (tau*Sigma) P' )
//
// i.e. a k x k diagonal matrix whose i-th diagonal entry is the prior variance
// of the i-th view portfolio, tau * P_i Sigma P_i'. This is THE convention
// this package pins to the published He-Litterman fixture, and the one under
// which the posterior mean is invariant to tau (see the package doc).
//
// P is the k x n view-pick matrix (row i = the portfolio expressing view i),
// Sigma is the n x n return covariance, tau > 0 is the prior scalar. Returns
// nil on a dimension mismatch, a non-finite input, or tau <= 0.
//
// Precision: ~15 significant digits (float64).
// Reference: He & Litterman (1999), footnote defining omega as the variance of
// the view portfolio; Appendix B.
func HeLittermanOmega(P [][]float64, Sigma [][]float64, tau float64) [][]float64 {
	sig, n, ok := flattenSquare(Sigma)
	if !ok {
		return nil
	}
	if tau <= 0 || math.IsNaN(tau) || math.IsInf(tau, 0) {
		return nil
	}
	pf, k, ok := flattenRect(P, n)
	if !ok {
		return nil
	}
	omega := make([][]float64, k)
	// omega_ii = tau * P_i Sigma P_i'
	tmp := make([]float64, n) // Sigma * P_i'
	for i := 0; i < k; i++ {
		row := pf[i*n : (i+1)*n]
		linalg.MatVecMul(sig, n, n, row, tmp)
		var v float64
		for j := 0; j < n; j++ {
			v += row[j] * tmp[j]
		}
		omega[i] = make([]float64, k)
		omega[i][i] = tau * v
	}
	return omega
}

// ---------------------------------------------------------------------------
// BlackLittermanPosterior — the composed posterior mean (master formula).
// ---------------------------------------------------------------------------

// BlackLittermanPosterior computes the Black-Litterman posterior expected-
// return vector by blending the equilibrium prior pi with K linear views
// (P mu = Q + noise, noise covariance Omega) under the He-Litterman /
// Satchell-Scowcroft master formula:
//
//	mu = [ (tau*Sigma)^-1 + P'Omega^-1 P ]^-1 [ (tau*Sigma)^-1 pi + P'Omega^-1 Q ]
//
// Dimensions are inferred: n = len(pi) = len(Q-independent), Sigma is n x n,
// P is k x n, Q is length k, Omega is k x k. When there are no views (P, Q,
// and Omega all empty), the posterior equals the prior and a copy of pi is
// returned (He & Litterman Appendix B: with no views the posterior matches the
// equilibrium).
//
// tau > 0 is the prior scalar. Under the canonical Omega = HeLittermanOmega(P,
// Sigma, tau) the returned mean is invariant to the value of tau (see package
// doc); tau still matters if the caller passes a fixed Omega unrelated to tau.
//
// Returns nil on any dimension mismatch, a non-finite input, tau <= 0, or a
// singular / near-singular Sigma, Omega, or precision matrix. The result is
// never clamped or re-normalised.
//
// Precision: ~15 significant digits (float64) for a well-conditioned system;
// the three inverses are LU with partial pivoting (linalg.Inverse).
// Reference: He & Litterman (1999) Appendix B item 4; Black & Litterman (1992).
func BlackLittermanPosterior(pi []float64, Sigma [][]float64, P [][]float64, Q []float64, Omega [][]float64, tau float64) []float64 {
	n := len(pi)
	if n == 0 || !finiteVec(pi) {
		return nil
	}
	if tau <= 0 || math.IsNaN(tau) || math.IsInf(tau, 0) {
		return nil
	}
	sig, sn, ok := flattenSquare(Sigma)
	if !ok || sn != n {
		return nil
	}

	k := len(Q)
	// No-views fast path: posterior mean equals the prior.
	if k == 0 && len(P) == 0 && len(Omega) == 0 {
		out := make([]float64, n)
		copy(out, pi)
		return out
	}
	if k == 0 || !finiteVec(Q) {
		return nil
	}

	pf, pRows, ok := flattenRect(P, n)
	if !ok || pRows != k {
		return nil
	}
	om, oN, ok := flattenSquare(Omega)
	if !ok || oN != k {
		return nil
	}

	// (tau*Sigma)^-1
	tauSig := make([]float64, n*n)
	for i := range sig {
		tauSig[i] = tau * sig[i]
	}
	invTauSig := make([]float64, n*n)
	if !linalg.Inverse(tauSig, n, invTauSig) {
		return nil
	}

	// Omega^-1
	invOmega := make([]float64, k*k)
	if !linalg.Inverse(om, k, invOmega) {
		return nil
	}

	// P' (n x k)
	pt := make([]float64, n*k)
	linalg.MatTranspose(pf, k, n, pt)

	// P'Omega^-1  (n x k)
	ptInvOmega := make([]float64, n*k)
	linalg.MatMul(pt, n, k, invOmega, k, ptInvOmega)

	// P'Omega^-1 P  (n x n)
	ptInvOmegaP := make([]float64, n*n)
	linalg.MatMul(ptInvOmega, n, k, pf, n, ptInvOmegaP)

	// A = (tau*Sigma)^-1 + P'Omega^-1 P
	a := make([]float64, n*n)
	for i := range a {
		a[i] = invTauSig[i] + ptInvOmegaP[i]
	}
	aInv := make([]float64, n*n)
	if !linalg.Inverse(a, n, aInv) {
		return nil
	}

	// b = (tau*Sigma)^-1 pi + P'Omega^-1 Q
	term1 := make([]float64, n)
	linalg.MatVecMul(invTauSig, n, n, pi, term1)
	term2 := make([]float64, n)
	linalg.MatVecMul(ptInvOmega, n, k, Q, term2)
	b := make([]float64, n)
	for i := 0; i < n; i++ {
		b[i] = term1[i] + term2[i]
	}

	// mu = A^-1 b
	mu := make([]float64, n)
	linalg.MatVecMul(aInv, n, n, b, mu)
	return mu
}

// BlackLittermanPosteriorCovariance returns the Black-Litterman posterior
// PARAMETER covariance
//
//	M = [ (tau*Sigma)^-1 + P'Omega^-1 P ]^-1
//
// the covariance of the estimate of the mean (NOT the return covariance). The
// full posterior return covariance used for optimisation is Sigma + M; this
// function returns the M term so callers can form it. With no views it reduces
// to tau*Sigma.
//
// Same dimension / finiteness / singularity contract as BlackLittermanPosterior;
// returns nil on any ill-posed input.
//
// Precision: ~15 significant digits (float64).
// Reference: He & Litterman (1999) Appendix B; Satchell & Scowcroft (2000).
func BlackLittermanPosteriorCovariance(Sigma [][]float64, P [][]float64, Omega [][]float64, tau float64) [][]float64 {
	sig, n, ok := flattenSquare(Sigma)
	if !ok {
		return nil
	}
	if tau <= 0 || math.IsNaN(tau) || math.IsInf(tau, 0) {
		return nil
	}

	tauSig := make([]float64, n*n)
	for i := range sig {
		tauSig[i] = tau * sig[i]
	}

	k := len(P)
	if k == 0 && len(Omega) == 0 {
		// No views: M = tau*Sigma.
		return unflatten(tauSig, n, n)
	}

	pf, pRows, ok := flattenRect(P, n)
	if !ok || pRows != k {
		return nil
	}
	om, oN, ok := flattenSquare(Omega)
	if !ok || oN != k {
		return nil
	}

	invTauSig := make([]float64, n*n)
	if !linalg.Inverse(tauSig, n, invTauSig) {
		return nil
	}
	invOmega := make([]float64, k*k)
	if !linalg.Inverse(om, k, invOmega) {
		return nil
	}
	pt := make([]float64, n*k)
	linalg.MatTranspose(pf, k, n, pt)
	ptInvOmega := make([]float64, n*k)
	linalg.MatMul(pt, n, k, invOmega, k, ptInvOmega)
	ptInvOmegaP := make([]float64, n*n)
	linalg.MatMul(ptInvOmega, n, k, pf, n, ptInvOmegaP)
	a := make([]float64, n*n)
	for i := range a {
		a[i] = invTauSig[i] + ptInvOmegaP[i]
	}
	m := make([]float64, n*n)
	if !linalg.Inverse(a, n, m) {
		return nil
	}
	return unflatten(m, n, n)
}

// ---------------------------------------------------------------------------
// Weight maps.
// ---------------------------------------------------------------------------

// MeanVarianceWeights computes the unconstrained Markowitz mean-variance
// optimal weights that maximise w'mu - (delta/2) w'Sigma w:
//
//	w* = (1/delta) Sigma^-1 mu
//
// The Sigma^-1 mu solve is delegated to the landed
// gametheory.KellyContinuousMulti (Gaussian elimination with partial
// pivoting) rather than re-implemented here; the mean-variance optimum is
// exactly the full continuous-Kelly allocation scaled by 1/delta.
//
// Returns nil if delta <= 0 / non-finite, mu is empty or non-finite, Sigma is
// not len(mu) x len(mu) or non-finite, or Sigma is singular. Weights are NOT
// normalised to sum to one and MAY be negative (short) or exceed one
// (levered); use MeanVarianceWeightsLongOnly for a fully-invested long-only
// portfolio.
//
// Precision: ~15 significant digits (float64) for a well-conditioned Sigma.
// Reference: Markowitz (1952); He & Litterman (1999) Appendix B item 5.
func MeanVarianceWeights(mu []float64, Sigma [][]float64, delta float64) []float64 {
	if len(mu) == 0 || !finiteVec(mu) {
		return nil
	}
	if delta <= 0 || math.IsNaN(delta) || math.IsInf(delta, 0) {
		return nil
	}
	if _, sn, ok := flattenSquare(Sigma); !ok || sn != len(mu) {
		return nil
	}
	raw := gametheory.KellyContinuousMulti(mu, Sigma) // Sigma^-1 mu
	if raw == nil {
		return nil
	}
	inv := 1.0 / delta
	for i := range raw {
		raw[i] *= inv
	}
	if !finiteVec(raw) {
		return nil
	}
	return raw
}

// ContinuousKellyWeights computes fractional continuous-Kelly portfolio
// weights for correlated assets:
//
//	w = fraction * Sigma^-1 mu
//
// where Sigma^-1 mu is the full (unit-fraction) continuous-Kelly allocation
// computed by the landed gametheory.KellyContinuousMulti, and fraction is the
// fractional-Kelly multiplier (full Kelly = 1; the common "quarter Kelly"
// convention = 0.25). This is a thin wrapper over the landed Wave-2 gametheory
// function — it does NOT re-implement the Sigma^-1 mu solve — with the single
// added step of applying the fraction to the whole weight vector.
//
// fraction is applied verbatim and is NOT clamped: a caller may pass a value
// above one (leverage) or a negative value (flip) if they mean to; the
// money-safety clamp is a caller policy, not a math fact. Returns nil if mu is
// empty / non-finite, Sigma is not len(mu) x len(mu) / non-finite, fraction is
// non-finite, or Sigma is singular.
//
// Precision: ~15 significant digits (float64) for a well-conditioned Sigma.
// Reference: Thorp (2006); MacLean, Thorp & Ziemba (2011). Base solve:
// gametheory.KellyContinuousMulti.
func ContinuousKellyWeights(mu []float64, Sigma [][]float64, fraction float64) []float64 {
	if len(mu) == 0 || !finiteVec(mu) {
		return nil
	}
	if math.IsNaN(fraction) || math.IsInf(fraction, 0) {
		return nil
	}
	if _, sn, ok := flattenSquare(Sigma); !ok || sn != len(mu) {
		return nil
	}
	raw := gametheory.KellyContinuousMulti(mu, Sigma)
	if raw == nil {
		return nil
	}
	for i := range raw {
		raw[i] *= fraction
	}
	if !finiteVec(raw) {
		return nil
	}
	return raw
}

// MeanVarianceWeightsLongOnly computes the mean-variance optimal weights and
// then Euclidean-projects them onto the fully-invested long-only simplex
// { w : sum w = 1, w >= 0 } via ProjectSimplex. This is the constrained
// counterpart of MeanVarianceWeights: the closest (in L2) fully-invested
// long-only portfolio to the unconstrained optimum.
//
// Note that the projection is the Euclidean projection of the unconstrained
// mean-variance solution, NOT the exact solution of the inequality-constrained
// QP; it is the standard cheap long-only heuristic. Returns nil whenever
// MeanVarianceWeights returns nil.
//
// Precision: the underlying solve is ~15 significant digits; the projection is
// exact (a sorted threshold search).
// Reference: Markowitz (1952) with a nonnegativity + budget constraint; the
// projection follows Duchi et al. (2008) / Held, Wolfe & Crowder (1974).
func MeanVarianceWeightsLongOnly(mu []float64, Sigma [][]float64, delta float64) []float64 {
	w := MeanVarianceWeights(mu, Sigma, delta)
	if w == nil {
		return nil
	}
	return ProjectSimplex(w)
}

// ---------------------------------------------------------------------------
// ProjectSimplex — Euclidean projection onto the probability simplex.
// ---------------------------------------------------------------------------

// ProjectSimplex returns the Euclidean projection of v onto the probability
// simplex { w : sum_i w_i = 1, w_i >= 0 }: the unique point on the simplex
// minimising ||w - v||_2. The algorithm sorts v descending, finds the largest
// index rho where the running threshold stays feasible, and shifts-and-clips:
//
//	w_i = max(v_i - theta, 0),  theta chosen so the result sums to 1.
//
// Returns nil for an empty or non-finite v. The result always sums to exactly
// 1 (up to float rounding) and is elementwise nonnegative.
//
// Precision: exact up to float64 rounding; O(n log n) from the sort.
// Reference: Held, Wolfe & Crowder (1974); Duchi, Shalev-Shwartz, Singer &
// Chandra (2008), "Efficient Projections onto the l1-Ball", Figure 1.
func ProjectSimplex(v []float64) []float64 {
	n := len(v)
	if n == 0 || !finiteVec(v) {
		return nil
	}
	// Sort a copy descending.
	u := make([]float64, n)
	copy(u, v)
	sort.Sort(sort.Reverse(sort.Float64Slice(u)))

	// Find rho = max { j : u_j - (cumsum_j - 1)/(j+1) > 0 }.
	cumsum := 0.0
	rho := -1
	theta := 0.0
	for j := 0; j < n; j++ {
		cumsum += u[j]
		t := (cumsum - 1.0) / float64(j+1)
		if u[j]-t > 0 {
			rho = j
			theta = t
		}
	}
	if rho < 0 {
		// Degenerate (all-equal pathological case); fall back to uniform.
		w := make([]float64, n)
		for i := range w {
			w[i] = 1.0 / float64(n)
		}
		return w
	}

	w := make([]float64, n)
	for i := 0; i < n; i++ {
		wi := v[i] - theta
		if wi < 0 {
			wi = 0
		}
		w[i] = wi
	}
	return w
}
