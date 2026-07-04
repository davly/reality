package linalg

// Estimation-error hardening for covariance/correlation matrices: James-Stein
// mean shrinkage, Ledoit-Wolf optimal linear shrinkage (scaled-identity and
// constant-correlation targets), Marchenko-Pastur (RMT) noise bounds, and
// eigenvalue-clipping correlation cleaning.
//
// ─────────────────────────────────────────────────────────────────────────────
// THE RD-4A LESSON (encoded as an API contract, not a comment):
//
// Shrinkage that reduces estimation error on a *location* estimate (the mean)
// is NOT free to apply to a *dispersion* estimate (variance, covariance, tail).
// A live risk system (RubberDuck) once fed James-Stein-shrunk returns into
// stdDev / CVaR / Sortino / drawdown by reassigning the whole return series to
// its shrunk copy. Because shrinkage pulls every observation toward the grand
// mean, it COMPRESSES the sample variance and the tail — systematically
// UNDERSTATING the risk that drives live position sizing. The dashboard then
// reported less risk than the book actually carried.
//
// Rule: shrink means to CENTRE a VaR/return estimate; NEVER route shrunk data
// into the dispersion inputs. This package keeps the two operations separate by
// construction:
//   - JamesSteinShrink returns shrunk *means* only (a location estimator).
//   - LedoitWolf* / CleanCorrelation operate on the *dispersion* matrix directly
//     and never touch the mean, so there is no path to leak a shrunk mean into
//     a variance. Callers must keep raw returns for every dispersion/tail
//     measure downstream.
//
// References:
//   - James, W. & Stein, C. (1961). "Estimation with quadratic loss."
//     Proc. 4th Berkeley Symp. Math. Statist. Prob., Vol. 1, 361-379.
//   - Efron, B. & Morris, C. (1977). "Stein's paradox in statistics."
//     Scientific American 236(5), 119-127.
//   - Jorion, P. (1986). "Bayes-Stein estimation for portfolio analysis."
//     J. Financial and Quantitative Analysis 21(3), 279-292.
//   - Ledoit, O. & Wolf, M. (2004). "A well-conditioned estimator for
//     large-dimensional covariance matrices." J. Multivariate Analysis 88(2),
//     365-411. (scaled-identity target)
//   - Ledoit, O. & Wolf, M. (2003/2004). "Honey, I shrunk the sample covariance
//     matrix." J. Portfolio Management 30(4), 110-119; and "Improved estimation
//     of the covariance matrix of stock returns with an application to portfolio
//     selection." J. Empirical Finance 10(5), 603-621. (constant-correlation
//     target)
//   - Marchenko, V.A. & Pastur, L.A. (1967). "Distribution of eigenvalues for
//     some sets of random matrices." Math. USSR-Sbornik 1(4), 457-483.
//   - Laloux, L., Cizeau, P., Bouchaud, J-P. & Potters, M. (2000). "Random
//     matrix theory and financial correlations." Int. J. Theor. Appl. Finance
//     3(3), 391-397. (eigenvalue-clipping correlation cleaning)

import "math"

// JamesSteinShrink applies the positive-part James-Stein estimator to a vector
// of p sample means, shrinking each toward the grand mean. It is a LOCATION
// estimator: the result is intended to CENTRE downstream estimates (e.g. a VaR
// mean), never to replace the data feeding a dispersion/tail measure.
//
// For p >= 3 the estimator provably dominates the raw sample mean in total
// squared error (Stein's paradox).
//
// means    is the p sample means (one per parameter), length p, not modified.
// variance is the KNOWN or EXTERNALLY-ESTIMATED variance of each mean (e.g. a
//
//	pooled within-group variance). It must NOT be estimated from `means`
//	itself: doing so makes the shrinkage factor a deterministic function
//	of p alone and defeats the estimator (this is the documented F-JS-1
//	failure mode in the consumer's review history).
//
// Returns (shrunk, c):
//
//	shrunk is a freshly-allocated length-p slice of shrunk means.
//	c      is the positive-part shrinkage factor in [0, 1] that was applied:
//	       theta_i = xbar + (1 - c) * (means_i - xbar).
//
// Formula: c = (p - 2) * variance / S, clamped to [0, 1], where
// S = sum_i (means_i - xbar)^2 and xbar is the grand mean.
//
// Edge cases: for p < 3 the estimator is inadmissible and the raw means are
// returned unchanged with c = 0. If all means are identical (S ~ 0) the grand
// mean is returned with c = 1 (full but vacuous shrinkage). A non-finite input
// (NaN/Inf in means or variance) returns a copy of means unchanged with c = 0.
//
// Precision: exact algebra in float64; no iterative error. Reference:
// James & Stein (1961); positive-part variant per Efron & Morris (1977).
func JamesSteinShrink(means []float64, variance float64) ([]float64, float64) {
	p := len(means)
	shrunk := make([]float64, p)
	copy(shrunk, means)

	if p < 3 {
		return shrunk, 0
	}

	// Reject non-finite inputs rather than propagate NaN into a risk path.
	if math.IsNaN(variance) || math.IsInf(variance, 0) {
		return shrunk, 0
	}
	for i := 0; i < p; i++ {
		if math.IsNaN(means[i]) || math.IsInf(means[i], 0) {
			return shrunk, 0
		}
	}

	// Grand mean.
	var xbar float64
	for i := 0; i < p; i++ {
		xbar += means[i]
	}
	xbar /= float64(p)

	// Sum of squared deviations from the grand mean.
	var s float64
	for i := 0; i < p; i++ {
		d := means[i] - xbar
		s += d * d
	}

	// Degenerate: all means identical => collapse to the grand mean.
	if s < 1e-300 {
		for i := 0; i < p; i++ {
			shrunk[i] = xbar
		}
		return shrunk, 1
	}

	// Positive-part shrinkage factor.
	c := float64(p-2) * variance / s
	if c < 0 {
		c = 0
	} else if c > 1 {
		c = 1
	}

	for i := 0; i < p; i++ {
		shrunk[i] = xbar + (1-c)*(means[i]-xbar)
	}
	return shrunk, c
}

// LedoitWolfShrinkageIdentity computes the Ledoit-Wolf (2004) optimal linear
// shrinkage of the sample covariance matrix toward a scaled-identity target
// F = m*I, where m is the average sample variance (trace(S)/N).
//
// This is a DISPERSION estimator. It consumes the return series directly and
// never touches the mean, so it cannot leak a shrunk mean into a variance
// (see the RD-4A lesson in the package doc).
//
// x is the T x N return series stored row-major (x[t*N + i] = observation t of
// variable i), not modified. T is the number of observations (rows), N the
// number of variables (columns).
//
// Returns (sigma, shrinkage):
//
//	sigma      is a freshly-allocated N*N row-major shrunk covariance matrix:
//	           sigma = shrinkage*m*I + (1-shrinkage)*S.
//	shrinkage  is the optimal intensity b^2/d^2 in [0, 1] toward the identity
//	           target.
//
// The sample covariance S uses the 1/T (MLE) normalisation, matching the
// published estimator. The optimal intensity is identifiable only from the raw
// sample (it depends on fourth moments), which is why this takes returns rather
// than a pre-formed covariance matrix.
//
// Definitions with inner product <A,B> = trace(A B^T)/N (Ledoit-Wolf 2004):
//
//	m    = <S, I> = trace(S)/N
//	d^2  = ||S - m I||^2
//	bbar^2 = (1/T^2) sum_t ||y_t y_t^T - S||^2   (y_t = centred observation t)
//	b^2  = min(bbar^2, d^2);  a^2 = d^2 - b^2
//	sigma = (b^2/d^2) m I + (a^2/d^2) S
//
// Edge cases: T < 2 or N < 1 panics (no covariance is defined). If d^2 ~ 0 (S
// already a scaled identity) the intensity is 0 and S is returned unchanged.
//
// Precision: ~1e-12 on the intensity for well-conditioned inputs; matrix
// entries accumulate to ~1e-9. Reference: Ledoit & Wolf (2004), JMVA 88(2).
func LedoitWolfShrinkageIdentity(x []float64, T, N int) ([]float64, float64) {
	if T < 2 {
		panic("linalg.LedoitWolfShrinkageIdentity: T < 2")
	}
	if N < 1 {
		panic("linalg.LedoitWolfShrinkageIdentity: N < 1")
	}
	if len(x) != T*N {
		panic("linalg.LedoitWolfShrinkageIdentity: len(x) != T*N")
	}

	y := centeredCopy(x, T, N)
	S := sampleCovMLE(y, T, N) // 1/T normalisation

	Tf := float64(T)
	Nf := float64(N)

	// m = trace(S)/N.
	var m float64
	for i := 0; i < N; i++ {
		m += S[i*N+i]
	}
	m /= Nf

	// d^2 = ||S - m I||^2, with ||A||^2 = trace(A A^T)/N = (1/N) sum_ij A_ij^2.
	var d2 float64
	for i := 0; i < N; i++ {
		for j := 0; j < N; j++ {
			a := S[i*N+j]
			if i == j {
				a -= m
			}
			d2 += a * a
		}
	}
	d2 /= Nf

	// bbar^2 = (1/T^2) sum_t (1/N) sum_ij (y_ti y_tj - S_ij)^2.
	var bbar2 float64
	for t := 0; t < T; t++ {
		var acc float64
		for i := 0; i < N; i++ {
			yti := y[t*N+i]
			for j := 0; j < N; j++ {
				e := yti*y[t*N+j] - S[i*N+j]
				acc += e * e
			}
		}
		bbar2 += acc / Nf
	}
	bbar2 /= Tf * Tf

	sigma := make([]float64, N*N)
	if d2 <= 0 {
		copy(sigma, S)
		return sigma, 0
	}

	b2 := bbar2
	if b2 > d2 {
		b2 = d2
	}
	a2 := d2 - b2
	shrinkage := b2 / d2 // intensity toward the identity target

	// sigma = shrinkage*m*I + (a2/d2)*S.
	rawW := a2 / d2
	for i := 0; i < N; i++ {
		for j := 0; j < N; j++ {
			v := rawW * S[i*N+j]
			if i == j {
				v += shrinkage * m
			}
			sigma[i*N+j] = v
		}
	}
	return sigma, shrinkage
}

// LedoitWolfShrinkageConstantCorr computes the Ledoit-Wolf (2003) optimal linear
// shrinkage of the sample covariance toward a constant-correlation target — the
// estimator designed for financial return covariances. Sample variances are
// kept on the diagonal; every off-diagonal correlation is pulled toward the
// average pairwise sample correlation rbar.
//
// This is a DISPERSION estimator and never touches the mean (RD-4A lesson).
//
// x is the T x N return series row-major (not modified); T rows, N columns.
//
// Returns (sigma, shrinkage):
//
//	sigma     is the N*N row-major shrunk covariance = shrinkage*F + (1-shrinkage)*S,
//	          where F is the constant-correlation target.
//	shrinkage is the optimal intensity delta = max(0, min(kappa/T, 1)).
//
// The sample covariance S uses 1/T. Target: F_ii = S_ii, and for i != j
// F_ij = rbar * sqrt(S_ii S_jj), where rbar is the mean of the off-diagonal
// sample correlations. Optimal intensity per Ledoit-Wolf (2003) Appendix A:
//
//	pi    = sum_ij (1/T) sum_t (y_ti y_tj - S_ij)^2
//	rho   = sum_i pi_ii + sum_{i!=j} (rbar/2)[ sqrt(S_jj/S_ii) th_ii,ij
//	                                         + sqrt(S_ii/S_jj) th_jj,ij ]
//	        th_kk,ij = (1/T) sum_t (y_tk^2 - S_kk)(y_ti y_tj - S_ij)
//	gamma = sum_ij (F_ij - S_ij)^2
//	kappa = (pi - rho) / gamma;  delta = clamp(kappa/T, 0, 1)
//
// Edge cases: T < 2 or N < 1 panics. N == 1 has no off-diagonal to shrink, so
// the sample variance is returned with intensity 0. If gamma ~ 0 (S already has
// constant correlation) the intensity is 0 and S is returned unchanged. Any
// zero sample variance makes correlations undefined and yields intensity 0.
//
// Precision: ~1e-12 intensity; matrix entries ~1e-9. Reference: Ledoit & Wolf
// (2003), J. Empirical Finance 10(5); (2004) JPM 30(4).
func LedoitWolfShrinkageConstantCorr(x []float64, T, N int) ([]float64, float64) {
	if T < 2 {
		panic("linalg.LedoitWolfShrinkageConstantCorr: T < 2")
	}
	if N < 1 {
		panic("linalg.LedoitWolfShrinkageConstantCorr: N < 1")
	}
	if len(x) != T*N {
		panic("linalg.LedoitWolfShrinkageConstantCorr: len(x) != T*N")
	}

	y := centeredCopy(x, T, N)
	S := sampleCovMLE(y, T, N)
	Tf := float64(T)

	sigma := make([]float64, N*N)

	// Standard deviations from the diagonal.
	sd := make([]float64, N)
	anyZeroVar := false
	for i := 0; i < N; i++ {
		v := S[i*N+i]
		if v <= 0 {
			anyZeroVar = true
			sd[i] = 0
		} else {
			sd[i] = math.Sqrt(v)
		}
	}

	// N == 1 or a degenerate variance: nothing to shrink toward.
	if N == 1 || anyZeroVar {
		copy(sigma, S)
		return sigma, 0
	}

	// Average off-diagonal sample correlation rbar.
	var rsum float64
	pairs := 0
	for i := 0; i < N; i++ {
		for j := i + 1; j < N; j++ {
			rsum += S[i*N+j] / (sd[i] * sd[j])
			pairs++
		}
	}
	rbar := rsum / float64(pairs)

	// Constant-correlation target F.
	F := make([]float64, N*N)
	for i := 0; i < N; i++ {
		F[i*N+i] = S[i*N+i]
		for j := i + 1; j < N; j++ {
			f := rbar * sd[i] * sd[j]
			F[i*N+j] = f
			F[j*N+i] = f
		}
	}

	// pi_ij = (1/T) sum_t (y_ti y_tj - S_ij)^2 ; pi = sum_ij pi_ij.
	// Precompute pi_ij matrix (needed again for rho diagonal).
	piMat := make([]float64, N*N)
	for i := 0; i < N; i++ {
		for j := 0; j < N; j++ {
			var acc float64
			sij := S[i*N+j]
			for t := 0; t < T; t++ {
				e := y[t*N+i]*y[t*N+j] - sij
				acc += e * e
			}
			piMat[i*N+j] = acc / Tf
		}
	}
	var pi float64
	for k := 0; k < N*N; k++ {
		pi += piMat[k]
	}

	// rho = sum_i pi_ii + sum_{i!=j} (rbar/2)[ sqrt(S_jj/S_ii) th_ii,ij
	//                                        + sqrt(S_ii/S_jj) th_jj,ij ]
	// th_kk,ij = (1/T) sum_t (y_tk^2 - S_kk)(y_ti y_tj - S_ij)
	var rho float64
	for i := 0; i < N; i++ {
		rho += piMat[i*N+i]
	}
	for i := 0; i < N; i++ {
		for j := 0; j < N; j++ {
			if i == j {
				continue
			}
			sij := S[i*N+j]
			sii := S[i*N+i]
			sjj := S[j*N+j]
			var thII, thJJ float64
			for t := 0; t < T; t++ {
				cross := y[t*N+i]*y[t*N+j] - sij
				thII += (y[t*N+i]*y[t*N+i] - sii) * cross
				thJJ += (y[t*N+j]*y[t*N+j] - sjj) * cross
			}
			thII /= Tf
			thJJ /= Tf
			rho += (rbar / 2) * (sd[j]/sd[i]*thII + sd[i]/sd[j]*thJJ)
		}
	}

	// gamma = ||F - S||_F^2.
	var gamma float64
	for k := 0; k < N*N; k++ {
		e := F[k] - S[k]
		gamma += e * e
	}

	if gamma <= 0 {
		copy(sigma, S)
		return sigma, 0
	}

	kappa := (pi - rho) / gamma
	delta := kappa / Tf
	if delta < 0 {
		delta = 0
	} else if delta > 1 {
		delta = 1
	}

	for k := 0; k < N*N; k++ {
		sigma[k] = delta*F[k] + (1-delta)*S[k]
	}
	return sigma, delta
}

// MarchenkoPasturBounds returns the theoretical support edges [lambdaMin,
// lambdaMax] of the Marchenko-Pastur distribution for the eigenvalues of a
// sample correlation/covariance matrix built from pure unit-variance noise, at
// aspect ratio q = N/T (variables / observations).
//
// Eigenvalues falling inside [lambdaMin, lambdaMax] are statistically
// indistinguishable from noise; those above lambdaMax carry signal.
//
// For unit variance the edges are lambda_+- = (1 +- sqrt(q))^2. This is the
// sigma^2 = 1 form appropriate for a correlation matrix (unit diagonal). For a
// covariance with average variance sigma^2, multiply both bounds by sigma^2.
//
// q must satisfy q > 0. For q > 1 (more variables than observations) the bulk
// edges are still given by the same formula, with N-T exact zero eigenvalues
// outside the bulk; lambdaMin is then the lower bulk edge, not the matrix
// minimum. Panics if q <= 0.
//
// Precision: exact to machine epsilon on sqrt. Reference: Marchenko & Pastur
// (1967).
func MarchenkoPasturBounds(q float64) (lambdaMin, lambdaMax float64) {
	if q <= 0 || math.IsNaN(q) || math.IsInf(q, 0) {
		panic("linalg.MarchenkoPasturBounds: q must be finite and > 0")
	}
	sq := math.Sqrt(q)
	lambdaMin = (1 - sq) * (1 - sq)
	lambdaMax = (1 + sq) * (1 + sq)
	return lambdaMin, lambdaMax
}

// CleanCorrelation performs random-matrix-theory eigenvalue clipping on a sample
// correlation matrix: eigenvalues inside the Marchenko-Pastur noise bulk are
// replaced by their common average (preserving the trace of the noise
// subspace), signal eigenvalues above the bulk are kept, the matrix is
// reconstructed, and its diagonal is renormalised back to unit correlation.
//
// This is a DISPERSION operation on the correlation matrix itself; it never
// touches means (RD-4A lesson).
//
// corr is the N x N sample correlation matrix row-major (unit diagonal assumed,
// symmetric; not modified). T is the number of observations used to form it, N
// the number of variables. The noise-bulk upper edge is
// lambdaMax = (1 + sqrt(N/T))^2.
//
// Returns a freshly-allocated N*N row-major cleaned correlation matrix with
// exact unit diagonal.
//
// Method (Laloux et al. 2000): symmetric eigendecomposition (Jacobi), clip
// eigenvalues <= lambdaMax to their mean, reconstruct V diag(lambda') V^T, then
// scale entry (i,j) by 1/sqrt(C_ii C_jj) and pin the diagonal to 1 to remove
// floating-point drift.
//
// Edge cases: N < 1 or T < 1 panics. N == 1 returns [1]. If every eigenvalue is
// noise the result is the average-eigenvalue matrix renormalised to the
// identity.
//
// Precision: eigendecomposition to ~1e-12; reconstruction accumulates to ~1e-9.
// Reference: Laloux, Cizeau, Bouchaud & Potters (2000); Marchenko & Pastur
// (1967).
func CleanCorrelation(corr []float64, T, N int) []float64 {
	if N < 1 {
		panic("linalg.CleanCorrelation: N < 1")
	}
	if T < 1 {
		panic("linalg.CleanCorrelation: T < 1")
	}
	if len(corr) != N*N {
		panic("linalg.CleanCorrelation: len(corr) != N*N")
	}

	out := make([]float64, N*N)
	if N == 1 {
		out[0] = 1
		return out
	}

	evals, evecs := jacobiEigenSymmetric(corr, N)

	q := float64(N) / float64(T)
	_, lambdaMax := MarchenkoPasturBounds(q)

	// Average the noise eigenvalues (<= lambdaMax).
	var noiseSum float64
	noiseCount := 0
	for i := 0; i < N; i++ {
		if evals[i] <= lambdaMax {
			noiseSum += evals[i]
			noiseCount++
		}
	}
	avgNoise := 0.0
	if noiseCount > 0 {
		avgNoise = noiseSum / float64(noiseCount)
	}

	cleaned := make([]float64, N)
	for i := 0; i < N; i++ {
		if evals[i] > lambdaMax {
			cleaned[i] = evals[i]
		} else {
			cleaned[i] = avgNoise
		}
	}

	// Reconstruct C = V diag(cleaned) V^T. evecs is row-major with column k the
	// k-th eigenvector: evecs[i*N + k].
	for i := 0; i < N; i++ {
		for j := i; j < N; j++ {
			var sum float64
			for k := 0; k < N; k++ {
				sum += evecs[i*N+k] * cleaned[k] * evecs[j*N+k]
			}
			out[i*N+j] = sum
			out[j*N+i] = sum
		}
	}

	// Renormalise to unit diagonal. Precompute scales to avoid double-division.
	scale := make([]float64, N)
	for i := 0; i < N; i++ {
		d := out[i*N+i]
		if d < 1e-300 {
			d = 1e-300
		}
		scale[i] = math.Sqrt(d)
	}
	for i := 0; i < N; i++ {
		for j := 0; j < N; j++ {
			out[i*N+j] /= scale[i] * scale[j]
		}
	}
	for i := 0; i < N; i++ {
		out[i*N+i] = 1
	}
	return out
}

// centeredCopy returns a fresh T*N row-major copy of x with each column's mean
// removed.
func centeredCopy(x []float64, T, N int) []float64 {
	y := make([]float64, T*N)
	copy(y, x)
	means := make([]float64, N)
	for t := 0; t < T; t++ {
		for i := 0; i < N; i++ {
			means[i] += x[t*N+i]
		}
	}
	Tf := float64(T)
	for i := 0; i < N; i++ {
		means[i] /= Tf
	}
	for t := 0; t < T; t++ {
		for i := 0; i < N; i++ {
			y[t*N+i] -= means[i]
		}
	}
	return y
}

// sampleCovMLE computes the N*N row-major sample covariance of centred data y
// (T*N row-major) using the 1/T (maximum-likelihood) normalisation, matching
// the Ledoit-Wolf published estimators.
func sampleCovMLE(y []float64, T, N int) []float64 {
	S := make([]float64, N*N)
	Tf := float64(T)
	for i := 0; i < N; i++ {
		for j := i; j < N; j++ {
			var sum float64
			for t := 0; t < T; t++ {
				sum += y[t*N+i] * y[t*N+j]
			}
			c := sum / Tf
			S[i*N+j] = c
			S[j*N+i] = c
		}
	}
	return S
}

// jacobiEigenSymmetric computes the full eigendecomposition of a real symmetric
// n x n matrix A (row-major, not modified) using the cyclic Jacobi rotation
// method.
//
// Returns (eigenvalues, eigenvectors):
//
//	eigenvalues  length n.
//	eigenvectors n*n row-major, orthonormal, with column k the eigenvector for
//	             eigenvalues[k]: eigenvectors[i*n + k].
//
// Eigenpairs are sorted by descending eigenvalue. The Jacobi method is
// backward-stable and returns a fully orthonormal eigenvector set for symmetric
// matrices, which is what correlation reconstruction requires.
//
// Reference: Golub & Van Loan, "Matrix Computations", 4th ed., Section 8.5;
// Press et al., "Numerical Recipes", jacobi routine.
func jacobiEigenSymmetric(A []float64, n int) (eigenvalues, eigenvectors []float64) {
	// Work on a mutable copy a; V accumulates the rotations.
	a := make([]float64, n*n)
	copy(a, A)
	V := make([]float64, n*n)
	for i := 0; i < n; i++ {
		V[i*n+i] = 1
	}

	if n == 1 {
		return []float64{a[0]}, V
	}

	const maxSweeps = 100
	for sweep := 0; sweep < maxSweeps; sweep++ {
		// Sum of off-diagonal magnitudes.
		off := 0.0
		for p := 0; p < n; p++ {
			for qi := p + 1; qi < n; qi++ {
				off += math.Abs(a[p*n+qi])
			}
		}
		if off == 0 {
			break
		}

		for p := 0; p < n; p++ {
			for qi := p + 1; qi < n; qi++ {
				apq := a[p*n+qi]
				if apq == 0 {
					continue
				}
				app := a[p*n+p]
				aqq := a[qi*n+qi]

				// Jacobi rotation angle: cot(2theta) = (aqq-app)/(2apq).
				phi := (aqq - app) / (2 * apq)
				var tang float64 // tan(theta)
				if phi >= 0 {
					tang = 1.0 / (phi + math.Sqrt(phi*phi+1))
				} else {
					tang = -1.0 / (-phi + math.Sqrt(phi*phi+1))
				}
				cos := 1.0 / math.Sqrt(tang*tang+1)
				sin := tang * cos

				// Rotate rows/cols p and qi of a.
				for k := 0; k < n; k++ {
					akp := a[k*n+p]
					akq := a[k*n+qi]
					a[k*n+p] = cos*akp - sin*akq
					a[k*n+qi] = sin*akp + cos*akq
				}
				for k := 0; k < n; k++ {
					apk := a[p*n+k]
					aqk := a[qi*n+k]
					a[p*n+k] = cos*apk - sin*aqk
					a[qi*n+k] = sin*apk + cos*aqk
				}
				// Force the annihilated off-diagonal to exact zero.
				a[p*n+qi] = 0
				a[qi*n+p] = 0

				// Accumulate the rotation into V.
				for k := 0; k < n; k++ {
					vkp := V[k*n+p]
					vkq := V[k*n+qi]
					V[k*n+p] = cos*vkp - sin*vkq
					V[k*n+qi] = sin*vkp + cos*vkq
				}
			}
		}
	}

	// Extract eigenvalues from the diagonal.
	eigenvalues = make([]float64, n)
	for i := 0; i < n; i++ {
		eigenvalues[i] = a[i*n+i]
	}

	// Sort eigenpairs by descending eigenvalue (insertion sort, columns of V).
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	for i := 1; i < n; i++ {
		key := order[i]
		kv := eigenvalues[key]
		j := i - 1
		for j >= 0 && eigenvalues[order[j]] < kv {
			order[j+1] = order[j]
			j--
		}
		order[j+1] = key
	}

	sortedVals := make([]float64, n)
	eigenvectors = make([]float64, n*n)
	for newCol, oldCol := range order {
		sortedVals[newCol] = eigenvalues[oldCol]
		for i := 0; i < n; i++ {
			eigenvectors[i*n+newCol] = V[i*n+oldCol]
		}
	}
	return sortedVals, eigenvectors
}
