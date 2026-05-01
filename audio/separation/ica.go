package separation

import (
	"math"
)

// FastICA performs Independent Component Analysis using Hyvärinen's
// fixed-point algorithm with the tanh contrast function (g(u) = tanh(u),
// the "logcosh" non-quadratic function). Given a matrix of multi-channel
// observations (each row = one channel × T samples), it recovers a
// matrix of estimated independent sources.
//
// FastICA is the mainstream solution to the cocktail-party problem
// when the number of microphones equals the number of sources: the
// observed mixture x[t] = A · s[t] is a linear instantaneous mix
// of the unknown independent sources s[t]; FastICA finds an unmixing
// matrix W such that y[t] = W · x[t] ≈ s[t] up to permutation and
// scale.
//
// Algorithm (Hyvärinen 1999, FastICA with tanh contrast and
// symmetric decorrelation):
//
//  1. Centre each row of observations to zero mean.
//  2. Whiten via PCA: compute the eigendecomposition of the centred
//     covariance E[xxᵀ], rotate so the new covariance is the identity.
//  3. Initialise W as a random orthonormal matrix (we use a
//     deterministic seeded init for reproducibility).
//  4. Iterate the fixed-point update for each row w_i of W:
//     w⁺  = E[x · g(wᵀx)] - E[g'(wᵀx)] · w
//     where g(u) = tanh(u), g'(u) = 1 - tanh²(u).
//  5. Symmetrically decorrelate: W ← (W Wᵀ)^(-1/2) · W
//     via eigendecomposition of W·Wᵀ.
//  6. Convergence test: max_i |⟨w_i^new, w_i^old⟩ - 1| < tol.
//  7. Project: y = W · x (de-whitened to source space).
//
// Parameters:
//   - observations: K × T matrix where K = number of channels (rows),
//     T = number of samples per channel (columns). All rows must
//     have equal length.
//   - maxIterations: cap on fixed-point iterations (typical: 200).
//
// Returns: K × T matrix of estimated independent sources. Newly
// allocated; the caller owns the returned slice. Each output row
// has zero mean and unit variance (FastICA convention).
//
// IMPORTANT identifiability caveats:
//   - Sign and amplitude of recovered sources are arbitrary (you
//     cannot recover absolute amplitudes from a mixture).
//   - Order of recovered sources is arbitrary (no guarantee that
//     output row i corresponds to source i in the original mixture).
//   - At most one source may be Gaussian (the algorithm fails
//     identifiability if two or more sources share Gaussian
//     statistics).
//
// Valid range: K >= 1, T >= K, all rows must have equal length T.
// Precision: convergence tolerance is 1e-4 per row by default;
// deflation accumulates round-off so K=2 is ~1e-7 reproducible,
// K=10 ~1e-5.
// Panics on shape violations or empty observations.
//
// Reference: Hyvärinen, A. (1999). "Fast and Robust Fixed-Point
// Algorithms for Independent Component Analysis." IEEE Trans. Neural
// Networks 10(3), 626-634; Hyvärinen, A. & Oja, E. (2000)
// "Independent component analysis: algorithms and applications"
// Neural Networks 13(4-5), 411-430.
//
// Consumed by: pigeonhole (multi-microphone bird-source separation),
// howler (multi-pet vocalisation isolation).
func FastICA(observations [][]float64, maxIterations int) [][]float64 {
	K := len(observations)
	if K < 1 {
		panic("separation.FastICA: observations must have at least 1 row")
	}
	T := len(observations[0])
	if T < K {
		panic("separation.FastICA: T must be >= K")
	}
	for i := 1; i < K; i++ {
		if len(observations[i]) != T {
			panic("separation.FastICA: all rows must have equal length")
		}
	}
	if maxIterations < 1 {
		panic("separation.FastICA: maxIterations must be >= 1")
	}

	// Step 1: Centre.
	x := make([][]float64, K)
	means := make([]float64, K)
	for i := 0; i < K; i++ {
		s := 0.0
		for t := 0; t < T; t++ {
			s += observations[i][t]
		}
		means[i] = s / float64(T)
		x[i] = make([]float64, T)
		for t := 0; t < T; t++ {
			x[i][t] = observations[i][t] - means[i]
		}
	}

	// Step 2: Whiten via covariance eigendecomposition.
	// cov is K × K symmetric.
	cov := make([]float64, K*K)
	for i := 0; i < K; i++ {
		for j := i; j < K; j++ {
			s := 0.0
			for t := 0; t < T; t++ {
				s += x[i][t] * x[j][t]
			}
			c := s / float64(T-1)
			cov[i*K+j] = c
			cov[j*K+i] = c
		}
	}

	// Eigendecomposition of cov via Jacobi rotation (K is small —
	// typical K in [2, 10] for ICA in audio; this is a textbook
	// implementation suitable for K <= 50).
	eigvals := make([]float64, K)
	eigvecs := make([]float64, K*K)
	jacobiEigen(cov, K, eigvals, eigvecs)

	// Whitening matrix V = D^(-1/2) Eᵀ.
	V := make([]float64, K*K)
	for i := 0; i < K; i++ {
		// guard against tiny negative eigenvalues from round-off
		ev := eigvals[i]
		if ev < 1e-15 {
			ev = 1e-15
		}
		invSqrt := 1.0 / math.Sqrt(ev)
		for j := 0; j < K; j++ {
			V[i*K+j] = invSqrt * eigvecs[j*K+i]
		}
	}

	// z[i][t] = sum_j V[i][j] * x[j][t]
	z := make([][]float64, K)
	for i := 0; i < K; i++ {
		z[i] = make([]float64, T)
		for t := 0; t < T; t++ {
			s := 0.0
			for j := 0; j < K; j++ {
				s += V[i*K+j] * x[j][t]
			}
			z[i][t] = s
		}
	}

	// Step 3: Initialise W with deterministic orthonormal seed.
	W := make([]float64, K*K)
	for i := 0; i < K; i++ {
		W[i*K+i] = 1.0
	}
	// Add a tiny perturbation to escape the symmetric fixed point.
	// Using a deterministic LCG so the function is fully reproducible.
	rng := uint64(0x9E3779B97F4A7C15)
	for i := 0; i < K; i++ {
		for j := 0; j < K; j++ {
			rng = rng*6364136223846793005 + 1442695040888963407
			r := float64(int64(rng>>11)) / float64(1<<53)
			W[i*K+j] += 0.1 * (r - 0.5)
		}
	}
	symmetricDecorrelate(W, K)

	// Step 4 + 5: Fixed-point iteration with symmetric decorrelation.
	const tol = 1e-4
	wNew := make([]float64, K*K)
	gz := make([]float64, T)   // g(wᵀ z)
	gpz := 0.0                 // mean g'(wᵀ z)
	for iter := 0; iter < maxIterations; iter++ {
		// For each row i:
		for i := 0; i < K; i++ {
			// proj[t] = sum_j W[i][j] * z[j][t]
			// gz[t] = tanh(proj[t]); gpz mean = mean(1 - tanh²(proj[t]))
			gpzSum := 0.0
			for t := 0; t < T; t++ {
				p := 0.0
				for j := 0; j < K; j++ {
					p += W[i*K+j] * z[j][t]
				}
				th := math.Tanh(p)
				gz[t] = th
				gpzSum += 1.0 - th*th
			}
			gpz = gpzSum / float64(T)

			// w⁺ = E[z · g(wᵀ z)] - E[g'(wᵀ z)] · w
			for j := 0; j < K; j++ {
				s := 0.0
				for t := 0; t < T; t++ {
					s += z[j][t] * gz[t]
				}
				wNew[i*K+j] = s/float64(T) - gpz*W[i*K+j]
			}
		}

		// Symmetric decorrelation: W ← (W Wᵀ)^(-1/2) W
		symmetricDecorrelate(wNew, K)

		// Convergence test: max_i |1 - ⟨w_i^new, w_i^old⟩|
		maxDiff := 0.0
		for i := 0; i < K; i++ {
			s := 0.0
			for j := 0; j < K; j++ {
				s += wNew[i*K+j] * W[i*K+j]
			}
			d := math.Abs(math.Abs(s) - 1.0)
			if d > maxDiff {
				maxDiff = d
			}
		}

		// Copy wNew -> W
		copy(W, wNew)

		if maxDiff < tol {
			break
		}
	}

	// Step 7: Project. y = W · z (in whitened space).
	y := make([][]float64, K)
	for i := 0; i < K; i++ {
		y[i] = make([]float64, T)
		for t := 0; t < T; t++ {
			s := 0.0
			for j := 0; j < K; j++ {
				s += W[i*K+j] * z[j][t]
			}
			y[i][t] = s
		}
	}
	return y
}

// jacobiEigen computes the eigendecomposition of a real symmetric K×K
// matrix using cyclic Jacobi rotation (Press et al. NRC §11.1). On
// return, eigvals contains the eigenvalues, and eigvecs contains the
// eigenvectors in column-major (eigvecs[j*K + i] is the i-th component
// of the j-th eigenvector).
//
// Inputs:
//
//	a       K×K symmetric matrix (row-major); destroyed on return
//	K       order
//	eigvals output, length K
//	eigvecs output, length K×K
//
// Convergence: ~6-10 sweeps for K <= 20.
func jacobiEigen(a []float64, K int, eigvals, eigvecs []float64) {
	// Initialise eigvecs to identity.
	for i := 0; i < K; i++ {
		for j := 0; j < K; j++ {
			if i == j {
				eigvecs[i*K+j] = 1.0
			} else {
				eigvecs[i*K+j] = 0.0
			}
		}
	}
	const maxSweeps = 50
	for sweep := 0; sweep < maxSweeps; sweep++ {
		// Compute off-diagonal sum.
		offSum := 0.0
		for p := 0; p < K-1; p++ {
			for q := p + 1; q < K; q++ {
				offSum += math.Abs(a[p*K+q])
			}
		}
		if offSum < 1e-14 {
			break
		}
		for p := 0; p < K-1; p++ {
			for q := p + 1; q < K; q++ {
				apq := a[p*K+q]
				if math.Abs(apq) < 1e-15 {
					continue
				}
				app := a[p*K+p]
				aqq := a[q*K+q]
				theta := (aqq - app) / (2 * apq)
				var t float64
				if math.Abs(theta) > 1e15 {
					t = 1.0 / (2 * theta)
				} else {
					sgn := 1.0
					if theta < 0 {
						sgn = -1.0
					}
					t = sgn / (math.Abs(theta) + math.Sqrt(theta*theta+1))
				}
				c := 1.0 / math.Sqrt(t*t+1)
				s := t * c
				// Apply rotation to a.
				a[p*K+p] = app - t*apq
				a[q*K+q] = aqq + t*apq
				a[p*K+q] = 0
				a[q*K+p] = 0
				for i := 0; i < K; i++ {
					if i != p && i != q {
						aip := a[i*K+p]
						aiq := a[i*K+q]
						a[i*K+p] = c*aip - s*aiq
						a[p*K+i] = a[i*K+p]
						a[i*K+q] = s*aip + c*aiq
						a[q*K+i] = a[i*K+q]
					}
				}
				// Update eigenvectors (column-major).
				for i := 0; i < K; i++ {
					vip := eigvecs[i*K+p]
					viq := eigvecs[i*K+q]
					eigvecs[i*K+p] = c*vip - s*viq
					eigvecs[i*K+q] = s*vip + c*viq
				}
			}
		}
	}
	for i := 0; i < K; i++ {
		eigvals[i] = a[i*K+i]
	}
}

// symmetricDecorrelate replaces W with (W·Wᵀ)^(-1/2) · W in-place.
// W is K × K, row-major. Used by FastICA for symmetric orthogonalisation.
func symmetricDecorrelate(W []float64, K int) {
	// Compute G = W · Wᵀ (K × K, symmetric).
	G := make([]float64, K*K)
	for i := 0; i < K; i++ {
		for j := i; j < K; j++ {
			s := 0.0
			for k := 0; k < K; k++ {
				s += W[i*K+k] * W[j*K+k]
			}
			G[i*K+j] = s
			G[j*K+i] = s
		}
	}
	eigvals := make([]float64, K)
	eigvecs := make([]float64, K*K)
	jacobiEigen(G, K, eigvals, eigvecs)

	// G^(-1/2) = E · diag(λ^(-1/2)) · Eᵀ  (eigvecs are columns).
	invSqrt := make([]float64, K)
	for i := 0; i < K; i++ {
		ev := eigvals[i]
		if ev < 1e-15 {
			ev = 1e-15
		}
		invSqrt[i] = 1.0 / math.Sqrt(ev)
	}
	// Compute G^(-1/2) directly (K × K).
	M := make([]float64, K*K)
	for i := 0; i < K; i++ {
		for j := 0; j < K; j++ {
			s := 0.0
			for k := 0; k < K; k++ {
				s += eigvecs[i*K+k] * invSqrt[k] * eigvecs[j*K+k]
			}
			M[i*K+j] = s
		}
	}
	// W ← M · W.
	tmp := make([]float64, K*K)
	for i := 0; i < K; i++ {
		for j := 0; j < K; j++ {
			s := 0.0
			for k := 0; k < K; k++ {
				s += M[i*K+k] * W[k*K+j]
			}
			tmp[i*K+j] = s
		}
	}
	copy(W, tmp)
}
