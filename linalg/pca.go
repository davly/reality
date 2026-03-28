package linalg

import "math"

// PCA performs Principal Component Analysis on a dataset via covariance matrix
// eigendecomposition. It computes the top nComponents principal components.
//
// data is nSamples x nFeatures stored row-major in a flat slice.
// nSamples is the number of observations. nFeatures is the dimensionality.
// nComponents is the number of principal components to extract (1 <= nComponents <= nFeatures).
//
// components is nComponents x nFeatures (pre-allocated): each row is a principal
// component vector (unit eigenvector of the covariance matrix), ordered by
// decreasing explained variance.
//
// explained is length nComponents (pre-allocated): the fraction of total variance
// explained by each component.
//
// Returns the cumulative fraction of total variance explained by the selected
// components (sum of explained[0..nComponents-1]).
//
// Algorithm:
//   1. Center the data (subtract column means).
//   2. Compute the covariance matrix (nFeatures x nFeatures).
//   3. Compute eigenvalues via QR algorithm.
//   4. For each eigenvalue, recover the eigenvector via inverse iteration.
//   5. Sort by decreasing eigenvalue; fill components and explained.
//
// Valid input range: nSamples >= 2, nFeatures >= 1, 1 <= nComponents <= nFeatures.
// Allocates internal workspace for covariance matrix and eigen computation.
//
// Panics if slice lengths are inconsistent with dimensions or if nComponents > nFeatures.
func PCA(data []float64, nSamples, nFeatures, nComponents int, components, explained []float64) float64 {
	if len(data) != nSamples*nFeatures {
		panic("linalg.PCA: len(data) != nSamples*nFeatures")
	}
	if nComponents > nFeatures {
		panic("linalg.PCA: nComponents > nFeatures")
	}
	if nComponents < 1 {
		panic("linalg.PCA: nComponents < 1")
	}
	if nSamples < 2 {
		panic("linalg.PCA: nSamples < 2")
	}
	if len(components) != nComponents*nFeatures {
		panic("linalg.PCA: len(components) != nComponents*nFeatures")
	}
	if len(explained) != nComponents {
		panic("linalg.PCA: len(explained) != nComponents")
	}

	nf := nFeatures

	// Step 1: Compute column means.
	means := make([]float64, nf)
	for s := 0; s < nSamples; s++ {
		for f := 0; f < nf; f++ {
			means[f] += data[s*nf+f]
		}
	}
	nsf := float64(nSamples)
	for f := 0; f < nf; f++ {
		means[f] /= nsf
	}

	// Step 2: Build covariance matrix (nFeatures x nFeatures).
	cov := make([]float64, nf*nf)
	denom := nsf - 1
	for i := 0; i < nf; i++ {
		for j := i; j < nf; j++ {
			var sum float64
			for s := 0; s < nSamples; s++ {
				sum += (data[s*nf+i] - means[i]) * (data[s*nf+j] - means[j])
			}
			c := sum / denom
			cov[i*nf+j] = c
			cov[j*nf+i] = c
		}
	}

	// Step 3: Compute eigenvalues.
	eigenvalues := make([]float64, nf)
	QRAlgorithm(cov, nf, eigenvalues, 1000)

	// Step 4: For each eigenvalue, recover eigenvector via inverse iteration.
	// Eigenvalues are already sorted descending by QRAlgorithm.
	totalVar := 0.0
	for i := 0; i < nf; i++ {
		totalVar += eigenvalues[i]
	}

	// Scratch space for inverse iteration.
	shifted := make([]float64, nf*nf)
	L := make([]float64, nf*nf)
	U := make([]float64, nf*nf)
	perm := make([]int, nf)
	bvec := make([]float64, nf)
	xvec := make([]float64, nf)

	for c := 0; c < nComponents; c++ {
		lambda := eigenvalues[c]

		// Compute eigenvector via inverse iteration: (A - lambda*I)^(-1) * b converges
		// to the eigenvector corresponding to the eigenvalue closest to lambda.
		// Use a small shift to avoid exact singularity.
		shift := lambda - 1e-10*(1+lambda*lambda)

		// Build (cov - shift*I).
		copy(shifted, cov)
		for i := 0; i < nf; i++ {
			shifted[i*nf+i] -= shift
		}

		ok := LUDecompose(shifted, nf, L, U, perm)
		if !ok {
			// Fallback: try a different shift.
			copy(shifted, cov)
			shift = lambda + 1e-8*(1+lambda*lambda)
			for i := 0; i < nf; i++ {
				shifted[i*nf+i] -= shift
			}
			ok = LUDecompose(shifted, nf, L, U, perm)
		}

		if ok {
			// Start with a random-ish vector (all ones).
			for i := 0; i < nf; i++ {
				bvec[i] = 1.0
			}

			// Iterate.
			for iter := 0; iter < 50; iter++ {
				LUSolve(L, U, nf, perm, bvec, xvec)

				// Normalize.
				norm := 0.0
				for i := 0; i < nf; i++ {
					norm += xvec[i] * xvec[i]
				}
				norm = math.Sqrt(norm)
				if norm < 1e-300 {
					break
				}
				for i := 0; i < nf; i++ {
					xvec[i] /= norm
				}

				// Check convergence: if bvec and xvec are parallel.
				maxDiff := 0.0
				for i := 0; i < nf; i++ {
					// Allow sign flip.
					d1 := xvec[i] - bvec[i]
					d2 := xvec[i] + bvec[i]
					if d1 < 0 {
						d1 = -d1
					}
					if d2 < 0 {
						d2 = -d2
					}
					d := d1
					if d2 < d {
						d = d2
					}
					if d > maxDiff {
						maxDiff = d
					}
				}

				copy(bvec, xvec)
				if maxDiff < 1e-12 {
					break
				}
			}

			// Orthogonalize against previously found components (Gram-Schmidt).
			for prev := 0; prev < c; prev++ {
				dot := 0.0
				for i := 0; i < nf; i++ {
					dot += bvec[i] * components[prev*nf+i]
				}
				for i := 0; i < nf; i++ {
					bvec[i] -= dot * components[prev*nf+i]
				}
			}

			// Re-normalize after orthogonalization.
			norm := 0.0
			for i := 0; i < nf; i++ {
				norm += bvec[i] * bvec[i]
			}
			norm = math.Sqrt(norm)
			if norm > 1e-300 {
				for i := 0; i < nf; i++ {
					bvec[i] /= norm
				}
			}

			copy(components[c*nf:(c+1)*nf], bvec)
		}

		if totalVar > 0 {
			explained[c] = eigenvalues[c] / totalVar
		} else {
			explained[c] = 0
		}
	}

	cumVar := 0.0
	for i := 0; i < nComponents; i++ {
		cumVar += explained[i]
	}
	return cumVar
}

