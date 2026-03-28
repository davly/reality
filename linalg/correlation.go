package linalg

import "math"

// PearsonCorrelation computes the Pearson product-moment correlation coefficient
// between two equal-length slices. Returns 0 if fewer than 2 data points or
// if either variable has zero variance.
//
// Definition: r = (n*sum(xy) - sum(x)*sum(y)) / sqrt((n*sum(x^2) - sum(x)^2) * (n*sum(y^2) - sum(y)^2))
// Result range: [-1, 1].
// Valid input range: len(x) == len(y) >= 2; at least one variable must have nonzero variance.
// Precision: exact for IEEE 754 float64.
//
// Source: extracted from aicore/echomath.PearsonCorrelation.
func PearsonCorrelation(x, y []float64) float64 {
	n := len(x)
	if n < 2 || len(y) != n {
		return 0
	}
	var sumX, sumY, sumXY, sumX2, sumY2 float64
	for i := 0; i < n; i++ {
		sumX += x[i]
		sumY += y[i]
		sumXY += x[i] * y[i]
		sumX2 += x[i] * x[i]
		sumY2 += y[i] * y[i]
	}
	nf := float64(n)
	denom := math.Sqrt((nf*sumX2 - sumX*sumX) * (nf*sumY2 - sumY*sumY))
	if denom == 0 {
		return 0
	}
	return (nf*sumXY - sumX*sumY) / denom
}

// SpearmanCorrelation computes the Spearman rank correlation coefficient
// between two equal-length slices. This is the Pearson correlation of the
// rank-transformed data, making it robust to nonlinear monotonic relationships.
//
// Ties are handled by assigning average ranks.
// Returns 0 if fewer than 2 data points.
//
// Definition: rho = Pearson(rank(x), rank(y))
// Result range: [-1, 1].
// Valid input range: len(x) == len(y) >= 2.
// Precision: exact for IEEE 754 float64 (limited by ranking step).
func SpearmanCorrelation(x, y []float64) float64 {
	n := len(x)
	if n < 2 || len(y) != n {
		return 0
	}

	rankX := ranks(x)
	rankY := ranks(y)

	return PearsonCorrelation(rankX, rankY)
}

// ranks computes the average ranks for a float64 slice.
// Ties receive the average of their ordinal positions (1-based).
// Allocates a temporary index slice and the output ranks slice.
func ranks(data []float64) []float64 {
	n := len(data)

	// Create indices sorted by data value.
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	// Insertion sort (stable, good for typical correlation dataset sizes).
	for i := 1; i < n; i++ {
		for j := i; j > 0 && data[idx[j]] < data[idx[j-1]]; j-- {
			idx[j], idx[j-1] = idx[j-1], idx[j]
		}
	}

	rnk := make([]float64, n)
	i := 0
	for i < n {
		j := i
		// Find the end of ties.
		for j < n-1 && data[idx[j+1]] == data[idx[j]] {
			j++
		}
		// Average rank for tied values (1-based).
		avgRank := float64(i+j)/2.0 + 1.0
		for k := i; k <= j; k++ {
			rnk[idx[k]] = avgRank
		}
		i = j + 1
	}
	return rnk
}

// Covariance computes the sample covariance between two equal-length slices.
// Uses n-1 (Bessel's correction) as the denominator for unbiased estimation.
// Returns 0 if fewer than 2 data points.
//
// Definition: cov(x, y) = sum((x_i - mean_x) * (y_i - mean_y)) / (n - 1)
// Valid input range: len(x) == len(y) >= 2.
// Precision: two-pass algorithm for improved numerical stability.
func Covariance(x, y []float64) float64 {
	n := len(x)
	if n < 2 || len(y) != n {
		return 0
	}

	// Pass 1: compute means.
	var sumX, sumY float64
	for i := 0; i < n; i++ {
		sumX += x[i]
		sumY += y[i]
	}
	nf := float64(n)
	meanX := sumX / nf
	meanY := sumY / nf

	// Pass 2: compute covariance.
	var cov float64
	for i := 0; i < n; i++ {
		cov += (x[i] - meanX) * (y[i] - meanY)
	}
	return cov / (nf - 1)
}

// CovarianceMatrix computes the sample covariance matrix from a dataset.
// data is organized as samples (rows) x features (columns), where each inner
// slice is one sample. out must be pre-allocated with length features*features.
//
// Definition: C[i][j] = cov(feature_i, feature_j)
// Uses n-1 denominator (Bessel's correction).
//
// Panics if data is empty, samples have inconsistent lengths, or out has wrong length.
func CovarianceMatrix(data [][]float64, out []float64) {
	if len(data) == 0 {
		panic("linalg.CovarianceMatrix: empty data")
	}
	nSamples := len(data)
	nFeatures := len(data[0])
	if nFeatures == 0 {
		panic("linalg.CovarianceMatrix: zero features")
	}
	if len(out) != nFeatures*nFeatures {
		panic("linalg.CovarianceMatrix: len(out) != features*features")
	}
	if nSamples < 2 {
		panic("linalg.CovarianceMatrix: need at least 2 samples")
	}

	// Validate consistent sample lengths.
	for s := 1; s < nSamples; s++ {
		if len(data[s]) != nFeatures {
			panic("linalg.CovarianceMatrix: inconsistent sample lengths")
		}
	}

	nf := float64(nSamples)

	// Compute means for each feature.
	// We allocate here because this is setup, not the hot inner loop.
	means := make([]float64, nFeatures)
	for s := 0; s < nSamples; s++ {
		for f := 0; f < nFeatures; f++ {
			means[f] += data[s][f]
		}
	}
	for f := 0; f < nFeatures; f++ {
		means[f] /= nf
	}

	// Compute covariance matrix (symmetric: only compute upper triangle, mirror).
	for i := 0; i < nFeatures; i++ {
		for j := i; j < nFeatures; j++ {
			var cov float64
			for s := 0; s < nSamples; s++ {
				cov += (data[s][i] - means[i]) * (data[s][j] - means[j])
			}
			cov /= (nf - 1)
			out[i*nFeatures+j] = cov
			out[j*nFeatures+i] = cov
		}
	}
}
