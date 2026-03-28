// Package compression provides lossless and lossy compression primitives
// plus information-theoretic measures. All functions are deterministic, use
// only the Go standard library, and target zero heap allocations in hot paths.
//
// Information theory: Shannon entropy, conditional/joint entropy, mutual
// information, KL divergence, cross-entropy.
//
// Lossless coding: run-length encoding, delta encoding.
//
// Lossy compression: uniform scalar quantization with dequantization.
//
// Consumed by: Recall (cache compression), Echo/Parallax (embedding
// quantization), Ingest (payload compression), Pistachio (texture
// compression), Oracle/RubberDuck (time series delta encoding).
package compression

import "math"

// ShannonEntropy computes the Shannon entropy of a discrete probability
// distribution in bits.
//
// Formula: H = -sum_i p_i * log2(p_i), skipping any p_i <= 0
// Valid range: each p_i in [0, 1]; caller is responsible for normalization
// Output range: [0 (single certain event), log2(n) (uniform)]
// Precision: limited by float64 log; ~15 significant digits
// Reference: Shannon, C.E. (1948) "A Mathematical Theory of Communication"
func ShannonEntropy(probs []float64) float64 {
	h := 0.0
	for _, p := range probs {
		if p > 0 {
			h -= p * math.Log2(p)
		}
	}
	return h
}

// JointEntropy computes the joint entropy H(X,Y) from a joint probability
// distribution matrix. Each element joint[i][j] represents P(X=i, Y=j).
//
// Formula: H(X,Y) = -sum_i sum_j P(i,j) * log2(P(i,j)), skipping zeros
// Valid range: each joint[i][j] in [0, 1]; entries should sum to 1
// Output range: [0, log2(rows*cols)]
// Precision: limited by float64 log
// Reference: Cover, Thomas (2006) "Elements of Information Theory", Ch. 2
func JointEntropy(joint [][]float64) float64 {
	h := 0.0
	for _, row := range joint {
		for _, p := range row {
			if p > 0 {
				h -= p * math.Log2(p)
			}
		}
	}
	return h
}

// ConditionalEntropy computes the conditional entropy H(Y|X) from a joint
// probability distribution matrix. This measures the remaining uncertainty
// in Y given knowledge of X.
//
// Formula: H(Y|X) = H(X,Y) - H(X)
//
//	where H(X) is the marginal entropy of X (row sums)
//
// Valid range: joint[i][j] in [0, 1]; entries should sum to 1
// Output range: [0, H(Y)] — conditioning can only reduce or maintain entropy
// Precision: limited by float64 log and summation
// Reference: Cover, Thomas (2006) "Elements of Information Theory", Ch. 2
func ConditionalEntropy(joint [][]float64) float64 {
	// Marginal distribution of X: sum each row.
	marginalX := make([]float64, len(joint))
	for i, row := range joint {
		s := 0.0
		for _, p := range row {
			s += p
		}
		marginalX[i] = s
	}
	return JointEntropy(joint) - ShannonEntropy(marginalX)
}

// MutualInformation computes the mutual information I(X;Y) from a joint
// probability distribution matrix. This measures how much knowing one
// variable reduces uncertainty about the other.
//
// Formula: I(X;Y) = H(X) + H(Y) - H(X,Y)
//
//	where H(X) and H(Y) are the marginal entropies
//
// Valid range: joint[i][j] in [0, 1]; entries should sum to 1
// Output range: [0 (independent), min(H(X), H(Y)) (perfectly correlated)]
// Precision: limited by float64 log and summation
// Reference: Cover, Thomas (2006) "Elements of Information Theory", Ch. 2
func MutualInformation(joint [][]float64) float64 {
	// Marginal distribution of X: sum each row.
	marginalX := make([]float64, len(joint))
	for i, row := range joint {
		s := 0.0
		for _, p := range row {
			s += p
		}
		marginalX[i] = s
	}

	// Marginal distribution of Y: sum each column.
	// Determine max column width.
	maxCols := 0
	for _, row := range joint {
		if len(row) > maxCols {
			maxCols = len(row)
		}
	}
	marginalY := make([]float64, maxCols)
	for _, row := range joint {
		for j, p := range row {
			marginalY[j] += p
		}
	}

	return ShannonEntropy(marginalX) + ShannonEntropy(marginalY) - JointEntropy(joint)
}

// KLDivergence computes the Kullback-Leibler divergence KL(P || Q), which
// measures how distribution P diverges from reference distribution Q.
//
// Formula: KL(P||Q) = sum_i p_i * log2(p_i / q_i)
// Skips entries where p_i <= 0. Returns +Inf if any p_i > 0 and q_i <= 0.
// Valid range: p and q must have the same length; entries in [0, 1]
// Output range: [0 (identical), +Inf (q has zero where p is positive)]
// Note: KL divergence is NOT symmetric: KL(P||Q) != KL(Q||P) in general.
// Precision: limited by float64 log
// Reference: Kullback, Leibler (1951) "On Information and Sufficiency"
func KLDivergence(p, q []float64) float64 {
	n := len(p)
	if n == 0 || n != len(q) {
		return 0
	}
	kl := 0.0
	for i := 0; i < n; i++ {
		if p[i] > 0 {
			if q[i] <= 0 {
				return math.Inf(1)
			}
			kl += p[i] * math.Log2(p[i]/q[i])
		}
	}
	return kl
}

// CrossEntropy computes the cross-entropy H(P, Q) between distributions P
// and Q. This measures the expected number of bits needed to encode events
// from P using an optimal code for Q.
//
// Formula: H(P, Q) = -sum_i p_i * log2(q_i)
// Skips entries where p_i <= 0. Returns +Inf if any p_i > 0 and q_i <= 0.
// Valid range: p and q must have the same length; entries in [0, 1]
// Output range: [H(P) (when Q==P), +Inf]
// Identity: H(P, Q) = H(P) + KL(P||Q)
// Precision: limited by float64 log
// Reference: Cover, Thomas (2006) "Elements of Information Theory", Ch. 2
func CrossEntropy(p, q []float64) float64 {
	n := len(p)
	if n == 0 || n != len(q) {
		return 0
	}
	ce := 0.0
	for i := 0; i < n; i++ {
		if p[i] > 0 {
			if q[i] <= 0 {
				return math.Inf(1)
			}
			ce -= p[i] * math.Log2(q[i])
		}
	}
	return ce
}
