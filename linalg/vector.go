// Package linalg provides linear algebra primitives: vector operations, matrix
// arithmetic, and correlation functions. All functions are deterministic, use
// only the Go standard library, and make zero heap allocations in hot paths.
//
// Matrix operations use flat []float64 slices in row-major order. Output
// parameters are always pre-allocated by the caller.
//
// Extracted from: github.com/davly/aicore/echomath (proven in production
// across 12 ecosystem projects) plus new operations implemented from
// standard mathematical definitions.
package linalg

import "math"

// CosineSimilarity computes the cosine similarity between two float64 vectors.
// Returns 0 if vectors differ in length or either has zero magnitude.
//
// Definition: cos(a, b) = (a . b) / (||a|| * ||b||)
// Result range: [-1, 1] for unit vectors; [0, 1] for non-negative vectors.
// Precision: exact for IEEE 754 float64 inputs (no iterative steps).
//
// Source: extracted from aicore/echomath.CosineSimilarity.
func CosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

// EncodingDistance computes the L2 (Euclidean) distance between two vectors,
// normalized by the square root of the dimension count.
// Returns 0 if vectors differ in length or are empty.
//
// Definition: d(a, b) = sqrt(sum((a_i - b_i)^2)) / sqrt(n)
// Valid input range: any finite float64 values; n > 0.
// Precision: exact for IEEE 754 float64.
//
// Source: extracted from aicore/echomath.EncodingDistance.
func EncodingDistance(a, b []float64) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var sumSq float64
	for i := range a {
		d := a[i] - b[i]
		sumSq += d * d
	}
	return math.Sqrt(sumSq) / math.Sqrt(float64(len(a)))
}

// DimensionWeightedDistance computes a weighted Euclidean distance between two
// vectors. Each dimension is scaled by its corresponding weight before computing
// the distance. The result is normalized by the sum of weights.
// Returns 0 if inputs are invalid (mismatched lengths, empty, zero total weight).
// Dimensions with non-positive weights are skipped.
//
// Definition: d(a, b, w) = sqrt(sum(w_i * (a_i - b_i)^2) / sum(w_i))
// Valid input range: any finite float64; weights > 0 are used.
// Precision: exact for IEEE 754 float64.
//
// Source: extracted from aicore/echomath.DimensionWeightedDistance.
func DimensionWeightedDistance(a, b, weights []float64) float64 {
	if len(a) != len(b) || len(a) != len(weights) || len(a) == 0 {
		return 0
	}
	var sumSq, totalWeight float64
	for i := range a {
		if weights[i] <= 0 {
			continue
		}
		d := a[i] - b[i]
		sumSq += weights[i] * d * d
		totalWeight += weights[i]
	}
	if totalWeight == 0 {
		return 0
	}
	return math.Sqrt(sumSq / totalWeight)
}

// L2Normalize normalizes a vector to unit length in-place.
// Returns false if the vector has zero magnitude (left unchanged).
//
// Definition: v_i = v_i / ||v||_2
// Valid input range: any finite float64 slice with at least one non-zero element.
// Precision: exact for IEEE 754 float64.
//
// Source: extracted from aicore/echomath.L2Normalize.
func L2Normalize(vec []float64) bool {
	var sumSq float64
	for _, v := range vec {
		sumSq += v * v
	}
	if sumSq == 0 {
		return false
	}
	norm := math.Sqrt(sumSq)
	for i := range vec {
		vec[i] /= norm
	}
	return true
}

// Clamp restricts v to the range [lo, hi].
// If lo > hi the behavior is undefined (caller must ensure lo <= hi).
//
// Definition: clamp(v, lo, hi) = max(lo, min(hi, v))
//
// Source: extracted from aicore/echomath.Clamp.
func Clamp(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// DotProduct computes the dot product (inner product) of two float64 vectors.
// Returns 0 if vectors differ in length or are empty.
//
// Definition: a . b = sum(a_i * b_i)
// Precision: exact for IEEE 754 float64.
func DotProduct(a, b []float64) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var sum float64
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

// L2Norm computes the Euclidean (L2) norm of a vector.
//
// Definition: ||v||_2 = sqrt(sum(v_i^2))
// Returns 0 for empty vectors.
// Precision: exact for IEEE 754 float64.
func L2Norm(v []float64) float64 {
	var sumSq float64
	for _, x := range v {
		sumSq += x * x
	}
	return math.Sqrt(sumSq)
}

// L1Norm computes the Manhattan (L1) norm of a vector.
//
// Definition: ||v||_1 = sum(|v_i|)
// Returns 0 for empty vectors.
// Precision: exact for IEEE 754 float64.
func L1Norm(v []float64) float64 {
	var sum float64
	for _, x := range v {
		sum += math.Abs(x)
	}
	return sum
}

// LInfNorm computes the infinity norm (max absolute value) of a vector.
//
// Definition: ||v||_inf = max(|v_i|)
// Returns 0 for empty vectors.
// Precision: exact for IEEE 754 float64.
func LInfNorm(v []float64) float64 {
	var m float64
	for _, x := range v {
		a := math.Abs(x)
		if a > m {
			m = a
		}
	}
	return m
}

// VectorAdd computes element-wise addition: out[i] = a[i] + b[i].
// All three slices must have the same length. The caller must pre-allocate out.
// Zero heap allocations.
//
// Panics if len(a) != len(b) or len(a) != len(out).
func VectorAdd(a, b, out []float64) {
	if len(a) != len(b) || len(a) != len(out) {
		panic("linalg.VectorAdd: length mismatch")
	}
	for i := range a {
		out[i] = a[i] + b[i]
	}
}

// VectorSub computes element-wise subtraction: out[i] = a[i] - b[i].
// All three slices must have the same length. The caller must pre-allocate out.
// Zero heap allocations.
//
// Panics if len(a) != len(b) or len(a) != len(out).
func VectorSub(a, b, out []float64) {
	if len(a) != len(b) || len(a) != len(out) {
		panic("linalg.VectorSub: length mismatch")
	}
	for i := range a {
		out[i] = a[i] - b[i]
	}
}

// VectorScale computes scalar multiplication: out[i] = a[i] * s.
// a and out must have the same length. The caller must pre-allocate out.
// Zero heap allocations.
//
// Panics if len(a) != len(out).
func VectorScale(a []float64, s float64, out []float64) {
	if len(a) != len(out) {
		panic("linalg.VectorScale: length mismatch")
	}
	for i := range a {
		out[i] = a[i] * s
	}
}

// StructuralOverlap computes the fraction of matching dimensions.
// matched is the number of matching dimensions out of total.
// Returns 0 if total is 0 or negative.
//
// Definition: overlap = matched / total
//
// Source: extracted from aicore/echomath.StructuralOverlap.
func StructuralOverlap(matched, total int) float64 {
	if total <= 0 {
		return 0
	}
	return float64(matched) / float64(total)
}
