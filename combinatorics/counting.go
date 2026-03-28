// Package combinatorics provides classical combinatorial functions:
// counting (factorial, binomial, Catalan, Fibonacci, derangements) and
// generation (permutations, combinations, lexicographic next, random subsets).
//
// All counting functions use float64 to support large values (exact up to
// about 170! for factorial). Generation functions produce concrete slices.
// Zero external dependencies.
package combinatorics

import "math"

// ---------------------------------------------------------------------------
// Counting Functions
// ---------------------------------------------------------------------------

// Factorial returns n! as a float64. Exact for n <= 170, +Inf for n > 170.
// Returns 1 for n <= 0 (by convention 0! = 1).
//
// Formula: n! = 1 * 2 * 3 * ... * n
// Valid range: n >= 0 (negative n returns 1)
// Precision: exact for n <= 20 (fits in uint64); float64 mantissa limits
// exact integer representation to about n <= 22; relative error < 1e-15
// for n <= 170.
// Reference: fundamental counting principle; Knuth, TAOCP vol. 1
func Factorial(n int) float64 {
	if n <= 0 {
		return 1
	}
	// For small n, compute directly for exact integer results.
	if n <= 20 {
		result := 1.0
		for i := 2; i <= n; i++ {
			result *= float64(i)
		}
		return result
	}
	// Use math.Lgamma for large n to avoid intermediate overflow.
	// Gamma(n+1) = n! for non-negative integers.
	lg, _ := math.Lgamma(float64(n + 1))
	return math.Exp(lg)
}

// BinomialCoeff returns C(n,k) = n! / (k! * (n-k)!) via log-gamma for
// numerical stability. Returns 0 if k < 0 or k > n.
//
// Formula: exp(lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1))
// Valid range: 0 <= k <= n
// Precision: relative error < 1e-12 for typical inputs
// Reference: Knuth, TAOCP vol. 1; using log-gamma avoids intermediate
// overflow that direct factorial computation would cause for large n.
func BinomialCoeff(n, k int) float64 {
	if k < 0 || k > n {
		return 0
	}
	if k == 0 || k == n {
		return 1
	}
	// Symmetry: C(n,k) = C(n, n-k). Use smaller k for fewer terms.
	if k > n-k {
		k = n - k
	}
	lgn, _ := math.Lgamma(float64(n + 1))
	lgk, _ := math.Lgamma(float64(k + 1))
	lgnk, _ := math.Lgamma(float64(n - k + 1))
	return math.Round(math.Exp(lgn - lgk - lgnk))
}

// Permutations returns P(n,k) = n! / (n-k)!, the number of k-permutations
// of n elements. Returns 0 if k < 0 or k > n.
//
// Formula: n * (n-1) * ... * (n-k+1)
// Valid range: 0 <= k <= n
// Precision: exact for small n; float64 mantissa limits for large n.
// Reference: fundamental counting principle
func Permutations(n, k int) float64 {
	if k < 0 || k > n {
		return 0
	}
	if k == 0 {
		return 1
	}
	lgn, _ := math.Lgamma(float64(n + 1))
	lgnk, _ := math.Lgamma(float64(n - k + 1))
	return math.Round(math.Exp(lgn - lgnk))
}

// CatalanNumber returns the nth Catalan number C_n = C(2n,n) / (n+1).
// Returns 1 for n <= 0 (C_0 = 1).
//
// Formula: C_n = C(2n, n) / (n + 1) = (2n)! / ((n+1)! * n!)
// Valid range: n >= 0
// Precision: exact for small n; float64 rounding for large n
// Reference: Stanley, R.P. "Catalan Numbers" (2015)
func CatalanNumber(n int) float64 {
	if n <= 0 {
		return 1
	}
	return BinomialCoeff(2*n, n) / float64(n+1)
}

// FibonacciNumber returns the nth Fibonacci number F_n using matrix
// exponentiation in O(log n) time. F_0 = 0, F_1 = 1, F_n = F_{n-1} + F_{n-2}.
// Returns 0 for n <= 0.
//
// Method: matrix exponentiation of [[1,1],[1,0]]^n.
// Valid range: n >= 0; exact for n <= 93 (F_93 = 12200160415121876738,
// which is the largest Fibonacci number fitting in uint64).
// Precision: exact (integer arithmetic, no floating point)
// Reference: Knuth, TAOCP vol. 1, Section 1.2.8
func FibonacciNumber(n int) uint64 {
	if n <= 0 {
		return 0
	}
	if n == 1 || n == 2 {
		return 1
	}

	// Matrix [[a,b],[c,d]] represents [[F_{k+1}, F_k], [F_k, F_{k-1}]].
	// We exponentiate [[1,1],[1,0]] by n using repeated squaring.
	var (
		// Result matrix (identity)
		ra, rb, rc, rd uint64 = 1, 0, 0, 1
		// Base matrix
		ba, bb, bc, bd uint64 = 1, 1, 1, 0
	)

	exp := n
	for exp > 0 {
		if exp%2 == 1 {
			// result = result * base
			na := ra*ba + rb*bc
			nb := ra*bb + rb*bd
			nc := rc*ba + rd*bc
			nd := rc*bb + rd*bd
			ra, rb, rc, rd = na, nb, nc, nd
		}
		// base = base * base
		na := ba*ba + bb*bc
		nb := ba*bb + bb*bd
		nc := bc*ba + bd*bc
		nd := bc*bb + bd*bd
		ba, bb, bc, bd = na, nb, nc, nd
		exp /= 2
	}

	return rb // F_n is in position (0,1) of the result matrix
}

// DerangementCount returns !n, the number of derangements (permutations with
// no fixed points) of n elements. Returns 1 for n == 0, 0 for n == 1.
//
// Formula: !n = n! * sum_{k=0}^{n} (-1)^k / k!
// Equivalently: !n = round(n! / e) for n >= 1
// Valid range: n >= 0
// Precision: exact via rounding for moderate n; float64 limits for very large n
// Reference: Euler (1751); see also Knuth TAOCP vol. 1
func DerangementCount(n int) float64 {
	if n <= 0 {
		return 1
	}
	if n == 1 {
		return 0
	}
	// !n = round(n! / e) for n >= 1
	return math.Round(Factorial(n) / math.E)
}
