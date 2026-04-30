package proximal

import (
	"math"
	"sort"
)

// A ProxOp is a proximal operator: prox_{gamma*g}(v) = argmin_x g(x) +
// (1/(2*gamma)) ||x - v||^2. Implementations write the result to out and
// must accept out aliased with v (in-place is a common case).
//
// gamma is the proximal step size (positive). Operators that represent
// indicator functions of feasible sets (box, non-negative, L2-ball, simplex)
// ignore gamma — projection has no scale.
type ProxOp func(v []float64, gamma float64, out []float64)

// =========================================================================
// L1 / sparsity prox operators
// =========================================================================

// ProxL1 applies the proximal operator of g(x) = ||x||_1, i.e. element-wise
// soft thresholding:
//
//	prox_{gamma*||.||_1}(v)_i = sign(v_i) * max(|v_i| - gamma, 0)
//
// This is the building block of LASSO, basis pursuit, and L1-regularised
// learning. See Parikh & Boyd 2014 §6.5.2.
func ProxL1(v []float64, gamma float64, out []float64) {
	for i, vi := range v {
		switch {
		case vi > gamma:
			out[i] = vi - gamma
		case vi < -gamma:
			out[i] = vi + gamma
		default:
			out[i] = 0
		}
	}
}

// ProxL0 applies the proximal operator of g(x) = ||x||_0 (non-convex):
// hard thresholding,
//
//	prox_{gamma*||.||_0}(v)_i = v_i  if v_i^2 > 2*gamma, else 0
//
// Included for completeness. Convergence guarantees of FBS/FISTA do not
// extend to non-convex g; use cautiously and with restarts.
func ProxL0(v []float64, gamma float64, out []float64) {
	thresh := 2 * gamma
	for i, vi := range v {
		if vi*vi > thresh {
			out[i] = vi
		} else {
			out[i] = 0
		}
	}
}

// =========================================================================
// Smooth quadratic prox operators
// =========================================================================

// ProxSquaredL2 applies the prox of g(x) = (1/2) ||x||^2:
//
//	prox_{gamma*g}(v) = v / (1 + gamma)
//
// This is the resolvent of the identity operator and appears as the simple
// Tikhonov / L2 regulariser. See Parikh & Boyd 2014 §6.1.1.
func ProxSquaredL2(v []float64, gamma float64, out []float64) {
	scale := 1.0 / (1.0 + gamma)
	for i, vi := range v {
		out[i] = scale * vi
	}
}

// =========================================================================
// Indicator (projection) prox operators — gamma is ignored
// =========================================================================

// ProxNonNeg projects v onto the non-negative orthant:
//
//	prox_{i_{R+}}(v)_i = max(v_i, 0)
//
// Equivalent to the indicator function of {x : x >= 0}. Ignores gamma.
func ProxNonNeg(v []float64, gamma float64, out []float64) {
	_ = gamma
	for i, vi := range v {
		if vi < 0 {
			out[i] = 0
		} else {
			out[i] = vi
		}
	}
}

// ProxBox returns a ProxOp that projects v onto the axis-aligned box
// [lo_i, hi_i]. The returned operator captures lo and hi by reference; do
// not mutate the slices after construction.
//
// Caller must ensure len(lo) == len(hi) and lo_i <= hi_i for all i. NaN
// bounds are not permitted; use math.Inf(-1) / math.Inf(1) for unbounded
// sides.
func ProxBox(lo, hi []float64) ProxOp {
	return func(v []float64, gamma float64, out []float64) {
		_ = gamma
		for i, vi := range v {
			switch {
			case vi < lo[i]:
				out[i] = lo[i]
			case vi > hi[i]:
				out[i] = hi[i]
			default:
				out[i] = vi
			}
		}
	}
}

// ProxL2Ball returns a ProxOp that projects v onto the closed L2-ball of
// radius r centred at the origin:
//
//	prox(v) = v               if ||v|| <= r
//	         r * v / ||v||    otherwise
//
// Caller must ensure r > 0.
func ProxL2Ball(r float64) ProxOp {
	return func(v []float64, gamma float64, out []float64) {
		_ = gamma
		var s float64
		for _, vi := range v {
			s += vi * vi
		}
		norm := math.Sqrt(s)
		if norm <= r {
			copy(out, v)
			return
		}
		scale := r / norm
		for i, vi := range v {
			out[i] = scale * vi
		}
	}
}

// ProxSimplex projects v onto the unit simplex
// { x : sum(x) = 1, x >= 0 } using the Held-Wolfe-Crowder sort-based
// algorithm. O(n log n).
//
// Reference: Held M., Wolfe P. & Crowder H. P. (1974). "Validation of
// subgradient optimization." Math. Prog. 6:62-88. Implementation follows
// the simplified projection in Wang & Carreira-Perpinan (2013) "Projection
// onto the probability simplex: An efficient algorithm with a simple proof".
//
// Allocates a sorted copy of v internally. Caller may reuse the function
// across many calls; allocation is a single len(v) slice per invocation.
func ProxSimplex(v []float64, gamma float64, out []float64) {
	_ = gamma
	n := len(v)
	if n == 0 {
		return
	}
	u := make([]float64, n)
	copy(u, v)
	sort.Sort(sort.Reverse(sort.Float64Slice(u)))
	cssv := 0.0
	rho := 0
	for i := 0; i < n; i++ {
		cssv += u[i]
		if u[i]+(1.0-cssv)/float64(i+1) > 0 {
			rho = i + 1
		}
	}
	cssv = 0.0
	for i := 0; i < rho; i++ {
		cssv += u[i]
	}
	tau := (cssv - 1.0) / float64(rho)
	for i, vi := range v {
		x := vi - tau
		if x > 0 {
			out[i] = x
		} else {
			out[i] = 0
		}
	}
}

// =========================================================================
// Linear prox — the affine minorant
// =========================================================================

// ProxLinear returns a ProxOp for g(x) = c^T x:
//
//	prox_{gamma*g}(v) = v - gamma * c
//
// Useful as a one-line term inside larger composite problems.
func ProxLinear(c []float64) ProxOp {
	return func(v []float64, gamma float64, out []float64) {
		for i, vi := range v {
			out[i] = vi - gamma*c[i]
		}
	}
}
