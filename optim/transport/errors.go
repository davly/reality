package transport

import "errors"

// ErrEmptyDistribution is returned when a distribution passed to
// Wasserstein1D or Sinkhorn is empty after NaN/Inf filtering.  An OT
// distance against an empty support is undefined; callers receive a
// typed signal rather than a quiet NaN.
//
// Note: the cross-substrate-precision contract with RubberDuck's
// Wasserstein1D requires the *Wasserstein1D* function specifically
// to surface emptiness as math.NaN (matching the C# `double.NaN`
// return), so the closed-form 1D path returns NaN-with-nil-error for
// empty inputs.  Sinkhorn's typed return surfaces the sentinel.
var ErrEmptyDistribution = errors.New("transport: distribution must be non-empty")

// ErrUnequalMass is returned by Sinkhorn when |sum(a) - sum(b)| > 1e-9
// (the entropic-OT problem requires equal-mass marginals; unbalanced
// OT is a v2 capability).  Detection is L^1 because the multiplicative
// renormalisation Sinkhorn does on the dual potentials produces
// numerically clean equality up to roughly 1e-12 when the input
// marginals already match in L^1.
var ErrUnequalMass = errors.New("transport: marginals must have equal total mass")

// ErrInvalidP is returned when the order p of the Wasserstein-p metric
// is non-finite or < 1.  Wasserstein-p for p in [1, +Inf) is a metric;
// p = 1 is the special closed-form 1D case and is the default for
// regime-detection consumers.
var ErrInvalidP = errors.New("transport: p must be a finite number in [1, +Inf)")

// ErrInvalidRegularisation is returned when the Sinkhorn regularisation
// parameter epsilon is non-finite, NaN, or non-positive.  The entropic
// OT problem
//
//	min <P, C> + epsilon * H(P)  s.t. P 1 = a, P^T 1 = b, P >= 0
//
// is convex iff epsilon > 0; epsilon = 0 reduces to the LP which this
// package does not solve.  Underflow risk grows as epsilon -> 0; the
// log-domain implementation pushes the floor to roughly 1e-9 of the
// cost-matrix scale before convergence stalls.
var ErrInvalidRegularisation = errors.New("transport: regularisation epsilon must be finite and positive")

// ErrSinkhornNonConvergent is returned when the Sinkhorn iteration
// fails to bring the marginal-deviation L^1 norm below tol within the
// supplied iteration cap.  Common causes: epsilon too small relative
// to the cost-matrix scale (recommended: epsilon >= 0.001 * mean(C)),
// pathological cost matrix (very large dynamic range), or maxIter too
// small for the supplied epsilon (rule of thumb: 100 * (mean(C) /
// epsilon)).  The R75 StandardEscapeReason::SINKHORN_NONCONVERGENT is
// the cross-substrate-canonical mapping of this sentinel.
var ErrSinkhornNonConvergent = errors.New("transport: Sinkhorn iteration did not converge to tolerance")

// ErrCostMatrixDimensionMismatch is returned when the cost matrix
// supplied to Sinkhorn has dimensions inconsistent with the marginal
// vectors.  For marginals a in R^n and b in R^m the cost matrix C
// must be n rows of m columns; ragged matrices are also rejected.
var ErrCostMatrixDimensionMismatch = errors.New("transport: cost matrix dimensions do not match marginals")
