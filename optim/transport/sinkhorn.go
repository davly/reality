package transport

import "math"

// SinkhornResult bundles the transport plan, total transport cost, and
// the convergence diagnostic for a successful run of Sinkhorn.  Plan
// is the n×m doubly-stochastic-style matrix (row-sums = a,
// column-sums = b); Cost is <Plan, CostMatrix> = sum_{i,j} P_ij * C_ij;
// Iterations is the number of full Sinkhorn passes consumed.
type SinkhornResult struct {
	Plan       [][]float64
	Cost       float64
	Iterations int
}

// Sinkhorn solves the entropic-regularised optimal-transport problem
//
//	min_{P >= 0}  <P, C> + epsilon * H(P)
//	    s.t.  P 1 = a,  P^T 1 = b
//
// where H(P) = sum_{i,j} P_ij (log P_ij - 1) is the negative-entropy
// regulariser (Cuturi 2013 "Sinkhorn distances").  The implementation
// uses the *log-domain* Sinkhorn iteration to avoid the
// underflow / overflow pathologies of the multiplicative algorithm
// when epsilon is small relative to the cost-matrix scale; concretely
// we update the dual potentials f and g via
//
//	f_i ← epsilon * (log a_i - LSE_j ( (g_j - C_ij) / epsilon ))
//	g_j ← epsilon * (log b_j - LSE_i ( (f_i - C_ij) / epsilon ))
//
// where LSE is the standard numerically-stable log-sum-exp.  The
// transport plan is recovered from the converged potentials as
//
//	P_ij = exp( (f_i + g_j - C_ij) / epsilon )
//
// Convergence is measured by the L^1 marginal deviation
//
//	||P 1 - a||_1   (column-sums equal b after the g-update)
//
// against tol.  We cap maxIter at 1000 internally; passing maxIter <= 0
// uses 200 as the default (enough for epsilon >= 0.01 * mean(C) on
// well-conditioned problems).  tol <= 0 falls back to 1e-7 — tighter
// thresholds (1e-9, 1e-12) tend to bottom out on the IEEE-754 noise
// floor of the LSE / exp round-trip rather than on actual residual
// progress; users who want tighter bounds can pass a smaller tol but
// should also raise maxIter.
//
// Returns ErrInvalidRegularisation, ErrCostMatrixDimensionMismatch,
// ErrEmptyDistribution, ErrUnequalMass, or ErrSinkhornNonConvergent
// per the failure mode.  On success, the Plan is freshly allocated
// and is owned by the caller; the input slices are not mutated.
//
// Cuturi 2013 noted Sinkhorn is differentiable in the marginals and
// in C; this implementation keeps the gradient-of-OT extension at the
// API surface (return the dual potentials) but defers exposing the
// gradient API to v2 — the present consumers (Reality regime drift,
// Nexus directional-shape comparison) need only the cost and the
// transport plan.
//
// Cross-substrate parity (R80b): Sinkhorn is not yet shipped in
// RubberDuck (the C# `Wasserstein2DSinkhorn` design-doc spec is
// unwired); R80b parity for n-D OT will be added when a sister
// implementation lands.  The current parity contract is the closed-
// form Wasserstein1D path against RubberDuck's reference.
func Sinkhorn(
	a, b []float64,
	costMatrix [][]float64,
	epsilon float64,
	maxIter int,
	tol float64,
) (SinkhornResult, error) {
	if !(epsilon > 0) || math.IsNaN(epsilon) || math.IsInf(epsilon, 0) {
		return SinkhornResult{}, ErrInvalidRegularisation
	}
	if len(a) == 0 || len(b) == 0 {
		return SinkhornResult{}, ErrEmptyDistribution
	}
	if len(costMatrix) != len(a) {
		return SinkhornResult{}, ErrCostMatrixDimensionMismatch
	}
	for _, row := range costMatrix {
		if len(row) != len(b) {
			return SinkhornResult{}, ErrCostMatrixDimensionMismatch
		}
	}

	// Validate marginals: non-negative, finite, equal mass.
	sumA, sumB := 0.0, 0.0
	for _, ai := range a {
		if math.IsNaN(ai) || math.IsInf(ai, 0) || ai < 0 {
			return SinkhornResult{}, ErrEmptyDistribution
		}
		sumA += ai
	}
	for _, bj := range b {
		if math.IsNaN(bj) || math.IsInf(bj, 0) || bj < 0 {
			return SinkhornResult{}, ErrEmptyDistribution
		}
		sumB += bj
	}
	if sumA == 0 || sumB == 0 {
		return SinkhornResult{}, ErrEmptyDistribution
	}
	if math.Abs(sumA-sumB) > 1e-9 {
		return SinkhornResult{}, ErrUnequalMass
	}

	if maxIter <= 0 {
		maxIter = 200
	}
	if maxIter > 1000 {
		maxIter = 1000
	}
	if !(tol > 0) {
		tol = 1e-7
	}

	n, m := len(a), len(b)

	// Pre-compute log a and log b; substitute -Inf for log 0 (Sinkhorn
	// handles zero-mass marginals cleanly because exp(-Inf + finite) = 0
	// when accumulated through LSE).
	logA := make([]float64, n)
	logB := make([]float64, m)
	for i, ai := range a {
		if ai == 0 {
			logA[i] = math.Inf(-1)
		} else {
			logA[i] = math.Log(ai)
		}
	}
	for j, bj := range b {
		if bj == 0 {
			logB[j] = math.Inf(-1)
		} else {
			logB[j] = math.Log(bj)
		}
	}

	// Initialise dual potentials at zero (equivalent to uniform
	// scaling vectors u, v = 1 in the multiplicative form).
	f := make([]float64, n)
	g := make([]float64, m)

	// Working buffer for LSE arguments.
	bufRow := make([]float64, m)
	bufCol := make([]float64, n)

	iterations := 0
	for it := 0; it < maxIter; it++ {
		iterations = it + 1

		// f_i ← epsilon * ( log a_i - LSE_j ( (g_j - C_ij) / epsilon ) )
		for i := 0; i < n; i++ {
			for j := 0; j < m; j++ {
				bufRow[j] = (g[j] - costMatrix[i][j]) / epsilon
			}
			f[i] = epsilon * (logA[i] - logSumExp(bufRow))
		}

		// g_j ← epsilon * ( log b_j - LSE_i ( (f_i - C_ij) / epsilon ) )
		for j := 0; j < m; j++ {
			for i := 0; i < n; i++ {
				bufCol[i] = (f[i] - costMatrix[i][j]) / epsilon
			}
			g[j] = epsilon * (logB[j] - logSumExp(bufCol))
		}

		// Convergence check on L^1 marginal deviation.  Column-sums
		// are equal to b after the g-update (modulo numerical error
		// from the LSE), so the residual signal is row-sums vs a.
		// Compute row-sums of the implicit plan.  When the L^1
		// residual is below tol, return.
		residual := 0.0
		for i := 0; i < n; i++ {
			rowSum := 0.0
			for j := 0; j < m; j++ {
				rowSum += math.Exp((f[i] + g[j] - costMatrix[i][j]) / epsilon)
			}
			residual += math.Abs(rowSum - a[i])
		}
		if residual < tol {
			plan := buildPlan(f, g, costMatrix, epsilon)
			cost := planCost(plan, costMatrix)
			return SinkhornResult{
				Plan:       plan,
				Cost:       cost,
				Iterations: iterations,
			}, nil
		}
	}

	return SinkhornResult{Iterations: iterations}, ErrSinkhornNonConvergent
}

// buildPlan reconstructs the transport plan from converged dual
// potentials: P_ij = exp( (f_i + g_j - C_ij) / epsilon ).  Allocates
// fresh; not aliased to any input.
func buildPlan(f, g []float64, costMatrix [][]float64, epsilon float64) [][]float64 {
	n, m := len(f), len(g)
	plan := make([][]float64, n)
	for i := 0; i < n; i++ {
		row := make([]float64, m)
		for j := 0; j < m; j++ {
			row[j] = math.Exp((f[i] + g[j] - costMatrix[i][j]) / epsilon)
		}
		plan[i] = row
	}
	return plan
}

// planCost is <P, C> = sum_{i,j} P_ij * C_ij.
func planCost(plan, costMatrix [][]float64) float64 {
	c := 0.0
	for i := range plan {
		for j := range plan[i] {
			c += plan[i][j] * costMatrix[i][j]
		}
	}
	return c
}

// logSumExp is the standard numerically-stable log-sum-exp:
//
//	LSE(x) = m + log( sum_i exp(x_i - m) )  with  m = max_i x_i.
//
// Returns -Inf if the slice is empty or every element is -Inf.  Pure
// stdlib math; no slice allocation.
func logSumExp(xs []float64) float64 {
	if len(xs) == 0 {
		return math.Inf(-1)
	}
	maxV := math.Inf(-1)
	for _, x := range xs {
		if x > maxV {
			maxV = x
		}
	}
	if math.IsInf(maxV, -1) {
		return math.Inf(-1)
	}
	sum := 0.0
	for _, x := range xs {
		sum += math.Exp(x - maxV)
	}
	return maxV + math.Log(sum)
}
