package optim

import (
	"errors"
	"math"
)

// ---------------------------------------------------------------------------
// Linear Programming
//
// Classical methods for solving linear programs of the form:
//   minimize   c'x
//   subject to Ax <= b, x >= 0
//
// All functions are self-contained with zero external dependencies.
// ---------------------------------------------------------------------------

// simplexMaxIter caps the number of simplex pivots before SimplexMethod
// declares non-convergence. With Bland's anti-cycling rule the method is
// finite, so this is a defensive ceiling against pathological / numerically
// degenerate inputs. It is a package variable (rather than a const) so tests
// can lower it to exercise the non-convergence guard.
var simplexMaxIter = 10000

// SimplexMethod solves a standard-form linear program using the revised simplex
// method with Bland's anti-cycling rule.
//
// Problem:
//
//	minimize   c'x
//	subject to Ax <= b, x >= 0
//
// where A is m x n, b is m x 1, c is n x 1. Slack variables are added
// internally to convert to the equality form Ax + s = b, s >= 0.
//
// Returns the optimal solution x (length n), the optimal objective value, and
// an error if the problem is infeasible or unbounded.
//
// Reference: Dantzig, "Linear Programming and Extensions," 1963.
//
//	Bland, "New finite pivoting rules for the simplex method," 1977.
func SimplexMethod(c []float64, A [][]float64, b []float64) ([]float64, float64, error) {
	m := len(A)
	n := len(c)
	if m == 0 || n == 0 {
		return nil, 0, errors.New("optim.SimplexMethod: empty problem")
	}
	if len(b) != m {
		return nil, 0, errors.New("optim.SimplexMethod: len(b) != len(A)")
	}
	for i := range A {
		if len(A[i]) != n {
			return nil, 0, errors.New("optim.SimplexMethod: inconsistent A column count")
		}
	}

	// Check that b >= 0 (standard form requirement after adding slacks).
	// If b[i] < 0, multiply the row by -1.
	aCopy := make([][]float64, m)
	bCopy := make([]float64, m)
	for i := 0; i < m; i++ {
		aCopy[i] = make([]float64, n)
		copy(aCopy[i], A[i])
		bCopy[i] = b[i]
		if bCopy[i] < 0 {
			for j := 0; j < n; j++ {
				aCopy[i][j] = -aCopy[i][j]
			}
			bCopy[i] = -bCopy[i]
		}
	}

	// Build the full tableau: m rows x (n + m) columns (original + slack).
	totalVars := n + m
	tableau := make([][]float64, m)
	for i := 0; i < m; i++ {
		row := make([]float64, totalVars)
		copy(row, aCopy[i])
		row[n+i] = 1 // slack variable
		tableau[i] = row
	}

	// Cost vector (extended with zeros for slacks).
	cost := make([]float64, totalVars)
	copy(cost, c)

	// Basis: initially the slack variables.
	basis := make([]int, m)
	for i := 0; i < m; i++ {
		basis[i] = n + i
	}

	// Simplex iterations with Bland's rule.
	maxIter := simplexMaxIter
	converged := false
	for iter := 0; iter < maxIter; iter++ {
		// Compute reduced costs: rc[j] = c[j] - c_B' * A_j.
		// Find entering variable (Bland: smallest index with rc < 0).
		entering := -1
		for j := 0; j < totalVars; j++ {
			rc := cost[j]
			for i := 0; i < m; i++ {
				rc -= cost[basis[i]] * tableau[i][j]
			}
			if rc < -1e-10 {
				entering = j
				break // Bland's rule: take first
			}
		}
		if entering == -1 {
			// Optimal.
			converged = true
			break
		}

		// Minimum ratio test to find leaving variable.
		leaving := -1
		minRatio := math.Inf(1)
		for i := 0; i < m; i++ {
			if tableau[i][entering] > 1e-10 {
				ratio := bCopy[i] / tableau[i][entering]
				if ratio < minRatio || (ratio == minRatio && basis[i] < basis[leaving]) {
					minRatio = ratio
					leaving = i
				}
			}
		}
		if leaving == -1 {
			return nil, 0, errors.New("optim.SimplexMethod: problem is unbounded")
		}

		// Pivot.
		pivot := tableau[leaving][entering]
		for j := 0; j < totalVars; j++ {
			tableau[leaving][j] /= pivot
		}
		bCopy[leaving] /= pivot

		for i := 0; i < m; i++ {
			if i == leaving {
				continue
			}
			factor := tableau[i][entering]
			for j := 0; j < totalVars; j++ {
				tableau[i][j] -= factor * tableau[leaving][j]
			}
			bCopy[i] -= factor * bCopy[leaving]
		}

		basis[leaving] = entering
	}

	// Guard: the loop exhausted its pivot budget without reaching optimality
	// (entering == -1 was never observed). Returning the current tableau would
	// be a silent non-optimal result, so report a non-nil error instead.
	if !converged {
		return nil, 0, errors.New("optim.SimplexMethod: did not converge within iteration limit")
	}

	// Extract solution.
	x := make([]float64, n)
	optVal := 0.0
	for i := 0; i < m; i++ {
		if basis[i] < n {
			x[basis[i]] = bCopy[i]
			optVal += c[basis[i]] * bCopy[i]
		}
	}

	// Feasibility verification against the ORIGINAL constraints. This method
	// starts from the all-slack origin (x = 0) and has no Phase I, so for an
	// infeasible LP — or a feasible LP whose origin is infeasible (e.g. a
	// constraint with b < 0, i.e. an effective ">=") — the pivots can terminate
	// at a point that is optimal for the relaxed tableau yet does not satisfy
	// Ax <= b. Returning that point with a nil error (the previous behavior)
	// silently handed back a wrong, infeasible "solution". Verify and return the
	// honest error the docstring promises for infeasible problems.
	const feasTol = 1e-7
	for j := 0; j < n; j++ {
		if x[j] < -feasTol {
			return nil, 0, errors.New("optim.SimplexMethod: no feasible solution found (negative variable; problem may be infeasible — Phase I not implemented)")
		}
	}
	for i := 0; i < m; i++ {
		lhs := 0.0
		for j := 0; j < n; j++ {
			lhs += A[i][j] * x[j]
		}
		tol := feasTol * (1 + math.Abs(b[i]))
		if lhs > b[i]+tol {
			return nil, 0, errors.New("optim.SimplexMethod: no feasible solution found (constraint violated; problem may be infeasible or its origin infeasible — Phase I not implemented)")
		}
	}

	return x, optVal, nil
}

// InteriorPoint solves a standard-form linear program (minimize c'x subject to
// Ax <= b, x >= 0), returning the optimal x, the optimal objective value, and an
// error if the problem is infeasible/unbounded or cannot be solved.
//
// NOTE: the previous bespoke barrier iteration was numerically unstable and
// returned NaN or divergent (~1e100) values with a nil error on well-posed LPs
// (it implemented no real Newton/KKT solve and used a positive-feedback dual
// update). Until a numerically-robust primal-dual implementation lands it
// delegates to SimplexMethod, which solves the same standard-form LP correctly
// and reports infeasibility honestly.
//
// Reference: Wright, "Primal-Dual Interior-Point Methods," SIAM, 1997.
func InteriorPoint(c []float64, A [][]float64, b []float64) ([]float64, float64, error) {
	return SimplexMethod(c, A, b)
}
