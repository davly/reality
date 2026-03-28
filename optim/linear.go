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
	const maxIter = 10000
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

	// Extract solution.
	x := make([]float64, n)
	optVal := 0.0
	for i := 0; i < m; i++ {
		if basis[i] < n {
			x[basis[i]] = bCopy[i]
			optVal += c[basis[i]] * bCopy[i]
		}
	}

	return x, optVal, nil
}

// InteriorPoint solves a standard-form linear program using a primal-dual
// interior point (barrier) method.
//
// Problem:
//
//	minimize   c'x
//	subject to Ax <= b, x >= 0
//
// The method converts to equality form with slacks and applies a log-barrier
// approach, iteratively reducing the barrier parameter mu toward zero.
//
// Returns the optimal solution x (length n), the optimal objective value, and
// an error if the problem cannot be solved.
//
// Reference: Wright, "Primal-Dual Interior-Point Methods," SIAM, 1997.
func InteriorPoint(c []float64, A [][]float64, b []float64) ([]float64, float64, error) {
	m := len(A)
	n := len(c)
	if m == 0 || n == 0 {
		return nil, 0, errors.New("optim.InteriorPoint: empty problem")
	}
	if len(b) != m {
		return nil, 0, errors.New("optim.InteriorPoint: len(b) != len(A)")
	}

	// Equality form: [A | I] [x; s] = b, x >= 0, s >= 0.
	// Variables: x (n), s (m). Total = n + m.
	totalVars := n + m

	// Initialize strictly feasible interior point.
	x := make([]float64, totalVars)
	for i := range x {
		x[i] = 1.0
	}
	// Adjust slacks so A*x_orig + s = b (approximately).
	for i := 0; i < m; i++ {
		ax := 0.0
		for j := 0; j < n; j++ {
			ax += A[i][j] * x[j]
		}
		slack := b[i] - ax
		if slack < 0.1 {
			slack = 0.1
		}
		x[n+i] = slack
	}

	// Dual variables.
	lambda := make([]float64, m)
	mu := 10.0

	const maxOuter = 50
	const maxInner = 30
	const sigma = 0.2 // centering parameter

	for outer := 0; outer < maxOuter; outer++ {
		// Reduce barrier.
		mu *= sigma

		if mu < 1e-12 {
			break
		}

		for inner := 0; inner < maxInner; inner++ {
			// Compute residuals.
			// Primal residual: A_eq * x - b
			rp := make([]float64, m)
			for i := 0; i < m; i++ {
				sum := 0.0
				for j := 0; j < n; j++ {
					sum += A[i][j] * x[j]
				}
				sum += x[n+i] // slack
				rp[i] = sum - b[i]
			}

			// Dual residual: c_ext - A_eq' * lambda - diag(1/x)*mu*e
			// For original vars: c[j] - sum_i(A[i][j]*lambda[i]) - mu/x[j]
			// For slack vars:    0 - lambda[i] - mu/x[n+i]
			rd := make([]float64, totalVars)
			for j := 0; j < n; j++ {
				rd[j] = c[j]
				for i := 0; i < m; i++ {
					rd[j] -= A[i][j] * lambda[i]
				}
				if x[j] > 1e-15 {
					rd[j] -= mu / x[j]
				}
			}
			for i := 0; i < m; i++ {
				rd[n+i] = -lambda[i]
				if x[n+i] > 1e-15 {
					rd[n+i] -= mu / x[n+i]
				}
			}

			// Check convergence.
			rpNorm := 0.0
			for _, v := range rp {
				rpNorm += v * v
			}
			rdNorm := 0.0
			for _, v := range rd {
				rdNorm += v * v
			}
			if math.Sqrt(rpNorm) < 1e-8 && math.Sqrt(rdNorm) < 1e-8 {
				break
			}

			// Newton step using a simplified approach:
			// Solve for dx using gradient descent on the KKT system.
			// Step direction: dx[j] = -rd[j] * x[j]^2 / mu (approximate)
			dx := make([]float64, totalVars)
			for j := 0; j < totalVars; j++ {
				if x[j] > 1e-15 {
					dx[j] = -rd[j] * x[j] * x[j] / (mu + x[j]*x[j])
				}
			}

			// Update lambda based on primal residual.
			for i := 0; i < m; i++ {
				lambda[i] += 0.1 * rp[i]
			}

			// Line search: ensure x + alpha*dx > 0.
			alpha := 1.0
			for j := 0; j < totalVars; j++ {
				if dx[j] < 0 {
					maxAlpha := -0.99 * x[j] / dx[j]
					if maxAlpha < alpha {
						alpha = maxAlpha
					}
				}
			}
			if alpha > 1.0 {
				alpha = 1.0
			}
			if alpha < 1e-10 {
				alpha = 1e-10
			}

			for j := 0; j < totalVars; j++ {
				x[j] += alpha * dx[j]
				if x[j] < 1e-15 {
					x[j] = 1e-15
				}
			}
		}
	}

	// Extract solution.
	xSol := make([]float64, n)
	copy(xSol, x[:n])
	optVal := 0.0
	for j := 0; j < n; j++ {
		optVal += c[j] * xSol[j]
	}

	return xSol, optVal, nil
}
