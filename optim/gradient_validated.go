package optim

import (
	"errors"
	"math"
)

// ---------------------------------------------------------------------------
// R123 — Validated-iterate convergence variants of GradientDescent / LBFGS.
//
// The original GradientDescent / LBFGS at gradient.go are R123-compliant only
// when the user-supplied gradient function never emits sentinel-zero on
// invalid input. Any caller whose gradient has a "RetainPreviousIterate"
// recovery path (the GARCH bug class — return zero gradient on Validate
// failure as a "skip this iterate" signal) MUST use the *Validated variants
// here instead.
//
// The *Validated variants take an extra `validate func([]float64) bool`
// parameter that is consulted before declaring convergence. The loop
// terminator is `(gnorm < tol) AND validate(x)` — never `gnorm < tol` alone.
// When `validate(x)` returns false, the loop continues (or exits with
// ConvergenceUnvalidated if maxIter is hit).
//
// See ECOSYSTEM_QUALITY_STANDARD.md §R123 — VALIDATED-ITERATE-CONVERGENCE.
// ---------------------------------------------------------------------------

// ConvergenceResult reports the outcome of a *Validated optimisation.
// Iters is the number of iterations actually executed (1-based for completed
// iterations; 0 if the loop never ran). Converged is true iff the loop exited
// via the validated-convergence break (gnorm < tol AND validate(x)).
//
// When Converged is false and Iters == maxIter, the optimiser exhausted its
// budget. When Converged is false and Iters < maxIter, the loop exited
// because validate(x) failed and there was no further progress to make
// (this can happen when grad returns zero on an invalid iterate that the
// outer optimiser cannot escape via its own step rule).
type ConvergenceResult struct {
	X         []float64
	Iters     int
	Converged bool
	Reason    string
}

// ErrNilValidate is returned when GradientDescentValidated or LBFGSValidated
// are called with a nil validate parameter. R123 requires an explicit
// validity predicate; pass `func(_ []float64) bool { return true }` if the
// iterate domain is unconstrained.
var ErrNilValidate = errors.New("optim: validate function must not be nil (R123)")

// GradientDescentValidated minimises f by steepest descent with R123-compliant
// convergence: the loop only declares converged when gnorm < tol AND
// validate(x) returns true. Any caller whose `grad` has a sentinel-zero
// recovery path on invalid x MUST pass a real `validate` predicate that
// returns false on the same inputs.
//
// Returns a ConvergenceResult; the X slice is freshly allocated.
func GradientDescentValidated(
	f func([]float64) float64,
	grad func([]float64, []float64),
	x0 []float64,
	lr float64,
	maxIter int,
	tol float64,
	validate func([]float64) bool,
) (ConvergenceResult, error) {
	_ = f // available for line-search extensions; unused in basic GD
	if validate == nil {
		return ConvergenceResult{}, ErrNilValidate
	}

	n := len(x0)
	x := make([]float64, n)
	copy(x, x0)
	g := make([]float64, n)

	var iter int
	for iter = 0; iter < maxIter; iter++ {
		grad(x, g)

		gnorm := 0.0
		for i := 0; i < n; i++ {
			gnorm += g[i] * g[i]
		}
		gnorm = math.Sqrt(gnorm)

		// R123 convergence terminator: tolerance AND iterate validity.
		// The order matters: validate(x) is the slower check, so we
		// short-circuit on the cheaper tolerance test first.
		if gnorm < tol {
			if validate(x) {
				return ConvergenceResult{
					X:         x,
					Iters:     iter + 1,
					Converged: true,
					Reason:    "validated convergence",
				}, nil
			}
			// gnorm < tol but iterate is invalid — this is the GARCH
			// bug class. The grad function returned sentinel-zero on
			// an invalid x. We cannot make progress (next iter will
			// see the same x and the same zero gradient), so report
			// non-convergence with the diagnostic.
			return ConvergenceResult{
				X:         x,
				Iters:     iter + 1,
				Converged: false,
				Reason:    "tolerance hit on invalid iterate (R123 trap caught)",
			}, nil
		}

		for i := 0; i < n; i++ {
			x[i] -= lr * g[i]
		}
	}

	// Budget exhausted.
	return ConvergenceResult{
		X:         x,
		Iters:     iter,
		Converged: false,
		Reason:    "max iterations exhausted",
	}, nil
}

// LBFGSValidated minimises f via L-BFGS with R123-compliant convergence.
// Same contract as GradientDescentValidated.
func LBFGSValidated(
	f func([]float64) float64,
	grad func([]float64, []float64),
	x0 []float64,
	m, maxIter int,
	tol float64,
	validate func([]float64) bool,
) (ConvergenceResult, error) {
	if validate == nil {
		return ConvergenceResult{}, ErrNilValidate
	}

	n := len(x0)
	x := make([]float64, n)
	copy(x, x0)

	g := make([]float64, n)
	gPrev := make([]float64, n)
	xPrev := make([]float64, n)
	d := make([]float64, n)

	sHist := make([][]float64, 0, m)
	yHist := make([][]float64, 0, m)
	rho := make([]float64, 0, m)
	alpha := make([]float64, m)

	grad(x, g)

	var iter int
	for iter = 0; iter < maxIter; iter++ {
		gnorm := vecNorm(g)
		// R123 convergence terminator: tolerance AND iterate validity.
		if gnorm < tol {
			if validate(x) {
				return ConvergenceResult{
					X:         x,
					Iters:     iter + 1,
					Converged: true,
					Reason:    "validated convergence",
				}, nil
			}
			return ConvergenceResult{
				X:         x,
				Iters:     iter + 1,
				Converged: false,
				Reason:    "tolerance hit on invalid iterate (R123 trap caught)",
			}, nil
		}

		// Two-loop recursion to compute d = -H_k * g_k.
		q := make([]float64, n)
		copy(q, g)
		k := len(sHist)
		for i := k - 1; i >= 0; i-- {
			alpha[i] = rho[i] * vecDot(sHist[i], q)
			vecAddScaled(q, q, yHist[i], -alpha[i])
		}
		gamma := 1.0
		if k > 0 {
			sy := vecDot(sHist[k-1], yHist[k-1])
			yy := vecDot(yHist[k-1], yHist[k-1])
			if yy > 0 {
				gamma = sy / yy
			}
		}
		for i := 0; i < n; i++ {
			d[i] = gamma * q[i]
		}
		for i := 0; i < k; i++ {
			beta := rho[i] * vecDot(yHist[i], d)
			vecAddScaled(d, d, sHist[i], alpha[i]-beta)
		}
		for i := 0; i < n; i++ {
			d[i] = -d[i]
		}

		step := lbfgsLineSearch(f, grad, x, g, d, n)

		copy(xPrev, x)
		copy(gPrev, g)
		for i := 0; i < n; i++ {
			x[i] += step * d[i]
		}
		grad(x, g)

		sk := make([]float64, n)
		yk := make([]float64, n)
		for i := 0; i < n; i++ {
			sk[i] = x[i] - xPrev[i]
			yk[i] = g[i] - gPrev[i]
		}
		sy := vecDot(sk, yk)
		if sy <= 0 {
			continue
		}
		rhok := 1.0 / sy
		if len(sHist) < m {
			sHist = append(sHist, sk)
			yHist = append(yHist, yk)
			rho = append(rho, rhok)
		} else {
			copy(sHist, sHist[1:])
			copy(yHist, yHist[1:])
			copy(rho, rho[1:])
			sHist[m-1] = sk
			yHist[m-1] = yk
			rho[m-1] = rhok
		}
	}

	return ConvergenceResult{
		X:         x,
		Iters:     iter,
		Converged: false,
		Reason:    "max iterations exhausted",
	}, nil
}
