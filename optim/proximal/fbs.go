package proximal

import (
	"errors"
	"math"
)

// GradOp evaluates the gradient of a smooth convex function f at x and
// writes it to out (which has len(x) entries). Implementations may also
// return f(x) for monitoring; if not available, return 0.
type GradOp func(x []float64, out []float64) (fval float64)

// FbsConfig configures Forward-Backward Splitting and FISTA.
type FbsConfig struct {
	// Step is the proximal step size gamma. For convergence Step must lie
	// in (0, 2/L) where L is the Lipschitz constant of grad f. A safe
	// default is 1/L. If Step <= 0 the solver returns an error.
	Step float64

	// MaxIter caps the number of outer iterations. If 0 defaults to 1000.
	MaxIter int

	// AbsTol terminates when ||x_{k+1} - x_k||_inf < AbsTol. If <= 0
	// defaults to 1e-9.
	AbsTol float64

	// Accelerate enables FISTA acceleration. False uses plain FBS.
	Accelerate bool
}

// FbsResult reports the outcome of an FBS / FISTA run.
type FbsResult struct {
	// Iter is the number of iterations actually executed.
	Iter int

	// Converged is true if the algorithm reached AbsTol; false on MaxIter.
	Converged bool

	// FinalDelta is the infinity-norm of the last update step.
	FinalDelta float64
}

// Fbs solves
//
//	minimize_x  f(x) + g(x)
//
// where grad evaluates the gradient of f (smooth, convex, L-Lipschitz) and
// prox is the proximal operator of g (convex, possibly non-smooth). The
// initial guess x is updated in place. work must be a buffer of length
// len(x); it is overwritten.
//
// Reference: Bauschke-Combettes 2011, Convex Analysis and Monotone Operator
// Theory in Hilbert Spaces, §28.5. With FISTA acceleration: Beck-Teboulle
// 2009, A Fast Iterative Shrinkage-Thresholding Algorithm.
//
// FISTA convergence rate is O(1/k^2) on the objective; plain FBS is O(1/k).
func Fbs(grad GradOp, prox ProxOp, x, work []float64, cfg FbsConfig) (FbsResult, error) {
	if cfg.Step <= 0 || math.IsInf(cfg.Step, 0) || math.IsNaN(cfg.Step) {
		return FbsResult{}, errors.New("proximal: FbsConfig.Step must be positive and finite")
	}
	if grad == nil || prox == nil {
		return FbsResult{}, errors.New("proximal: grad and prox must be non-nil")
	}
	if len(work) < len(x) {
		return FbsResult{}, errors.New("proximal: work buffer must have len >= len(x)")
	}
	maxIter := cfg.MaxIter
	if maxIter == 0 {
		maxIter = 1000
	}
	tol := cfg.AbsTol
	if !(tol > 0) {
		tol = 1e-9
	}

	n := len(x)
	gradBuf := work[:n]

	if !cfg.Accelerate {
		return fbsLoop(grad, prox, x, gradBuf, cfg.Step, maxIter, tol)
	}
	return fistaLoop(grad, prox, x, gradBuf, cfg.Step, maxIter, tol)
}

func fbsLoop(grad GradOp, prox ProxOp, x, gradBuf []float64, step float64, maxIter int, tol float64) (FbsResult, error) {
	n := len(x)
	prev := make([]float64, n)
	for k := 0; k < maxIter; k++ {
		copy(prev, x)
		grad(x, gradBuf)
		// y = x - step * grad
		for i, xi := range x {
			gradBuf[i] = xi - step*gradBuf[i]
		}
		// x = prox_{step*g}(y)
		prox(gradBuf, step, x)

		delta := infDelta(x, prev)
		if delta < tol {
			return FbsResult{Iter: k + 1, Converged: true, FinalDelta: delta}, nil
		}
	}
	return FbsResult{Iter: maxIter, Converged: false, FinalDelta: infDelta(x, prev)}, nil
}

func fistaLoop(grad GradOp, prox ProxOp, x, gradBuf []float64, step float64, maxIter int, tol float64) (FbsResult, error) {
	n := len(x)
	xNew := make([]float64, n)
	y := make([]float64, n)
	copy(y, x)
	t := 1.0

	// Loop invariant at the start of iteration k (k >= 1):
	//   x = x_{k-1},  y = y_k,  t = t_k
	// Each iteration computes x_k = prox-grad(y_k), then
	//   t_{k+1} = (1 + sqrt(1 + 4 t_k^2)) / 2
	//   y_{k+1} = x_k + ((t_k - 1)/t_{k+1}) * (x_k - x_{k-1})
	// Then rotates  x <- x_k.
	for k := 0; k < maxIter; k++ {
		grad(y, gradBuf)
		for i, yi := range y {
			gradBuf[i] = yi - step*gradBuf[i]
		}
		prox(gradBuf, step, xNew)

		tNew := 0.5 * (1.0 + math.Sqrt(1.0+4.0*t*t))
		w := (t - 1.0) / tNew
		for i := 0; i < n; i++ {
			y[i] = xNew[i] + w*(xNew[i]-x[i])
		}

		delta := infDelta(xNew, x)
		copy(x, xNew)
		t = tNew

		if delta < tol {
			return FbsResult{Iter: k + 1, Converged: true, FinalDelta: delta}, nil
		}
	}
	return FbsResult{Iter: maxIter, Converged: false, FinalDelta: infDelta(x, xNew)}, nil
}

func infDelta(a, b []float64) float64 {
	var m float64
	for i, ai := range a {
		d := math.Abs(ai - b[i])
		if d > m {
			m = d
		}
	}
	return m
}
