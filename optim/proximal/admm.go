package proximal

import (
	"errors"
	"math"
)

// AdmmConfig configures consensus ADMM.
type AdmmConfig struct {
	// Rho is the augmented-Lagrangian penalty parameter (positive). Larger
	// rho enforces the consensus constraint x = z more aggressively but can
	// slow primal convergence. A safe default is 1.0 for unit-scale data.
	Rho float64

	// MaxIter caps the outer loop. If 0 defaults to 1000.
	MaxIter int

	// AbsTol terminates when both primal residual ||x-z||_inf and dual
	// residual rho*||z-z_prev||_inf fall below AbsTol. If <= 0 defaults to
	// 1e-7.
	AbsTol float64
}

// AdmmResult reports the outcome of an ADMM run.
type AdmmResult struct {
	Iter         int
	Converged    bool
	PrimalResid  float64 // ||x - z||_inf
	DualResid    float64 // rho * ||z - z_prev||_inf
}

// Admm solves
//
//	minimize_x  f(x) + g(z)
//	subject to  x = z
//
// the consensus form of ADMM (a.k.a. Douglas-Rachford on f + g). proxF and
// proxG are the proximal operators of f and g respectively. The primal
// variable x is updated in place; z and u must be buffers of length len(x)
// (each will be overwritten).
//
// Scaled-form iteration (Boyd 2011 §3.1.1):
//
//	x^{k+1} = prox_{f/rho}(z^k - u^k)
//	z^{k+1} = prox_{g/rho}(x^{k+1} + u^k)
//	u^{k+1} = u^k + x^{k+1} - z^{k+1}
//
// Convergence: linear under standard convexity + Slater conditions.
//
// Reference: Boyd S. et al. (2011). Distributed Optimization and Statistical
// Learning via the Alternating Direction Method of Multipliers. Found.
// Trends Mach. Learn. 3(1):1-122. §3.
func Admm(proxF, proxG ProxOp, x, z, u []float64, cfg AdmmConfig) (AdmmResult, error) {
	if cfg.Rho <= 0 || math.IsInf(cfg.Rho, 0) || math.IsNaN(cfg.Rho) {
		return AdmmResult{}, errors.New("proximal: AdmmConfig.Rho must be positive and finite")
	}
	if proxF == nil || proxG == nil {
		return AdmmResult{}, errors.New("proximal: proxF and proxG must be non-nil")
	}
	n := len(x)
	if len(z) != n || len(u) != n {
		return AdmmResult{}, errors.New("proximal: x, z, u must have equal length")
	}
	maxIter := cfg.MaxIter
	if maxIter == 0 {
		maxIter = 1000
	}
	tol := cfg.AbsTol
	if !(tol > 0) {
		tol = 1e-7
	}

	gamma := 1.0 / cfg.Rho
	tmp := make([]float64, n)
	zPrev := make([]float64, n)

	for k := 0; k < maxIter; k++ {
		// x = prox_{f/rho}(z - u)
		for i := range tmp {
			tmp[i] = z[i] - u[i]
		}
		proxF(tmp, gamma, x)

		copy(zPrev, z)

		// z = prox_{g/rho}(x + u)
		for i := range tmp {
			tmp[i] = x[i] + u[i]
		}
		proxG(tmp, gamma, z)

		// u = u + (x - z)
		var primal, dual float64
		for i := range u {
			u[i] += x[i] - z[i]
			if d := math.Abs(x[i] - z[i]); d > primal {
				primal = d
			}
			if d := cfg.Rho * math.Abs(z[i]-zPrev[i]); d > dual {
				dual = d
			}
		}

		if primal < tol && dual < tol {
			return AdmmResult{Iter: k + 1, Converged: true, PrimalResid: primal, DualResid: dual}, nil
		}
	}

	// Final residuals.
	var primal, dual float64
	for i := range x {
		if d := math.Abs(x[i] - z[i]); d > primal {
			primal = d
		}
		if d := cfg.Rho * math.Abs(z[i]-zPrev[i]); d > dual {
			dual = d
		}
	}
	return AdmmResult{Iter: maxIter, Converged: false, PrimalResid: primal, DualResid: dual}, nil
}
