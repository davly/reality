package copula

import (
	"errors"
	"math"
)

// h-functions for Archimedean copulas — load-bearing primitive for vine
// copula construction. The h-function is the conditional CDF:
//
//	h(u | v; θ) = ∂C(u, v; θ) / ∂v
//
// which equals P(U ≤ u | V = v) under the copula C(·, ·; θ). Vine
// copulas use h-functions to convert observations into pseudo-observations
// for higher-tree fitting (Aas, Czado et al. 2009).
//
// Both Clayton and Gumbel admit closed-form h-functions; no quadrature
// required. This is the cheap-to-ship path to vine copula support.
//
// References:
//   - Aas, K., Czado, C., Frigessi, A., Bakken, H. (2009). "Pair-copula
//     constructions of multiple dependence." Insurance: Mathematics and
//     Economics 44: 182-198. (h-function definitions: §2.4.)
//   - Joe, H. (1997). "Multivariate Models and Multivariate Dependence
//     Concepts." Chapman & Hall §5.1.

// ErrCopulaHFnInvalidTheta is returned by the h-function constructors
// when the supplied theta is outside the family's admissible range.
var ErrCopulaHFnInvalidTheta = errors.New(
	"copula: h-function theta must be in the admissible range")

// ClaytonHFn returns the conditional CDF h(u | v; θ) for the Clayton
// copula:
//
//	h(u | v; θ) = v^(-θ-1) * ( u^(-θ) + v^(-θ) - 1 )^(-(1+θ)/θ)
//
// Defined for θ > 0; returns ErrCopulaHFnInvalidTheta otherwise.
//
// Boundary behaviour: u, v ∈ (0, 1). The closure clamps u, v to that
// open interval to avoid log-of-zero / divide-by-zero (consistent with
// ClaytonCopulaCDFFn).
func ClaytonHFn(theta float64) (func(u, v float64) float64, error) {
	if !(theta > 0) || math.IsNaN(theta) || math.IsInf(theta, 0) {
		return nil, ErrCopulaHFnInvalidTheta
	}
	return func(u, v float64) float64 {
		// Clamp to (0, 1) — closed-form is undefined at the boundaries.
		if u <= 0 {
			return 0
		}
		if u >= 1 {
			return 1
		}
		if v <= 0 {
			return 0
		}
		if v >= 1 {
			return u
		}

		// h(u | v) = v^(-θ-1) · ( u^(-θ) + v^(-θ) - 1 )^(-(1+θ)/θ)
		uNegT := math.Pow(u, -theta)
		vNegT := math.Pow(v, -theta)
		sum := uNegT + vNegT - 1.0
		if sum <= 0 {
			// Numerical floor; the conditional CDF is bounded in [0, 1].
			return 0
		}
		exponent := -(1.0 + theta) / theta
		return math.Pow(v, -theta-1.0) * math.Pow(sum, exponent)
	}, nil
}

// GumbelHFn returns the conditional CDF h(u | v; θ) for the Gumbel
// copula:
//
//	h(u | v; θ) = C(u, v; θ) / v · ( -ln v )^(θ-1) · ( (-ln u)^θ + (-ln v)^θ )^(1/θ - 1)
//
// where C(u, v; θ) = exp( -( (-ln u)^θ + (-ln v)^θ )^(1/θ) ) is the
// Gumbel copula CDF.
//
// Defined for θ ≥ 1; returns ErrCopulaHFnInvalidTheta otherwise.
// At θ = 1 the Gumbel copula reduces to the independence copula and
// h(u | v) = u (verified by the closure).
func GumbelHFn(theta float64) (func(u, v float64) float64, error) {
	if math.IsNaN(theta) || math.IsInf(theta, 0) || theta < 1 {
		return nil, ErrCopulaHFnInvalidTheta
	}
	return func(u, v float64) float64 {
		if u <= 0 {
			return 0
		}
		if u >= 1 {
			return 1
		}
		if v <= 0 {
			return 0
		}
		if v >= 1 {
			return u
		}
		if theta == 1.0 {
			// Independence copula corner case — h(u | v) = u.
			return u
		}

		nu := -math.Log(u)
		nv := -math.Log(v)
		nuT := math.Pow(nu, theta)
		nvT := math.Pow(nv, theta)
		sum := nuT + nvT
		// C(u, v; θ) closed-form
		cuv := math.Exp(-math.Pow(sum, 1.0/theta))
		// h(u | v) = C(u, v) / v · (-ln v)^(θ-1) · sum^(1/θ - 1)
		return (cuv / v) * math.Pow(nv, theta-1.0) * math.Pow(sum, 1.0/theta-1.0)
	}, nil
}

// HFnForFamily dispatches to the appropriate h-function constructor by
// family enum. Convenience wrapper for vine code that selects family at
// runtime.
func HFnForFamily(family ArchimedeanFamily, theta float64) (func(u, v float64) float64, error) {
	switch family {
	case FamilyClayton:
		return ClaytonHFn(theta)
	case FamilyGumbel:
		return GumbelHFn(theta)
	default:
		return nil, ErrCopulaHFnInvalidTheta
	}
}
