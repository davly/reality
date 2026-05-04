package copula

import (
	"errors"
	"math"
)

// Bivariate copula PDF closures — Clayton + Gumbel families. The PDF
// c(u, v; θ) is the density of the copula CDF C(u, v; θ); for vine
// copula log-likelihood evaluation we need log c(u, v; θ) at every
// pair-copula edge.
//
// Closed-form expressions:
//
//   Clayton:
//     c(u, v; θ) = (1 + θ) · (uv)^(-1-θ) · ( u^(-θ) + v^(-θ) - 1 )^(-2 - 1/θ)
//
//   Gumbel:
//     Let Lu = -ln u, Lv = -ln v, T = Lu^θ + Lv^θ.
//     C(u, v; θ) = exp(-T^(1/θ))
//     c(u, v; θ) = C(u,v;θ) / (uv)
//                · ( T^(1/θ - 2) ) · ( Lu · Lv )^(θ - 1)
//                · ( T^(1/θ) + θ - 1 )
//
// References:
//   - Joe, H. (1997). "Multivariate Models and Multivariate Dependence
//     Concepts." Chapman & Hall §5.1 (Archimedean PDFs).
//   - Aas, K., Czado, C., Frigessi, A., Bakken, H. (2009). "Pair-copula
//     constructions of multiple dependence." Insurance: Mathematics and
//     Economics 44: 182-198. (Vine likelihood § 2.5.)

// ErrCopulaPDFInvalidTheta is returned by the PDF constructors when the
// supplied theta is outside the family's admissible range. Mirrors the
// h-function error sentinel.
var ErrCopulaPDFInvalidTheta = errors.New(
	"copula: PDF theta must be in the admissible range")

// ClaytonPDFFn returns the bivariate Clayton copula density closure.
// Defined for θ > 0.
//
// Formula (Joe 1997 §5.1):
//
//   c(u, v; θ) = (1 + θ) · (uv)^(-1-θ) · ( u^(-θ) + v^(-θ) - 1 )^(-2 - 1/θ)
//
// At the boundaries (u or v ∈ {0, 1}) the density is degenerate; the
// closure clamps to (0, 1) and returns 0 outside.
func ClaytonPDFFn(theta float64) (func(u, v float64) float64, error) {
	if !(theta > 0) || math.IsNaN(theta) || math.IsInf(theta, 0) {
		return nil, ErrCopulaPDFInvalidTheta
	}
	return func(u, v float64) float64 {
		if u <= 0 || u >= 1 || v <= 0 || v >= 1 {
			return 0
		}
		uNegT := math.Pow(u, -theta)
		vNegT := math.Pow(v, -theta)
		sum := uNegT + vNegT - 1.0
		if sum <= 0 {
			// Numerical floor; the density is non-negative everywhere
			// the copula is defined.
			return 0
		}
		exp := -2.0 - 1.0/theta
		return (1.0 + theta) *
			math.Pow(u*v, -1.0-theta) *
			math.Pow(sum, exp)
	}, nil
}

// ClaytonLogPDFFn is the log-density variant — preferred for vine
// log-likelihood evaluation because it accumulates additively over edges
// rather than multiplicatively, avoiding underflow when N is large.
func ClaytonLogPDFFn(theta float64) (func(u, v float64) float64, error) {
	if !(theta > 0) || math.IsNaN(theta) || math.IsInf(theta, 0) {
		return nil, ErrCopulaPDFInvalidTheta
	}
	return func(u, v float64) float64 {
		if u <= 0 || u >= 1 || v <= 0 || v >= 1 {
			return math.Inf(-1)
		}
		uNegT := math.Pow(u, -theta)
		vNegT := math.Pow(v, -theta)
		sum := uNegT + vNegT - 1.0
		if sum <= 0 {
			return math.Inf(-1)
		}
		// log c = log(1+θ) + (-1-θ)·log(uv) + (-2 - 1/θ)·log(sum)
		return math.Log1p(theta) +
			(-1.0-theta)*(math.Log(u)+math.Log(v)) +
			(-2.0-1.0/theta)*math.Log(sum)
	}, nil
}

// GumbelPDFFn returns the bivariate Gumbel copula density closure.
// Defined for θ ≥ 1. At θ = 1 the density reduces to 1 (independence).
//
// Formula (Joe 1997 §5.1):
//
//   Lu = -ln u; Lv = -ln v; T = Lu^θ + Lv^θ; C = exp(-T^(1/θ))
//   c(u, v; θ) = C / (uv) · T^(2/θ - 2) · (Lu·Lv)^(θ-1) · (T^(1/θ) + θ - 1)
func GumbelPDFFn(theta float64) (func(u, v float64) float64, error) {
	if math.IsNaN(theta) || math.IsInf(theta, 0) || theta < 1 {
		return nil, ErrCopulaPDFInvalidTheta
	}
	return func(u, v float64) float64 {
		if u <= 0 || u >= 1 || v <= 0 || v >= 1 {
			return 0
		}
		if theta == 1.0 {
			// Independence copula c(u, v) = 1.
			return 1.0
		}
		Lu := -math.Log(u)
		Lv := -math.Log(v)
		T := math.Pow(Lu, theta) + math.Pow(Lv, theta)
		if T <= 0 {
			return 0
		}
		t1 := math.Pow(T, 1.0/theta)
		C := math.Exp(-t1)
		factor := math.Pow(T, 1.0/theta-2.0) *
			math.Pow(Lu*Lv, theta-1.0) *
			(t1 + theta - 1.0)
		return (C / (u * v)) * factor
	}, nil
}

// GumbelLogPDFFn is the log-density variant — see ClaytonLogPDFFn for
// rationale.
func GumbelLogPDFFn(theta float64) (func(u, v float64) float64, error) {
	if math.IsNaN(theta) || math.IsInf(theta, 0) || theta < 1 {
		return nil, ErrCopulaPDFInvalidTheta
	}
	return func(u, v float64) float64 {
		if u <= 0 || u >= 1 || v <= 0 || v >= 1 {
			return math.Inf(-1)
		}
		if theta == 1.0 {
			return 0.0
		}
		Lu := -math.Log(u)
		Lv := -math.Log(v)
		T := math.Pow(Lu, theta) + math.Pow(Lv, theta)
		if T <= 0 {
			return math.Inf(-1)
		}
		t1 := math.Pow(T, 1.0/theta)
		// log c = log C - log(uv) + (1/θ - 2)·log T
		//       + (θ - 1)·log(Lu·Lv) + log(T^(1/θ) + θ - 1)
		// log C = -t1
		return -t1 -
			math.Log(u) - math.Log(v) +
			(1.0/theta-2.0)*math.Log(T) +
			(theta-1.0)*(math.Log(Lu)+math.Log(Lv)) +
			math.Log(t1+theta-1.0)
	}, nil
}

// LogPDFFnForFamily dispatches to the appropriate log-density constructor.
func LogPDFFnForFamily(family ArchimedeanFamily, theta float64) (func(u, v float64) float64, error) {
	switch family {
	case FamilyClayton:
		return ClaytonLogPDFFn(theta)
	case FamilyGumbel:
		return GumbelLogPDFFn(theta)
	default:
		return nil, ErrCopulaPDFInvalidTheta
	}
}
