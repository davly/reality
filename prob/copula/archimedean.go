package copula

import (
	"errors"
	"math"
)

// Archimedean copula closures — Clayton + Gumbel families.
//
// Both are bivariate; both have closed-form CDFs (no quadrature required)
// and closed-form theta-from-Kendall-tau inversions, making them the
// cheapest joint-risk additions to ship.
//
// Compared to the elliptical Gaussian + Student-t copulas already in this
// package:
//
//   - **Clayton** has *lower* tail dependence (positive λ_L) and *zero*
//     upper tail dependence — models crisis clustering without
//     symmetrically inflating bull-regime co-movement.
//   - **Gumbel** is the mirror: *upper* tail dependence, *zero* lower —
//     useful for modelling joint upside (e.g. correlated rallies in
//     positively-correlated assets).
//
// Both are members of the Archimedean family: C(u, v) = ψ⁻¹(ψ(u) + ψ(v))
// for a generator ψ. Closed-form CDF means we ship as scalar functions,
// not the integral-form CopulaCDF closure used by Gaussian/Student-t.
//
// References:
//   - Clayton, D.G. (1978). "A model for association in bivariate life
//     tables and its application in epidemiological studies of familial
//     tendency in chronic disease incidence." Biometrika 65(1): 141-151.
//   - Gumbel, E.J. (1960). "Distributions des valeurs extrêmes en
//     plusieurs dimensions." Publications de l'Institut de Statistique
//     de l'Université de Paris 9: 171-173.
//   - Embrechts, P., Lindskog, F., McNeil, A.J. (2003). "Modelling
//     dependence with copulas and applications to risk management."
//     Handbook of Heavy-Tailed Distributions in Finance §8.

// ArchimedeanFamily is the discriminator for ThetaFromKendallTau.
type ArchimedeanFamily int

const (
	// FamilyClayton — lower tail dependence λ_L = 2^(-1/θ), upper = 0.
	FamilyClayton ArchimedeanFamily = iota
	// FamilyGumbel — upper tail dependence λ_U = 2 - 2^(1/θ), lower = 0.
	FamilyGumbel
)

// ErrArchimedeanThetaOutOfRange is returned by ThetaFromKendallTau when
// the requested Kendall tau falls outside the family's admissible range.
var ErrArchimedeanThetaOutOfRange = errors.New(
	"copula: Archimedean theta out of admissible range")

// ErrArchimedeanInvalidTheta is returned by the CDF closures when
// constructed with an invalid theta parameter.
var ErrArchimedeanInvalidTheta = errors.New(
	"copula: Archimedean copula theta must be in the admissible range")

// ClaytonCopulaCDFFn returns the bivariate Clayton copula CDF closure.
//
//	C(u, v; θ) = ( u^(-θ) + v^(-θ) - 1 )^(-1/θ),  θ > 0
//
// As θ → 0 the copula approaches the independence copula Π(u,v) = uv.
// As θ → ∞ it approaches the comonotone copula M(u,v) = min(u,v).
//
// Lower tail dependence: λ_L = 2^(-1/θ) > 0 for any θ > 0.
// Upper tail dependence: λ_U = 0.
//
// theta must be > 0 (returns ErrArchimedeanInvalidTheta otherwise).
func ClaytonCopulaCDFFn(theta float64) (func(u, v float64) float64, error) {
	if !(theta > 0) || math.IsNaN(theta) || math.IsInf(theta, 0) {
		return nil, ErrArchimedeanInvalidTheta
	}
	return func(u, v float64) float64 {
		// Clamp to (0,1) — formula is undefined at the boundaries.
		if u <= 0 || v <= 0 {
			return 0
		}
		if u >= 1 {
			return v
		}
		if v >= 1 {
			return u
		}
		// (u^-θ + v^-θ - 1)^(-1/θ)
		s := math.Pow(u, -theta) + math.Pow(v, -theta) - 1.0
		if s <= 0 {
			return 0
		}
		return math.Pow(s, -1.0/theta)
	}, nil
}

// GumbelCopulaCDFFn returns the bivariate Gumbel copula CDF closure.
//
//	C(u, v; θ) = exp( -( (-ln u)^θ + (-ln v)^θ )^(1/θ) ),  θ ≥ 1
//
// At θ = 1 it reduces to the independence copula. As θ → ∞ it approaches
// the comonotone copula M(u, v) = min(u, v).
//
// Upper tail dependence: λ_U = 2 - 2^(1/θ) > 0 for any θ > 1.
// Lower tail dependence: λ_L = 0.
//
// theta must be ≥ 1 (returns ErrArchimedeanInvalidTheta otherwise).
func GumbelCopulaCDFFn(theta float64) (func(u, v float64) float64, error) {
	if math.IsNaN(theta) || math.IsInf(theta, 0) || theta < 1 {
		return nil, ErrArchimedeanInvalidTheta
	}
	return func(u, v float64) float64 {
		if u <= 0 || v <= 0 {
			return 0
		}
		if u >= 1 {
			return v
		}
		if v >= 1 {
			return u
		}
		// exp(-( (-ln u)^θ + (-ln v)^θ )^(1/θ))
		nu := -math.Log(u)
		nv := -math.Log(v)
		s := math.Pow(nu, theta) + math.Pow(nv, theta)
		return math.Exp(-math.Pow(s, 1.0/theta))
	}, nil
}

// ThetaFromKendallTau inverts Kendall's tau to the Archimedean theta
// parameter for the named family.
//
//   - Clayton: θ = 2τ / (1 - τ),  admissible τ ∈ (0, 1).
//   - Gumbel:  θ = 1 / (1 - τ),   admissible τ ∈ [0, 1).
//
// Returns ErrArchimedeanThetaOutOfRange when tau is outside the family's
// admissible interval (e.g. negative tau for Clayton, since the closed-form
// inversion is only defined for positive concordance under both families).
//
// References:
//   - Genest, C., MacKay, J. (1986). "Copules archimédiennes et familles
//     de lois bidimensionnelles dont les marges sont données." Canadian
//     Journal of Statistics 14(2): 145-159.
func ThetaFromKendallTau(tau float64, family ArchimedeanFamily) (float64, error) {
	if math.IsNaN(tau) {
		return 0, ErrArchimedeanThetaOutOfRange
	}
	switch family {
	case FamilyClayton:
		// θ = 2τ / (1 - τ); admissible τ ∈ (0, 1).
		if tau <= 0 || tau >= 1 {
			return 0, ErrArchimedeanThetaOutOfRange
		}
		return 2.0 * tau / (1.0 - tau), nil
	case FamilyGumbel:
		// θ = 1 / (1 - τ); admissible τ ∈ [0, 1). At τ = 0 → θ = 1
		// (independence copula).
		if tau < 0 || tau >= 1 {
			return 0, ErrArchimedeanThetaOutOfRange
		}
		return 1.0 / (1.0 - tau), nil
	default:
		return 0, ErrArchimedeanThetaOutOfRange
	}
}

// ClaytonLowerTailDependence returns λ_L = 2^(-1/θ) for the Clayton
// copula. Returns 0 for theta ≤ 0.
func ClaytonLowerTailDependence(theta float64) float64 {
	if theta <= 0 || math.IsNaN(theta) || math.IsInf(theta, 0) {
		return 0
	}
	return math.Pow(2.0, -1.0/theta)
}

// GumbelUpperTailDependence returns λ_U = 2 - 2^(1/θ) for the Gumbel
// copula. Returns 0 for theta < 1.
func GumbelUpperTailDependence(theta float64) float64 {
	if math.IsNaN(theta) || math.IsInf(theta, 0) || theta < 1 {
		return 0
	}
	return 2.0 - math.Pow(2.0, 1.0/theta)
}
