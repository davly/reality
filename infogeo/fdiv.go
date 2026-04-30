package infogeo

import (
	"errors"
	"math"
)

// ErrInvalidDistribution is returned when a probability vector contains
// negative entries, NaN/Inf, has length 0, or fails to sum to 1 within
// float64 precision (1e-9 tolerance).
var ErrInvalidDistribution = errors.New("infogeo: probability vector must be non-negative and sum to 1")

// ErrLengthMismatch is returned when two probability vectors have different
// lengths.
var ErrLengthMismatch = errors.New("infogeo: probability vectors must have equal length")

// ErrInvalidParameter is returned when a divergence parameter (e.g. the
// alpha of Renyi-alpha) is outside its valid domain.
var ErrInvalidParameter = errors.New("infogeo: divergence parameter outside valid domain")

const probTol = 1e-9

// validatePair returns an error if p or q is not a valid probability vector
// or if they have unequal length.
func validatePair(p, q []float64) error {
	if len(p) == 0 {
		return ErrInvalidDistribution
	}
	if len(p) != len(q) {
		return ErrLengthMismatch
	}
	if err := validate(p); err != nil {
		return err
	}
	return validate(q)
}

func validate(p []float64) error {
	var s float64
	for _, v := range p {
		if math.IsNaN(v) || math.IsInf(v, 0) || v < 0 {
			return ErrInvalidDistribution
		}
		s += v
	}
	if math.Abs(s-1.0) > probTol {
		return ErrInvalidDistribution
	}
	return nil
}

// =========================================================================
// f-divergences over discrete probability vectors
// =========================================================================

// KL returns the Kullback-Leibler divergence KL(p || q) in nats.
//
//	KL(p || q) = sum_i p_i * log(p_i / q_i)
//
// Convention: 0 * log(0/q_i) = 0 for any q_i.  Returns +Inf if p_i > 0 and
// q_i = 0 (absolute continuity violated).
//
// Reference: Kullback & Leibler (1951). On information and sufficiency.
// Ann. Math. Stat. 22(1):79-86.
func KL(p, q []float64) (float64, error) {
	if err := validatePair(p, q); err != nil {
		return 0, err
	}
	var sum float64
	for i, pi := range p {
		if pi == 0 {
			continue
		}
		qi := q[i]
		if qi == 0 {
			return math.Inf(1), nil
		}
		sum += pi * math.Log(pi/qi)
	}
	return sum, nil
}

// ReverseKL returns KL(q || p) in nats.  Convenience wrapper for use in
// asymmetric model-comparison settings (e.g., variational inference).
func ReverseKL(p, q []float64) (float64, error) {
	return KL(q, p)
}

// JS returns the Jensen-Shannon divergence JSD(p || q) in nats:
//
//	JSD(p || q) = 0.5 KL(p || m) + 0.5 KL(q || m),  m = 0.5 (p + q)
//
// JSD is symmetric and bounded above by log(2).  Its square root is a
// metric.
//
// Reference: Lin J. (1991). Divergence measures based on the Shannon
// entropy.  IEEE Trans. Inf. Theory 37(1):145-151.
func JS(p, q []float64) (float64, error) {
	if err := validatePair(p, q); err != nil {
		return 0, err
	}
	var sum float64
	for i, pi := range p {
		qi := q[i]
		mi := 0.5 * (pi + qi)
		if mi == 0 {
			continue
		}
		if pi > 0 {
			sum += 0.5 * pi * math.Log(pi/mi)
		}
		if qi > 0 {
			sum += 0.5 * qi * math.Log(qi/mi)
		}
	}
	return sum, nil
}

// TotalVariation returns the total-variation distance:
//
//	TV(p, q) = 0.5 sum_i |p_i - q_i|
//
// Bounded in [0, 1].  TV satisfies the triangle inequality and is a true
// metric on the simplex.
func TotalVariation(p, q []float64) (float64, error) {
	if err := validatePair(p, q); err != nil {
		return 0, err
	}
	var sum float64
	for i, pi := range p {
		sum += math.Abs(pi - q[i])
	}
	return 0.5 * sum, nil
}

// Hellinger returns the Hellinger distance:
//
//	H(p, q) = sqrt(0.5 sum_i (sqrt(p_i) - sqrt(q_i))^2)
//
// Bounded in [0, 1].  H is a metric and is invariant to absolutely
// continuous reparameterisations.
//
// Reference: Hellinger E. (1909). Neue Begründung der Theorie quadratischer
// Formen von unendlichvielen Veränderlichen.  J. Reine Angew. Math. 136.
func Hellinger(p, q []float64) (float64, error) {
	if err := validatePair(p, q); err != nil {
		return 0, err
	}
	var sum float64
	for i, pi := range p {
		d := math.Sqrt(pi) - math.Sqrt(q[i])
		sum += d * d
	}
	return math.Sqrt(0.5 * sum), nil
}

// ChiSquared returns the Pearson chi-squared divergence:
//
//	chi^2(p || q) = sum_i (p_i - q_i)^2 / q_i
//
// Returns +Inf if any q_i = 0 with p_i > 0.  Used as the second-order term
// in the Pinsker-Csiszar inequality KL >= 0.5 * chi^2 - higher-order terms.
func ChiSquared(p, q []float64) (float64, error) {
	if err := validatePair(p, q); err != nil {
		return 0, err
	}
	var sum float64
	for i, pi := range p {
		qi := q[i]
		d := pi - qi
		if d == 0 {
			continue
		}
		if qi == 0 {
			return math.Inf(1), nil
		}
		sum += d * d / qi
	}
	return sum, nil
}

// Renyi returns the Renyi-alpha divergence in nats:
//
//	D_alpha(p || q) = (1/(alpha-1)) log( sum_i p_i^alpha q_i^{1-alpha} )
//
// Valid for alpha in (0, 1) U (1, infinity).  Limits:
//
//	D_0       -> -log(sum_{i: p_i > 0} q_i)            (Renyi-0)
//	D_1       -> KL(p || q)                             (Renyi-1)
//	D_2       -> log(sum_i p_i^2 / q_i)                 (Renyi-2)
//	D_infty   -> log(sup_i p_i / q_i)                   (Renyi-infty)
//
// Returns ErrInvalidParameter if alpha is exactly 1 (use KL) or non-finite.
//
// Reference: Renyi (1961).
func Renyi(p, q []float64, alpha float64) (float64, error) {
	if err := validatePair(p, q); err != nil {
		return 0, err
	}
	if math.IsNaN(alpha) || math.IsInf(alpha, 0) || alpha == 1.0 || alpha <= 0 {
		return 0, ErrInvalidParameter
	}
	var sum float64
	for i, pi := range p {
		qi := q[i]
		switch {
		case pi == 0 && qi == 0:
			continue
		case pi == 0:
			// p^alpha = 0 for alpha > 0; term is 0.
			continue
		case qi == 0:
			// p^alpha * q^{1-alpha}: q^{1-alpha} = +Inf if alpha > 1, 0 if
			// alpha < 1.  When alpha > 1 and p_i > 0, the divergence is
			// +Inf; when alpha < 1 the term is 0.
			if alpha > 1 {
				return math.Inf(1), nil
			}
			continue
		}
		sum += math.Pow(pi, alpha) * math.Pow(qi, 1.0-alpha)
	}
	if sum <= 0 {
		return math.Inf(1), nil
	}
	return math.Log(sum) / (alpha - 1.0), nil
}
