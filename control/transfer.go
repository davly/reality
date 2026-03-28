package control

import "math/cmplx"

// TransferFunction represents a continuous-time linear transfer function H(s)
// as a ratio of polynomials in the Laplace variable s:
//
//	H(s) = N(s) / D(s)
//	     = (num[0]*s^n + num[1]*s^(n-1) + ... + num[n]) /
//	       (den[0]*s^m + den[1]*s^(m-1) + ... + den[m])
//
// Coefficients are ordered highest-degree first (descending powers of s).
// For example, the transfer function H(s) = 1 / (s + 1) is represented as:
//
//	Numerator:   []float64{1}       // constant 1
//	Denominator: []float64{1, 1}    // s + 1
//
// The denominator must not be empty. The numerator must not be empty.
//
// Consumers: Pulse (control loop stability analysis), Sentinel (filter design).
type TransferFunction struct {
	Numerator   []float64
	Denominator []float64
}

// Evaluate computes H(s) at the given complex frequency s.
//
// It evaluates both numerator and denominator polynomials at s using Horner's
// method, then returns N(s) / D(s).
//
// Panics if Numerator or Denominator is empty.
// If D(s) = 0 (pole at s), returns complex infinity.
// Zero heap allocations.
//
// Definition: H(s) = sum(num[i] * s^(n-i)) / sum(den[j] * s^(m-j))
func (tf *TransferFunction) Evaluate(s complex128) complex128 {
	if len(tf.Numerator) == 0 {
		panic("control.TransferFunction.Evaluate: Numerator must not be empty")
	}
	if len(tf.Denominator) == 0 {
		panic("control.TransferFunction.Evaluate: Denominator must not be empty")
	}

	num := evalPoly(tf.Numerator, s)
	den := evalPoly(tf.Denominator, s)

	return num / den
}

// evalPoly evaluates a polynomial with the given coefficients at complex point s
// using Horner's method. Coefficients are ordered highest-degree first.
//
// p(s) = coeffs[0]*s^(n-1) + coeffs[1]*s^(n-2) + ... + coeffs[n-1]
func evalPoly(coeffs []float64, s complex128) complex128 {
	result := complex(coeffs[0], 0)
	for i := 1; i < len(coeffs); i++ {
		result = result*s + complex(coeffs[i], 0)
	}
	return result
}

// Poles returns the roots of the denominator polynomial. These are the values
// of s where H(s) is undefined (D(s) = 0).
//
// For a first-order denominator (as + b), the pole is at s = -b/a.
// For a second-order denominator (as^2 + bs + c), the poles are found using
// the quadratic formula.
// For higher-order denominators, the Durand-Kerner method is used.
//
// Panics if Denominator is empty or has length 1 (constant — no poles).
//
// Consumers: Pulse (stability check), Sentinel (filter pole placement).
func (tf *TransferFunction) Poles() []complex128 {
	d := tf.Denominator
	if len(d) == 0 {
		panic("control.TransferFunction.Poles: Denominator must not be empty")
	}
	if len(d) == 1 {
		// Constant denominator has no poles.
		return nil
	}

	degree := len(d) - 1

	// Normalize to monic polynomial (leading coefficient = 1).
	lead := d[0]
	if lead == 0 {
		panic("control.TransferFunction.Poles: leading coefficient must not be zero")
	}
	monic := make([]float64, len(d))
	for i := range d {
		monic[i] = d[i] / lead
	}

	switch degree {
	case 1:
		// s + monic[1] = 0 => s = -monic[1]
		return []complex128{complex(-monic[1], 0)}
	case 2:
		// s^2 + bs + c = 0
		b := monic[1]
		c := monic[2]
		disc := complex(b*b-4*c, 0)
		sqrtDisc := cmplx.Sqrt(disc)
		return []complex128{
			(-complex(b, 0) + sqrtDisc) / 2,
			(-complex(b, 0) - sqrtDisc) / 2,
		}
	default:
		return durandKerner(monic, degree)
	}
}

// durandKerner finds all roots of a monic polynomial using the Durand-Kerner
// (Weierstrass) iteration method. This is a simultaneous root-finding algorithm
// that converges for most polynomials.
//
// The monic polynomial has coefficients [1, a_{n-1}, ..., a_0] in descending
// order. Returns n roots where n = degree.
//
// Reference: Durand (1960), Kerner (1966).
func durandKerner(monic []float64, degree int) []complex128 {
	const maxIter = 1000
	const tol = 1e-12

	// Initial guesses: distribute on a circle of radius r.
	// r is chosen based on Cauchy's bound: max(1, sum|a_i|).
	r := 1.0
	for i := 1; i <= degree; i++ {
		a := monic[i]
		if a < 0 {
			a = -a
		}
		if a > r {
			r = a
		}
	}

	roots := make([]complex128, degree)
	for i := 0; i < degree; i++ {
		// Spread initial guesses asymmetrically to avoid symmetry traps.
		angle := 2.0*3.141592653589793*float64(i)/float64(degree) + 0.4
		roots[i] = complex(r*realCos(angle), r*realSin(angle))
	}

	for iter := 0; iter < maxIter; iter++ {
		maxDelta := 0.0
		for i := 0; i < degree; i++ {
			// Evaluate polynomial at roots[i].
			val := complex(1.0, 0)
			s := roots[i]
			for j := 0; j < len(monic); j++ {
				if j == 0 {
					val = complex(monic[0], 0)
				} else {
					val = val*s + complex(monic[j], 0)
				}
			}

			// Product of (roots[i] - roots[j]) for j != i.
			denom := complex(1.0, 0)
			for j := 0; j < degree; j++ {
				if j != i {
					denom *= roots[i] - roots[j]
				}
			}

			delta := val / denom
			roots[i] -= delta

			d := cmplx.Abs(delta)
			if d > maxDelta {
				maxDelta = d
			}
		}

		if maxDelta < tol {
			break
		}
	}

	return roots
}

// realCos and realSin avoid importing math for just cos/sin, keeping the
// dependency list minimal. These use the Taylor series which converges
// well for the small angles used in initial guess generation.
// However, since we need full-circle accuracy, we use the standard
// reduction + Taylor approach.

func realCos(x float64) float64 {
	// Reduce to [0, 2pi].
	const twoPi = 2 * 3.141592653589793
	for x < 0 {
		x += twoPi
	}
	for x >= twoPi {
		x -= twoPi
	}
	// Taylor series for cos(x), 12 terms for good accuracy.
	sum := 1.0
	term := 1.0
	for n := 1; n <= 12; n++ {
		term *= -x * x / float64((2*n-1)*(2*n))
		sum += term
	}
	return sum
}

func realSin(x float64) float64 {
	const twoPi = 2 * 3.141592653589793
	for x < 0 {
		x += twoPi
	}
	for x >= twoPi {
		x -= twoPi
	}
	sum := x
	term := x
	for n := 1; n <= 12; n++ {
		term *= -x * x / float64((2*n)*(2*n+1))
		sum += term
	}
	return sum
}

// IsStable returns true if all poles of the transfer function have strictly
// negative real parts (the system is BIBO stable).
//
// A continuous-time LTI system is stable if and only if all poles of its
// transfer function lie in the open left half of the complex plane
// (Re(pole) < 0 for all poles).
//
// A constant denominator (degree 0) is considered stable (no poles).
//
// Reference: Ogata, Modern Control Engineering, Chapter 5.
//
// Consumers: Pulse (verify feedback loop stability), Sentinel (filter safety).
func (tf *TransferFunction) IsStable() bool {
	if len(tf.Denominator) <= 1 {
		// No poles — stable by convention.
		return true
	}

	poles := tf.Poles()
	for _, p := range poles {
		if real(p) >= 0 {
			return false
		}
	}
	return true
}
