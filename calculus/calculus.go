// Package calculus provides numerical differentiation and integration
// primitives. Every function is pure, deterministic, and uses only the Go
// standard library. No external dependencies.
//
// Consumers:
//   - Oracle:     sensitivity analysis (numerical derivatives)
//   - Causal:     counterfactual gradient estimation (numerical gradient)
//   - RubberDuck: option pricing integrals (Gauss-Legendre, Monte Carlo)
//   - Pistachio:  physics simulation integrals (trapezoidal, Simpson's)
//   - Horizon:    trend derivative estimation (numerical derivative)
package calculus

import "math"

// ---------------------------------------------------------------------------
// Numerical Differentiation
//
// Central difference approximation for first derivatives and gradients.
// These methods are O(h^2) accurate for smooth functions.
// ---------------------------------------------------------------------------

// NumericalDerivative approximates the first derivative of f at x using the
// central difference formula:
//
//	f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
//
// Valid range: h > 0 (too small → cancellation error; too large → truncation error).
// Recommended h ≈ cbrt(machineEpsilon) * max(1, |x|) ≈ 1e-5 for float64.
// Precision: O(h^2) truncation error + O(eps/h) roundoff error.
// Reference: Burden & Faires, Numerical Analysis, Chapter 4.
func NumericalDerivative(f func(float64) float64, x, h float64) float64 {
	return (f(x+h) - f(x-h)) / (2 * h)
}

// NumericalGradient approximates the gradient of f at x using central
// differences. Each partial derivative is computed independently by
// perturbing one coordinate at a time.
//
// The gradient is written into out, which must have the same length as x.
// No allocations are performed (caller provides the output buffer).
//
//	∂f/∂x_i ≈ (f(x + h*e_i) - f(x - h*e_i)) / (2h)
//
// Valid range: h > 0, len(out) == len(x)
// Precision: O(h^2) per component
// Reference: Nocedal & Wright, Numerical Optimization, Chapter 8.
func NumericalGradient(f func([]float64) float64, x []float64, h float64, out []float64) {
	n := len(x)
	// We need to perturb x in-place and restore it to avoid allocation.
	// This is safe because we only modify one element at a time.
	for i := 0; i < n; i++ {
		orig := x[i]

		x[i] = orig + h
		fPlus := f(x)

		x[i] = orig - h
		fMinus := f(x)

		out[i] = (fPlus - fMinus) / (2 * h)

		x[i] = orig // restore
	}
}

// ---------------------------------------------------------------------------
// Numerical Integration (Quadrature)
//
// Classical quadrature rules for approximating definite integrals. These
// span from simple (trapezoidal) to high-accuracy (Gauss-Legendre) to
// multi-dimensional (Monte Carlo).
// ---------------------------------------------------------------------------

// TrapezoidalRule approximates the definite integral of f over [a, b] using
// the composite trapezoidal rule with n subintervals.
//
// Formula:
//
//	∫_a^b f(x) dx ≈ h/2 * [f(a) + 2*f(a+h) + 2*f(a+2h) + ... + 2*f(b-h) + f(b)]
//
// where h = (b-a)/n.
//
// Valid range: n >= 1, a < b
// Precision: O(h^2) — error proportional to h^2 * max|f''|
// Reference: Burden & Faires, Numerical Analysis, Chapter 4.
func TrapezoidalRule(f func(float64) float64, a, b float64, n int) float64 {
	if n < 1 {
		n = 1
	}
	h := (b - a) / float64(n)
	sum := 0.5 * (f(a) + f(b))
	for i := 1; i < n; i++ {
		sum += f(a + float64(i)*h)
	}
	return sum * h
}

// SimpsonsRule approximates the definite integral of f over [a, b] using
// the composite Simpson's 1/3 rule with n subintervals.
//
// n must be even; if odd, it is incremented by 1. If n < 2, it is set to 2.
//
// Formula:
//
//	∫_a^b f(x) dx ≈ h/3 * [f(a) + 4*f(a+h) + 2*f(a+2h) + 4*f(a+3h) + ... + f(b)]
//
// where h = (b-a)/n.
//
// Valid range: n >= 2 (even), a < b
// Precision: O(h^4) — error proportional to h^4 * max|f⁴|
// Reference: Burden & Faires, Numerical Analysis, Chapter 4.
func SimpsonsRule(f func(float64) float64, a, b float64, n int) float64 {
	if n < 2 {
		n = 2
	}
	if n%2 != 0 {
		n++
	}
	h := (b - a) / float64(n)
	sum := f(a) + f(b)
	for i := 1; i < n; i++ {
		x := a + float64(i)*h
		if i%2 == 0 {
			sum += 2 * f(x)
		} else {
			sum += 4 * f(x)
		}
	}
	return sum * h / 3
}

// GaussLegendre approximates the definite integral of f over [a, b] using
// Gauss-Legendre quadrature with the specified number of points.
//
// Supported point counts: 2, 3, 4, 5. Other values are clamped to the
// nearest supported value.
//
// Gauss-Legendre quadrature with n points is exact for polynomials of
// degree 2n-1 or less, making it far more accurate than Newton-Cotes rules
// of comparable evaluation count.
//
// The rule transforms [a, b] to [-1, 1] via x = ((b-a)*t + (a+b))/2.
//
// Valid range: a < b, points in {2, 3, 4, 5}
// Precision: exact for polynomials of degree <= 2*points - 1
// Reference: Abramowitz & Stegun, Table 25.4;
//
//	Press et al., Numerical Recipes, Chapter 4.
func GaussLegendre(f func(float64) float64, a, b float64, points int) float64 {
	// Gauss-Legendre nodes and weights on [-1, 1].
	type nw struct {
		nodes   []float64
		weights []float64
	}

	// Precomputed nodes and weights for 2–5 point rules.
	rules := map[int]nw{
		2: {
			nodes:   []float64{-1.0 / math.Sqrt(3), 1.0 / math.Sqrt(3)},
			weights: []float64{1.0, 1.0},
		},
		3: {
			nodes:   []float64{-math.Sqrt(3.0 / 5.0), 0, math.Sqrt(3.0 / 5.0)},
			weights: []float64{5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0},
		},
		4: {
			nodes: []float64{
				-math.Sqrt((3.0 + 2.0*math.Sqrt(6.0/5.0)) / 7.0),
				-math.Sqrt((3.0 - 2.0*math.Sqrt(6.0/5.0)) / 7.0),
				math.Sqrt((3.0 - 2.0*math.Sqrt(6.0/5.0)) / 7.0),
				math.Sqrt((3.0 + 2.0*math.Sqrt(6.0/5.0)) / 7.0),
			},
			weights: []float64{
				(18.0 - math.Sqrt(30.0)) / 36.0,
				(18.0 + math.Sqrt(30.0)) / 36.0,
				(18.0 + math.Sqrt(30.0)) / 36.0,
				(18.0 - math.Sqrt(30.0)) / 36.0,
			},
		},
		5: {
			nodes: []float64{
				-math.Sqrt(5.0+2.0*math.Sqrt(10.0/7.0)) / 3.0,
				-math.Sqrt(5.0-2.0*math.Sqrt(10.0/7.0)) / 3.0,
				0,
				math.Sqrt(5.0-2.0*math.Sqrt(10.0/7.0)) / 3.0,
				math.Sqrt(5.0+2.0*math.Sqrt(10.0/7.0)) / 3.0,
			},
			weights: []float64{
				(322.0 - 13.0*math.Sqrt(70.0)) / 900.0,
				(322.0 + 13.0*math.Sqrt(70.0)) / 900.0,
				128.0 / 225.0,
				(322.0 + 13.0*math.Sqrt(70.0)) / 900.0,
				(322.0 - 13.0*math.Sqrt(70.0)) / 900.0,
			},
		},
	}

	// Clamp to supported range.
	if points < 2 {
		points = 2
	}
	if points > 5 {
		points = 5
	}

	rule := rules[points]

	// Transform from [-1, 1] to [a, b]: x = ((b-a)*t + (a+b))/2
	halfLen := (b - a) / 2.0
	midpoint := (a + b) / 2.0

	sum := 0.0
	for i, t := range rule.nodes {
		x := halfLen*t + midpoint
		sum += rule.weights[i] * f(x)
	}
	return sum * halfLen
}

// MonteCarloIntegrate approximates a multi-dimensional integral using
// Monte Carlo sampling with a user-provided random number generator.
//
//	∫_{lower}^{upper} f(x) dx ≈ V * (1/N) * Σ f(x_i)
//
// where V is the hypervolume of the integration region, N is the number
// of samples, and x_i are uniformly distributed random points in the
// integration region.
//
// Parameters:
//   - f:       integrand R^dim → R
//   - dim:     number of dimensions
//   - lower:   lower bounds for each dimension (length dim)
//   - upper:   upper bounds for each dimension (length dim)
//   - samples: number of Monte Carlo samples
//   - rng:     random number generator with Float64() method returning [0, 1)
//
// Returns the estimated integral value.
//
// Valid range: dim >= 1, samples >= 1, lower[i] < upper[i] for all i
// Precision: O(1/sqrt(N)) — convergence is independent of dimension
// Reference: Press et al., Numerical Recipes, Chapter 7;
//
//	Metropolis & Ulam, "The Monte Carlo Method," 1949.
func MonteCarloIntegrate(
	f func([]float64) float64,
	dim int,
	lower, upper []float64,
	samples int,
	rng interface{ Float64() float64 },
) float64 {
	if samples < 1 {
		samples = 1
	}

	// Compute hypervolume of the integration region.
	volume := 1.0
	for i := 0; i < dim; i++ {
		volume *= upper[i] - lower[i]
	}

	// Sample point workspace (reused across iterations).
	point := make([]float64, dim)
	sum := 0.0

	for s := 0; s < samples; s++ {
		// Generate a uniform random point in [lower, upper].
		for i := 0; i < dim; i++ {
			point[i] = lower[i] + rng.Float64()*(upper[i]-lower[i])
		}
		sum += f(point)
	}

	return volume * sum / float64(samples)
}
