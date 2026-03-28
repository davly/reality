package chaos

import "math"

// LyapunovExponent estimates the Lyapunov exponent for a 1D iterated map.
//
// The Lyapunov exponent measures the average exponential rate of divergence
// of nearby orbits. For a 1D map x_{n+1} = f(x_n):
//
//	lambda = (1/n) * sum_{i=0}^{n-1} ln|f'(x_i)|
//
// Since the exact derivative may not be available, this implementation uses
// a numerical derivative with a small perturbation epsilon = 1e-10:
//
//	f'(x) ≈ (f(x + eps) - f(x - eps)) / (2*eps)
//
// Parameters:
//   - f: the iterated map function
//   - x0: initial condition
//   - n: number of iterations to average over
//
// Returns the estimated Lyapunov exponent. Positive values indicate chaos,
// negative values indicate convergence to a fixed point or periodic orbit.
//
// Reference: Strogatz, Nonlinear Dynamics and Chaos, Chapter 10.
func LyapunovExponent(f func(float64) float64, x0 float64, n int) float64 {
	if n <= 0 {
		return 0
	}

	const eps = 1e-10
	sum := 0.0
	x := x0

	for i := 0; i < n; i++ {
		// Numerical derivative: f'(x) ≈ (f(x+eps) - f(x-eps)) / (2*eps)
		deriv := (f(x+eps) - f(x-eps)) / (2 * eps)
		absDeriv := math.Abs(deriv)
		if absDeriv > 0 {
			sum += math.Log(absDeriv)
		} else {
			// If derivative is exactly zero, Lyapunov exponent → -inf.
			// In practice this means super-stable fixed point.
			sum += math.Log(eps) // Use a very negative value.
		}
		x = f(x)
	}

	return sum / float64(n)
}

// BifurcationPoint holds a single (parameter, state) pair from a bifurcation
// diagram scan.
type BifurcationPoint struct {
	R float64 // Parameter value.
	X float64 // State value after transient warmup.
}

// BifurcationDiagram computes a bifurcation diagram by scanning a parameter
// range for a 1D iterated map of the form x_{n+1} = f(r, x).
//
// For each of rSteps evenly spaced values of r in [rMin, rMax]:
//  1. Iterate the map warmup times to discard transients.
//  2. Record the next samples state values.
//
// This reveals fixed points, periodic orbits, and chaotic regions as r varies.
//
// Parameters:
//   - f: the parameterized map function (r, x) -> x_next
//   - rMin, rMax: parameter range to scan
//   - rSteps: number of parameter values to sample
//   - warmup: iterations to discard before sampling
//   - samples: number of state values to record per parameter
//
// Returns a slice of BifurcationPoint structs.
func BifurcationDiagram(f func(r, x float64) float64, rMin, rMax float64, rSteps, warmup, samples int) []BifurcationPoint {
	if rSteps <= 0 || samples <= 0 {
		return nil
	}

	result := make([]BifurcationPoint, 0, rSteps*samples)
	dr := (rMax - rMin) / float64(rSteps)

	for i := 0; i <= rSteps; i++ {
		r := rMin + float64(i)*dr

		// Start from a fixed initial condition.
		x := 0.5

		// Warmup: discard transient behavior.
		for j := 0; j < warmup; j++ {
			x = f(r, x)
		}

		// Sample the attractor.
		for j := 0; j < samples; j++ {
			x = f(r, x)
			result = append(result, BifurcationPoint{R: r, X: x})
		}
	}

	return result
}

// RecurrencePlot computes a binary recurrence matrix from a trajectory.
//
// The recurrence plot is an N x N boolean matrix where entry (i, j) is true
// if the Euclidean distance between trajectory[i] and trajectory[j] is less
// than or equal to threshold.
//
// Recurrence plots reveal temporal patterns in dynamical systems:
//   - Diagonal lines: deterministic/periodic behavior
//   - Vertical/horizontal lines: laminar states
//   - Isolated points: rare/chaotic states
//   - Uniform density: stochastic behavior
//
// Parameters:
//   - trajectory: sequence of state vectors (each []float64 of same length)
//   - threshold: distance threshold for recurrence
//
// Returns an N x N boolean matrix. The matrix is symmetric: R[i][j] == R[j][i].
//
// Reference: Eckmann, Kamphorst, Ruelle (1987). "Recurrence plots of
// dynamical systems."
func RecurrencePlot(trajectory [][]float64, threshold float64) [][]bool {
	n := len(trajectory)
	if n == 0 {
		return nil
	}

	result := make([][]bool, n)
	for i := 0; i < n; i++ {
		result[i] = make([]bool, n)
	}

	for i := 0; i < n; i++ {
		result[i][i] = true // A point is always recurrent with itself.
		for j := i + 1; j < n; j++ {
			d := euclideanDist(trajectory[i], trajectory[j])
			if d <= threshold {
				result[i][j] = true
				result[j][i] = true
			}
		}
	}

	return result
}

// euclideanDist computes the Euclidean distance between two vectors.
func euclideanDist(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		d := a[i] - b[i]
		sum += d * d
	}
	return math.Sqrt(sum)
}
