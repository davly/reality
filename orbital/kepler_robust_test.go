package orbital

import (
	"math"
	"testing"
)

// ulp returns the spacing of float64 values at x.
func ulp(x float64) float64 {
	x = math.Abs(x)
	return math.Nextafter(x, math.Inf(1)) - x
}

// keplerTol is the accepted error of a computed true anomaly. The Kepler
// residual E - e*sin(E) - M is evaluated in float64, which perturbs the
// equation by a few ulp of the larger of E and M; the true anomaly amplifies
// that by dnu/dM (the oracle's conditioning, see kepler_robust_data_test.go).
// The second term covers the rounding of the final angle itself.
func keplerTol(M, eAnom, nu, dnudM float64) float64 {
	const (
		ulpsOfResidual = 8  // perturbation of the equation, in ulp of max(E, M)
		ulpsOfNu       = 16 // rounding of the angle conversion, in ulp of nu
	)
	return ulpsOfResidual*ulp(math.Max(eAnom, M))*dnudM + ulpsOfNu*ulp(nu)
}

// nuDistance is the distance between two angles on the circle.
func nuDistance(a, b float64) float64 {
	d := math.Abs(a - b)
	return math.Min(d, math.Abs(d-2*math.Pi))
}

func TestTrueAnomalyFromMean_MatchesTheExactSolution(t *testing.T) {
	worst := 0.0
	for _, row := range keplerGridRows {
		for i, M := range keplerGridM {
			got := TrueAnomalyFromMean(M, row.e, 30)
			want := row.nu[i]
			tol := keplerTol(M, row.eAnom[i], want, row.dnudM[i])
			if d := nuDistance(got, want); !(d <= tol) {
				t.Errorf("e=%v M=%v: got %v, want %v (error %.3g, tolerance %.3g)", row.e, M, got, want, d, tol)
			} else if r := d / tol; r > worst {
				worst = r
			}
		}
	}
	t.Logf("%d cases; worst error / tolerance = %.3g", len(keplerGridRows)*len(keplerGridM), worst)
}

// Newton's method started at E = M returned a wrong angle on all of these with
// 30 iterations, and on two of them with 200.
func TestTrueAnomalyFromMean_HighEccentricityCases(t *testing.T) {
	for _, c := range keplerHardCases {
		for _, maxIter := range []int{30, 200} {
			got := TrueAnomalyFromMean(c.M, c.e, maxIter)
			if d, tol := nuDistance(got, c.nu), keplerTol(c.M, c.eAnom, c.nu, c.dnudM); !(d <= tol) {
				t.Errorf("e=%v M=%v maxIter=%d: got %v, want %v (error %.3g, tolerance %.3g)", c.e, c.M, maxIter, got, c.nu, d, tol)
			}
		}
	}
}

// The failure is signalled, not returned as an angle: NaN when the inputs are
// outside the valid range or the iteration budget is too small.
func TestTrueAnomalyFromMean_NaNInsteadOfAWrongAngle(t *testing.T) {
	nan := math.NaN()
	for _, c := range []struct {
		name    string
		M, e    float64
		maxIter int
	}{
		{"e = 1", 1, 1, 30},
		{"e > 1", 1, 1.5, 30},
		{"e < 0", 1, -0.1, 30},
		{"e NaN", 1, nan, 30},
		{"M NaN", nan, 0.5, 30},
		{"M +Inf", math.Inf(1), 0.5, 30},
		{"M -Inf", math.Inf(-1), 0.5, 30},
		{"maxIter 0", 1, 0.5, 0},
		{"maxIter negative", 1, 0.5, -3},
		{"budget of one step is not enough", 0.18, 0.999, 1},
	} {
		if got := TrueAnomalyFromMean(c.M, c.e, c.maxIter); !math.IsNaN(got) {
			t.Errorf("%s: got %v, want NaN", c.name, got)
		}
	}
	// ...while an exact start needs no iterations at all.
	if got := TrueAnomalyFromMean(0, 0.9, 0); got != 0 {
		t.Errorf("M = 0, maxIter 0: got %v, want 0", got)
	}
	if got := TrueAnomalyFromMean(1, 0, 0); math.Abs(got-1) > 1e-15 {
		t.Errorf("e = 0, maxIter 0: got %v, want 1", got)
	}
	// ...and a sufficient budget converges: 10 iterations are plenty.
	if got, want := TrueAnomalyFromMean(0.18, 0.999, 10), TrueAnomalyFromMean(0.18, 0.999, 200); got != want || math.IsNaN(got) {
		t.Errorf("maxIter 10 gives %v, maxIter 200 gives %v", got, want)
	}
}

// solveKepler converges in a handful of iterations for every eccentricity, to
// a root of Kepler's equation within rounding error of its terms. The check
// uses the equation itself (a root has a tiny residual), not another solver.
func TestSolveKepler_ConvergesInAHandfulOfIterations(t *testing.T) {
	es := []float64{0, 1e-6, 0.01, 0.3, 0.7, 0.8, 0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999, 1 - 1e-8, 1 - 1e-10, 1 - 1e-12}
	ms := []float64{0, 1e-300, 1e-100, 1e-30, 1e-15, 1e-12, 1e-9, 1e-6, 1e-4, 1e-3, math.Nextafter(2*math.Pi, 0)}
	for k := 0; k <= 4000; k++ {
		ms = append(ms, 2*math.Pi*float64(k)/4001)
	}
	for _, e := range es {
		for _, M := range ms {
			E, ok := solveKepler(M, e, 6)
			if !ok {
				t.Fatalf("e=%v M=%v: not converged in 6 iterations", e, M)
			}
			if res := math.Abs(E - e*math.Sin(E) - M); res > 4e-15*(math.Abs(E)+M) {
				t.Fatalf("e=%v M=%v: E=%v has residual %.3g", e, M, E, res)
			}
		}
	}
}
