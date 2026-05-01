package copula

import (
	"math"
	"testing"
)

// =========================================================================
// StudentTCDF + StudentTQuantile — round-trip + special values
// =========================================================================

func TestStudentTCDF_StandardCases(t *testing.T) {
	// CDF(0; df) = 0.5 for any df (symmetric).
	for _, df := range []float64{1, 2, 5, 10, 30, 100} {
		got := StudentTCDF(0, df)
		if math.Abs(got-0.5) > 1e-10 {
			t.Errorf("CDF(0; df=%v) = %v, want 0.5", df, got)
		}
	}
}

func TestStudentTCDF_LargeDfApproachesNormal(t *testing.T) {
	// As df -> inf, t-CDF -> normal CDF.
	for _, x := range []float64{-1.5, -0.5, 0.5, 1.5} {
		gotT := StudentTCDF(x, 1000)
		gotN := 0.5 * math.Erfc(-x/math.Sqrt2)
		if math.Abs(gotT-gotN) > 1e-3 {
			t.Errorf("df=1000 at x=%v: t-CDF %v vs normal-CDF %v", x, gotT, gotN)
		}
	}
}

func TestStudentTQuantile_RoundTrip(t *testing.T) {
	for _, df := range []float64{2, 5, 10, 30} {
		for _, p := range []float64{0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99} {
			x := StudentTQuantile(p, df)
			pBack := StudentTCDF(x, df)
			if math.Abs(p-pBack) > 1e-7 {
				t.Errorf("round-trip df=%v p=%v: x=%v, p'=%v",
					df, p, x, pBack)
			}
		}
	}
}

func TestStudentTQuantile_RejectsBadInputs(t *testing.T) {
	if !math.IsNaN(StudentTQuantile(0, 5)) {
		t.Error("p=0 should NaN")
	}
	if !math.IsNaN(StudentTQuantile(1, 5)) {
		t.Error("p=1 should NaN")
	}
	if !math.IsNaN(StudentTQuantile(0.5, 0)) {
		t.Error("df=0 should NaN")
	}
}

// =========================================================================
// BivariateTCDF — closed-form boundaries
// =========================================================================

func TestBivariateTCDF_PerfectPositiveCorrelation(t *testing.T) {
	got := BivariateTCDF(1.0, 2.0, 1.0, 5)
	want := StudentTCDF(1.0, 5)
	if math.Abs(got-want) > 1e-9 {
		t.Errorf("r=1: got %v, want %v", got, want)
	}
}

func TestBivariateTCDF_Independence(t *testing.T) {
	got := BivariateTCDF(0.5, -0.3, 0.0, 5)
	want := StudentTCDF(0.5, 5) * StudentTCDF(-0.3, 5)
	if math.Abs(got-want) > 1e-7 {
		t.Errorf("r=0: got %v, want %v", got, want)
	}
}

func TestBivariateTCDF_Symmetry(t *testing.T) {
	r := 0.4
	df := 7.0
	a := BivariateTCDF(0.6, -0.2, r, df)
	b := BivariateTCDF(-0.2, 0.6, r, df)
	if math.Abs(a-b) > 1e-7 {
		t.Errorf("symmetry: %v vs %v", a, b)
	}
}

func TestBivariateTCDF_LargeDfApproachesGaussian(t *testing.T) {
	// As df -> inf the t-copula CDF converges to the Gaussian copula CDF.
	for _, r := range []float64{-0.5, 0, 0.3, 0.7} {
		gotT := BivariateTCDF(0.5, 1.2, r, 1000)
		gotG := BivariateNormalCDF(0.5, 1.2, r)
		if math.Abs(gotT-gotG) > 5e-3 {
			t.Errorf("df=1000 r=%v: t %v vs normal %v", r, gotT, gotG)
		}
	}
}

// =========================================================================
// StudentTCopulaCDF — bivariate
// =========================================================================

func TestStudentTCopulaCDF_Bivariate_Independence(t *testing.T) {
	sigma := [][]float64{{1, 0}, {0, 1}}
	u := []float64{0.7, 0.3}
	got, err := StudentTCopulaCDF(u, sigma, 5)
	if err != nil {
		t.Fatal(err)
	}
	want := 0.7 * 0.3
	if math.Abs(got-want) > 1e-7 {
		t.Errorf("t independent: got %v, want %v", got, want)
	}
}

func TestStudentTCopulaCDF_LargeDfApproachesGaussian(t *testing.T) {
	sigma := [][]float64{{1, 0.5}, {0.5, 1}}
	u := []float64{0.6, 0.4}
	gotT, err := StudentTCopulaCDF(u, sigma, 1000)
	if err != nil {
		t.Fatal(err)
	}
	gotG, err := GaussianCopulaCDF(u, sigma)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(gotT-gotG) > 5e-3 {
		t.Errorf("df=1000 t-copula vs Gaussian-copula: %v vs %v", gotT, gotG)
	}
}

func TestStudentTCopulaCDF_RejectsLowDf(t *testing.T) {
	sigma := [][]float64{{1, 0.5}, {0.5, 1}}
	u := []float64{0.5, 0.5}
	if _, err := StudentTCopulaCDF(u, sigma, 0.5); err == nil {
		t.Error("df=0.5 should error")
	}
	if _, err := StudentTCopulaCDF(u, sigma, math.NaN()); err == nil {
		t.Error("df=NaN should error")
	}
}

// =========================================================================
// Tail-dependence: the load-bearing comparison for cat-cluster aggregation
// =========================================================================
//
// The Gaussian copula has zero asymptotic tail dependence; the
// Student-t copula has positive tail dependence determined by df and
// rho.  The Embrechts-Lindskog-McNeil 2003 closed form is
//
//   lambda_U = lambda_L = 2 * T_{df+1}(-sqrt((df+1)*(1-rho)/(1+rho)))
//
// We don't ship lambda_U/L as a public function in v1 (deferred to v2's
// Archimedean cohort) but we test the CDF shape that drives them: at
// high quantiles (u_1 = u_2 = 0.99) and moderate-positive rho, the
// t-copula CDF should exceed the Gaussian-copula CDF, reflecting the
// stronger joint-extreme co-movement.

func TestStudentTCopulaCDF_HighTailExceedsGaussian(t *testing.T) {
	sigma := [][]float64{{1, 0.5}, {0.5, 1}}
	u := []float64{0.99, 0.99}
	gotT, err := StudentTCopulaCDF(u, sigma, 4)
	if err != nil {
		t.Fatal(err)
	}
	gotG, err := GaussianCopulaCDF(u, sigma)
	if err != nil {
		t.Fatal(err)
	}
	// t-copula at low df concentrates more mass in the upper tail —
	// joint CDF approaches Phi(min) = 0.99 faster than Gaussian.  The
	// directional inequality is the load-bearing cat-cluster property.
	if gotT < gotG {
		t.Errorf("t-tail dependence: t-copula CDF %v should >= Gaussian-copula %v",
			gotT, gotG)
	}
	if gotT > 0.99 {
		t.Errorf("t-copula CDF %v exceeds Phi(min)=0.99", gotT)
	}
}

// =========================================================================
// Trivariate t
// =========================================================================

func TestTrivariateTCDF_Independence(t *testing.T) {
	got := TrivariateTCDF(0.5, -0.3, 1.0, 0, 0, 0, 5)
	want := StudentTCDF(0.5, 5) * StudentTCDF(-0.3, 5) * StudentTCDF(1.0, 5)
	if math.Abs(got-want) > 1e-7 {
		t.Errorf("trivariate t independent: got %v, want %v", got, want)
	}
}

func TestStudentTCopulaCDF_Trivariate_Independence(t *testing.T) {
	sigma := [][]float64{
		{1, 0, 0},
		{0, 1, 0},
		{0, 0, 1},
	}
	u := []float64{0.4, 0.6, 0.7}
	got, err := StudentTCopulaCDF(u, sigma, 5)
	if err != nil {
		t.Fatal(err)
	}
	want := u[0] * u[1] * u[2]
	if math.Abs(got-want) > 1e-7 {
		t.Errorf("trivariate t independent copula: got %v, want %v", got, want)
	}
}

// =========================================================================
// Determinism
// =========================================================================

func TestStudentTCopulaCDF_Deterministic(t *testing.T) {
	sigma := [][]float64{{1, 0.5}, {0.5, 1}}
	u := []float64{0.6, 0.4}
	a, _ := StudentTCopulaCDF(u, sigma, 5)
	b, _ := StudentTCopulaCDF(u, sigma, 5)
	if a != b {
		t.Errorf("non-deterministic: %v vs %v", a, b)
	}
}
