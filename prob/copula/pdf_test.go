package copula

import (
	"errors"
	"math"
	"testing"
)

// Tests for Clayton + Gumbel bivariate copula PDFs (Phase 15).
// Pattern: closed-form PDF verified against finite-difference of CDF
// (∂²C/∂u∂v) at a grid of test points; log-PDF verified against
// log of PDF.

func TestClaytonPDFFn_RejectsInvalidTheta(t *testing.T) {
	for _, theta := range []float64{0, -1, math.NaN(), math.Inf(1)} {
		if _, err := ClaytonPDFFn(theta); !errors.Is(err, ErrCopulaPDFInvalidTheta) {
			t.Errorf("ClaytonPDFFn(%v): expected ErrCopulaPDFInvalidTheta, got %v", theta, err)
		}
		if _, err := ClaytonLogPDFFn(theta); !errors.Is(err, ErrCopulaPDFInvalidTheta) {
			t.Errorf("ClaytonLogPDFFn(%v): expected ErrCopulaPDFInvalidTheta, got %v", theta, err)
		}
	}
}

func TestGumbelPDFFn_RejectsInvalidTheta(t *testing.T) {
	for _, theta := range []float64{0.5, 0, -1, math.NaN(), math.Inf(1)} {
		if _, err := GumbelPDFFn(theta); !errors.Is(err, ErrCopulaPDFInvalidTheta) {
			t.Errorf("GumbelPDFFn(%v): expected ErrCopulaPDFInvalidTheta, got %v", theta, err)
		}
		if _, err := GumbelLogPDFFn(theta); !errors.Is(err, ErrCopulaPDFInvalidTheta) {
			t.Errorf("GumbelLogPDFFn(%v): expected ErrCopulaPDFInvalidTheta, got %v", theta, err)
		}
	}
}

// numericalPDFFromCDF approximates c(u, v) = ∂²C/∂u∂v via a central
// finite-difference scheme: c ≈ ( C(u+h,v+h) - C(u-h,v+h) - C(u+h,v-h) + C(u-h,v-h) ) / (4 h²)
func numericalPDFFromCDF(cdf func(u, v float64) float64, u, v, h float64) float64 {
	return (cdf(u+h, v+h) - cdf(u-h, v+h) - cdf(u+h, v-h) + cdf(u-h, v-h)) / (4.0 * h * h)
}

func TestClaytonPDFFn_MatchesNumericalSecondDerivative(t *testing.T) {
	const h = 1e-3
	const tol = 5e-2 // central diff is O(h²) and second-order is noisier

	for _, theta := range []float64{0.5, 1.0, 2.0} {
		pdf, err := ClaytonPDFFn(theta)
		if err != nil {
			t.Fatalf("ClaytonPDFFn(%v) err: %v", theta, err)
		}
		cdf, err := ClaytonCopulaCDFFn(theta)
		if err != nil {
			t.Fatalf("ClaytonCopulaCDFFn(%v) err: %v", theta, err)
		}

		for _, u := range []float64{0.3, 0.5, 0.7} {
			for _, v := range []float64{0.3, 0.5, 0.7} {
				got := pdf(u, v)
				want := numericalPDFFromCDF(cdf, u, v, h)
				rel := math.Abs(got-want) / (1.0 + math.Abs(want))
				if rel > tol {
					t.Errorf("ClaytonPDF(θ=%v)(u=%v, v=%v): got %v, want %v (rel=%v)",
						theta, u, v, got, want, rel)
				}
			}
		}
	}
}

func TestGumbelPDFFn_MatchesNumericalSecondDerivative(t *testing.T) {
	const h = 1e-3
	const tol = 5e-2

	for _, theta := range []float64{1.5, 2.0, 3.0} {
		pdf, err := GumbelPDFFn(theta)
		if err != nil {
			t.Fatalf("GumbelPDFFn(%v) err: %v", theta, err)
		}
		cdf, err := GumbelCopulaCDFFn(theta)
		if err != nil {
			t.Fatalf("GumbelCopulaCDFFn(%v) err: %v", theta, err)
		}

		for _, u := range []float64{0.3, 0.5, 0.7} {
			for _, v := range []float64{0.3, 0.5, 0.7} {
				got := pdf(u, v)
				want := numericalPDFFromCDF(cdf, u, v, h)
				rel := math.Abs(got-want) / (1.0 + math.Abs(want))
				if rel > tol {
					t.Errorf("GumbelPDF(θ=%v)(u=%v, v=%v): got %v, want %v (rel=%v)",
						theta, u, v, got, want, rel)
				}
			}
		}
	}
}

func TestGumbelPDFFn_IndependenceCornerCase(t *testing.T) {
	pdf, err := GumbelPDFFn(1.0)
	if err != nil {
		t.Fatalf("GumbelPDFFn(1.0) err: %v", err)
	}
	for _, u := range []float64{0.1, 0.3, 0.7, 0.9} {
		for _, v := range []float64{0.1, 0.5, 0.9} {
			if got := pdf(u, v); got != 1.0 {
				t.Errorf("GumbelPDF(1.0)(u=%v, v=%v) = %v, want 1.0 (independence)", u, v, got)
			}
		}
	}
}

func TestClaytonLogPDFFn_AgreesWithLogOfClaytonPDFFn(t *testing.T) {
	for _, theta := range []float64{0.5, 1.5, 3.0} {
		pdf, _ := ClaytonPDFFn(theta)
		logPdf, _ := ClaytonLogPDFFn(theta)
		for _, u := range []float64{0.2, 0.5, 0.8} {
			for _, v := range []float64{0.2, 0.5, 0.8} {
				want := math.Log(pdf(u, v))
				got := logPdf(u, v)
				if math.Abs(got-want) > 1e-9 {
					t.Errorf("ClaytonLogPDF(θ=%v)(u=%v, v=%v): got %v, want log(%v)=%v",
						theta, u, v, got, pdf(u, v), want)
				}
			}
		}
	}
}

func TestGumbelLogPDFFn_AgreesWithLogOfGumbelPDFFn(t *testing.T) {
	for _, theta := range []float64{1.5, 2.5, 4.0} {
		pdf, _ := GumbelPDFFn(theta)
		logPdf, _ := GumbelLogPDFFn(theta)
		for _, u := range []float64{0.2, 0.5, 0.8} {
			for _, v := range []float64{0.2, 0.5, 0.8} {
				want := math.Log(pdf(u, v))
				got := logPdf(u, v)
				if math.Abs(got-want) > 1e-9 {
					t.Errorf("GumbelLogPDF(θ=%v)(u=%v, v=%v): got %v, want log(%v)=%v",
						theta, u, v, got, pdf(u, v), want)
				}
			}
		}
	}
}

func TestPDFs_ReturnZeroOnBoundary(t *testing.T) {
	pdf, _ := ClaytonPDFFn(2.0)
	if pdf(0, 0.5) != 0 || pdf(1, 0.5) != 0 || pdf(0.5, 0) != 0 || pdf(0.5, 1) != 0 {
		t.Errorf("ClaytonPDF should return 0 at hypercube boundaries")
	}
	pdf2, _ := GumbelPDFFn(2.0)
	if pdf2(0, 0.5) != 0 || pdf2(1, 0.5) != 0 || pdf2(0.5, 0) != 0 || pdf2(0.5, 1) != 0 {
		t.Errorf("GumbelPDF should return 0 at hypercube boundaries")
	}
}

func TestLogPDFs_ReturnNegInfOnBoundary(t *testing.T) {
	logPdf, _ := ClaytonLogPDFFn(2.0)
	if !math.IsInf(logPdf(0, 0.5), -1) {
		t.Errorf("ClaytonLogPDF should return -∞ at boundary")
	}
	logPdf2, _ := GumbelLogPDFFn(2.0)
	if !math.IsInf(logPdf2(0, 0.5), -1) {
		t.Errorf("GumbelLogPDF should return -∞ at boundary")
	}
}

func TestLogPDFFnForFamily_DispatchesCorrectly(t *testing.T) {
	cl, err := LogPDFFnForFamily(FamilyClayton, 2.0)
	if err != nil {
		t.Fatalf("LogPDFFnForFamily(Clayton, 2.0) err: %v", err)
	}
	gu, err := LogPDFFnForFamily(FamilyGumbel, 2.0)
	if err != nil {
		t.Fatalf("LogPDFFnForFamily(Gumbel, 2.0) err: %v", err)
	}
	// Sanity: distinct families produce distinct outputs.
	if cl(0.5, 0.5) == gu(0.5, 0.5) {
		t.Errorf("Clayton + Gumbel log-PDFs produced same value at (0.5, 0.5); families should differ")
	}
}
