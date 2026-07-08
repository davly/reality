package evt

import (
	"math"
	"testing"
)

// GPD CDF/PDF cross-check against the direct math.Pow form
//
//	F = 1 - (1+xi y/sigma)^{-1/xi},  f = (1/sigma)(1+xi y/sigma)^{-1/xi-1}
//
// (Coles 2001 eq. 4.2).
func TestGPD_CDFPDF_MatchDirectPowForm(t *testing.T) {
	params := []GPDParams{{Sigma: 1, Xi: 0.3}, {Sigma: 2.5, Xi: -0.2}, {Sigma: 0.5, Xi: 0.8}}
	for _, p := range params {
		for _, y := range []float64{0.1, 0.5, 1, 2, 4} {
			arg := 1 + p.Xi*y/p.Sigma
			if arg <= 0 {
				continue
			}
			wantCDF := 1 - math.Pow(arg, -1/p.Xi)
			wantPDF := math.Pow(arg, -1/p.Xi-1) / p.Sigma
			assertClose(t, "GPD CDF", GPDCDF(y, p), wantCDF, 1e-12)
			assertClose(t, "GPD PDF", GPDPDF(y, p), wantPDF, 1e-12)
		}
	}
}

func TestGPD_QuantileCDF_RoundTrip(t *testing.T) {
	params := []GPDParams{{Sigma: 1, Xi: 0.3}, {Sigma: 2, Xi: -0.25}, {Sigma: 1.5, Xi: 0}}
	for _, p := range params {
		for _, pr := range []float64{0.01, 0.25, 0.5, 0.9, 0.99} {
			y := GPDQuantile(pr, p)
			assertClose(t, "GPD F(Q(p))=p", GPDCDF(y, p), pr, 1e-10)
		}
	}
}

// Exponential limit (xi -> 0): F = 1 - exp(-y/sigma).
func TestGPD_ExponentialLimit(t *testing.T) {
	exp := GPDParams{Sigma: 2, Xi: 0}
	for _, y := range []float64{0.5, 1, 3, 7} {
		want := 1 - math.Exp(-y/2)
		assertClose(t, "exp CDF", GPDCDF(y, exp), want, 1e-15)
		// xi != 0 branch converges to it.
		near := GPDParams{Sigma: 2, Xi: 1e-9}
		assertClose(t, "GPD->exp", GPDCDF(y, near), want, 1e-6)
	}
}

func TestGPD_PDF_IsDerivativeOfCDF(t *testing.T) {
	p := GPDParams{Sigma: 1, Xi: 0.3}
	const h = 1e-6
	for _, y := range []float64{0.2, 1, 2, 4} {
		fd := (GPDCDF(y+h, p) - GPDCDF(y-h, p)) / (2 * h)
		assertClose(t, "GPD PDF~dCDF", GPDPDF(y, p), fd, 1e-6)
	}
}

// Weibull-type GPD (xi<0) has a finite upper endpoint -sigma/xi; above it the
// survival is 0 (CDF=1) and the density is 0.
func TestGPD_FiniteUpperEndpoint(t *testing.T) {
	p := GPDParams{Sigma: 1, Xi: -0.5}
	endpoint := -p.Sigma / p.Xi // = 2
	assertClose(t, "F at endpoint", GPDCDF(endpoint, p), 1, 1e-12)
	if GPDCDF(endpoint+0.5, p) != 1 {
		t.Error("above endpoint CDF should be 1")
	}
	if GPDPDF(endpoint+0.5, p) != 0 {
		t.Error("above endpoint PDF should be 0")
	}
	assertClose(t, "Q(1) endpoint", GPDQuantile(1, p), endpoint, 1e-12)
}

func TestGPD_Degenerate(t *testing.T) {
	if !math.IsNaN(GPDCDF(1, GPDParams{Sigma: 0})) {
		t.Error("sigma=0 should be NaN")
	}
	if GPDCDF(-1, GPDParams{Sigma: 1, Xi: 0.3}) != 0 {
		t.Error("y<0 CDF should be 0")
	}
}
