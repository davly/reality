package evt

import (
	"math"
	"testing"
)

// assertClose fails t if got is not within tol of want.
func assertClose(t *testing.T, name string, got, want, tol float64) {
	t.Helper()
	if math.IsNaN(got) || math.Abs(got-want) > tol {
		t.Errorf("%s: got %.15g, want %.15g (tol %g, diff %g)", name, got, want, tol, math.Abs(got-want))
	}
}

// ---------------------------------------------------------------------------
// GEV CDF/PDF cross-check against the direct math.Pow form.
//
// GEVCDF/GEVPDF assemble the tail via log1p/expm1 for numerical stability;
// here we cross-check that assembly against the textbook direct form
//   F = exp(-t^{-1/xi}),  f = (1/sigma) t^{-1-1/xi} exp(-t^{-1/xi})
// computed independently with math.Pow (Coles 2001 eqs. 3.2-3.3).
// ---------------------------------------------------------------------------

func TestGEV_CDFPDF_MatchDirectPowForm(t *testing.T) {
	params := []GEVParams{
		{Mu: 0, Sigma: 1, Xi: 0.2},
		{Mu: 10, Sigma: 2.5, Xi: -0.15},
		{Mu: -3, Sigma: 0.7, Xi: 0.6},
	}
	for _, p := range params {
		for _, x := range []float64{-1, 0, 0.5, 1, 3, 8, 12} {
			t2 := 1 + p.Xi*(x-p.Mu)/p.Sigma
			if t2 <= 0 {
				continue // outside support; handled by dedicated boundary test
			}
			tpow := math.Pow(t2, -1/p.Xi)
			wantCDF := math.Exp(-tpow)
			wantPDF := math.Pow(t2, -1-1/p.Xi) * math.Exp(-tpow) / p.Sigma
			assertClose(t, "CDF", GEVCDF(x, p), wantCDF, 1e-12)
			assertClose(t, "PDF", GEVPDF(x, p), wantPDF, 1e-12)
		}
	}
}

// At x = mu the reduced variable t = 1 for every xi, so F(mu) = exp(-1) = 1/e
// exactly, independent of the shape (Coles 2001 eq. 3.2).  A clean exact golden.
func TestGEV_AtLocation_IsInvE(t *testing.T) {
	for _, xi := range []float64{-0.4, -0.1, 0, 0.1, 0.3, 0.9} {
		p := GEVParams{Mu: 5, Sigma: 2, Xi: xi}
		assertClose(t, "F(mu)=1/e", GEVCDF(5, p), math.Exp(-1), 1e-12)
		assertClose(t, "Q(1/e)=mu", GEVQuantile(math.Exp(-1), p), 5, 1e-9)
	}
}

// CDF and Quantile must be mutual inverses to machine precision.
func TestGEV_QuantileCDF_RoundTrip(t *testing.T) {
	params := []GEVParams{
		{Mu: 0, Sigma: 1, Xi: 0.2},
		{Mu: 3.87, Sigma: 0.198, Xi: -0.05}, // Coles 2001 Port Pirie order of magnitude
		{Mu: -2, Sigma: 4, Xi: 0},           // Gumbel
	}
	for _, p := range params {
		for _, pr := range []float64{0.01, 0.1, 0.5, 0.9, 0.99, 0.999} {
			x := GEVQuantile(pr, p)
			assertClose(t, "F(Q(p))=p", GEVCDF(x, p), pr, 1e-10)
		}
	}
}

// The xi != 0 branches must converge to the explicit Gumbel (xi == 0) branch
// as xi -> 0.  This validates the "unified xi with Gumbel limit handled
// explicitly" contract.
func TestGEV_GumbelLimit_Continuity(t *testing.T) {
	base := GEVParams{Mu: 1.5, Sigma: 2.0}
	gum := base
	gum.Xi = 0
	for _, x := range []float64{-2, 0, 1.5, 4, 9} {
		for _, xi := range []float64{1e-8, -1e-8, 1e-6, -1e-6} {
			p := base
			p.Xi = xi
			assertClose(t, "CDF->Gumbel", GEVCDF(x, p), GEVCDF(x, gum), 1e-5)
			assertClose(t, "PDF->Gumbel", GEVPDF(x, p), GEVPDF(x, gum), 1e-5)
		}
		assertClose(t, "Q->Gumbel", GEVQuantile(0.7, GEVParams{Mu: 1.5, Sigma: 2, Xi: 1e-9}),
			GEVQuantile(0.7, gum), 1e-6)
	}
}

// PDF equals the numerical derivative of CDF (central difference).
func TestGEV_PDF_IsDerivativeOfCDF(t *testing.T) {
	p := GEVParams{Mu: 0, Sigma: 1, Xi: 0.2}
	const h = 1e-6
	for _, x := range []float64{-1, 0, 1, 2, 5} {
		fd := (GEVCDF(x+h, p) - GEVCDF(x-h, p)) / (2 * h)
		assertClose(t, "PDF~dCDF", GEVPDF(x, p), fd, 1e-6)
	}
}

// Gumbel closed forms at a hand-checkable point.  mu=0, sigma=1, x=0:
//
//	F = exp(-exp(0)) = exp(-1);  f = exp(0)exp(-1) = 1/e.
func TestGEV_Gumbel_ExactPoints(t *testing.T) {
	p := GEVParams{Mu: 0, Sigma: 1, Xi: 0}
	assertClose(t, "Gumbel F(0)", GEVCDF(0, p), math.Exp(-1), 1e-15)
	assertClose(t, "Gumbel f(0)", GEVPDF(0, p), math.Exp(-1), 1e-15)
	// Quantile at p = exp(-1): x = -ln(-ln(1/e)) = -ln(1) = 0.
	assertClose(t, "Gumbel Q(1/e)", GEVQuantile(math.Exp(-1), p), 0, 1e-15)
}

func TestGEV_Support_Boundary(t *testing.T) {
	// Frechet xi>0: lower endpoint mu - sigma/xi; below it F=0.
	pF := GEVParams{Mu: 0, Sigma: 1, Xi: 0.5}
	lower := -1.0 / 0.5 // mu - sigma/xi = -2
	if GEVCDF(lower-0.1, pF) != 0 {
		t.Error("Frechet below lower endpoint should have F=0")
	}
	if GEVPDF(lower-0.1, pF) != 0 {
		t.Error("Frechet below lower endpoint should have f=0")
	}
	// Weibull xi<0: upper endpoint mu - sigma/xi; above it F=1.
	pW := GEVParams{Mu: 0, Sigma: 1, Xi: -0.5}
	upper := -1.0 / -0.5 // = 2
	if GEVCDF(upper+0.1, pW) != 1 {
		t.Error("Weibull above upper endpoint should have F=1")
	}
}

func TestGEV_Kind(t *testing.T) {
	if (GEVParams{Xi: 0.3}).Kind(0.05) != Frechet {
		t.Error("xi=0.3 should be Frechet")
	}
	if (GEVParams{Xi: -0.3}).Kind(0.05) != Weibull {
		t.Error("xi=-0.3 should be Weibull")
	}
	if (GEVParams{Xi: 0.01}).Kind(0.05) != Gumbel {
		t.Error("xi=0.01 within dead band should be Gumbel")
	}
}

func TestGEV_ReturnLevel(t *testing.T) {
	p := GEVParams{Mu: 0, Sigma: 1, Xi: 0.2}
	// Return level at T equals quantile at 1-1/T.
	for _, T := range []float64{10, 100, 1000} {
		assertClose(t, "returnlevel", GEVReturnLevel(T, p), GEVQuantile(1-1/T, p), 1e-12)
	}
	if !math.IsNaN(GEVReturnLevel(1, p)) {
		t.Error("T<=1 should be NaN")
	}
}

func TestGEV_Degenerate(t *testing.T) {
	if !math.IsNaN(GEVCDF(0, GEVParams{Sigma: 0})) {
		t.Error("sigma=0 CDF should be NaN")
	}
	if !math.IsNaN(GEVPDF(0, GEVParams{Sigma: -1})) {
		t.Error("sigma<0 PDF should be NaN")
	}
	if !math.IsNaN(GEVQuantile(1.5, GEVParams{Sigma: 1})) {
		t.Error("pr>1 should be NaN")
	}
}
