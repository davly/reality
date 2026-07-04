package evt

import (
	"math"
	"testing"
)

// LMoments3 hand-derived golden on data = [1,2,3,4].
//
//	b0 = 10/4 = 2.5
//	b1 = (1/4)[0 + (1/3)2 + (2/3)3 + (3/3)4] = (1/4)(20/3) = 5/3
//	b2 = (1/4)[0 + 0 + (2/6)3 + (6/6)4] = (1/4)(5) = 1.25
//	l1 = 2.5,  l2 = 2(5/3)-2.5 = 5/6,  l3 = 6(1.25)-6(5/3)+2.5 = 0
func TestLMoments3_HandDerived(t *testing.T) {
	l1, l2, l3, ok := LMoments3([]float64{4, 1, 3, 2}) // unsorted input
	if !ok {
		t.Fatal("expected ok")
	}
	assertClose(t, "l1", l1, 2.5, 1e-12)
	assertClose(t, "l2", l2, 5.0/6.0, 1e-12)
	assertClose(t, "l3", l3, 0, 1e-12) // symmetric data -> zero L-skewness
	if _, _, _, ok := LMoments3([]float64{1, 2}); ok {
		t.Error("n<3 should be ok=false")
	}
}

// FitGPDPWM hand-derived goldens (Hosking-Wallis 1987: xi = 2 - l1/l2,
// sigma = l1(1-xi)).
//
// data = [1,2,4]:  l1 = 7/3, l2 = 1  -> xi = 2 - 7/3 = -1/3, sigma = (7/3)(4/3) = 28/9.
// data = [1,2,9]:  l1 = 4,   l2 = 8/3 -> xi = 2 - 12/8 = 1/2, sigma = 4(1/2) = 2.
func TestFitGPDPWM_HandDerived(t *testing.T) {
	p1, ok := FitGPDPWM([]float64{4, 1, 2})
	if !ok {
		t.Fatal("fit 1 failed")
	}
	assertClose(t, "xi1", p1.Xi, -1.0/3.0, 1e-12)
	assertClose(t, "sigma1", p1.Sigma, 28.0/9.0, 1e-12)

	p2, ok := FitGPDPWM([]float64{9, 2, 1})
	if !ok {
		t.Fatal("fit 2 failed")
	}
	assertClose(t, "xi2", p2.Xi, 0.5, 1e-12)
	assertClose(t, "sigma2", p2.Sigma, 2.0, 1e-12)
}

func TestFitGPDPWM_Degenerate(t *testing.T) {
	if _, ok := FitGPDPWM([]float64{5, 5, 5, 5}); ok {
		t.Error("constant sample should fail (l2=0)")
	}
	if _, ok := FitGPDPWM([]float64{1}); ok {
		t.Error("n<2 should fail")
	}
}

// FitGEVLMoments: the shape depends only on t3 = l3/l2 through the published
// Hosking map c = 2/(3+t3) - ln2/ln3, k = 7.8590c + 2.9554c^2, xi = -k.
// For data = [1,2,3,4], t3 = 0, giving (hand-computed):
//
//	c = 2/3 - ln2/ln3 = 0.0357369131
//	k = 7.8590c + 2.9554c^2 = 0.2846311...  -> xi = -0.2846311
func TestFitGEVLMoments_ShapeFromLSkew(t *testing.T) {
	p, ok := FitGEVLMoments([]float64{1, 2, 3, 4})
	if !ok {
		t.Fatal("fit failed")
	}
	c := 2.0/3.0 - math.Ln2/math.Log(3)
	k := 7.8590*c + 2.9554*c*c
	assertClose(t, "xi = -k", p.Xi, -k, 1e-12)
	// sigma, mu regression-pinned against the same closed form (locks behavior).
	g := math.Gamma(1 + k)
	wantSigma := (5.0 / 6.0) * k / ((1 - math.Pow(2, -k)) * g)
	wantMu := 2.5 - wantSigma*(1-g)/k
	assertClose(t, "sigma", p.Sigma, wantSigma, 1e-12)
	assertClose(t, "mu", p.Mu, wantMu, 1e-12)
}

// Recovery: fit ideal GEV order statistics (quantiles at Gringorten plotting
// positions) for known parameters and confirm the L-moment fit recovers them
// to statistical tolerance.  Coles (2001) §3.3.3 workflow.
func TestFitGEVLMoments_RecoversKnownParams(t *testing.T) {
	true_ := GEVParams{Mu: 3.87, Sigma: 0.198, Xi: -0.05} // Port Pirie scale (Coles 2001 §3.4.1)
	n := 200
	data := make([]float64, n)
	for i := 0; i < n; i++ {
		pp := (float64(i) + 1 - 0.44) / (float64(n) + 0.12) // Gringorten
		data[i] = GEVQuantile(pp, true_)
	}
	fit, ok := FitGEVLMoments(data)
	if !ok {
		t.Fatal("fit failed")
	}
	assertClose(t, "recover mu", fit.Mu, true_.Mu, 0.02)
	assertClose(t, "recover sigma", fit.Sigma, true_.Sigma, 0.02)
	assertClose(t, "recover xi", fit.Xi, true_.Xi, 0.05)
}

// Gumbel branch: t3 near the Gumbel L-skewness (t3 ~= 0.1699) drives k -> 0,
// so the estimator returns xi == 0 exactly via the explicit Gumbel back-out.
func TestFitGEVLMoments_GumbelBranch(t *testing.T) {
	// Build ideal Gumbel order statistics; its L-skew hits the c=0 point.
	gum := GEVParams{Mu: 0, Sigma: 1, Xi: 0}
	n := 300
	data := make([]float64, n)
	for i := 0; i < n; i++ {
		pp := (float64(i) + 1 - 0.44) / (float64(n) + 0.12)
		data[i] = GEVQuantile(pp, gum)
	}
	fit, ok := FitGEVLMoments(data)
	if !ok {
		t.Fatal("fit failed")
	}
	if math.Abs(fit.Xi) > 0.03 {
		t.Errorf("Gumbel data should give xi near 0, got %g", fit.Xi)
	}
	assertClose(t, "gumbel mu", fit.Mu, 0, 0.05)
	assertClose(t, "gumbel sigma", fit.Sigma, 1, 0.05)
}
