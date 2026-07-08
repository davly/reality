package evt

import (
	"math"
	"testing"
)

// GEVLogLik hand-checkable point: Gumbel (xi=0), single obs x=mu, sigma=1.
//
//	ll = -ln(1) - z - exp(-z), z=0  ->  0 - 0 - 1 = -1.
func TestGEVLogLik_GumbelPoint(t *testing.T) {
	ll := GEVLogLik([]float64{5}, GEVParams{Mu: 5, Sigma: 1, Xi: 0})
	assertClose(t, "gev loglik", ll, -1, 1e-12)
	// Outside support -> -Inf.
	if !math.IsInf(GEVLogLik([]float64{100}, GEVParams{Mu: 0, Sigma: 1, Xi: -0.5}), -1) {
		t.Error("out-of-support GEV loglik should be -Inf")
	}
}

// GPDLogLik hand-checkable: exponential (xi=0), obs y=sigma=1 -> ll = -0 - 1 = -1.
func TestGPDLogLik_ExpPoint(t *testing.T) {
	ll := GPDLogLik([]float64{1}, GPDParams{Sigma: 1, Xi: 0})
	assertClose(t, "gpd loglik", ll, -1, 1e-12)
}

// FitGEVMLE must never return a fit with lower log-likelihood than the
// L-moment start (the deterministic safety contract), and should recover
// known parameters on a clean sample.
func TestFitGEVMLE_NeverWorseAndRecovers(t *testing.T) {
	true_ := GEVParams{Mu: 10, Sigma: 2, Xi: 0.15}
	n := 300
	data := make([]float64, n)
	for i := 0; i < n; i++ {
		pp := (float64(i) + 0.5) / float64(n)
		data[i] = GEVQuantile(pp, true_)
	}
	start, _ := FitGEVLMoments(data)
	fit, ok := FitGEVMLE(data)
	if !ok {
		t.Fatal("MLE failed")
	}
	if GEVLogLik(data, fit) < GEVLogLik(data, start)-1e-9 {
		t.Errorf("MLE loglik %.6f worse than L-moment start %.6f",
			GEVLogLik(data, fit), GEVLogLik(data, start))
	}
	assertClose(t, "mle mu", fit.Mu, true_.Mu, 0.2)
	assertClose(t, "mle sigma", fit.Sigma, true_.Sigma, 0.2)
	assertClose(t, "mle xi", fit.Xi, true_.Xi, 0.1)
}

func TestFitGPDMLE_NeverWorseAndRecovers(t *testing.T) {
	true_ := GPDParams{Sigma: 1.5, Xi: 0.25}
	n := 400
	exc := make([]float64, n)
	for i := 0; i < n; i++ {
		pp := (float64(i) + 0.5) / float64(n)
		exc[i] = GPDQuantile(pp, true_)
	}
	start, _ := FitGPDPWM(exc)
	fit, ok := FitGPDMLE(exc)
	if !ok {
		t.Fatal("MLE failed")
	}
	if GPDLogLik(exc, fit) < GPDLogLik(exc, start)-1e-9 {
		t.Errorf("GPD MLE loglik worse than PWM start")
	}
	assertClose(t, "gpd mle sigma", fit.Sigma, true_.Sigma, 0.2)
	assertClose(t, "gpd mle xi", fit.Xi, true_.Xi, 0.1)
}

// Determinism: identical input yields identical output (no random restarts).
func TestFitMLE_Deterministic(t *testing.T) {
	data := []float64{2, 3, 1, 5, 4, 6, 2.5, 3.5, 7, 1.5, 8, 2.2}
	a, _ := FitGEVMLE(data)
	b, _ := FitGEVMLE(data)
	if a != b {
		t.Errorf("FitGEVMLE not deterministic: %+v vs %+v", a, b)
	}
}
