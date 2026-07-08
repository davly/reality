package evt

import (
	"math"
	"testing"
)

// EvtVaR / EvtES hand-derived golden.
//
// POTModel u=0, sigma=1, xi=0.5, exceedance rate zeta_u=0.1; conf=0.99 (p=0.01).
//
//	ratio = zeta_u/p = 10
//	VaR = u + (sigma/xi)(ratio^xi - 1) = 2*(sqrt(10)-1) = 4.324555320336759
//	ES  = VaR/(1-xi) + (sigma - xi*u)/(1-xi) = 2*VaR + 2 = 10.649110640673518
func TestEvtVaR_ES_HandDerived(t *testing.T) {
	m := POTModel{Threshold: 0, Params: GPDParams{Sigma: 1, Xi: 0.5}, ExceedanceRate: 0.1, NumExceed: 10, NumTotal: 100}
	wantVaR := 2 * (math.Sqrt(10) - 1)
	assertClose(t, "EvtVaR", EvtVaR(m, 0.99), wantVaR, 1e-12)
	assertClose(t, "EvtES", EvtES(m, 0.99), 2*wantVaR+2, 1e-12)
}

// Exponential-tail VaR: xi=0 -> VaR = u + sigma ln(zeta_u/p).
func TestEvtVaR_ExponentialTail(t *testing.T) {
	m := POTModel{Threshold: 0, Params: GPDParams{Sigma: 2, Xi: 0}, ExceedanceRate: 0.1, NumTotal: 100, NumExceed: 10}
	assertClose(t, "EvtVaR xi=0", EvtVaR(m, 0.99), 2*math.Log(10), 1e-12)
}

// EvtReturnLevel and EvtReturnPeriod are mutual inverses.
func TestEvt_ReturnLevelPeriod_Inverse(t *testing.T) {
	m := POTModel{Threshold: 0, Params: GPDParams{Sigma: 1, Xi: 0.5}, ExceedanceRate: 0.1, NumTotal: 100, NumExceed: 10}
	// m*zeta = 100*0.1 = 10 = ratio above, so return level == the VaR golden.
	rl := EvtReturnLevel(m, 100)
	assertClose(t, "returnlevel(100)", rl, 2*(math.Sqrt(10)-1), 1e-12)
	assertClose(t, "returnperiod(rl)=100", EvtReturnPeriod(m, rl), 100, 1e-9)
}

func TestEvtES_HeavyTail_InfiniteMean(t *testing.T) {
	m := POTModel{Threshold: 0, Params: GPDParams{Sigma: 1, Xi: 1.2}, ExceedanceRate: 0.1, NumTotal: 100, NumExceed: 10}
	if !math.IsNaN(EvtES(m, 0.99)) {
		t.Error("xi>=1 (infinite mean excess) ES should be NaN")
	}
}

// FitPOT end-to-end: extract exceedances above a threshold, fit GPD by PWM,
// and confirm the fitted VaR is finite and above the threshold.
func TestFitPOT_EndToEnd(t *testing.T) {
	// Exact GPD(sigma=1, xi=0.2) exceedances at plotting positions, shifted by u=5.
	gpd := GPDParams{Sigma: 1, Xi: 0.2}
	u := 5.0
	n := 60
	data := make([]float64, 0, n+40)
	// 40 below-threshold points.
	for i := 0; i < 40; i++ {
		data = append(data, u-1-float64(i)*0.01)
	}
	for i := 0; i < n; i++ {
		pp := (float64(i) + 0.5) / float64(n)
		data = append(data, u+GPDQuantile(pp, gpd))
	}
	m, ok := FitPOT(data, u)
	if !ok {
		t.Fatal("FitPOT failed")
	}
	if m.NumExceed != n {
		t.Errorf("expected %d exceedances, got %d", n, m.NumExceed)
	}
	assertClose(t, "recover xi", m.Params.Xi, 0.2, 0.15)
	assertClose(t, "recover sigma", m.Params.Sigma, 1.0, 0.25)
	v := EvtVaR(m, 0.99)
	if !(v > u) || math.IsNaN(v) {
		t.Errorf("VaR should exceed threshold, got %g", v)
	}
}

func TestThresholdAtRate(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	// rate 0.2 -> top 2 are the tail (9,10); threshold is the largest non-tail = 8.
	assertClose(t, "threshold@0.2", ThresholdAtRate(data, 0.2), 8, 1e-12)
	exc := Exceedances(data, 8)
	if len(exc) != 2 {
		t.Errorf("expected 2 exceedances above 8, got %d", len(exc))
	}
}
