package evt

import (
	"math"
	"testing"
)

// ---------------------------------------------------------------------------
// R80b cross-substrate output parity vs RubberDuck C#.
//
// These tests pin this Go package against the RubberDuck originals
//   flagships/rubberduck/RubberDuck.Core/Analysis/ExtremeValueTheory.cs
//   flagships/rubberduck/RubberDuck.Core/Analysis/GevMaxDrawdown.cs
// on the convention-identical surfaces, following the output-parity (not
// strict-byte) tier used by prob/copula (see prob/copula/doc.go).  The
// reference constants below are the outputs of the identical closed forms as
// implemented in the C# originals; tolerance <= 1e-9 absolute.
//
// Parity is deliberately NOT asserted for the GEV *L-moment shape* estimator:
// RubberDuck.GevMaxDrawdown.EstimateLMoments uses a non-standard polynomial in
// L-skewness, whereas FitGEVLMoments here implements the published
// Hosking-Wallis map.  That divergence is the single-implementation risk this
// reference exists to arbitrate (see doc.go).
// ---------------------------------------------------------------------------

const parityTol = 1e-9

// GevCdf/GevPdf/GevQuantile — matches GevMaxDrawdown.cs general branch
// (|xi| > 0.01 so both substrates use the non-Gumbel form).
func TestCrossSubstratePrecision_RubberDuck_GevDistribution(t *testing.T) {
	p := GEVParams{Mu: 0, Sigma: 1, Xi: 0.2}
	assertClose(t, "GevCdf(1)", GEVCDF(1, p), 0.66906265266781884, parityTol)
	assertClose(t, "GevPdf(1)", GEVPDF(1, p), 0.22406772865086313, parityTol)
	assertClose(t, "GevQuantile(.99)", GEVQuantile(0.99, p), 7.5468264085857832, parityTol)
}

// GpdCdf — matches ExtremeValueTheory.cs GPD survival/cdf convention.
func TestCrossSubstratePrecision_RubberDuck_GpdDistribution(t *testing.T) {
	p := GPDParams{Sigma: 1, Xi: 0.3}
	assertClose(t, "GpdCdf(2)", GPDCDF(2, p), 0.79126270182169234, parityTol)
}

// FitGpd PWM — RubberDuck.ExtremeValueTheory.FitGpd uses the same
// Hosking-Wallis 1987 PWM estimator: shape = 2 + mean/(mean-2b1),
// scale = mean*(1-shape).  On exceedances [1,2,9]: xi=0.5, sigma=2.
func TestCrossSubstratePrecision_RubberDuck_FitGpdPWM(t *testing.T) {
	fit, ok := FitGPDPWM([]float64{1, 2, 9})
	if !ok {
		t.Fatal("fit failed")
	}
	assertClose(t, "shape", fit.Xi, 0.5, parityTol)
	assertClose(t, "scale", fit.Sigma, 2.0, parityTol)
}

// EvtVaR / EvtExpectedShortfall — matches ExtremeValueTheory.cs:
//
//	VaR = u + (beta/xi)((fu/p)^xi - 1)
//	ES  = VaR/(1-xi) + (beta - xi*u)/(1-xi)
//
// with u=0, beta=1, xi=0.5, fu=0.1, conf=0.99.
func TestCrossSubstratePrecision_RubberDuck_EvtVaR_ES(t *testing.T) {
	m := POTModel{Threshold: 0, Params: GPDParams{Sigma: 1, Xi: 0.5}, ExceedanceRate: 0.1, NumTotal: 100, NumExceed: 10}
	assertClose(t, "EvtVaR", EvtVaR(m, 0.99), 4.324555320336759, parityTol)
	assertClose(t, "EvtES", EvtES(m, 0.99), 10.649110640673518, parityTol)
}

// ReturnPeriod — matches ExtremeValueTheory.cs ReturnPeriod:
//
//	1 / (rate * (1 + xi*excess/beta)^(-1/xi)).
func TestCrossSubstratePrecision_RubberDuck_ReturnPeriod(t *testing.T) {
	m := POTModel{Threshold: 0, Params: GPDParams{Sigma: 1, Xi: 0.5}, ExceedanceRate: 0.1, NumTotal: 100, NumExceed: 10}
	assertClose(t, "ReturnPeriod", EvtReturnPeriod(m, 4.324555320336759), 100.0, 1e-7)
}

// Divergence guard: for xi >= 1 (infinite mean excess) this reference returns
// NaN (honest), whereas RubberDuck's EvtExpectedShortfall returns a VaR*1.5
// heuristic.  Pin the honest behaviour so the divergence is explicit.
func TestEvtES_DivergesFromRD_HeuristicAtHeavyTail(t *testing.T) {
	m := POTModel{Threshold: 0, Params: GPDParams{Sigma: 1, Xi: 1.0}, ExceedanceRate: 0.1, NumTotal: 100, NumExceed: 10}
	if !math.IsNaN(EvtES(m, 0.99)) {
		t.Error("xi>=1 ES should be NaN in the reference (RD uses a VaR*1.5 heuristic)")
	}
}
