package evt

import (
	"fmt"
	"math"
	"testing"
)

// Tests of the maximum-likelihood fitters against independently computed
// maximum-likelihood estimates (see mle_oracle_data_test.go for how the
// expected values were obtained).

const (
	// mleLLTol bounds the log-likelihood difference to the oracle.  The fitters
	// agree with the oracle to ~1e-12; the former optimiser missed the maximum
	// by 2e-3 or more on every sample used here (or returned a fit with zero
	// likelihood).
	mleLLTol = 1e-9
	// mleParamTol bounds the parameter difference to the oracle.  The
	// numerical-gradient optimiser is accurate to ~1e-6 (sigma and mu scale ~1).
	mleParamTol = 1e-5
)

// mleGridUniforms returns the n uniform draws behind one sample of the seeded
// grid (the grid applies the distribution's quantile function to them): a
// SplitMix64 stream (Steele, Lea & Flood, 2014) started from seed.  The
// generator is written out here, not taken from the standard library, so the
// samples (and the stored expected values) cannot change with the Go release.
func mleGridUniforms(seed uint64, n int) []float64 {
	u := make([]float64, n)
	state := seed
	for i := range u {
		state += 0x9E3779B97F4A7C15
		z := state
		z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
		z = (z ^ (z >> 27)) * 0x94D049BB133111EB
		z ^= z >> 31
		u[i] = (float64(z>>12) + 0.5) / (1 << 52) // exact: a 52-bit integer plus one half
	}
	return u
}

func assertMLE(t *testing.T, got []float64, gotLL float64, want []float64, wantLL float64) {
	t.Helper()
	if math.IsNaN(gotLL) || math.IsInf(gotLL, 0) {
		t.Fatalf("fit %v has non-finite log-likelihood %v", got, gotLL)
	}
	if d := math.Abs(gotLL - wantLL); d > mleLLTol {
		t.Errorf("log-likelihood %.12f, oracle %.12f (difference %.3g)", gotLL, wantLL, d)
	}
	for i := range want {
		if d := math.Abs(got[i] - want[i]); d > mleParamTol {
			t.Errorf("parameter %d = %.9g, oracle %.9g (difference %.3g)", i, got[i], want[i], d)
		}
	}
}

// On every sample the former optimiser either returned the closed-form
// estimate although the likelihood has an interior maximum, or returned a
// closed-form estimate with zero likelihood (an observation outside its
// support).
func TestFitGEVMLE_FindsTheMaximum(t *testing.T) {
	for _, c := range gevMLECases {
		t.Run(c.name, func(t *testing.T) {
			start, ok := FitGEVLMoments(c.data)
			if !ok {
				t.Fatal("closed-form estimate could not be formed")
			}
			startLL := GEVLogLik(c.data, start)
			if c.startOutside {
				if !math.IsInf(startLL, -1) {
					t.Fatalf("premise: closed-form estimate should have zero likelihood, log-likelihood %v", startLL)
				}
			} else if !(startLL < c.mleLL-1e-3) {
				t.Fatalf("premise: closed-form log-likelihood %v should be below the maximum %v", startLL, c.mleLL)
			}

			fit, ok := FitGEVMLE(c.data)
			if !ok {
				t.Fatal("FitGEVMLE reported failure")
			}
			assertMLE(t, []float64{fit.Mu, fit.Sigma, fit.Xi}, GEVLogLik(c.data, fit), c.mle, c.mleLL)
		})
	}
}

func TestFitGPDMLE_FindsTheMaximum(t *testing.T) {
	for _, c := range gpdMLECases {
		t.Run(c.name, func(t *testing.T) {
			start, ok := FitGPDPWM(c.data)
			if !ok {
				t.Fatal("closed-form estimate could not be formed")
			}
			startLL := GPDLogLik(c.data, start)
			if c.startOutside {
				if !math.IsInf(startLL, -1) {
					t.Fatalf("premise: closed-form estimate should have zero likelihood, log-likelihood %v", startLL)
				}
			} else if !(startLL < c.mleLL-1e-3) {
				t.Fatalf("premise: closed-form log-likelihood %v should be below the maximum %v", startLL, c.mleLL)
			}

			fit, ok := FitGPDMLE(c.data)
			if !ok {
				t.Fatal("FitGPDMLE reported failure")
			}
			assertMLE(t, []float64{fit.Sigma, fit.Xi}, GPDLogLik(c.data, fit), c.mle, c.mleLL)
		})
	}
}

// With no interior maximum the likelihood is unbounded toward xi = -1, so the
// closed-form estimate (a valid fit here) is returned unchanged and ok stays true.
func TestFitGEVMLE_NoInteriorMaximumReturnsClosedForm(t *testing.T) {
	for _, c := range gevNoMaximumCases {
		t.Run(c.name, func(t *testing.T) {
			start, ok := FitGEVLMoments(c.data)
			if !ok {
				t.Fatal("closed-form estimate could not be formed")
			}
			fit, ok := FitGEVMLE(c.data)
			if !ok {
				t.Fatal("FitGEVMLE reported failure")
			}
			if fit != start {
				t.Errorf("got %+v, want the closed-form estimate %+v", fit, start)
			}
		})
	}
}

// The former optimiser ran past xi = -1 (where the likelihood is unbounded)
// and returned that point.
func TestFitGPDMLE_NoInteriorMaximumReturnsClosedForm(t *testing.T) {
	for _, c := range gpdNoMaximumCases {
		t.Run(c.name, func(t *testing.T) {
			start, ok := FitGPDPWM(c.data)
			if !ok {
				t.Fatal("closed-form estimate could not be formed")
			}
			fit, ok := FitGPDMLE(c.data)
			if !ok {
				t.Fatal("FitGPDMLE reported failure")
			}
			if fit.Xi <= -1 {
				t.Errorf("shape %v is in the range where the likelihood is unbounded", fit.Xi)
			}
			if fit != start {
				t.Errorf("got %+v, want the closed-form estimate %+v", fit, start)
			}
		})
	}
}

// When there is no interior maximum and the closed-form estimate lies outside
// the support of the data, the fit returned must still have finite likelihood
// (the former optimiser returned the closed-form estimate with zero
// likelihood) and be at least as good as the exponential fit.
func TestFitGPDMLE_NoInteriorMaximumInvalidStartStaysFinite(t *testing.T) {
	for _, c := range gpdNoMaximumInvalidStartCases {
		t.Run(c.name, func(t *testing.T) {
			start, ok := FitGPDPWM(c.data)
			if !ok {
				t.Fatal("closed-form estimate could not be formed")
			}
			if !math.IsInf(GPDLogLik(c.data, start), -1) {
				t.Fatal("premise: closed-form estimate should have zero likelihood")
			}
			fit, ok := FitGPDMLE(c.data)
			if !ok {
				t.Fatal("FitGPDMLE reported failure")
			}
			ll := GPDLogLik(c.data, fit)
			if math.IsNaN(ll) || math.IsInf(ll, 0) {
				t.Fatalf("fit %+v has non-finite log-likelihood %v", fit, ll)
			}
			if fit.Xi <= -1 {
				t.Errorf("shape %v is in the range where the likelihood is unbounded", fit.Xi)
			}
			mean := 0.0
			for _, y := range c.data {
				mean += y
			}
			mean /= float64(len(c.data))
			if exp := GPDLogLik(c.data, GPDParams{Sigma: mean, Xi: 0}); ll < exp {
				t.Errorf("log-likelihood %v is below the exponential fit's %v", ll, exp)
			}
		})
	}
}

// Seeded samples from a fixed generator, in a grid over sample size and shape.
// Each configuration includes a seed on which the former optimiser missed the
// maximum and one on which it found it.
func TestFitGEVMLE_SeededGrid(t *testing.T) {
	for _, r := range gevMLEGrid {
		t.Run(fmt.Sprintf("n=%d xi=%g seed=%d", r.n, r.xi, r.seed), func(t *testing.T) {
			u := mleGridUniforms(r.seed, r.n)
			data := make([]float64, r.n)
			for i := range data {
				data[i] = GEVQuantile(u[i], GEVParams{Mu: 0, Sigma: 1, Xi: r.xi})
			}
			fit, ok := FitGEVMLE(data)
			if !ok {
				t.Fatal("FitGEVMLE reported failure")
			}
			assertMLE(t, []float64{fit.Mu, fit.Sigma, fit.Xi}, GEVLogLik(data, fit), r.mle, r.mleLL)
		})
	}
}

func TestFitGPDMLE_SeededGrid(t *testing.T) {
	for _, r := range gpdMLEGrid {
		t.Run(fmt.Sprintf("n=%d xi=%g seed=%d", r.n, r.xi, r.seed), func(t *testing.T) {
			u := mleGridUniforms(r.seed, r.n)
			data := make([]float64, r.n)
			for i := range data {
				data[i] = GPDQuantile(u[i], GPDParams{Sigma: 1, Xi: r.xi})
			}
			fit, ok := FitGPDMLE(data)
			if !ok {
				t.Fatal("FitGPDMLE reported failure")
			}
			assertMLE(t, []float64{fit.Sigma, fit.Xi}, GPDLogLik(data, fit), r.mle, r.mleLL)
		})
	}
}

// ok keeps its meaning: false only when the closed-form estimate cannot be
// formed (here, too few observations or a constant sample).
func TestFitMLE_OKMeansTheClosedFormEstimateExists(t *testing.T) {
	if _, ok := FitGEVMLE([]float64{1, 2}); ok {
		t.Error("FitGEVMLE with 2 observations: ok = true")
	}
	if _, ok := FitGEVMLE([]float64{3, 3, 3, 3}); ok {
		t.Error("FitGEVMLE of a constant sample: ok = true")
	}
	if _, ok := FitGPDMLE([]float64{1}); ok {
		t.Error("FitGPDMLE with 1 observation: ok = true")
	}
	if _, ok := FitGPDMLE([]float64{2, 2, 2}); ok {
		t.Error("FitGPDMLE of a constant sample: ok = true")
	}
}

// numGrad must not mix the penalty value into a difference quotient.  Here the
// point is closer to the edge of the domain (x < 0 is infeasible) than the
// difference step, so a central difference spans the penalty and gives ~5e17.
func TestNumGrad_OneSidedAtTheEdgeOfTheDomain(t *testing.T) {
	f := func(v []float64) float64 {
		if v[0] < 0 {
			return bigPenalty
		}
		return (v[0] - 2) * (v[0] - 2)
	}
	x := []float64{5e-7}
	g := make([]float64, 1)
	numGrad(f, x, g)
	if want := 2 * (x[0] - 2); math.Abs(g[0]-want) > 1e-5 {
		t.Errorf("gradient %v, want %v", g[0], want)
	}
}

// When both neighbours at the default step are infeasible the step is halved
// until a feasible pair is found.
func TestNumGrad_ShrinksTheStepInANarrowDomain(t *testing.T) {
	f := func(v []float64) float64 {
		if math.Abs(v[0]-0.5) > 1e-9 {
			return bigPenalty
		}
		return 3 * v[0]
	}
	x := []float64{0.5}
	g := make([]float64, 1)
	numGrad(f, x, g)
	if math.Abs(g[0]-3) > 1e-3 {
		t.Errorf("gradient %v, want 3", g[0])
	}
}
