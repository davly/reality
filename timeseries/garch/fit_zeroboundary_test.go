package garch

import (
	"math"
	"math/rand"
	"testing"
)

// TestFit_ZeroBoundaryWarmStart pins the fix for a valid zero-boundary warm-start.
// Alpha==0 (pure persistence) and Beta==0 (ARCH(1)) are accepted by Validate(),
// but the reparameterisation log(Alpha/slack) / log(Beta/slack) used to evaluate
// log(0) = -Inf, seeding theta with -Inf and poisoning the optimizer (Fit returned
// a zeroed model + "parameters must satisfy omega>0" error). Fit must now accept
// the boundary warm-start and return a finite, valid fitted model.
func TestFit_ZeroBoundaryWarmStart(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	const n = 200
	eps := make([]float64, n)
	for i := range eps {
		eps[i] = 0.01 * rng.NormFloat64()
	}
	for _, init := range []Model{
		{Omega: 1e-6, Alpha: 0.0, Beta: 0.90},  // pure persistence (Alpha==0)
		{Omega: 1e-6, Alpha: 0.05, Beta: 0.0},  // ARCH(1) (Beta==0)
	} {
		init.UncondVar = init.Omega / (1 - init.Alpha - init.Beta)
		fitted, res, err := Fit(eps, init, FitConfig{MaxIter: 50})
		if err != nil {
			t.Errorf("Fit with boundary warm-start {A=%v,B=%v} errored: %v", init.Alpha, init.Beta, err)
			continue
		}
		if math.IsNaN(res.FinalLogLik) || math.IsInf(res.FinalLogLik, 0) {
			t.Errorf("boundary warm-start {A=%v,B=%v}: FinalLogLik=%v, want finite", init.Alpha, init.Beta, res.FinalLogLik)
		}
		if err := fitted.Validate(); err != nil {
			t.Errorf("boundary warm-start {A=%v,B=%v}: fitted model invalid: %v", init.Alpha, init.Beta, err)
		}
	}
}
