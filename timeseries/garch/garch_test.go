package garch

import (
	"math"
	"math/rand"
	"testing"
)

// =========================================================================
// Validation
// =========================================================================

func TestModel_Validate(t *testing.T) {
	cases := []struct {
		m    Model
		good bool
	}{
		{Model{Omega: 1e-6, Alpha: 0.05, Beta: 0.9}, true},
		{Model{Omega: 0, Alpha: 0.05, Beta: 0.9}, false},
		{Model{Omega: 1e-6, Alpha: -0.01, Beta: 0.9}, false},
		{Model{Omega: 1e-6, Alpha: 0.6, Beta: 0.6}, false}, // sum >= 1
		{Model{Omega: math.NaN(), Alpha: 0.05, Beta: 0.9}, false},
	}
	for i, c := range cases {
		err := c.m.Validate()
		if c.good && err != nil {
			t.Errorf("case %d: want valid, got %v", i, err)
		}
		if !c.good && err == nil {
			t.Errorf("case %d: want invalid, got nil", i)
		}
	}
}

// =========================================================================
// Filter
// =========================================================================

func TestFilter_RecursionMatchesByHand(t *testing.T) {
	m := Model{Omega: 0.01, Alpha: 0.1, Beta: 0.85, UncondVar: 0.01 / 0.05}
	eps := []float64{0.5, -0.3, 0.2, 0.0, 0.4}
	sigma2 := make([]float64, len(eps))
	z := make([]float64, len(eps))
	if err := m.Filter(eps, sigma2, z); err != nil {
		t.Fatal(err)
	}

	// Reproduce by hand.
	wantS2 := make([]float64, len(eps))
	prevS2 := m.UncondVar
	prevEps := 0.0
	for i, e := range eps {
		s2 := m.Omega + m.Alpha*prevEps*prevEps + m.Beta*prevS2
		wantS2[i] = s2
		prevS2 = s2
		prevEps = e
	}
	for i := range eps {
		if math.Abs(sigma2[i]-wantS2[i]) > 1e-12 {
			t.Errorf("sigma2[%d] = %v, want %v", i, sigma2[i], wantS2[i])
		}
		if math.Abs(z[i]-eps[i]/math.Sqrt(wantS2[i])) > 1e-12 {
			t.Errorf("z[%d] mismatched", i)
		}
	}
}

func TestFilter_StandardisedResidualsHaveUnitVariance(t *testing.T) {
	rng := rand.New(rand.NewSource(2026))
	m := Model{Omega: 0.01, Alpha: 0.1, Beta: 0.85, UncondVar: 0.01 / 0.05}
	const n = 3000
	shocks := make([]float64, n)
	for i := range shocks {
		shocks[i] = rng.NormFloat64()
	}
	eps := make([]float64, n)
	sigma2Sim := make([]float64, n)
	if err := m.Simulate(shocks, eps, sigma2Sim); err != nil {
		t.Fatal(err)
	}

	sigma2 := make([]float64, n)
	z := make([]float64, n)
	if err := m.Filter(eps, sigma2, z); err != nil {
		t.Fatal(err)
	}
	// z should have mean ~0, variance ~1.
	var mean, m2 float64
	for _, zi := range z {
		mean += zi
	}
	mean /= float64(n)
	for _, zi := range z {
		m2 += (zi - mean) * (zi - mean)
	}
	variance := m2 / float64(n-1)
	if math.Abs(mean) > 0.05 {
		t.Errorf("z mean = %v, want close to 0", mean)
	}
	if math.Abs(variance-1.0) > 0.1 {
		t.Errorf("z variance = %v, want close to 1", variance)
	}
}

// =========================================================================
// ForecastVariance
// =========================================================================

func TestForecastVariance_ConvergesToUncondVar(t *testing.T) {
	m := Model{Omega: 0.01, Alpha: 0.1, Beta: 0.85, UncondVar: 0.01 / 0.05}
	out, err := m.ForecastVariance(0.5*0.5, 0.04, 200)
	if err != nil {
		t.Fatal(err)
	}
	last := out[len(out)-1]
	if math.Abs(last-m.UncondVar) > 0.001 {
		t.Errorf("h=200 forecast = %v, want close to UncondVar = %v", last, m.UncondVar)
	}
}

func TestForecastVariance_ClosedForm(t *testing.T) {
	m := Model{Omega: 0.02, Alpha: 0.1, Beta: 0.8, UncondVar: 0.02 / 0.1}
	eps2T := 0.16
	sigma2T := 0.10
	out, _ := m.ForecastVariance(eps2T, sigma2T, 5)
	// Step 1: omega + alpha*eps2T + beta*sigma2T.
	want := m.Omega + m.Alpha*eps2T + m.Beta*sigma2T
	if math.Abs(out[0]-want) > 1e-12 {
		t.Errorf("h=1 = %v, want %v", out[0], want)
	}
	// Closed form: sigma^2_{t+h} = v + (alpha+beta)^{h-1} * (sigma^2_{t+1} - v).
	v := m.UncondVar
	persist := m.Alpha + m.Beta
	for h := 2; h <= 5; h++ {
		expected := v + math.Pow(persist, float64(h-1))*(out[0]-v)
		if math.Abs(out[h-1]-expected) > 1e-12 {
			t.Errorf("h=%d = %v, want %v", h, out[h-1], expected)
		}
	}
}

// =========================================================================
// Simulate -> Fit recovers approximate parameters
// =========================================================================

func TestFit_RecoversApproximateParameters(t *testing.T) {
	rng := rand.New(rand.NewSource(2026))
	true_ := Model{Omega: 0.01, Alpha: 0.08, Beta: 0.9, UncondVar: 0}
	true_.UncondVar = true_.Omega / (1 - true_.Alpha - true_.Beta)
	const n = 5000
	shocks := make([]float64, n)
	for i := range shocks {
		shocks[i] = rng.NormFloat64()
	}
	eps := make([]float64, n)
	sigma2 := make([]float64, n)
	if err := true_.Simulate(shocks, eps, sigma2); err != nil {
		t.Fatal(err)
	}

	// Cold start: invalid receiver triggers default (omega=1e-6, alpha=0.05,
	// beta=0.9) per Fit's docs.
	fitted, res, err := Fit(eps, Model{}, FitConfig{MaxIter: 2000, LearningRate: 0.05, AbsTol: 1e-8, TikhonovLambda: 1e-5})
	if err != nil {
		t.Fatal(err)
	}
	if !res.Converged {
		t.Logf("not converged in %d iters; final logL = %.3f", res.Iter, res.FinalLogLik)
	}
	// Calibration is ill-posed for omega; alpha and beta are usually
	// recovered to within 0.05 with this sample size.
	if math.Abs(fitted.Alpha-true_.Alpha) > 0.06 {
		t.Errorf("alpha = %v, want close to %v", fitted.Alpha, true_.Alpha)
	}
	if math.Abs(fitted.Beta-true_.Beta) > 0.06 {
		t.Errorf("beta = %v, want close to %v", fitted.Beta, true_.Beta)
	}
	// Persistence (alpha + beta) is the most identifiable quantity.
	if math.Abs((fitted.Alpha+fitted.Beta)-(true_.Alpha+true_.Beta)) > 0.03 {
		t.Errorf("persistence %v, want close to %v",
			fitted.Alpha+fitted.Beta, true_.Alpha+true_.Beta)
	}
}

// =========================================================================
// Determinism
// =========================================================================

func TestFilter_Deterministic(t *testing.T) {
	m := Model{Omega: 0.01, Alpha: 0.1, Beta: 0.85, UncondVar: 0.2}
	eps := []float64{0.1, -0.2, 0.05, 0.3, -0.1}
	run := func() ([]float64, []float64) {
		s := make([]float64, 5)
		z := make([]float64, 5)
		_ = m.Filter(eps, s, z)
		return s, z
	}
	s1, z1 := run()
	s2, z2 := run()
	for i := range s1 {
		if s1[i] != s2[i] || z1[i] != z2[i] {
			t.Errorf("non-deterministic at i=%d", i)
		}
	}
}

// =========================================================================
// Validation
// =========================================================================

func TestFit_RejectsTooFewSamples(t *testing.T) {
	if _, _, err := Fit([]float64{1, 2, 3}, Model{}, FitConfig{}); err == nil {
		t.Error("Fit with n<50 should error")
	}
}

func TestFilter_RejectsBadInputs(t *testing.T) {
	if err := (Model{}).Filter([]float64{0.1}, []float64{0}, []float64{0}); err == nil {
		t.Error("invalid model should error")
	}
	m := Model{Omega: 0.01, Alpha: 0.1, Beta: 0.85, UncondVar: 0.2}
	if err := m.Filter([]float64{}, []float64{}, []float64{}); err == nil {
		t.Error("empty eps should error")
	}
	if err := m.Filter([]float64{0.1, 0.2}, []float64{0}, []float64{0, 0}); err == nil {
		t.Error("short sigma2 buffer should error")
	}
}

func TestForecastVariance_RejectsBadH(t *testing.T) {
	m := Model{Omega: 0.01, Alpha: 0.1, Beta: 0.85, UncondVar: 0.2}
	if _, err := m.ForecastVariance(0.04, 0.04, 0); err == nil {
		t.Error("h=0 should error")
	}
}
