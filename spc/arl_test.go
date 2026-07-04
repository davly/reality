package spc

import (
	"errors"
	"math"
	"testing"
)

// Golden ARL vectors are drawn from two published sources and cited per-case:
//
//   - Montgomery, D.C., "Introduction to Statistical Quality Control", 7th ed.
//     (2013), Table 9.3: ARLs of the two-sided tabular CUSUM with k = 0.5, h = 5.
//     Published ARLs (shift in sigma -> ARL): 0->465, 0.25->139, 0.5->38.0,
//     0.75->17.0, 1.0->10.4, 1.5->5.75. Siegmund's approximation reproduces these
//     to within ~1% for shift <= 1.5 sigma (it under-estimates for larger shifts,
//     as the package doc states).
//
//   - Lucas, J.M. & Saccucci, M.S. (1990), Technometrics 32(1): EWMA ARLs by the
//     Brook-Evans Markov chain. The exact lambda->1 reduction to the Shewhart
//     individuals chart, ARL0 = 1/(2*(1-Phi(L))), is used as a tight first-
//     principles anchor; the lambda<1 values are regression goldens pinned at the
//     default grid, with the published design ARL0 (~370) cross-checked.
//
// Each CUSUM case asserts BOTH an exact regression value (catches any formula
// regression to ~1e-4) AND agreement with the published table (documents model
// fidelity). A wrong sign, a 2-vs-3 factor slip, or a dropped overshoot term
// breaks at least one assertion.

const arlTol = 1e-4

// ---------------------------------------------------------------------------
// CUSUMARL — two-sided, Montgomery Table 9.3 (k=0.5, h=5)
// ---------------------------------------------------------------------------

func TestCUSUMARLTwoSidedGolden(t *testing.T) {
	tests := []struct {
		shift     float64
		wantExact float64 // Siegmund closed form (regression golden)
		published float64 // Montgomery Table 9.3
		pubTol    float64
	}{
		{0.00, 469.111182, 465.0, 6.0},
		{0.25, 139.776943, 139.0, 1.5},
		{0.50, 38.006815, 38.0, 0.2},
		{0.75, 17.030390, 17.0, 0.2},
		{1.00, 10.336195, 10.4, 0.15},
		{1.50, 5.666002, 5.75, 0.15},
	}
	for _, tt := range tests {
		got, err := CUSUMARL(tt.shift, 0.5, 5.0)
		if err != nil {
			t.Fatalf("CUSUMARL(%v) error: %v", tt.shift, err)
		}
		if math.Abs(got-tt.wantExact) > arlTol {
			t.Errorf("CUSUMARL(%v,0.5,5) = %.6f, want exact %.6f", tt.shift, got, tt.wantExact)
		}
		if math.Abs(got-tt.published) > tt.pubTol {
			t.Errorf("CUSUMARL(%v,0.5,5) = %.4f, Montgomery Table 9.3 = %.2f (tol %v)",
				tt.shift, got, tt.published, tt.pubTol)
		}
	}
}

func TestCUSUMARLMonotoneInShift(t *testing.T) {
	// ARL must fall monotonically as the shift grows (larger shift = faster detect).
	prev := math.Inf(1)
	for _, s := range []float64{0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0} {
		got, err := CUSUMARL(s, 0.5, 5.0)
		if err != nil {
			t.Fatal(err)
		}
		if got >= prev {
			t.Errorf("CUSUMARL not decreasing: shift %v -> %.4f, prev %.4f", s, got, prev)
		}
		if got < 1.0 {
			t.Errorf("ARL below the floor of 1: shift %v -> %.4f", s, got)
		}
		prev = got
	}
}

func TestCUSUMARLSymmetry(t *testing.T) {
	// The two-sided chart is symmetric: a +delta and -delta shift share one ARL.
	for _, s := range []float64{0.3, 0.8, 1.4, 2.2} {
		up, err := CUSUMARL(s, 0.5, 5.0)
		if err != nil {
			t.Fatal(err)
		}
		dn, err := CUSUMARL(-s, 0.5, 5.0)
		if err != nil {
			t.Fatal(err)
		}
		if math.Abs(up-dn) > 1e-9 {
			t.Errorf("asymmetric ARL: +%v -> %.6f, -%v -> %.6f", s, up, s, dn)
		}
	}
}

// ---------------------------------------------------------------------------
// CUSUMARLOneSided — Siegmund one-sided; ARL0 = 2 * two-sided at shift 0
// ---------------------------------------------------------------------------

func TestCUSUMARLOneSidedGolden(t *testing.T) {
	got, err := CUSUMARLOneSided(0, 0.5, 5.0)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(got-938.222364) > 1e-3 {
		t.Errorf("one-sided ARL0 = %.6f, want 938.222364", got)
	}
	// The two-sided ARL0 is exactly half the one-sided ARL0 (equal competing risks).
	two, _ := CUSUMARL(0, 0.5, 5.0)
	if math.Abs(two-got/2) > 1e-6 {
		t.Errorf("two-sided ARL0 (%.6f) should be one-sided/2 (%.6f)", two, got/2)
	}
}

func TestCUSUMARLValidation(t *testing.T) {
	if _, err := CUSUMARL(0, 0, 5); !errors.Is(err, ErrNonPositiveK) {
		t.Error("k=0 should error ErrNonPositiveK")
	}
	if _, err := CUSUMARL(0, 0.5, 0); !errors.Is(err, ErrNonPositiveH) {
		t.Error("h=0 should error ErrNonPositiveH")
	}
	if _, err := CUSUMARLOneSided(0, -1, 5); !errors.Is(err, ErrNonPositiveK) {
		t.Error("negative k should error")
	}
}

// ---------------------------------------------------------------------------
// CUSUMThresholdForARL — inverse solve, round-trips against CUSUMARL
// ---------------------------------------------------------------------------

func TestCUSUMThresholdForARLGolden(t *testing.T) {
	// Calibrating the nexus magic h=5.0: a target ARL0 of 465 with k=0.5 recovers
	// h ~ 4.99, i.e. the folklore h=5.0 corresponds to ~one false alarm per 465.
	tests := []struct {
		arl0, k, wantH float64
	}{
		{465, 0.5, 4.9913},
		{370, 0.5, 4.7661},
		{500, 0.5, 5.0630},
	}
	for _, tt := range tests {
		h, err := CUSUMThresholdForARL(tt.arl0, tt.k)
		if err != nil {
			t.Fatalf("CUSUMThresholdForARL(%v,%v) error: %v", tt.arl0, tt.k, err)
		}
		if math.Abs(h-tt.wantH) > 1e-3 {
			t.Errorf("CUSUMThresholdForARL(%v,%v) = %.4f, want %.4f", tt.arl0, tt.k, h, tt.wantH)
		}
		// Round-trip: feeding h back must reproduce the target ARL0.
		back, _ := CUSUMARL(0, tt.k, h)
		if math.Abs(back-tt.arl0) > 1e-4 {
			t.Errorf("round-trip ARL0 = %.6f, want %v", back, tt.arl0)
		}
	}
}

func TestCUSUMThresholdForARLUnattainable(t *testing.T) {
	// An ARL0 target at or below the h->0 minimum is not attainable.
	if _, err := CUSUMThresholdForARL(1.0, 0.5); !errors.Is(err, ErrARLTarget) {
		t.Error("ARL0=1.0 should be unattainable (ErrARLTarget)")
	}
	if _, err := CUSUMThresholdForARL(465, 0); !errors.Is(err, ErrNonPositiveK) {
		t.Error("k=0 should error ErrNonPositiveK")
	}
}

// ---------------------------------------------------------------------------
// EWMALimits — exact variance-inflation factor
// ---------------------------------------------------------------------------

func TestEWMALimitsGolden(t *testing.T) {
	// lambda=0.2, sigma=1, L=3. At t=1 the exact variance is lambda^2 = 0.04
	// (sigmaZ=0.2); the steady-state variance is lambda/(2-lambda) = 1/9.
	lim, err := EWMALimits(0.2, 3.0, 1.0, 1)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(lim.VarInflation-0.04) > 1e-12 {
		t.Errorf("t=1 var-inflation = %.12f, want 0.04", lim.VarInflation)
	}
	if math.Abs(lim.SigmaZ-0.2) > 1e-12 {
		t.Errorf("t=1 sigmaZ = %.12f, want 0.2", lim.SigmaZ)
	}
	if math.Abs(lim.HalfWidth-0.6) > 1e-12 || math.Abs(lim.UCL-0.6) > 1e-12 || math.Abs(lim.LCL+0.6) > 1e-12 {
		t.Errorf("t=1 limits = +/-%.12f (UCL %v LCL %v), want +/-0.6", lim.HalfWidth, lim.UCL, lim.LCL)
	}
	// The inflation factor rises monotonically toward the steady-state 1/9.
	steady := 0.2 / 1.8
	var prev float64
	for tt := 1; tt <= 60; tt++ {
		lim, err := EWMALimits(0.2, 3.0, 1.0, tt)
		if err != nil {
			t.Fatal(err)
		}
		if lim.VarInflation < prev {
			t.Errorf("var-inflation not monotone at t=%d", tt)
		}
		if lim.VarInflation > steady+1e-12 {
			t.Errorf("var-inflation %.9f exceeds steady state %.9f at t=%d", lim.VarInflation, steady, tt)
		}
		prev = lim.VarInflation
	}
	if math.Abs(prev-steady) > 1e-6 {
		t.Errorf("var-inflation at t=60 = %.9f, want ~steady %.9f", prev, steady)
	}
}

func TestEWMASteadyStateSigma(t *testing.T) {
	// lambda=0.5, sigma=2: sqrt(0.5/1.5)*2 = 2*sqrt(1/3).
	got, err := EWMASteadyStateSigma(0.5, 2.0)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(got-2*math.Sqrt(1.0/3.0)) > 1e-12 {
		t.Errorf("steady sigma = %.12f, want %.12f", got, 2*math.Sqrt(1.0/3.0))
	}
}

func TestEWMALimitsValidation(t *testing.T) {
	if _, err := EWMALimits(0, 3, 1, 1); !errors.Is(err, ErrLambdaRange) {
		t.Error("lambda=0 should error ErrLambdaRange")
	}
	if _, err := EWMALimits(1.5, 3, 1, 1); !errors.Is(err, ErrLambdaRange) {
		t.Error("lambda>1 should error ErrLambdaRange")
	}
	if _, err := EWMALimits(0.2, 0, 1, 1); !errors.Is(err, ErrNonPositiveL) {
		t.Error("L=0 should error ErrNonPositiveL")
	}
	if _, err := EWMALimits(0.2, 3, 0, 1); !errors.Is(err, ErrNonPositiveSigmaARL) {
		t.Error("sigma=0 should error ErrNonPositiveSigmaARL")
	}
	if _, err := EWMALimits(0.2, 3, 1, 0); !errors.Is(err, ErrTimeIndex) {
		t.Error("t=0 should error ErrTimeIndex")
	}
}

// ---------------------------------------------------------------------------
// EWMAARL — Markov chain; exact Shewhart limit + published cross-check
// ---------------------------------------------------------------------------

func TestEWMAARLShewhartLimit(t *testing.T) {
	// lambda=1 collapses the EWMA to a Shewhart individuals chart, whose exact
	// two-sided in-control ARL0 is 1/(2*(1-Phi(L))). The Markov chain reproduces
	// this independently of grid resolution.
	for _, L := range []float64{2.0, 2.5, 3.0} {
		want := 1.0 / (2.0 * (1.0 - normalCDF(L)))
		got, err := EWMAARL(1.0, L, 0)
		if err != nil {
			t.Fatal(err)
		}
		if math.Abs(got-want) > 1e-2 {
			t.Errorf("EWMAARL(1,%v,0) = %.6f, want Shewhart %.6f", L, got, want)
		}
	}
	// L=2.5 is sentinel's z-threshold: ARL0 ~ 80.5 -> a false page every ~80 obs.
	got, _ := EWMAARL(1.0, 2.5, 0)
	if math.Abs(got-80.519637) > 1e-2 {
		t.Errorf("sentinel z=2.5 ARL0 = %.6f, want 80.519637", got)
	}
}

func TestEWMAARLLambdaQuarterGolden(t *testing.T) {
	// Lucas & Saccucci (1990): an EWMA with lambda=0.25 designed near L=2.9 targets
	// ARL0 ~ 370. Regression goldens are the default-grid Markov-chain values; the
	// published design point is cross-checked to ~1%.
	cases := []struct {
		shift, want float64
	}{
		{0.0, 372.446389},
		{0.5, 41.262232},
		{1.0, 10.266928},
		{2.0, 3.466629},
	}
	for _, c := range cases {
		got, err := EWMAARL(0.25, 2.9, c.shift)
		if err != nil {
			t.Fatal(err)
		}
		if math.Abs(got-c.want) > 1e-3 {
			t.Errorf("EWMAARL(0.25,2.9,%v) = %.6f, want %.6f", c.shift, got, c.want)
		}
	}
	arl0, _ := EWMAARL(0.25, 2.9, 0)
	if arl0 < 366 || arl0 > 378 {
		t.Errorf("ARL0 %.3f outside ~1%% of the Lucas-Saccucci design ARL0 ~370", arl0)
	}
}

func TestEWMAARLGridConvergence(t *testing.T) {
	// Coarser and finer grids must agree to within the discretisation error, and
	// the odd-forcing means an even m is treated as m+1.
	a, _ := EWMAARLGrid(0.25, 2.9, 0, 101)
	b, _ := EWMAARLGrid(0.25, 2.9, 0, 301)
	if math.Abs(a-b) > 1.0 {
		t.Errorf("grid convergence poor: m=101 -> %.4f, m=301 -> %.4f", a, b)
	}
	even, _ := EWMAARLGrid(0.25, 2.9, 0, 200)
	odd, _ := EWMAARLGrid(0.25, 2.9, 0, 201)
	if math.Abs(even-odd) > 1e-12 {
		t.Errorf("even m should be promoted to odd: m=200 -> %.6f, m=201 -> %.6f", even, odd)
	}
}

func TestEWMAARLMonotoneInShift(t *testing.T) {
	prev := math.Inf(1)
	for _, s := range []float64{0, 0.5, 1.0, 1.5, 2.0, 3.0} {
		got, err := EWMAARL(0.2, 3.0, s)
		if err != nil {
			t.Fatal(err)
		}
		if got >= prev {
			t.Errorf("EWMAARL not decreasing in shift at %v: %.4f >= %.4f", s, got, prev)
		}
		prev = got
	}
}

func TestEWMAARLValidation(t *testing.T) {
	if _, err := EWMAARL(0, 3, 0); !errors.Is(err, ErrLambdaRange) {
		t.Error("lambda=0 should error ErrLambdaRange")
	}
	if _, err := EWMAARL(0.2, 0, 0); !errors.Is(err, ErrNonPositiveL) {
		t.Error("L=0 should error ErrNonPositiveL")
	}
	if _, err := EWMAARLGrid(0.2, 3, 0, 5); !errors.Is(err, ErrStates) {
		t.Error("m<11 should error ErrStates")
	}
}
