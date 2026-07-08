package gametheory

import (
	"math"
	"testing"
)

// ---------------------------------------------------------------------------
// KellyContinuous — single-asset f* = mu / sigma^2
//
// All golden values are hand-derived from the closed form f* = mu / sigma^2;
// the derivation for each case is shown inline.
// ---------------------------------------------------------------------------

func TestKellyContinuous_Basic(t *testing.T) {
	// mu=0.10, sigma=0.20 -> variance = 0.04 -> f* = 0.10/0.04 = 2.5
	assertClose(t, "mu=.10 sig=.20", KellyContinuous(0.10, 0.20), 2.5, 1e-12)
	// mu=0.08, sigma=0.40 -> variance = 0.16 -> f* = 0.08/0.16 = 0.5
	assertClose(t, "mu=.08 sig=.40", KellyContinuous(0.08, 0.40), 0.5, 1e-12)
	// mu=0.02, sigma=0.20 -> variance = 0.04 -> f* = 0.02/0.04 = 0.5
	assertClose(t, "mu=.02 sig=.20", KellyContinuous(0.02, 0.20), 0.5, 1e-12)
}

func TestKellyContinuous_NegativeEdge(t *testing.T) {
	// mu=-0.05, sigma=0.10 -> variance = 0.01 -> f* = -0.05/0.01 = -5.0 (short/abstain)
	assertClose(t, "negative mu", KellyContinuous(-0.05, 0.10), -5.0, 1e-12)
}

func TestKellyContinuous_Degenerate(t *testing.T) {
	if KellyContinuous(0.10, 0.0) != 0 {
		t.Error("zero sigma should return 0")
	}
	if KellyContinuous(0.10, -0.20) != 0 {
		t.Error("negative sigma should return 0")
	}
	if KellyContinuous(math.NaN(), 0.20) != 0 {
		t.Error("NaN mu should return 0")
	}
	if KellyContinuous(math.Inf(1), 0.20) != 0 {
		t.Error("Inf mu should return 0")
	}
	if KellyContinuous(0.10, math.Inf(1)) != 0 {
		t.Error("Inf sigma should return 0")
	}
}

// ---------------------------------------------------------------------------
// KellyContinuousMulti — multi-asset f* = Sigma^{-1} mu
// ---------------------------------------------------------------------------

func TestKellyContinuousMulti_Diagonal(t *testing.T) {
	// Uncorrelated assets: Sigma diagonal -> each f*_i = mu_i / sigma_ii,
	// i.e. the single-asset formula per asset.
	//   Sigma = [[0.04, 0],[0, 0.09]], mu = [0.10, 0.06]
	//   f1 = 0.10/0.04 = 2.5
	//   f2 = 0.06/0.09 = 0.6666666...
	cov := [][]float64{{0.04, 0.0}, {0.0, 0.09}}
	mu := []float64{0.10, 0.06}
	f := KellyContinuousMulti(mu, cov)
	if len(f) != 2 {
		t.Fatalf("expected 2 fractions, got %d", len(f))
	}
	assertClose(t, "diag f1", f[0], 2.5, 1e-12)
	assertClose(t, "diag f2", f[1], 0.06/0.09, 1e-12)
}

func TestKellyContinuousMulti_Correlated(t *testing.T) {
	// Correlated 2-asset case. Sigma = [[0.04, 0.012],[0.012, 0.09]], mu=[0.02, 0.03].
	//   det(Sigma) = 0.04*0.09 - 0.012^2 = 0.0036 - 0.000144 = 0.003456
	//   Sigma^{-1} = (1/det) * [[0.09, -0.012],[-0.012, 0.04]]
	//   f1 = (0.09*0.02 - 0.012*0.03)/det = (0.0018 - 0.00036)/0.003456
	//      = 0.00144/0.003456 = 0.4166666...
	//   f2 = (-0.012*0.02 + 0.04*0.03)/det = (-0.00024 + 0.0012)/0.003456
	//      = 0.00096/0.003456 = 0.2777777...
	cov := [][]float64{{0.04, 0.012}, {0.012, 0.09}}
	mu := []float64{0.02, 0.03}
	f := KellyContinuousMulti(mu, cov)
	if len(f) != 2 {
		t.Fatalf("expected 2 fractions, got %d", len(f))
	}
	assertClose(t, "corr f1", f[0], 0.00144/0.003456, 1e-12)
	assertClose(t, "corr f2", f[1], 0.00096/0.003456, 1e-12)
}

func TestKellyContinuousMulti_ResidualCheck(t *testing.T) {
	// Sanity: Sigma * f must reproduce mu (solving the linear system exactly).
	cov := [][]float64{
		{0.10, 0.02, 0.01},
		{0.02, 0.08, 0.015},
		{0.01, 0.015, 0.05},
	}
	mu := []float64{0.03, 0.02, 0.04}
	f := KellyContinuousMulti(mu, cov)
	if len(f) != 3 {
		t.Fatalf("expected 3 fractions, got %d", len(f))
	}
	for i := 0; i < 3; i++ {
		got := 0.0
		for j := 0; j < 3; j++ {
			got += cov[i][j] * f[j]
		}
		assertClose(t, "residual row", got, mu[i], 1e-10)
	}
}

func TestKellyContinuousMulti_MatchesSingleAsset(t *testing.T) {
	// 1x1 system must equal KellyContinuous with sigma = sqrt(variance).
	cov := [][]float64{{0.04}}
	mu := []float64{0.10}
	f := KellyContinuousMulti(mu, cov)
	assertClose(t, "1x1", f[0], KellyContinuous(0.10, 0.20), 1e-12)
}

func TestKellyContinuousMulti_BadInput(t *testing.T) {
	if KellyContinuousMulti(nil, nil) != nil {
		t.Error("empty should return nil")
	}
	if KellyContinuousMulti([]float64{0.1}, [][]float64{{0.04, 0.0}}) != nil {
		t.Error("non-square row should return nil")
	}
	if KellyContinuousMulti([]float64{0.1, 0.2}, [][]float64{{0.04, 0.0}}) != nil {
		t.Error("dimension mismatch should return nil")
	}
	// Singular covariance (two identical rows) -> nil.
	if KellyContinuousMulti([]float64{0.1, 0.2}, [][]float64{{0.04, 0.02}, {0.04, 0.02}}) != nil {
		t.Error("singular matrix should return nil")
	}
}

// ---------------------------------------------------------------------------
// FractionalKelly — clamp lambda into [0,1] then scale
// ---------------------------------------------------------------------------

func TestFractionalKelly(t *testing.T) {
	// f=2.5, lambda=0.25 (quarter Kelly) -> 0.625
	assertClose(t, "quarter", FractionalKelly(2.5, 0.25), 0.625, 1e-12)
	// full Kelly leaves f unchanged
	assertClose(t, "full", FractionalKelly(2.5, 1.0), 2.5, 1e-12)
	// lambda > 1 is capped at full Kelly (never levers up)
	assertClose(t, "over-one", FractionalKelly(2.5, 1.5), 2.5, 1e-12)
	// lambda <= 0 -> no bet
	assertClose(t, "zero", FractionalKelly(2.5, 0.0), 0.0, 1e-12)
	assertClose(t, "negative", FractionalKelly(2.5, -0.3), 0.0, 1e-12)
	// scales a negative (short) raw fraction too: -5 * 0.25 = -1.25
	assertClose(t, "short", FractionalKelly(-5.0, 0.25), -1.25, 1e-12)
}

// ---------------------------------------------------------------------------
// ContinuousGrowthRate — g(f) = f*mu - f^2*sigma^2/2, maximised at f*=mu/sigma^2
// ---------------------------------------------------------------------------

func TestContinuousGrowthRate_Value(t *testing.T) {
	// f=2.5, mu=0.10, sigma=0.20:
	//   g = 2.5*0.10 - 2.5^2 * 0.04 / 2 = 0.25 - 6.25*0.04/2 = 0.25 - 0.125 = 0.125
	assertClose(t, "g at f*", ContinuousGrowthRate(2.5, 0.10, 0.20), 0.125, 1e-12)
	// Zero position -> zero growth.
	assertClose(t, "g at 0", ContinuousGrowthRate(0.0, 0.10, 0.20), 0.0, 1e-12)
}

func TestContinuousGrowthRate_MaximisedAtKelly(t *testing.T) {
	mu, sigma := 0.10, 0.20
	fStar := KellyContinuous(mu, sigma) // 2.5
	// Closed-form maximum: g(f*) = mu^2 / (2*sigma^2) = 0.01/0.08 = 0.125
	assertClose(t, "max value", ContinuousGrowthRate(fStar, mu, sigma),
		mu*mu/(2*sigma*sigma), 1e-12)
	// Perturbing either side of f* must strictly lower growth (concavity).
	gStar := ContinuousGrowthRate(fStar, mu, sigma)
	if ContinuousGrowthRate(fStar-0.5, mu, sigma) >= gStar {
		t.Error("g should decrease below f*")
	}
	if ContinuousGrowthRate(fStar+0.5, mu, sigma) >= gStar {
		t.Error("g should decrease above f*")
	}
}

// ---------------------------------------------------------------------------
// Ergodicity — ensemble mean vs time-average growth, and the gap
//
// Classic example: a +50% / -50% coin. Ensemble mean E[r] = 0, but a single
// trajectory alternately multiplies by 1.5 and 0.5 (product 0.75 < 1), so it
// decays — the time-average growth is negative. This is the ergodicity trap.
// ---------------------------------------------------------------------------

func TestErgodicity_FiftyFiftyCoin(t *testing.T) {
	returns := []float64{0.5, -0.5}
	// Ensemble mean = (0.5 + (-0.5))/2 = 0
	assertClose(t, "ensemble", EnsembleMeanReturn(returns), 0.0, 1e-12)
	// Time-average = (ln(1.5) + ln(0.5))/2
	//   ln(1.5) =  0.4054651081...
	//   ln(0.5) = -0.6931471805...
	//   mean    = -0.2876820724.../2? No: sum = -0.2876820724, /2 = -0.1438410362...
	wantG := (math.Log(1.5) + math.Log(0.5)) / 2
	assertClose(t, "time-avg", TimeAverageGrowthRate(returns), wantG, 1e-12)
	// Gap = 0 - wantG = 0.1438410362... (>= 0 by Jensen)
	assertClose(t, "gap", ErgodicityGap(returns), -wantG, 1e-12)
	if ErgodicityGap(returns) <= 0 {
		t.Error("gap must be positive by Jensen")
	}
}

func TestErgodicity_SmallReturns(t *testing.T) {
	// Constant +10% per bar: ensemble = 0.10, time-avg = ln(1.1) = 0.0953101798...
	returns := []float64{0.1, 0.1, 0.1}
	assertClose(t, "ensemble", EnsembleMeanReturn(returns), 0.10, 1e-12)
	assertClose(t, "time-avg", TimeAverageGrowthRate(returns), math.Log(1.1), 1e-12)
	// gap = 0.10 - ln(1.1) = 0.0046898201...
	assertClose(t, "gap", ErgodicityGap(returns), 0.10-math.Log(1.1), 1e-12)
}

func TestErgodicity_TotalLossCap(t *testing.T) {
	// A -100% (or worse) bar is capped at -99.99% to avoid ln(0) = -Inf.
	returns := []float64{-1.0}
	g := TimeAverageGrowthRate(returns)
	if math.IsInf(g, 0) || math.IsNaN(g) {
		t.Errorf("total-loss bar must be capped, got %v", g)
	}
	// Capped value = ln(1 - 0.9999) = ln(0.0001)
	assertClose(t, "capped", g, math.Log(0.0001), 1e-12)
}

func TestErgodicity_Empty(t *testing.T) {
	if !math.IsNaN(EnsembleMeanReturn(nil)) {
		t.Error("empty ensemble should be NaN")
	}
	if !math.IsNaN(TimeAverageGrowthRate(nil)) {
		t.Error("empty time-avg should be NaN")
	}
}

// ---------------------------------------------------------------------------
// Gap-aware shrinkage — both C# twin shapes, pinned to one Go reference
// ---------------------------------------------------------------------------

func TestErgodicityShrinkage_UnitRatio(t *testing.T) {
	// From the 50/50 coin: gap = 0.1438410362..., timeAvg = -0.1438410362...
	// gapRatio = gap / max(|timeAvg|, eps) = 0.14384.../0.14384... = 1.0
	//   Exp variant:        exp(-1)      = 0.3678794411...
	//   Reciprocal variant: 1/(1+1)      = 0.5
	returns := []float64{0.5, -0.5}
	gap := ErgodicityGap(returns)
	tAvg := TimeAverageGrowthRate(returns)
	assertClose(t, "exp", ErgodicityShrinkageExp(gap, tAvg), math.Exp(-1), 1e-12)
	assertClose(t, "recip", ErgodicityShrinkageReciprocal(gap, tAvg), 0.5, 1e-12)
}

func TestErgodicityShrinkage_ZeroGap(t *testing.T) {
	// gap = 0 -> ratio 0 -> both shrinkages = 1 (no extra shrinkage).
	assertClose(t, "exp zero-gap", ErgodicityShrinkageExp(0.0, 0.05), 1.0, 1e-12)
	assertClose(t, "recip zero-gap", ErgodicityShrinkageReciprocal(0.0, 0.05), 1.0, 1e-12)
}

func TestErgodicityShrinkage_SmallRatio(t *testing.T) {
	// From constant +10%: gap = 0.0046898201..., timeAvg = ln(1.1) = 0.0953101798...
	// ratio = 0.0046898.../0.0953101... = 0.0492065...
	//   exp(-0.0492065)   = 0.9519854...
	//   1/(1+0.0492065)   = 0.9531015...
	returns := []float64{0.1, 0.1, 0.1}
	gap := ErgodicityGap(returns)
	tAvg := TimeAverageGrowthRate(returns)
	ratio := gap / tAvg
	assertClose(t, "exp small", ErgodicityShrinkageExp(gap, tAvg), math.Exp(-ratio), 1e-12)
	assertClose(t, "recip small", ErgodicityShrinkageReciprocal(gap, tAvg), 1/(1+ratio), 1e-12)
}

func TestErgodicityShrinkage_Monotone(t *testing.T) {
	// Both shrinkages decrease as the gap widens, and stay in (0,1].
	tAvg := 0.05
	prevExp, prevRec := 2.0, 2.0
	for _, gap := range []float64{0.0, 0.01, 0.05, 0.2, 1.0} {
		e := ErgodicityShrinkageExp(gap, tAvg)
		r := ErgodicityShrinkageReciprocal(gap, tAvg)
		if e <= 0 || e > 1 || r <= 0 || r > 1 {
			t.Errorf("shrinkage out of (0,1]: exp=%v recip=%v", e, r)
		}
		if e > prevExp || r > prevRec {
			t.Error("shrinkage must be non-increasing in gap")
		}
		prevExp, prevRec = e, r
	}
}

// ---------------------------------------------------------------------------
// PriorShrink — shrink-only ecosystem win-probability prior
// ---------------------------------------------------------------------------

func TestPriorShrink(t *testing.T) {
	// Neutral prior (0.5): priorShrink = 0.5/0.5 = 1 -> unchanged.
	assertClose(t, "neutral", PriorShrink(0.625, 0.5), 0.625, 1e-12)
	// Below neutral (0.25): priorShrink = clamp(0.5,0,1)=0.5 -> 0.625*0.5 = 0.3125.
	assertClose(t, "below", PriorShrink(0.625, 0.25), 0.3125, 1e-12)
	// Above neutral (0.8): priorShrink = clamp(1.6,0,1)=1 -> unchanged (never inflates).
	assertClose(t, "above", PriorShrink(0.625, 0.8), 0.625, 1e-12)
	// Zero win prob -> abstain.
	assertClose(t, "zero", PriorShrink(0.625, 0.0), 0.0, 1e-12)
}

func TestPriorShrink_MissingPrior(t *testing.T) {
	// Out-of-range / NaN prior returns the fraction unchanged (additive safety).
	assertClose(t, "nan", PriorShrink(0.625, math.NaN()), 0.625, 1e-12)
	assertClose(t, "neg", PriorShrink(0.625, -0.1), 0.625, 1e-12)
	assertClose(t, "over-one", PriorShrink(0.625, 1.5), 0.625, 1e-12)
}

// ---------------------------------------------------------------------------
// End-to-end: reproduce a RubberDuck ergodicity-sizer pipeline from a return
// series, exercising the full Go reference the C# twins are pinned against.
// ---------------------------------------------------------------------------

func TestContinuousKelly_SizerPipeline(t *testing.T) {
	// A modest positive-edge series.
	returns := []float64{0.02, -0.01, 0.03, 0.01, -0.02, 0.04}
	mu := TimeAverageGrowthRate(returns) // continuous Kelly uses log-return mean
	// sigma from the log-returns (population-ish); here we just need a positive vol.
	// Compute std of log-returns for a realistic sigma.
	n := len(returns)
	logs := make([]float64, n)
	lsum := 0.0
	for i, r := range returns {
		logs[i] = math.Log(1 + r)
		lsum += logs[i]
	}
	lmean := lsum / float64(n)
	sse := 0.0
	for _, l := range logs {
		sse += (l - lmean) * (l - lmean)
	}
	sigma := math.Sqrt(sse / float64(n-1))

	raw := KellyContinuous(mu, sigma)
	if raw <= 0 {
		t.Fatalf("expected positive raw Kelly for positive-edge series, got %v", raw)
	}
	// Quarter Kelly.
	frac := FractionalKelly(raw, 0.25)
	assertClose(t, "quarter of raw", frac, 0.25*raw, 1e-12)

	// Gap-aware shrinkage (Core exp variant), then clamp to a 1.0 cap.
	gap := ErgodicityGap(returns)
	shrunk := frac * ErgodicityShrinkageExp(gap, mu)
	if shrunk <= 0 || shrunk > frac {
		t.Errorf("shrinkage should reduce (or hold) the fraction: frac=%v shrunk=%v", frac, shrunk)
	}
	// Prior blend with a below-neutral ecosystem win rate halves it.
	final := PriorShrink(shrunk, 0.25)
	assertClose(t, "prior halves", final, shrunk*0.5, 1e-12)
}
