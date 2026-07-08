package portfolio

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/davly/reality/gametheory"
)

// =========================================================================
// He-Litterman (1999) published-fixture parity.
//
// testdata/he_litterman_1999.json encodes the seven-country worked example
// from He, G. & Litterman, R. (1999) "The Intuition Behind Black-Litterman
// Model Portfolios" (Goldman Sachs Investment Management Research), Appendix A
// Tables 1-2 and Appendix B. The `published_equilibrium_returns_pct` field is
// the paper's OWN Table 1 equilibrium-return column (external ground truth);
// the `expected` block is the canonical Go output that ports validate against.
// =========================================================================

type hlView struct {
	Description string      `json:"description"`
	P           [][]float64 `json:"P"`
	Q           []float64   `json:"Q"`
	Omega       [][]float64 `json:"Omega"`
}

type hlExpected struct {
	EquilibriumReturns           []float64 `json:"equilibrium_returns"`
	PosteriorReturns             []float64 `json:"posterior_returns"`
	MeanVarianceWeightsFromPrior []float64 `json:"meanvariance_weights_from_prior"`
	OptimalWeightsWithView       []float64 `json:"optimal_weights_with_view"`
	DeviationsFromEquilibrium    []float64 `json:"deviations_from_equilibrium"`
}

type hlGolden struct {
	Source                         string      `json:"source"`
	Assets                         []string    `json:"assets"`
	Delta                          float64     `json:"delta"`
	Tau                            float64     `json:"tau"`
	VolatilityPct                  []float64   `json:"volatility_pct"`
	MarketWeightPct                []float64   `json:"market_weight_pct"`
	Correlation                    [][]float64 `json:"correlation"`
	PublishedEquilibriumReturnsPct []float64   `json:"published_equilibrium_returns_pct"`
	View                           hlView      `json:"view"`
	Expected                       hlExpected  `json:"expected"`
}

func loadHL(t *testing.T) hlGolden {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("testdata", "he_litterman_1999.json"))
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g hlGolden
	if err := json.Unmarshal(data, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	return g
}

// buildSigma reconstructs the covariance Sigma = D corr D from the published
// per-asset volatilities (in %) and the correlation matrix, exactly as an
// analyst reading Tables 1-2 would.
func buildSigma(volPct []float64, corr [][]float64) [][]float64 {
	n := len(volPct)
	vol := make([]float64, n)
	for i := range volPct {
		vol[i] = volPct[i] / 100
	}
	Sigma := make([][]float64, n)
	for i := 0; i < n; i++ {
		Sigma[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			Sigma[i][j] = vol[i] * corr[i][j] * vol[j]
		}
	}
	return Sigma
}

func assertVecClose(t *testing.T, name string, got, want []float64, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length got %d want %d", name, len(got), len(want))
	}
	for i := range want {
		if math.Abs(got[i]-want[i]) > tol {
			t.Errorf("%s[%d]: got %.12g want %.12g (|diff|=%.3g > tol %.3g)",
				name, i, got[i], want[i], math.Abs(got[i]-want[i]), tol)
		}
	}
}

// TestHeLitterman_EquilibriumReturns_MatchPublishedTable pins
// ImpliedEquilibriumReturns against the paper's OWN Table 1 equilibrium-return
// column (3.9, 6.9, 8.4, 9.0, 4.3, 6.8, 7.6 %). Tolerance 5e-4 = the paper's
// one-decimal-% rounding.
func TestHeLitterman_EquilibriumReturns_MatchPublishedTable(t *testing.T) {
	g := loadHL(t)
	Sigma := buildSigma(g.VolatilityPct, g.Correlation)
	w := make([]float64, len(g.MarketWeightPct))
	for i := range w {
		w[i] = g.MarketWeightPct[i] / 100
	}
	pi := ImpliedEquilibriumReturns(w, Sigma, g.Delta)
	if pi == nil {
		t.Fatal("ImpliedEquilibriumReturns returned nil")
	}
	pubDecimal := make([]float64, len(g.PublishedEquilibriumReturnsPct))
	for i := range pubDecimal {
		pubDecimal[i] = g.PublishedEquilibriumReturnsPct[i] / 100
	}
	assertVecClose(t, "pi vs He-Litterman Table 1", pi, pubDecimal, 5e-4)
	// And exact against the canonical golden.
	assertVecClose(t, "pi vs golden", pi, g.Expected.EquilibriumReturns, 1e-12)
}

// TestHeLitterman_ReverseOptimizationRecoversMarketWeights checks that
// MeanVarianceWeights(pi, Sigma, delta) recovers the published market-cap
// weights — the exact inverse of ImpliedEquilibriumReturns (He-Litterman
// Appendix B item 5, no-view case). This is a hard published-fixture pin.
func TestHeLitterman_ReverseOptimizationRecoversMarketWeights(t *testing.T) {
	g := loadHL(t)
	Sigma := buildSigma(g.VolatilityPct, g.Correlation)
	pi := g.Expected.EquilibriumReturns
	w := MeanVarianceWeights(pi, Sigma, g.Delta)
	if w == nil {
		t.Fatal("MeanVarianceWeights returned nil")
	}
	want := make([]float64, len(g.MarketWeightPct))
	for i := range want {
		want[i] = g.MarketWeightPct[i] / 100
	}
	assertVecClose(t, "reverse weights vs Table 1", w, want, 1e-9)
}

// TestHeLitterman_PosteriorReturns_MatchGolden pins BlackLittermanPosterior on
// the "Germany outperforms Europe by 5%" view against the canonical golden.
// The paper reports posterior returns only via charts, so the numeric contract
// is the golden; the intuition the paper states in words (Germany's expected
// return rises above equilibrium, and France & UK rise too because they are
// correlated with Germany) is asserted separately below.
func TestHeLitterman_PosteriorReturns_MatchGolden(t *testing.T) {
	g := loadHL(t)
	Sigma := buildSigma(g.VolatilityPct, g.Correlation)
	pi := ImpliedEquilibriumReturns(vecDiv100(g.MarketWeightPct), Sigma, g.Delta)
	Omega := HeLittermanOmega(g.View.P, Sigma, g.Tau)
	if Omega == nil {
		t.Fatal("HeLittermanOmega returned nil")
	}
	// The generated golden Omega must equal the recomputed one.
	if len(g.View.Omega) != 1 || math.Abs(g.View.Omega[0][0]-Omega[0][0]) > 1e-15 {
		t.Fatalf("golden Omega %v != recomputed %v", g.View.Omega, Omega)
	}
	mu := BlackLittermanPosterior(pi, Sigma, g.View.P, g.View.Q, Omega, g.Tau)
	if mu == nil {
		t.Fatal("BlackLittermanPosterior returned nil")
	}
	assertVecClose(t, "posterior mu vs golden", mu, g.Expected.PosteriorReturns, 1e-11)

	// Paper's stated intuition (§ "One View on Germany versus the Rest of
	// Europe"): posterior returns for Germany, France AND the UK all rise above
	// equilibrium; the others move little.
	names := g.Assets
	for _, idx := range []int{2, 3, 5} { // France, Germany, UK
		if mu[idx] <= pi[idx] {
			t.Errorf("%s posterior %.4f should exceed equilibrium %.4f", names[idx], mu[idx], pi[idx])
		}
	}
}

// TestHeLitterman_CentralTheorem_OptimalDeviationIsViewPortfolio verifies the
// paper's headline result (Appendix B item 5, Chart 2C): the optimal weight
// deviation from equilibrium is EXACTLY proportional to the view portfolio —
// nonzero only in Germany / France / UK, with France:Germany and UK:Germany in
// the market-cap ratios -5.2/17.6 and -12.4/17.6. This grounds
// BlackLittermanPosterior + MeanVarianceWeights jointly against the paper.
func TestHeLitterman_CentralTheorem_OptimalDeviationIsViewPortfolio(t *testing.T) {
	g := loadHL(t)
	Sigma := buildSigma(g.VolatilityPct, g.Correlation)
	pi := ImpliedEquilibriumReturns(vecDiv100(g.MarketWeightPct), Sigma, g.Delta)
	Omega := HeLittermanOmega(g.View.P, Sigma, g.Tau)
	mu := BlackLittermanPosterior(pi, Sigma, g.View.P, g.View.Q, Omega, g.Tau)
	wView := MeanVarianceWeights(mu, Sigma, g.Delta)
	wEq := vecDiv100(g.MarketWeightPct)

	dev := make([]float64, len(wEq))
	for i := range dev {
		dev[i] = wView[i] - wEq[i]
	}
	// Australia(0), Canada(1), Japan(4), USA(6): deviation ~ 0.
	for _, idx := range []int{0, 1, 4, 6} {
		if math.Abs(dev[idx]) > 1e-9 {
			t.Errorf("%s deviation should be ~0, got %.3g", g.Assets[idx], dev[idx])
		}
	}
	// Germany(3) > 0; France(2), UK(5) < 0.
	if dev[3] <= 0 {
		t.Errorf("Germany deviation should be positive, got %.4f", dev[3])
	}
	// Ratio pins to the market-cap view weights.
	wantFR := -(5.2 / 17.6)
	wantUK := -(12.4 / 17.6)
	if r := dev[2] / dev[3]; math.Abs(r-wantFR) > 1e-9 {
		t.Errorf("France/Germany deviation ratio: got %.9f want %.9f", r, wantFR)
	}
	if r := dev[5] / dev[3]; math.Abs(r-wantUK) > 1e-9 {
		t.Errorf("UK/Germany deviation ratio: got %.9f want %.9f", r, wantUK)
	}
	// Cross-check golden deviations.
	assertVecClose(t, "deviations vs golden", dev, g.Expected.DeviationsFromEquilibrium, 1e-9)
}

// TestHeLitterman_PosteriorMeanIsTauInvariant verifies the key convention
// property: under Omega = HeLittermanOmega(P, Sigma, tau) the posterior mean is
// invariant to tau (see package doc / Appendix B). Three decades of tau give an
// identical mean.
func TestHeLitterman_PosteriorMeanIsTauInvariant(t *testing.T) {
	g := loadHL(t)
	Sigma := buildSigma(g.VolatilityPct, g.Correlation)
	pi := ImpliedEquilibriumReturns(vecDiv100(g.MarketWeightPct), Sigma, g.Delta)

	var ref []float64
	for _, tau := range []float64{0.01, 0.05, 0.5, 1.0} {
		Omega := HeLittermanOmega(g.View.P, Sigma, tau)
		mu := BlackLittermanPosterior(pi, Sigma, g.View.P, g.View.Q, Omega, tau)
		if mu == nil {
			t.Fatalf("nil posterior at tau=%g", tau)
		}
		if ref == nil {
			ref = mu
			continue
		}
		assertVecClose(t, "tau-invariance", mu, ref, 1e-10)
	}
}

// =========================================================================
// Structural / contract tests.
// =========================================================================

// TestBLPosterior_NoViews_ReturnsPrior: with no views the posterior equals the
// equilibrium prior (He-Litterman Appendix B).
func TestBLPosterior_NoViews_ReturnsPrior(t *testing.T) {
	pi := []float64{0.03, 0.05, 0.07}
	Sigma := [][]float64{
		{0.04, 0.01, 0.00},
		{0.01, 0.05, 0.02},
		{0.00, 0.02, 0.06},
	}
	mu := BlackLittermanPosterior(pi, Sigma, nil, nil, nil, 0.05)
	if mu == nil {
		t.Fatal("nil posterior for no-views")
	}
	assertVecClose(t, "no-views posterior == prior", mu, pi, 0)
	// And it must be a copy, not an alias.
	mu[0] = 999
	if pi[0] == 999 {
		t.Error("posterior aliases the prior slice")
	}
}

// TestImpliedEquilibrium_RoundTrip: MeanVarianceWeights inverts
// ImpliedEquilibriumReturns for an arbitrary SPD Sigma and weight vector.
func TestImpliedEquilibrium_RoundTrip(t *testing.T) {
	Sigma := [][]float64{
		{0.10, 0.02, 0.01},
		{0.02, 0.08, 0.03},
		{0.01, 0.03, 0.12},
	}
	w := []float64{0.2, 0.5, 0.3}
	delta := 3.0
	pi := ImpliedEquilibriumReturns(w, Sigma, delta)
	back := MeanVarianceWeights(pi, Sigma, delta)
	assertVecClose(t, "round-trip w", back, w, 1e-12)
}

// TestContinuousKellyWeights_WrapsGametheory: the weights equal
// fraction * gametheory.KellyContinuousMulti (the landed Wave-2 solve), and at
// fraction = 1/delta they coincide with MeanVarianceWeights.
func TestContinuousKellyWeights_WrapsGametheory(t *testing.T) {
	mu := []float64{0.06, 0.09, 0.04}
	Sigma := [][]float64{
		{0.09, 0.01, 0.02},
		{0.01, 0.10, 0.00},
		{0.02, 0.00, 0.07},
	}
	base := gametheory.KellyContinuousMulti(mu, Sigma)
	frac := 0.25
	got := ContinuousKellyWeights(mu, Sigma, frac)
	for i := range base {
		if math.Abs(got[i]-frac*base[i]) > 1e-12 {
			t.Errorf("kelly weight[%d]: got %.12g want %.12g", i, got[i], frac*base[i])
		}
	}
	// fraction = 1/delta -> MeanVarianceWeights.
	delta := 2.5
	mv := MeanVarianceWeights(mu, Sigma, delta)
	kw := ContinuousKellyWeights(mu, Sigma, 1.0/delta)
	assertVecClose(t, "kelly(1/delta) == meanvariance", kw, mv, 1e-12)
}

// TestPosteriorCovariance_NoViews_IsTauSigma and symmetry.
func TestPosteriorCovariance_NoViews_IsTauSigma(t *testing.T) {
	Sigma := [][]float64{
		{0.04, 0.01},
		{0.01, 0.05},
	}
	tau := 0.05
	M := BlackLittermanPosteriorCovariance(Sigma, nil, nil, tau)
	if M == nil {
		t.Fatal("nil covariance")
	}
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			if math.Abs(M[i][j]-tau*Sigma[i][j]) > 1e-15 {
				t.Errorf("M[%d][%d]=%.6g want %.6g", i, j, M[i][j], tau*Sigma[i][j])
			}
		}
	}
	// With a view, M must be symmetric and "shrunk" (posterior variance of the
	// mean estimate is smaller than the prior tau*Sigma on the view direction).
	P := [][]float64{{1, -1}}
	Omega := HeLittermanOmega(P, Sigma, tau)
	Mv := BlackLittermanPosteriorCovariance(Sigma, P, Omega, tau)
	if Mv == nil {
		t.Fatal("nil covariance with view")
	}
	if math.Abs(Mv[0][1]-Mv[1][0]) > 1e-15 {
		t.Errorf("posterior covariance not symmetric: %v", Mv)
	}
}

// =========================================================================
// ProjectSimplex.
// =========================================================================

func TestProjectSimplex_Properties(t *testing.T) {
	cases := [][]float64{
		{0.5, 0.3, 0.2},       // already on the simplex -> unchanged
		{3.0, 1.0, -2.0, 0.5}, // mixed sign
		{-1, -2, -3},          // all negative
		{10, 10, 10},          // equal -> uniform
		{0.7},                 // singleton -> 1
	}
	for _, v := range cases {
		w := ProjectSimplex(v)
		if w == nil {
			t.Fatalf("nil projection for %v", v)
		}
		sum := 0.0
		for _, x := range w {
			if x < -1e-15 {
				t.Errorf("negative weight %.6g in %v", x, w)
			}
			sum += x
		}
		if math.Abs(sum-1) > 1e-12 {
			t.Errorf("projection of %v sums to %.12g, want 1", v, sum)
		}
	}
	// Already-on-simplex is a fixed point.
	w := ProjectSimplex([]float64{0.5, 0.3, 0.2})
	assertVecClose(t, "simplex fixed point", w, []float64{0.5, 0.3, 0.2}, 1e-12)
	// Known projection: project [0.6, 0.6] -> [0.5, 0.5].
	w2 := ProjectSimplex([]float64{0.6, 0.6})
	assertVecClose(t, "simplex [0.6,0.6]", w2, []float64{0.5, 0.5}, 1e-12)
}

func TestMeanVarianceWeightsLongOnly_OnSimplex(t *testing.T) {
	mu := []float64{0.06, 0.09, 0.04}
	Sigma := [][]float64{
		{0.09, 0.01, 0.02},
		{0.01, 0.10, 0.00},
		{0.02, 0.00, 0.07},
	}
	w := MeanVarianceWeightsLongOnly(mu, Sigma, 2.5)
	if w == nil {
		t.Fatal("nil long-only weights")
	}
	sum := 0.0
	for _, x := range w {
		if x < -1e-15 {
			t.Errorf("negative long-only weight %.6g", x)
		}
		sum += x
	}
	if math.Abs(sum-1) > 1e-12 {
		t.Errorf("long-only weights sum to %.12g, want 1", sum)
	}
}

// =========================================================================
// Ill-posed input -> nil (honesty rule: never silently clamp/normalise).
// =========================================================================

func TestNilOnIllPosedInput(t *testing.T) {
	goodSigma := [][]float64{{0.04, 0.01}, {0.01, 0.05}}
	nan := math.NaN()
	inf := math.Inf(1)

	// ImpliedEquilibriumReturns
	if ImpliedEquilibriumReturns(nil, goodSigma, 2.5) != nil {
		t.Error("empty w should be nil")
	}
	if ImpliedEquilibriumReturns([]float64{1, 2, 3}, goodSigma, 2.5) != nil {
		t.Error("dim mismatch should be nil")
	}
	if ImpliedEquilibriumReturns([]float64{nan, 1}, goodSigma, 2.5) != nil {
		t.Error("non-finite w should be nil")
	}
	if ImpliedEquilibriumReturns([]float64{1, 1}, [][]float64{{inf, 0}, {0, 1}}, 2.5) != nil {
		t.Error("non-finite Sigma should be nil")
	}

	// MeanVarianceWeights: delta<=0 and singular Sigma.
	if MeanVarianceWeights([]float64{1, 2}, goodSigma, 0) != nil {
		t.Error("delta=0 should be nil")
	}
	if MeanVarianceWeights([]float64{1, 2}, goodSigma, -1) != nil {
		t.Error("delta<0 should be nil")
	}
	singular := [][]float64{{1, 1}, {1, 1}}
	if MeanVarianceWeights([]float64{1, 2}, singular, 2.5) != nil {
		t.Error("singular Sigma should be nil")
	}

	// BlackLittermanPosterior: mismatched P columns, singular Omega.
	pi := []float64{0.03, 0.05}
	if BlackLittermanPosterior(pi, goodSigma, [][]float64{{1, 0, 0}}, []float64{0.02}, [][]float64{{0.001}}, 0.05) != nil {
		t.Error("P column mismatch should be nil")
	}
	if BlackLittermanPosterior(pi, goodSigma, [][]float64{{1, -1}}, []float64{0.02}, [][]float64{{0}}, 0.05) != nil {
		t.Error("singular Omega should be nil")
	}
	if BlackLittermanPosterior(pi, goodSigma, [][]float64{{1, -1}}, []float64{0.02}, [][]float64{{0.001}}, 0) != nil {
		t.Error("tau=0 should be nil")
	}

	// HeLittermanOmega: bad tau / dims.
	if HeLittermanOmega([][]float64{{1, -1}}, goodSigma, 0) != nil {
		t.Error("tau=0 Omega should be nil")
	}
	if HeLittermanOmega([][]float64{{1, -1, 0}}, goodSigma, 0.05) != nil {
		t.Error("P column mismatch Omega should be nil")
	}

	// ContinuousKellyWeights: non-finite fraction, singular.
	if ContinuousKellyWeights([]float64{1, 2}, goodSigma, nan) != nil {
		t.Error("non-finite fraction should be nil")
	}
	if ContinuousKellyWeights([]float64{1, 2}, singular, 0.25) != nil {
		t.Error("singular Sigma kelly should be nil")
	}

	// ProjectSimplex
	if ProjectSimplex(nil) != nil {
		t.Error("empty v should be nil")
	}
	if ProjectSimplex([]float64{1, nan}) != nil {
		t.Error("non-finite v should be nil")
	}
}

// vecDiv100 scales a percentage vector to decimals.
func vecDiv100(pct []float64) []float64 {
	out := make([]float64, len(pct))
	for i := range pct {
		out[i] = pct[i] / 100
	}
	return out
}
