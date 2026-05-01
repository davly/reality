package mdl

import (
	"errors"
	"math"
	"testing"
)

// =========================================================================
// UniversalIntegerCodeLength — Rissanen 1983 closed-form smoke tests.
// =========================================================================

// TestUniversalIntegerCodeLength_N1 tests the n = 1 base case:
// log(1) = 0 terminates the recursion immediately, leaving only the
// Rissanen constant log(2.865064) ≈ 1.0524 nats.
func TestUniversalIntegerCodeLength_N1(t *testing.T) {
	got, err := UniversalIntegerCodeLength(1)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	want := math.Log(2.865064)
	if math.Abs(got-want) > 1e-12 {
		t.Errorf("L*(1): got %v, want %v", got, want)
	}
}

// TestUniversalIntegerCodeLength_N2 tests the n = 2 case:
// L*(2) = log(2) + log(2.865064) ≈ 0.6931 + 1.0524 = 1.7455 nats.
// Inner loop: x = log(2) ≈ 0.693 > 0; next iter x = log(0.693) < 0
// terminates.  Total: log(2) + Rissanen constant.
func TestUniversalIntegerCodeLength_N2(t *testing.T) {
	got, err := UniversalIntegerCodeLength(2)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	want := math.Log(2) + math.Log(2.865064)
	if math.Abs(got-want) > 1e-12 {
		t.Errorf("L*(2): got %v, want %v", got, want)
	}
}

// TestUniversalIntegerCodeLength_N10 tests n = 10:
// L*(10) = log(10) + log(log(10)) + log(2.865064)
//        = log(10) + log(log(10)) + 1.0524
// log(10) ≈ 2.3026; log(log(10)) ≈ log(2.3026) ≈ 0.8340; subsequent
// log(0.8340) is < 0 and terminates.
func TestUniversalIntegerCodeLength_N10(t *testing.T) {
	got, err := UniversalIntegerCodeLength(10)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	x := math.Log(10.0)
	want := x + math.Log(x) + math.Log(2.865064)
	if math.Abs(got-want) > 1e-12 {
		t.Errorf("L*(10): got %v, want %v", got, want)
	}
}

// TestUniversalIntegerCodeLength_Monotone tests that L*(n+1) >= L*(n)
// for n >= 1: the universal codelength is monotone non-decreasing in
// the integer being coded.
func TestUniversalIntegerCodeLength_Monotone(t *testing.T) {
	prev, _ := UniversalIntegerCodeLength(1)
	for n := 2; n <= 100; n++ {
		curr, err := UniversalIntegerCodeLength(n)
		if err != nil {
			t.Fatalf("L*(%d): err=%v", n, err)
		}
		if curr < prev-1e-12 {
			t.Errorf("L*(%d)=%v < L*(%d)=%v", n, curr, n-1, prev)
		}
		prev = curr
	}
}

// TestUniversalIntegerCodeLength_InvalidN tests the n < 1 error
// path: 0 and -1 both surface ErrInvalidUniversalInt.
func TestUniversalIntegerCodeLength_InvalidN(t *testing.T) {
	for _, n := range []int{0, -1, -100} {
		_, err := UniversalIntegerCodeLength(n)
		if !errors.Is(err, ErrInvalidUniversalInt) {
			t.Errorf("L*(%d): err=%v, want ErrInvalidUniversalInt", n, err)
		}
	}
}

// TestUniversalIntegerCodeLengthBits tests that the bits variant
// equals the nats variant divided by ln(2).
func TestUniversalIntegerCodeLengthBits(t *testing.T) {
	for _, n := range []int{1, 2, 10, 100, 1000} {
		nats, err := UniversalIntegerCodeLength(n)
		if err != nil {
			t.Fatalf("nats err: %v", err)
		}
		bits, err := UniversalIntegerCodeLengthBits(n)
		if err != nil {
			t.Fatalf("bits err: %v", err)
		}
		want := nats / math.Ln2
		if math.Abs(bits-want) > 1e-12 {
			t.Errorf("L*_bits(%d): got %v, want %v", n, bits, want)
		}
	}
}

// =========================================================================
// NMLMultinomial — Kontkanen-Myllymäki 2007 linear-time recursion.
// =========================================================================

// TestNMLMultinomial_K1_ZeroRegret tests the k = 1 edge case: a
// single-category multinomial has zero parametric complexity (the
// MLE is always 1; nothing to encode).
func TestNMLMultinomial_K1_ZeroRegret(t *testing.T) {
	got, err := NMLMultinomial([]int{10})
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if math.Abs(got) > 1e-12 {
		t.Errorf("k=1: got %v, want 0", got)
	}
}

// TestNMLMultinomial_EmptyData tests n = 0: empty data has zero
// codelength regardless of k.  C(0, k) = 1 -> log(1) = 0.
func TestNMLMultinomial_EmptyData(t *testing.T) {
	got, err := NMLMultinomial([]int{0, 0, 0})
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if math.Abs(got) > 1e-12 {
		t.Errorf("empty: got %v, want 0", got)
	}
}

// TestNMLMultinomial_K2_MatchesDirectSum tests that k = 2 with n
// observations produces the direct Bernoulli-mass sum from the
// Shtarkov 1987 formula.  We compute the reference value by hand
// and compare against the implementation.
//
// For n = 4, k = 2:
//   C(4, 2) = sum_{r=0}^{4} C(4, r) * (r/4)^r * ((4-r)/4)^(4-r)
//           = (4/4)^4 + 4*(1/4)^1*(3/4)^3 + 6*(2/4)^2*(2/4)^2
//             + 4*(3/4)^3*(1/4)^1 + (4/4)^4
//           = 1 + 4*(0.25)*(0.421875) + 6*(0.25)*(0.25)
//             + 4*(0.421875)*(0.25) + 1
//           = 1 + 0.421875 + 0.375 + 0.421875 + 1
//           = 3.21875
//   regret = log(3.21875) ≈ 1.16904
func TestNMLMultinomial_K2_MatchesDirectSum(t *testing.T) {
	got, err := NMLMultinomial([]int{2, 2})
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	// Hand-computed reference.
	cn2 := 1.0 + 4*0.25*math.Pow(0.75, 3) +
		6*math.Pow(0.5, 2)*math.Pow(0.5, 2) +
		4*math.Pow(0.75, 3)*0.25 + 1.0
	want := math.Log(cn2)
	if math.Abs(got-want) > 1e-10 {
		t.Errorf("k=2, n=4: got %v, want %v (cn2=%v)", got, want, cn2)
	}
}

// TestNMLMultinomial_K3_KontkanenRecurrence tests the recurrence step
// k -> k+1 by computing C(n, 3) two ways and asserting equality.
//
// For n = 10:
//   C(10, 3) = C(10, 2) + (10/2) * C(10, 1)
//            = C(10, 2) + 5 * 1.0
// Compute C(10, 2) directly via the Bernoulli-mass sum, then compare
// against NMLMultinomial([10/3, 10/3, 10/3 + 1]) = NMLMultinomial([3, 3, 4]).
func TestNMLMultinomial_K3_KontkanenRecurrence(t *testing.T) {
	cn2 := computeCn2(10)
	wantC10_3 := cn2 + 5*1.0
	want := math.Log(wantC10_3)
	got, err := NMLMultinomial([]int{3, 3, 4})
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if math.Abs(got-want) > 1e-10 {
		t.Errorf("Kontkanen k=3: got %v, want %v", got, want)
	}
}

// TestNMLMultinomial_RegretIndependentOfCounts verifies that the
// NML *regret* depends only on (n, k), not on the specific count
// vector.  Same n, same k -> same regret.
func TestNMLMultinomial_RegretIndependentOfCounts(t *testing.T) {
	r1, _ := NMLMultinomial([]int{5, 5})
	r2, _ := NMLMultinomial([]int{1, 9})
	r3, _ := NMLMultinomial([]int{0, 10})
	if math.Abs(r1-r2) > 1e-10 {
		t.Errorf("regret(5,5)=%v != regret(1,9)=%v", r1, r2)
	}
	if math.Abs(r1-r3) > 1e-10 {
		t.Errorf("regret(5,5)=%v != regret(0,10)=%v", r1, r3)
	}
}

// TestNMLMultinomial_RegretMonotoneInK tests that for fixed n the
// regret is monotone non-decreasing in k: larger model classes
// have larger parametric complexity.
func TestNMLMultinomial_RegretMonotoneInK(t *testing.T) {
	// Hold n = 12 fixed; k = 2, 3, 4, 6, 12.
	prev := math.Inf(-1)
	for _, counts := range [][]int{
		{6, 6},
		{4, 4, 4},
		{3, 3, 3, 3},
		{2, 2, 2, 2, 2, 2},
		{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
	} {
		got, err := NMLMultinomial(counts)
		if err != nil {
			t.Fatalf("counts=%v: err=%v", counts, err)
		}
		if got < prev-1e-10 {
			t.Errorf("counts=%v regret=%v < prev=%v (non-monotone in k)", counts, got, prev)
		}
		prev = got
	}
}

// TestNMLMultinomial_EmptyCounts tests the error path.
func TestNMLMultinomial_EmptyCounts(t *testing.T) {
	_, err := NMLMultinomial([]int{})
	if !errors.Is(err, ErrEmptyCounts) {
		t.Errorf("empty: err=%v, want ErrEmptyCounts", err)
	}
}

// TestNMLMultinomial_NegativeInput tests the negative-count error
// path.
func TestNMLMultinomial_NegativeInput(t *testing.T) {
	_, err := NMLMultinomial([]int{5, -1, 3})
	if !errors.Is(err, ErrNegativeInput) {
		t.Errorf("negative: err=%v, want ErrNegativeInput", err)
	}
}

// =========================================================================
// NMLBernoulli — special case k = 2.
// =========================================================================

// TestNMLBernoulli_AgreesWithMultinomial tests that NMLBernoulli is
// indistinguishable from NMLMultinomial with k = 2 categories.
func TestNMLBernoulli_AgreesWithMultinomial(t *testing.T) {
	for _, n := range []int{4, 10, 50} {
		s := n / 2
		bern, err := NMLBernoulli(s, n)
		if err != nil {
			t.Fatalf("Bern(%d, %d) err: %v", s, n, err)
		}
		multi, err := NMLMultinomial([]int{s, n - s})
		if err != nil {
			t.Fatalf("Multi err: %v", err)
		}
		if math.Abs(bern-multi) > 1e-12 {
			t.Errorf("n=%d: Bern=%v != Multi=%v", n, bern, multi)
		}
	}
}

// TestNMLBernoulli_InvalidTrials tests trials < 1 and successes >
// trials error paths.
func TestNMLBernoulli_InvalidTrials(t *testing.T) {
	_, err := NMLBernoulli(0, 0)
	if !errors.Is(err, ErrInvalidTrials) {
		t.Errorf("trials=0: err=%v, want ErrInvalidTrials", err)
	}
	_, err = NMLBernoulli(11, 10)
	if !errors.Is(err, ErrInvalidTrials) {
		t.Errorf("s>n: err=%v, want ErrInvalidTrials", err)
	}
}

// TestNMLBernoulli_NegativeSuccesses tests negative successes.
func TestNMLBernoulli_NegativeSuccesses(t *testing.T) {
	_, err := NMLBernoulli(-1, 10)
	if !errors.Is(err, ErrNegativeInput) {
		t.Errorf("s=-1: err=%v, want ErrNegativeInput", err)
	}
}

// =========================================================================
// BernoulliCodeLength — full MDL codelength NLL + regret.
// =========================================================================

// TestBernoulliCodeLength_AllSuccess_ZeroNLL tests that
// (successes=n, trials=n) produces zero negative log-likelihood
// (the MLE p = 1 perfectly predicts the all-success data) and a
// codelength equal to the NML regret alone.
func TestBernoulliCodeLength_AllSuccess_ZeroNLL(t *testing.T) {
	cl, err := BernoulliCodeLength(10, 10)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	regret, _ := NMLBernoulli(10, 10)
	if math.Abs(cl-regret) > 1e-12 {
		t.Errorf("all-success: codelength=%v, want regret=%v", cl, regret)
	}
}

// TestBernoulliCodeLength_HalfHalf tests that successes = n/2
// produces NLL = n*log(2) + regret = n*0.6931 + regret.
func TestBernoulliCodeLength_HalfHalf(t *testing.T) {
	n, s := 10, 5
	cl, err := BernoulliCodeLength(s, n)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	regret, _ := NMLBernoulli(s, n)
	wantNLL := -float64(s)*math.Log(0.5) - float64(n-s)*math.Log(0.5)
	want := wantNLL + regret
	if math.Abs(cl-want) > 1e-10 {
		t.Errorf("half-half: codelength=%v, want %v (NLL=%v + regret=%v)",
			cl, want, wantNLL, regret)
	}
}

// =========================================================================
// GaussianCodeLength — fixed-Gaussian-hypothesis codelength.
// =========================================================================

// TestGaussianCodeLength_AtMean tests that samples all at the mean
// produce codelength = (n/2) * log(2*pi*sigma^2), the entropy term
// alone with no SSR penalty.
func TestGaussianCodeLength_AtMean(t *testing.T) {
	samples := []float64{5, 5, 5, 5, 5}
	mu, sigma := 5.0, 1.0
	cl := GaussianCodeLength(samples, mu, sigma)
	const logTwoPi = 1.8378770664093454836
	want := 0.5 * float64(len(samples)) * logTwoPi
	if math.Abs(cl-want) > 1e-10 {
		t.Errorf("at-mean: cl=%v, want %v", cl, want)
	}
}

// TestGaussianCodeLength_NonpositiveStdev tests sigma <= 0 returns
// +Inf (no model emits zero stdev for non-degenerate data).
func TestGaussianCodeLength_NonpositiveStdev(t *testing.T) {
	cl := GaussianCodeLength([]float64{1, 2, 3}, 0, 0)
	if !math.IsInf(cl, 1) {
		t.Errorf("zero-stdev: cl=%v, want +Inf", cl)
	}
	cl = GaussianCodeLength([]float64{1, 2, 3}, 0, -1)
	if !math.IsInf(cl, 1) {
		t.Errorf("neg-stdev: cl=%v, want +Inf", cl)
	}
}

// TestGaussianCodeLength_Empty tests empty samples -> 0 codelength.
func TestGaussianCodeLength_Empty(t *testing.T) {
	cl := GaussianCodeLength([]float64{}, 0, 1)
	if cl != 0 {
		t.Errorf("empty: cl=%v, want 0", cl)
	}
}

// =========================================================================
// BICShape + AICShape — adapter validation.
// =========================================================================

// TestBICShape_MatchesFormula tests BIC = -ll + (k/2)*log(n).
func TestBICShape_MatchesFormula(t *testing.T) {
	negLL := 100.0
	k, n := 3, 50
	got := BICShape(negLL, k, n)
	want := negLL + 0.5*float64(k)*math.Log(float64(n))
	if math.Abs(got-want) > 1e-12 {
		t.Errorf("BIC: got %v, want %v", got, want)
	}
}

// TestAICShape_MatchesFormula tests AIC = -ll + k.
func TestAICShape_MatchesFormula(t *testing.T) {
	negLL := 100.0
	k := 3
	got := AICShape(negLL, k)
	want := negLL + float64(k)
	if math.Abs(got-want) > 1e-12 {
		t.Errorf("AIC: got %v, want %v", got, want)
	}
}

// TestBICShape_NonFiniteLikelihood tests +Inf passthrough.
func TestBICShape_NonFiniteLikelihood(t *testing.T) {
	cl := BICShape(math.NaN(), 3, 50)
	if !math.IsInf(cl, 1) {
		t.Errorf("NaN ll: cl=%v, want +Inf", cl)
	}
	cl = BICShape(math.Inf(1), 3, 50)
	if !math.IsInf(cl, 1) {
		t.Errorf("+Inf ll: cl=%v, want +Inf", cl)
	}
}

// TestModelCodeLength_TinySample tests sampleSize <= 1 -> 0.
func TestModelCodeLength_TinySample(t *testing.T) {
	if ModelCodeLength(3, 1) != 0 {
		t.Error("n=1: want 0")
	}
	if ModelCodeLength(3, 0) != 0 {
		t.Error("n=0: want 0")
	}
}

// =========================================================================
// SelectMDL + SelectMDLWithMargin — argmin-on-codelengths.
// =========================================================================

// TestSelectMDL_BasicArgmin tests the canonical argmin path.
func TestSelectMDL_BasicArgmin(t *testing.T) {
	codeLengths := []float64{10.5, 9.2, 11.1, 9.0, 10.0}
	got, err := SelectMDL(codeLengths)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if got != 3 {
		t.Errorf("argmin: got %d, want 3", got)
	}
}

// TestSelectMDL_TieBreakLowerIndex tests that ties are resolved to
// the lower index.
func TestSelectMDL_TieBreakLowerIndex(t *testing.T) {
	codeLengths := []float64{5.0, 4.0, 4.0, 4.0, 6.0}
	got, err := SelectMDL(codeLengths)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if got != 1 {
		t.Errorf("tie-break: got %d, want 1", got)
	}
}

// TestSelectMDL_Empty tests the empty-list error.
func TestSelectMDL_Empty(t *testing.T) {
	_, err := SelectMDL([]float64{})
	if !errors.Is(err, ErrEmptyModelList) {
		t.Errorf("empty: err=%v, want ErrEmptyModelList", err)
	}
}

// TestSelectMDL_NonFinite tests the NaN / +Inf error.
func TestSelectMDL_NonFinite(t *testing.T) {
	_, err := SelectMDL([]float64{1.0, math.NaN(), 3.0})
	if !errors.Is(err, ErrNonFiniteCodeLength) {
		t.Errorf("NaN: err=%v, want ErrNonFiniteCodeLength", err)
	}
	_, err = SelectMDL([]float64{1.0, math.Inf(1), 3.0})
	if !errors.Is(err, ErrNonFiniteCodeLength) {
		t.Errorf("+Inf: err=%v, want ErrNonFiniteCodeLength", err)
	}
}

// TestSelectMDLWithMargin_GapToSecondBest tests that the margin is
// correctly computed.
func TestSelectMDLWithMargin_GapToSecondBest(t *testing.T) {
	codeLengths := []float64{10.0, 5.0, 8.0, 7.0}
	idx, margin, err := SelectMDLWithMargin(codeLengths)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if idx != 1 {
		t.Errorf("idx: got %d, want 1", idx)
	}
	// Best = 5 at idx 1; second-best = 7 at idx 3; margin = 2.
	if math.Abs(margin-2.0) > 1e-12 {
		t.Errorf("margin: got %v, want 2.0", margin)
	}
}

// TestSelectMDLWithMargin_SingleModel tests single-model -> +Inf
// margin.
func TestSelectMDLWithMargin_SingleModel(t *testing.T) {
	idx, margin, err := SelectMDLWithMargin([]float64{42.0})
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if idx != 0 {
		t.Errorf("idx: got %d, want 0", idx)
	}
	if !math.IsInf(margin, 1) {
		t.Errorf("margin: got %v, want +Inf", margin)
	}
}

// TestSelectMDL_ARp_ScenarioPicksBestLag exercises the canonical
// AR(p) lag-selection scenario described in the L12 entry §8: an
// AR(2)-best codelength surface should produce idx = 1 (lag 2 = idx
// in 0-indexed lag-1, lag-2, lag-3, lag-4 list).
func TestSelectMDL_ARp_ScenarioPicksBestLag(t *testing.T) {
	// Synthetic AR(p) codelengths: lag 2 wins.
	codeLengths := []float64{
		152.5, // lag 1: under-fit
		148.3, // lag 2: best fit
		149.0, // lag 3: over-fit
		149.7, // lag 4: more over-fit
	}
	idx, margin, err := SelectMDLWithMargin(codeLengths)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if idx != 1 {
		t.Errorf("AR(p): selected lag idx=%d, want 1 (lag-2)", idx)
	}
	if margin <= 0 {
		t.Errorf("AR(p): margin=%v, want > 0", margin)
	}
}
