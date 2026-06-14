package moments

import (
	"math"
	"testing"
)

// goldenData is the classic textbook series: mean = 5, population variance = 4,
// sample variance = 32/7.
var goldenData = []float64{2, 4, 4, 4, 5, 5, 7, 9}

const (
	goldenMean   = 5.0
	goldenPopVar = 4.0
	goldenSamVar = 32.0 / 7.0 // ~4.5714285714285714
)

func feed(data []float64) *Welford {
	var w Welford
	for _, x := range data {
		w.Update(x)
	}
	return &w
}

// approx returns true when a and b agree to within tol in absolute terms.
func approx(a, b, tol float64) bool {
	return math.Abs(a-b) <= tol
}

func TestGoldenMeanVariance(t *testing.T) {
	w := feed(goldenData)

	if got := w.Count(); got != len(goldenData) {
		t.Fatalf("Count = %d, want %d", got, len(goldenData))
	}
	if got := w.Mean(); got != goldenMean {
		t.Errorf("Mean = %v, want exactly %v", got, goldenMean)
	}
	if got := w.PopVariance(); got != goldenPopVar {
		t.Errorf("PopVariance = %v, want exactly %v", got, goldenPopVar)
	}
	// Sample variance is 32/7, an inexact float; require tight agreement to the
	// same expression so any divisor/rounding regression is caught.
	if got := w.Variance(); !approx(got, goldenSamVar, 1e-12) {
		t.Errorf("Variance (sample) = %v, want %v", got, goldenSamVar)
	}
	if got, want := w.StdDev(), math.Sqrt(goldenSamVar); !approx(got, want, 1e-12) {
		t.Errorf("StdDev = %v, want %v", got, want)
	}
}

// naiveVariance is the textbook two-pass identity that forms sum(x^2) and
// subtracts n*mean^2 — the formulation Welford is meant to replace. It is
// CORRECT in exact arithmetic but suffers catastrophic cancellation in float64
// when the data sit far from zero, because sum(x^2) and n*mean^2 become huge and
// nearly equal. We use it as the foil for the offset-invariance test.
func naiveSampleVariance(data []float64) float64 {
	n := float64(len(data))
	var sum, sumSq float64
	for _, x := range data {
		sum += x
		sumSq += x * x
	}
	mean := sum / n
	return (sumSq - n*mean*mean) / (n - 1)
}

// TestNumericalStabilityOffsetInvariance is the KEY discriminating test. The
// same series shifted by a large constant offset has the SAME variance
// (variance is translation-invariant), yet the naive sum-of-squares formula
// loses nearly all precision at a 1e9 offset while Welford stays exact to a
// tight tolerance. This is what justifies the package's existence, not mere
// correctness on small inputs.
func TestNumericalStabilityOffsetInvariance(t *testing.T) {
	const offset = 1e9

	shifted := make([]float64, len(goldenData))
	for i, x := range goldenData {
		shifted[i] = offset + x
	}

	w := feed(shifted)

	// Welford must be offset-invariant: population variance still 4, sample
	// variance still 32/7, to a tight tolerance despite the 1e9 baseline.
	if got := w.PopVariance(); !approx(got, goldenPopVar, 1e-6) {
		t.Errorf("Welford PopVariance at 1e9 offset = %v, want %v (offset-invariant)", got, goldenPopVar)
	}
	if got := w.Variance(); !approx(got, goldenSamVar, 1e-6) {
		t.Errorf("Welford Variance at 1e9 offset = %v, want %v (offset-invariant)", got, goldenSamVar)
	}
	// Welford's residual error at a 1e9 offset is bounded by the float64
	// relative precision around 1e9 (~1e-7), i.e. roughly 1e-7 in absolute
	// terms on a ~4.57 variance. Assert it stays comfortably within that
	// regime — orders of magnitude tighter than the naive form checked below.
	if got := w.Variance(); !approx(got, goldenSamVar, 1e-6) {
		t.Errorf("Welford Variance at 1e9 offset = %v drifted beyond the float64-precision floor (want %v)", got, goldenSamVar)
	}

	// The naive formula DEGRADES badly here: demonstrate it is materially wrong
	// so the comparison is real and not vacuous. On the unshifted data the naive
	// formula is fine; on the shifted data it loses precision catastrophically.
	naiveUnshifted := naiveSampleVariance(goldenData)
	if !approx(naiveUnshifted, goldenSamVar, 1e-9) {
		t.Fatalf("sanity: naive on unshifted data = %v, want %v", naiveUnshifted, goldenSamVar)
	}
	naiveShifted := naiveSampleVariance(shifted)
	naiveErr := math.Abs(naiveShifted - goldenSamVar)
	welfordErr := math.Abs(w.Variance() - goldenSamVar)
	if !(naiveErr > welfordErr) {
		t.Errorf("expected naive (err=%v) to degrade beyond Welford (err=%v) at 1e9 offset; "+
			"naiveShifted=%v welford=%v want=%v", naiveErr, welfordErr, naiveShifted, w.Variance(), goldenSamVar)
	}
	// Make the magnitude of the naive failure explicit: at 1e9 the naive error
	// should be gross (off by more than a whole unit of variance), whereas
	// Welford's error is sub-1e-6.
	if naiveErr < 0.5 {
		t.Errorf("naive error at 1e9 offset only %v — expected catastrophic cancellation (>0.5); "+
			"test may not be exercising the failure mode", naiveErr)
	}
	t.Logf("offset=1e9: welford err=%.3e, naive err=%.3e (naive degraded by %.3gx)",
		welfordErr, naiveErr, naiveErr/math.Max(welfordErr, math.SmallestNonzeroFloat64))
}

// TestMergeEqualsSingleStream verifies Chan-Golub-LeVeque parallel merge: a
// Welford over the first half merged with a Welford over the second half equals
// a single Welford streamed over the whole series, in BOTH mean and variance.
func TestMergeEqualsSingleStream(t *testing.T) {
	series := []float64{
		3.1, 7.4, -2.0, 5.5, 11.2, 0.3, -8.8, 4.4,
		9.9, 2.2, 6.7, -1.1, 13.0, 0.0, 5.05, -3.33,
	}
	// Deliberately uneven split to exercise nA != nB.
	mid := 5
	first := series[:mid]
	second := series[mid:]

	whole := feed(series)
	a := feed(first)
	b := feed(second)
	merged := Merge(*a, *b)

	if merged.Count() != whole.Count() {
		t.Fatalf("merged Count = %d, want %d", merged.Count(), whole.Count())
	}
	if !approx(merged.Mean(), whole.Mean(), 1e-12) {
		t.Errorf("merged Mean = %v, want %v", merged.Mean(), whole.Mean())
	}
	if !approx(merged.Variance(), whole.Variance(), 1e-12) {
		t.Errorf("merged Variance = %v, want %v", merged.Variance(), whole.Variance())
	}
	if !approx(merged.PopVariance(), whole.PopVariance(), 1e-12) {
		t.Errorf("merged PopVariance = %v, want %v", merged.PopVariance(), whole.PopVariance())
	}

	// The method form must agree with the package function.
	mMethod := a.Merge(*b)
	if mMethod != merged {
		t.Errorf("Welford.Merge method = %+v, want %+v", mMethod, merged)
	}
}

func TestMergeEmptyIdentity(t *testing.T) {
	var empty Welford
	w := feed(goldenData)

	if got := Merge(*w, empty); got != *w {
		t.Errorf("Merge(w, empty) = %+v, want %+v", got, *w)
	}
	if got := Merge(empty, *w); got != *w {
		t.Errorf("Merge(empty, w) = %+v, want %+v", got, *w)
	}
	if got := Merge(empty, empty); got != (Welford{}) {
		t.Errorf("Merge(empty, empty) = %+v, want zero", got)
	}
}

func TestEmptyAndSingleConventions(t *testing.T) {
	var empty Welford
	if got := empty.Count(); got != 0 {
		t.Errorf("empty Count = %d, want 0", got)
	}
	if got := empty.Mean(); got != 0 {
		t.Errorf("empty Mean = %v, want 0 (folio convention, not NaN)", got)
	}
	if got := empty.Variance(); got != 0 {
		t.Errorf("empty Variance = %v, want 0", got)
	}
	if got := empty.PopVariance(); got != 0 {
		t.Errorf("empty PopVariance = %v, want 0", got)
	}
	if got := empty.StdDev(); got != 0 {
		t.Errorf("empty StdDev = %v, want 0", got)
	}
	if got := empty.ZScore(5); got != 0 {
		t.Errorf("empty ZScore = %v, want 0 (no baseline)", got)
	}

	var one Welford
	one.Update(42)
	if got := one.Count(); got != 1 {
		t.Errorf("single Count = %d, want 1", got)
	}
	if got := one.Mean(); got != 42 {
		t.Errorf("single Mean = %v, want 42", got)
	}
	if got := one.Variance(); got != 0 {
		t.Errorf("single Variance = %v, want 0 (n<2)", got)
	}
	if got := one.PopVariance(); got != 0 {
		t.Errorf("single PopVariance = %v, want 0", got)
	}
	if got := one.StdDev(); got != 0 {
		t.Errorf("single StdDev = %v, want 0", got)
	}
}

func TestZScore(t *testing.T) {
	w := feed(goldenData) // mean 5, sample stddev sqrt(32/7)
	sd := math.Sqrt(goldenSamVar)

	if got, want := w.ZScore(goldenMean), 0.0; got != want {
		t.Errorf("ZScore(mean) = %v, want %v", got, want)
	}
	if got, want := w.ZScore(goldenMean+sd), 1.0; !approx(got, want, 1e-12) {
		t.Errorf("ZScore(mean+sd) = %v, want %v", got, want)
	}
	if got, want := w.ZScore(goldenMean-2*sd), -2.0; !approx(got, want, 1e-12) {
		t.Errorf("ZScore(mean-2sd) = %v, want %v", got, want)
	}

	// Degenerate baseline: constant stream => stddev 0.
	var c Welford
	for i := 0; i < 5; i++ {
		c.Update(7)
	}
	if got := c.ZScore(7); got != 0 {
		t.Errorf("ZScore at mean of constant stream = %v, want 0", got)
	}
	if got := c.ZScore(8); !math.IsInf(got, 1) {
		t.Errorf("ZScore above constant stream = %v, want +Inf", got)
	}
	if got := c.ZScore(6); !math.IsInf(got, -1) {
		t.Errorf("ZScore below constant stream = %v, want -Inf", got)
	}
}

func TestWelfordVecParityWithScalar(t *testing.T) {
	// Build a 3-dimensional stream; dimension 0 is the golden series (padded to
	// the vector length), dimensions 1 and 2 are independent series. Each
	// dimension must match a scalar Welford fed the same coordinate values.
	dim := 3
	rows := [][]float64{
		{2, 10.0, -1.0},
		{4, 12.5, -1.5},
		{4, 9.0, 0.0},
		{4, 11.0, 2.0},
		{5, 13.0, -3.0},
		{5, 8.5, 4.0},
		{7, 10.0, 1.0},
		{9, 14.0, -2.0},
	}

	wv := NewWelfordVec(dim)
	scalars := make([]*Welford, dim)
	for d := range scalars {
		scalars[d] = &Welford{}
	}
	for _, r := range rows {
		wv.Update(r)
		for d := 0; d < dim; d++ {
			scalars[d].Update(r[d])
		}
	}

	if wv.Count() != len(rows) {
		t.Fatalf("WelfordVec Count = %d, want %d", wv.Count(), len(rows))
	}
	if wv.Dim() != dim {
		t.Fatalf("WelfordVec Dim = %d, want %d", wv.Dim(), dim)
	}

	mean := wv.Mean()
	variance := wv.Variance()
	popVar := wv.PopVariance()
	stddev := wv.StdDev()
	for d := 0; d < dim; d++ {
		if !approx(mean[d], scalars[d].Mean(), 1e-12) {
			t.Errorf("dim %d Mean = %v, want %v", d, mean[d], scalars[d].Mean())
		}
		if !approx(variance[d], scalars[d].Variance(), 1e-12) {
			t.Errorf("dim %d Variance = %v, want %v", d, variance[d], scalars[d].Variance())
		}
		if !approx(popVar[d], scalars[d].PopVariance(), 1e-12) {
			t.Errorf("dim %d PopVariance = %v, want %v", d, popVar[d], scalars[d].PopVariance())
		}
		if !approx(stddev[d], scalars[d].StdDev(), 1e-12) {
			t.Errorf("dim %d StdDev = %v, want %v", d, stddev[d], scalars[d].StdDev())
		}
	}

	// Dimension 0 carries the golden series — pin it to the hand-computed value.
	if !approx(mean[0], goldenMean, 1e-12) {
		t.Errorf("dim 0 Mean = %v, want golden %v", mean[0], goldenMean)
	}
	if !approx(popVar[0], goldenPopVar, 1e-12) {
		t.Errorf("dim 0 PopVariance = %v, want golden %v", popVar[0], goldenPopVar)
	}
	if !approx(variance[0], goldenSamVar, 1e-12) {
		t.Errorf("dim 0 Variance = %v, want golden %v", variance[0], goldenSamVar)
	}
}

func TestWelfordVecReturnsCopies(t *testing.T) {
	wv := NewWelfordVec(2)
	wv.Update([]float64{1, 2})
	wv.Update([]float64{3, 4})

	m1 := wv.Mean()
	m1[0] = 999 // mutate the returned slice
	m2 := wv.Mean()
	if m2[0] == 999 {
		t.Errorf("Mean returned an aliased slice; internal state was mutated")
	}
}

func TestWelfordVecEmptyConventions(t *testing.T) {
	wv := NewWelfordVec(3)
	if got := wv.Count(); got != 0 {
		t.Errorf("empty WelfordVec Count = %d, want 0", got)
	}
	for d, v := range wv.Mean() {
		if v != 0 {
			t.Errorf("empty Mean[%d] = %v, want 0", d, v)
		}
	}
	for d, v := range wv.Variance() {
		if v != 0 {
			t.Errorf("empty Variance[%d] = %v, want 0", d, v)
		}
	}
}

func TestWelfordVecDimMismatchPanics(t *testing.T) {
	wv := NewWelfordVec(3)
	defer func() {
		if recover() == nil {
			t.Errorf("Update with wrong dimension did not panic")
		}
	}()
	wv.Update([]float64{1, 2}) // dim 2 != 3
}

func TestNewWelfordVecRejectsBadDim(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Errorf("NewWelfordVec(0) did not panic")
		}
	}()
	NewWelfordVec(0)
}

// TestM2Accessor pins the new M2 accessor against the variance accessors it
// feeds: M2 is the sum of squared deviations, so Variance == M2/(n-1) and
// PopVariance == M2/n exactly. For the golden series M2 = popVar*n = 4*8 = 32.
func TestM2Accessor(t *testing.T) {
	w := feed(goldenData)

	if got, want := w.M2(), goldenPopVar*float64(len(goldenData)); !approx(got, want, 1e-12) {
		t.Errorf("M2 = %v, want %v (popVar*n)", got, want)
	}
	// M2 must be the exact numerator of both variance accessors.
	if got, want := w.Variance(), w.M2()/float64(w.Count()-1); !approx(got, want, 1e-15) {
		t.Errorf("Variance = %v but M2/(n-1) = %v — M2 is not the sample-variance numerator", got, want)
	}
	if got, want := w.PopVariance(), w.M2()/float64(w.Count()); !approx(got, want, 1e-15) {
		t.Errorf("PopVariance = %v but M2/n = %v — M2 is not the pop-variance numerator", got, want)
	}

	// Empty and single-observation accumulators carry no dispersion => M2 == 0.
	var empty Welford
	if got := empty.M2(); got != 0 {
		t.Errorf("empty M2 = %v, want 0", got)
	}
	var one Welford
	one.Update(42)
	if got := one.M2(); got != 0 {
		t.Errorf("single-observation M2 = %v, want 0", got)
	}
}

// TestNewWelfordRoundTrip is the headline persistence test: a Welford built by
// streaming, decomposed into its (Count, Mean, M2) triple, and reconstructed via
// NewWelford must be byte-for-byte identical downstream — every accessor equal
// EXACTLY, not just to a tolerance, since reconstruction copies the same scalars.
func TestNewWelfordRoundTrip(t *testing.T) {
	w1 := feed(goldenData)

	// Decompose to the persisted triple, then rehydrate.
	w2 := NewWelford(w1.Count(), w1.Mean(), w1.M2())

	if w2.Count() != w1.Count() {
		t.Errorf("round-trip Count = %d, want %d", w2.Count(), w1.Count())
	}
	if w2.Mean() != w1.Mean() {
		t.Errorf("round-trip Mean = %v, want exactly %v", w2.Mean(), w1.Mean())
	}
	if w2.M2() != w1.M2() {
		t.Errorf("round-trip M2 = %v, want exactly %v", w2.M2(), w1.M2())
	}
	if w2.Variance() != w1.Variance() {
		t.Errorf("round-trip Variance = %v, want exactly %v", w2.Variance(), w1.Variance())
	}
	if w2.PopVariance() != w1.PopVariance() {
		t.Errorf("round-trip PopVariance = %v, want exactly %v", w2.PopVariance(), w1.PopVariance())
	}
	if w2.StdDev() != w1.StdDev() {
		t.Errorf("round-trip StdDev = %v, want exactly %v", w2.StdDev(), w1.StdDev())
	}
	// The reconstructed value must be struct-identical to the streamed one, so it
	// is interchangeable everywhere a Welford value flows (e.g. Merge, ==).
	if w2 != *w1 {
		t.Errorf("round-trip struct = %+v, want %+v", w2, *w1)
	}
}

// TestNewWelfordMergeEquivalence proves the reconstructed state behaves
// identically DOWNSTREAM: merging a rehydrated w2 with a third stream w3 yields
// the same result as merging the original streamed w1 with w3. If NewWelford
// dropped or mangled any field, the merge (which reads n, mean AND m2) diverges.
func TestNewWelfordMergeEquivalence(t *testing.T) {
	w1 := feed(goldenData)
	w2 := NewWelford(w1.Count(), w1.Mean(), w1.M2())

	w3 := feed([]float64{100.5, -7.25, 3.0, 88.0, 12.5, -1.0, 0.0})

	fromStreamed := Merge(*w1, *w3)
	fromRehydrated := Merge(w2, *w3)

	if fromRehydrated != fromStreamed {
		t.Errorf("Merge(rehydrated, w3) = %+v, want %+v (Merge(streamed, w3)) — "+
			"reconstructed state is not downstream-equivalent", fromRehydrated, fromStreamed)
	}
}

// TestNewWelfordGolden pins NewWelford against hand-computed values and the
// documented edge conventions, independent of any streaming.
func TestNewWelfordGolden(t *testing.T) {
	// Direct golden: n=8, mean=5, M2=32 => PopVar 4, SamVar 32/7, M2() 32.
	w := NewWelford(8, 5.0, 32.0)
	if got := w.Count(); got != 8 {
		t.Errorf("Count = %d, want 8", got)
	}
	if got := w.Mean(); got != 5.0 {
		t.Errorf("Mean = %v, want 5", got)
	}
	if got := w.M2(); got != 32.0 {
		t.Errorf("M2 = %v, want 32", got)
	}
	if got := w.PopVariance(); got != 4.0 {
		t.Errorf("PopVariance = %v, want exactly 4", got)
	}
	if got, want := w.Variance(), 32.0/7.0; !approx(got, want, 1e-12) {
		t.Errorf("Variance = %v, want %v (32/7)", got, want)
	}

	// Empty: NewWelford(0,0,0) is the zero accumulator.
	if got := NewWelford(0, 0, 0); got != (Welford{}) {
		t.Errorf("NewWelford(0,0,0) = %+v, want zero Welford", got)
	}
	// n<0 is clamped to empty; mean/m2 ignored.
	if got := NewWelford(-3, 99, 99); got != (Welford{}) {
		t.Errorf("NewWelford(-3,99,99) = %+v, want zero Welford (n<0 => empty)", got)
	}
	empty := NewWelford(0, 0, 0)
	if empty.Count() != 0 {
		t.Errorf("empty Count = %d, want 0", empty.Count())
	}
	if empty.Variance() != 0 {
		t.Errorf("empty Variance = %v, want 0", empty.Variance())
	}

	// n=1: a single observation carries no dispersion => Variance 0 (n<2).
	one := NewWelford(1, 42.0, 0)
	if got := one.Count(); got != 1 {
		t.Errorf("n=1 Count = %d, want 1", got)
	}
	if got := one.Mean(); got != 42.0 {
		t.Errorf("n=1 Mean = %v, want 42", got)
	}
	if got := one.Variance(); got != 0 {
		t.Errorf("n=1 Variance = %v, want 0 (n<2)", got)
	}
}

// TestNewWelfordM2IsLoadBearing is the MUTATION test: it proves M2 is actually
// carried through reconstruction. If NewWelford ignored its m2 argument (e.g.
// hard-coded 0), the rehydrated dispersion would collapse to 0 and these
// assertions would fire — so a passing run is positive evidence that M2 is
// load-bearing in the round-trip, not incidental.
func TestNewWelfordM2IsLoadBearing(t *testing.T) {
	w1 := feed(goldenData)

	// The faithful reconstruction must reproduce the non-zero dispersion.
	good := NewWelford(w1.Count(), w1.Mean(), w1.M2())
	if good.Variance() == 0 {
		t.Fatalf("sanity: golden series has non-zero variance but reconstruction is 0")
	}
	if good.Variance() != w1.Variance() || good.M2() != w1.M2() {
		t.Errorf("faithful reconstruction diverged: M2 %v/%v variance %v/%v",
			good.M2(), w1.M2(), good.Variance(), w1.Variance())
	}

	// The MUTANT — dropping m2 to 0 — MUST differ. This is the discriminating
	// check: if it did NOT differ, M2 would be dead weight in NewWelford.
	mutant := NewWelford(w1.Count(), w1.Mean(), 0)
	if mutant.M2() != 0 {
		t.Fatalf("mutant M2 = %v, want 0 (test wiring)", mutant.M2())
	}
	if mutant.Variance() != 0 {
		t.Errorf("mutant (m2=0) Variance = %v, want 0", mutant.Variance())
	}
	if mutant.M2() == good.M2() {
		t.Errorf("mutant M2 (%v) equals faithful M2 (%v) — NewWelford is ignoring m2; "+
			"round-trip would silently lose dispersion", mutant.M2(), good.M2())
	}
	if mutant.Variance() == good.Variance() {
		t.Errorf("mutant Variance equals faithful Variance — m2 is not load-bearing")
	}
}
