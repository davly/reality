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
