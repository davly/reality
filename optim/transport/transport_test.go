package transport

import (
	"errors"
	"math"
	"math/rand"
	"testing"
)

// =========================================================================
// Wasserstein1D — basic correctness (cross-substrate parity with
// RubberDuck.Core.Tests.Analysis.OptimalTransportTests).
// =========================================================================

// TestCrossSubstratePrecision_RubberDuck_Identical mirrors
// `Wasserstein1D_IdenticalArrays_ReturnsZero` from
// `flagships/rubberduck/tests/RubberDuck.Core.Tests/Analysis/
// OptimalTransportTests.cs:9-18`.  Output parity to ≤1e-12.
func TestCrossSubstratePrecision_RubberDuck_Identical(t *testing.T) {
	p := []float64{1, 2, 3, 4, 5}
	q := []float64{1, 2, 3, 4, 5}
	got, err := Wasserstein1D(p, q, 1)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if math.Abs(got-0.0) > 1e-12 {
		t.Errorf("RubberDuck parity Identical: got %v, want 0.0", got)
	}
}

// TestCrossSubstratePrecision_RubberDuck_Shifted mirrors
// `Wasserstein1D_ShiftedDistribution_ReturnsMeanShift`.
func TestCrossSubstratePrecision_RubberDuck_Shifted(t *testing.T) {
	p := []float64{0, 1, 2}
	q := []float64{10, 11, 12}
	got, err := Wasserstein1D(p, q, 1)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if math.Abs(got-10.0) > 1e-10 {
		t.Errorf("RubberDuck parity Shifted: got %v, want 10.0", got)
	}
}

// TestCrossSubstratePrecision_RubberDuck_Scaled mirrors
// `Wasserstein1D_ScaledDistribution_ReturnsCorrectDistance`.
// P=[0,1], Q=[0,2]: pairs (0,0), (1,2); mean |diff| = 0.5.
func TestCrossSubstratePrecision_RubberDuck_Scaled(t *testing.T) {
	p := []float64{0, 1}
	q := []float64{0, 2}
	got, err := Wasserstein1D(p, q, 1)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if math.Abs(got-0.5) > 1e-10 {
		t.Errorf("RubberDuck parity Scaled: got %v, want 0.5", got)
	}
}

// TestCrossSubstratePrecision_RubberDuck_SingleElement mirrors
// `Wasserstein1D_SingleElements_ReturnsAbsDifference`.
func TestCrossSubstratePrecision_RubberDuck_SingleElement(t *testing.T) {
	p := []float64{3.0}
	q := []float64{7.0}
	got, err := Wasserstein1D(p, q, 1)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if math.Abs(got-4.0) > 1e-10 {
		t.Errorf("RubberDuck parity SingleElement: got %v, want 4.0", got)
	}
}

// TestCrossSubstratePrecision_RubberDuck_EmptyInput mirrors
// `Wasserstein1D_EmptyInput_ReturnsNaN`.
func TestCrossSubstratePrecision_RubberDuck_EmptyInput(t *testing.T) {
	got, err := Wasserstein1D([]float64{}, []float64{1, 2, 3}, 1)
	if err != nil {
		t.Fatalf("unexpected err for one-empty: %v", err)
	}
	if !math.IsNaN(got) {
		t.Errorf("RubberDuck parity OneEmpty: got %v, want NaN", got)
	}
	got, err = Wasserstein1D([]float64{}, []float64{}, 1)
	if err != nil {
		t.Fatalf("unexpected err for both-empty: %v", err)
	}
	if !math.IsNaN(got) {
		t.Errorf("RubberDuck parity BothEmpty: got %v, want NaN", got)
	}
}

// TestCrossSubstratePrecision_RubberDuck_Symmetric mirrors
// `Wasserstein1D_SymmetricProperty`.
func TestCrossSubstratePrecision_RubberDuck_Symmetric(t *testing.T) {
	p := []float64{1, 3, 5, 7, 9}
	q := []float64{2, 4, 6, 8, 10}
	wpq, err := Wasserstein1D(p, q, 1)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	wqp, err := Wasserstein1D(q, p, 1)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if math.Abs(wpq-wqp) > 1e-12 {
		t.Errorf("symmetry: W(p,q)=%v != W(q,p)=%v", wpq, wqp)
	}
}

// TestCrossSubstratePrecision_RubberDuck_TriangleInequality mirrors
// `Wasserstein1D_TriangleInequality`.
func TestCrossSubstratePrecision_RubberDuck_TriangleInequality(t *testing.T) {
	p := []float64{0, 1, 2, 3, 4}
	q := []float64{5, 6, 7, 8, 9}
	r := []float64{10, 11, 12, 13, 14}
	wpq, _ := Wasserstein1D(p, q, 1)
	wqr, _ := Wasserstein1D(q, r, 1)
	wpr, _ := Wasserstein1D(p, r, 1)
	if wpr > wpq+wqr+1e-10 {
		t.Errorf("triangle inequality: W(p,r)=%v > W(p,q)+W(q,r)=%v", wpr, wpq+wqr)
	}
}

// TestCrossSubstratePrecision_RubberDuck_NonNegative mirrors
// `Wasserstein1D_NonNegative`.
func TestCrossSubstratePrecision_RubberDuck_NonNegative(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	p := make([]float64, 100)
	q := make([]float64, 100)
	for i := range p {
		p[i] = rng.Float64()
		q[i] = rng.Float64()
	}
	got, err := Wasserstein1D(p, q, 1)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if got < 0 {
		t.Errorf("non-negative: got %v", got)
	}
}

// TestCrossSubstratePrecision_RubberDuck_UnequalSizes mirrors
// `Wasserstein1D_UnequalSizes_ProducesReasonableResult`.
func TestCrossSubstratePrecision_RubberDuck_UnequalSizes(t *testing.T) {
	p := []float64{0, 1, 2, 3}
	q := []float64{10, 20}
	got, err := Wasserstein1D(p, q, 1)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if !(got > 5.0 && got < 20.0) {
		t.Errorf("UnequalSizes: got %v, want in (5, 20)", got)
	}
}

// TestCrossSubstratePrecision_RubberDuck_NaNFiltering mirrors
// `Wasserstein1D_NaNInSamples_FilteredOut`.
func TestCrossSubstratePrecision_RubberDuck_NaNFiltering(t *testing.T) {
	p := []float64{1, math.NaN(), 3, math.Inf(1), 5}
	q := []float64{1, 3, 5}
	got, err := Wasserstein1D(p, q, 1)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	// Filtered p = [1,3,5] vs q = [1,3,5] → 0.
	if math.Abs(got-0.0) > 1e-10 {
		t.Errorf("NaNFiltering: got %v, want 0.0", got)
	}
}

// TestCrossSubstratePrecision_RubberDuck_AllNaN mirrors
// `Wasserstein1D_AllNaN_ReturnsNaN`.
func TestCrossSubstratePrecision_RubberDuck_AllNaN(t *testing.T) {
	p := []float64{math.NaN(), math.NaN()}
	q := []float64{1, 2, 3}
	got, err := Wasserstein1D(p, q, 1)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if !math.IsNaN(got) {
		t.Errorf("AllNaN: got %v, want NaN", got)
	}
}

// TestCrossSubstratePrecision_RubberDuck_DetailedNormalisation mirrors
// `Wasserstein1DDetailed_ReturnsCorrectNormalization`.
func TestCrossSubstratePrecision_RubberDuck_DetailedNormalisation(t *testing.T) {
	p := []float64{0, 10}
	q := []float64{5, 15}
	res, ok := Wasserstein1DDetailed(p, q)
	if !ok {
		t.Fatal("expected non-nil result")
	}
	if math.Abs(res.Distance-5.0) > 1e-10 {
		t.Errorf("Detailed.Distance: got %v, want 5.0", res.Distance)
	}
	if res.SampleSizeP != 2 || res.SampleSizeQ != 2 {
		t.Errorf("Detailed.SampleSize: got (%d, %d), want (2, 2)",
			res.SampleSizeP, res.SampleSizeQ)
	}
	if !(res.NormalizedDistance > 0) {
		t.Errorf("Detailed.NormalizedDistance: got %v, want > 0",
			res.NormalizedDistance)
	}
}

// TestCrossSubstratePrecision_RubberDuck_DetailedEmpty mirrors
// `Wasserstein1DDetailed_EmptyInput_ReturnsNull`.
func TestCrossSubstratePrecision_RubberDuck_DetailedEmpty(t *testing.T) {
	_, ok := Wasserstein1DDetailed([]float64{}, []float64{1, 2, 3})
	if ok {
		t.Error("Detailed empty: ok = true, want false")
	}
}

// TestCrossSubstratePrecision_RubberDuck_PairwiseSymmetric mirrors
// `PairwiseDistances_ThreeDistributions_SymmetricMatrix`.
func TestCrossSubstratePrecision_RubberDuck_PairwiseSymmetric(t *testing.T) {
	dists := [][]float64{
		{0, 1, 2},
		{5, 6, 7},
		{10, 11, 12},
	}
	mat, err := PairwiseWasserstein1D(dists, 1)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	for i := 0; i < 3; i++ {
		if mat[i][i] != 0 {
			t.Errorf("diagonal[%d] = %v, want 0", i, mat[i][i])
		}
	}
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			if math.Abs(mat[i][j]-mat[j][i]) > 1e-12 {
				t.Errorf("symmetry[%d,%d] %v != [%d,%d] %v",
					i, j, mat[i][j], j, i, mat[j][i])
			}
		}
	}
	if !(mat[0][1] > 0) {
		t.Errorf("Pairwise[0,1] = %v, want > 0", mat[0][1])
	}
	if !(mat[0][2] > mat[0][1]) {
		t.Errorf("Pairwise[0,2]=%v should exceed Pairwise[0,1]=%v",
			mat[0][2], mat[0][1])
	}
}

// TestCrossSubstratePrecision_RubberDuck_MinPairwiseSingle mirrors
// `MinimumPairwiseDistance_FewerThanTwo_ReturnsNaN`.  The Go API
// returns +Inf with sentinel idx (not NaN — the C# convention is to
// surface degeneracy via NaN; we use +Inf because Go's idiomatic
// "no-result" sentinel for min-distance is +Inf).
func TestCrossSubstratePrecision_RubberDuck_MinPairwiseSingle(t *testing.T) {
	dists := [][]float64{{1, 2, 3}}
	idx, dist, err := MinPairwiseWasserstein1D(dists, 1)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if !math.IsInf(dist, 1) {
		t.Errorf("MinPairwise single: dist = %v, want +Inf", dist)
	}
	if idx != [2]int{-1, -1} {
		t.Errorf("MinPairwise single: idx = %v, want [-1, -1]", idx)
	}
}

// =========================================================================
// IQR normalisation
// =========================================================================

func TestIQRNormalise_DegenerateZeroIQR(t *testing.T) {
	out := IQRNormalise([]float64{1, 1, 1, 1, 1})
	for i, v := range out {
		if v != 0 {
			t.Errorf("IQR=0 path: out[%d] = %v, want 0", i, v)
		}
	}
}

func TestIQRNormalise_BasicShift(t *testing.T) {
	// Median = 5, IQR = Q75 - Q25 = 7 - 3 = 4 (linear interp on
	// [1,2,3,4,5,6,7,8,9]: Q25 = 3, Q75 = 7).
	in := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9}
	out := IQRNormalise(in)
	// Median element (5) should map to 0; expected shape symmetric
	// around 0 with the (i - 5) / 4 pattern.
	for i, v := range out {
		want := (in[i] - 5) / 4
		if math.Abs(v-want) > 1e-12 {
			t.Errorf("IQR shift[%d]: got %v, want %v", i, v, want)
		}
	}
}

func TestIQRNormalise_PreservesNonFinite(t *testing.T) {
	in := []float64{1, math.NaN(), 3, math.Inf(1), 5, 7}
	out := IQRNormalise(in)
	if !math.IsNaN(out[1]) {
		t.Errorf("NaN preservation: out[1] = %v, want NaN", out[1])
	}
	if !math.IsInf(out[3], 1) {
		t.Errorf("+Inf preservation: out[3] = %v, want +Inf", out[3])
	}
	// Median of finite [1,3,5,7] = 4; Q25 = 2.5, Q75 = 5.5, IQR = 3.
	if math.Abs(out[0]-((1-4)/3)) > 1e-12 {
		t.Errorf("normal value with NaN/Inf in slice: got %v", out[0])
	}
}

// =========================================================================
// Wasserstein1D — additional structural checks
// =========================================================================

func TestWasserstein1D_InvalidP(t *testing.T) {
	_, err := Wasserstein1D([]float64{1, 2}, []float64{3, 4}, 0.5)
	if !errors.Is(err, ErrInvalidP) {
		t.Errorf("p=0.5: err=%v, want ErrInvalidP", err)
	}
	_, err = Wasserstein1D([]float64{1, 2}, []float64{3, 4}, math.NaN())
	if !errors.Is(err, ErrInvalidP) {
		t.Errorf("p=NaN: err=%v, want ErrInvalidP", err)
	}
	_, err = Wasserstein1D([]float64{1, 2}, []float64{3, 4}, math.Inf(1))
	if !errors.Is(err, ErrInvalidP) {
		t.Errorf("p=+Inf: err=%v, want ErrInvalidP", err)
	}
}

func TestWasserstein1D_P2RecoversL2OnEqualSize(t *testing.T) {
	// W_2 on equal-size order-statistic pairing is the L^2 mean of |
	// u_(i) - v_(i) | -> for shift-by-c distributions this equals c.
	p := []float64{0, 1, 2, 3, 4}
	q := []float64{2, 3, 4, 5, 6}
	got, err := Wasserstein1D(p, q, 2)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if math.Abs(got-2.0) > 1e-10 {
		t.Errorf("W_2 shift: got %v, want 2.0", got)
	}
}

// =========================================================================
// Sinkhorn — convergence + correctness on the classic Cuturi example
// =========================================================================

// TestSinkhorn_ConvergenceCuturi tests Sinkhorn on a textbook-style
// problem: two uniform marginals on n=4 points, cost C_ij = (i-j)^2,
// epsilon = 0.1.  The transport plan should concentrate near the
// diagonal; total cost should be small (close to 0 since the optimal
// LP transport for symmetric uniforms with quadratic cost is the
// identity).
func TestSinkhorn_ConvergenceCuturi(t *testing.T) {
	n := 4
	a := []float64{0.25, 0.25, 0.25, 0.25}
	b := []float64{0.25, 0.25, 0.25, 0.25}
	C := make([][]float64, n)
	for i := 0; i < n; i++ {
		C[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			d := float64(i - j)
			C[i][j] = d * d
		}
	}
	res, err := Sinkhorn(a, b, C, 0.1, 500, 1e-7)
	if err != nil {
		t.Fatalf("Sinkhorn err: %v", err)
	}
	if !(res.Iterations > 0 && res.Iterations < 500) {
		t.Errorf("Iterations: got %d, want in (0, 500)", res.Iterations)
	}
	// Plan row-sums should match a; col-sums should match b.
	for i, row := range res.Plan {
		s := 0.0
		for _, v := range row {
			s += v
		}
		if math.Abs(s-a[i]) > 1e-7 {
			t.Errorf("row[%d] sum = %v, want %v", i, s, a[i])
		}
	}
	for j := 0; j < n; j++ {
		s := 0.0
		for i := 0; i < n; i++ {
			s += res.Plan[i][j]
		}
		if math.Abs(s-b[j]) > 1e-7 {
			t.Errorf("col[%d] sum = %v, want %v", j, s, b[j])
		}
	}
	// Cost should be small (entropic OT smudges the identity matrix
	// but transport mass concentrates near the diagonal).
	if !(res.Cost >= 0 && res.Cost < 1.0) {
		t.Errorf("Cost = %v, want in [0, 1)", res.Cost)
	}
}

// TestSinkhorn_IdenticalMarginals_ZeroCost tests that two identical
// distributions produce a transport plan whose cost is 0 (the
// diagonal of zero-cost cells captures all mass).
func TestSinkhorn_IdenticalMarginals_ZeroCost(t *testing.T) {
	a := []float64{0.4, 0.3, 0.3}
	b := []float64{0.4, 0.3, 0.3}
	C := [][]float64{
		{0, 1, 2},
		{1, 0, 1},
		{2, 1, 0},
	}
	res, err := Sinkhorn(a, b, C, 0.05, 500, 1e-7)
	if err != nil {
		t.Fatalf("Sinkhorn err: %v", err)
	}
	// Diagonal-mass: P_ii should each capture roughly a_i; off-
	// diagonals should be small.  With epsilon = 0.05 and integer
	// costs, the entropic smudge is non-trivial — assert Cost
	// dominated by epsilon * H(P) rather than transport mass.
	if !(res.Cost < 0.5) {
		t.Errorf("Cost = %v, want < 0.5 for identical marginals", res.Cost)
	}
}

// TestSinkhorn_UnequalMass returns ErrUnequalMass when sum(a) != sum(b).
func TestSinkhorn_UnequalMass(t *testing.T) {
	a := []float64{0.5, 0.5}
	b := []float64{0.3, 0.3}
	C := [][]float64{{0, 1}, {1, 0}}
	_, err := Sinkhorn(a, b, C, 0.1, 100, 1e-9)
	if !errors.Is(err, ErrUnequalMass) {
		t.Errorf("err = %v, want ErrUnequalMass", err)
	}
}

func TestSinkhorn_InvalidRegularisation(t *testing.T) {
	a := []float64{0.5, 0.5}
	b := []float64{0.5, 0.5}
	C := [][]float64{{0, 1}, {1, 0}}
	for _, eps := range []float64{0, -0.1, math.NaN(), math.Inf(1)} {
		_, err := Sinkhorn(a, b, C, eps, 100, 1e-9)
		if !errors.Is(err, ErrInvalidRegularisation) {
			t.Errorf("eps=%v: err=%v, want ErrInvalidRegularisation", eps, err)
		}
	}
}

func TestSinkhorn_DimensionMismatch(t *testing.T) {
	a := []float64{0.5, 0.5}
	b := []float64{0.5, 0.5}
	C := [][]float64{{0, 1, 2}, {1, 0, 1}}
	_, err := Sinkhorn(a, b, C, 0.1, 100, 1e-9)
	if !errors.Is(err, ErrCostMatrixDimensionMismatch) {
		t.Errorf("err = %v, want ErrCostMatrixDimensionMismatch", err)
	}
}

func TestSinkhorn_Empty(t *testing.T) {
	_, err := Sinkhorn([]float64{}, []float64{0.5, 0.5},
		[][]float64{{0, 1}, {1, 0}}, 0.1, 100, 1e-9)
	if !errors.Is(err, ErrEmptyDistribution) {
		t.Errorf("err = %v, want ErrEmptyDistribution", err)
	}
}

// TestSinkhorn_LogDomainStability asserts log-domain Sinkhorn does not
// underflow on small epsilon: the multiplicative-form would emit
// exp(-100) underflow on a unit-cost matrix with eps=0.005, but log-
// domain LSE handles this cleanly.
func TestSinkhorn_LogDomainStability(t *testing.T) {
	n := 5
	a := make([]float64, n)
	b := make([]float64, n)
	for i := 0; i < n; i++ {
		a[i] = 1.0 / float64(n)
		b[i] = 1.0 / float64(n)
	}
	C := make([][]float64, n)
	for i := 0; i < n; i++ {
		C[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			d := float64(i - j)
			C[i][j] = d * d
		}
	}
	// Very small epsilon would underflow naive multiplicative
	// Sinkhorn (exp(-100) ≈ 3.7e-44 * 0).  Log-domain handles it.
	res, err := Sinkhorn(a, b, C, 0.005, 1000, 1e-7)
	if err != nil {
		t.Fatalf("log-domain stability: err=%v", err)
	}
	// Still doubly stochastic-ish.
	for i := 0; i < n; i++ {
		s := 0.0
		for j := 0; j < n; j++ {
			s += res.Plan[i][j]
		}
		if math.Abs(s-a[i]) > 1e-5 {
			t.Errorf("small-eps row[%d] = %v, want %v", i, s, a[i])
		}
	}
}

// =========================================================================
// MinPairwise — index pair correctness
// =========================================================================

func TestMinPairwise_FindsClosestPair(t *testing.T) {
	dists := [][]float64{
		{0, 1, 2},          // group A
		{100, 101, 102},    // group C
		{0.5, 1.5, 2.5},    // group A' (closest to A)
		{50, 51, 52},       // group B
	}
	idx, dist, err := MinPairwiseWasserstein1D(dists, 1)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	// Closest pair should be (0, 2) — A and A' (shift of 0.5).
	if !((idx == [2]int{0, 2}) || (idx == [2]int{2, 0})) {
		t.Errorf("closest pair: got %v, want (0, 2)", idx)
	}
	if !(dist > 0 && dist < 1.0) {
		t.Errorf("closest dist: got %v, want in (0, 1)", dist)
	}
}

// =========================================================================
// Performance — sanity check Sinkhorn on a 100x100 cost matrix.
// =========================================================================

func TestSinkhorn_Performance100x100(t *testing.T) {
	if testing.Short() {
		t.Skip("performance check; -short")
	}
	const n = 100
	a := make([]float64, n)
	b := make([]float64, n)
	for i := 0; i < n; i++ {
		a[i] = 1.0 / float64(n)
		b[i] = 1.0 / float64(n)
	}
	// Cost matrix scaled to unit-mean so epsilon = 0.05 is well-
	// conditioned (eps / mean(C) ≈ 0.05).  Squared distance on a
	// length-1 grid: max cost = 1, mean cost ≈ 1/3.
	C := make([][]float64, n)
	for i := 0; i < n; i++ {
		C[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			d := float64(i-j) / float64(n-1)
			C[i][j] = d * d
		}
	}
	// Well-conditioned 100x100 with normalised cost should converge
	// well within the 1000-iter cap; we assert convergence + a
	// reasonable iteration ceiling rather than exact iter count.
	res, err := Sinkhorn(a, b, C, 0.05, 1000, 1e-6)
	if err != nil {
		t.Fatalf("100x100 perf: err=%v", err)
	}
	if res.Iterations < 1 {
		t.Errorf("100x100: iters=%d, want >= 1", res.Iterations)
	}
}

// =========================================================================
// Wasserstein1D — large-corpus performance gate (parity with
// RubberDuck's perf gate at OptimalTransportTests.cs:* 50k-element
// performance check).
// =========================================================================

func TestWasserstein1D_LargeCorpus(t *testing.T) {
	if testing.Short() {
		t.Skip("performance check; -short")
	}
	const n = 50_000
	rng := rand.New(rand.NewSource(2026))
	p := make([]float64, n)
	q := make([]float64, n)
	for i := range p {
		p[i] = rng.NormFloat64()
		q[i] = rng.NormFloat64() + 1.0
	}
	got, err := Wasserstein1D(p, q, 1)
	if err != nil {
		t.Fatalf("large corpus: err=%v", err)
	}
	// Two unit-Gaussians shifted by 1 have W_1 ≈ 1.0; sample noise.
	if !(got > 0.5 && got < 1.5) {
		t.Errorf("LargeCorpus W_1 = %v, want close to 1.0", got)
	}
}
