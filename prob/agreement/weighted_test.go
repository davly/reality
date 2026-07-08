package agreement

import "testing"

// TestWeightedKappa_TwoCategoryReductionIdentity: with exactly 2
// categories there is only one possible rank distance, so both Linear and
// Quadratic weights collapse to the unweighted disagreement weight (0 on
// the diagonal, 1 off it) — WeightedKappa on Cohen's (1960) 50-proposal
// example must equal the unweighted CohenKappa result exactly, under
// either scheme.
func TestWeightedKappa_TwoCategoryReductionIdentity(t *testing.T) {
	a, b := buildFromCounts([]int{0, 1}, [][]int{
		{20, 5},
		{10, 15},
	})

	unweighted, err := CohenKappa(a, b)
	if err != nil {
		t.Fatalf("CohenKappa: unexpected error: %v", err)
	}

	for _, scheme := range []WeightScheme{Linear, Quadratic} {
		got, err := WeightedKappa(a, b, scheme)
		if err != nil {
			t.Fatalf("WeightedKappa(scheme=%v): unexpected error: %v", scheme, err)
		}
		if diff := got - unweighted; diff > 1e-9 || diff < -1e-9 {
			t.Errorf("WeightedKappa(scheme=%v) = %v, want %v (k=2 reduction identity)", scheme, got, unweighted)
		}
		if diff := got - 0.40; diff > 1e-9 || diff < -1e-9 {
			t.Errorf("WeightedKappa(scheme=%v) = %v, want 0.40 (Cohen 1960)", scheme, got)
		}
	}
}

// TestWeightedKappa_ThreeCategoryHandDerived cross-checks WeightedKappa
// against an independent, algebraically-equivalent implementation of the
// AGREEMENT-weight form of the same statistic:
//
//	kappa = (po(w) - pe(w)) / (1 - pe(w)),  w(c,k) = 1 - v(c,k)/v_max
//
// where v is the disagreement weight WeightedKappa itself uses. This is
// mathematically the same statistic viewed through the complementary
// weight convention (see weighted.go's doc comment for the derivation), so
// agreement between the two independent code paths is a strong correctness
// signal, not a tautology against a single hardcoded constant.
func TestWeightedKappa_ThreeCategoryHandDerived(t *testing.T) {
	// Never=0, Sometimes=1, Often=2.
	counts := [][]int{
		{5, 3, 2},
		{2, 11, 4},
		{2, 4, 22},
	}
	a, b := buildFromCounts([]int{0, 1, 2}, counts)
	if len(a) != 55 {
		t.Fatalf("expected 55 ratings, got %d", len(a))
	}

	cases := []struct {
		scheme WeightScheme
		want   float64 // exact rational, see weighted.go doc comment
	}{
		{Linear, 1222.0 / 2377.0},
		{Quadratic, 1846.0 / 3441.0},
	}

	for _, tc := range cases {
		got, err := WeightedKappa(a, b, tc.scheme)
		if err != nil {
			t.Fatalf("WeightedKappa(scheme=%v): unexpected error: %v", tc.scheme, err)
		}
		if diff := got - tc.want; diff > 1e-6 || diff < -1e-6 {
			t.Errorf("WeightedKappa(scheme=%v) = %v, want %v", tc.scheme, got, tc.want)
		}

		// Independent cross-check via the agreement-weight formulation.
		cross := agreementWeightKappa(counts, tc.scheme)
		if diff := got - cross; diff > 1e-9 || diff < -1e-9 {
			t.Errorf("scheme=%v: WeightedKappa=%v disagrees with independent agreement-weight derivation=%v", tc.scheme, got, cross)
		}
	}
}

// agreementWeightKappa is a from-scratch, independent implementation of
// weighted kappa using Cohen (1968)'s AGREEMENT-weight convention
// (w=1 on the diagonal, shrinking toward 0 for categories far apart),
// used only to cross-validate WeightedKappa's disagreement-weight
// implementation in tests.
func agreementWeightKappa(counts [][]int, scheme WeightScheme) float64 {
	k := len(counts)
	n := 0.0
	rowSum := make([]float64, k)
	colSum := make([]float64, k)
	for i := 0; i < k; i++ {
		for j := 0; j < k; j++ {
			v := float64(counts[i][j])
			rowSum[i] += v
			colSum[j] += v
			n += v
		}
	}

	vmax := float64(k - 1)
	if scheme == Quadratic {
		vmax = vmax * vmax
	}

	agreementWeight := func(i, j int) float64 {
		d := float64(i - j)
		if d < 0 {
			d = -d
		}
		v := d
		if scheme == Quadratic {
			v = d * d
		}
		return 1 - v/vmax
	}

	var poW, peW float64
	for i := 0; i < k; i++ {
		for j := 0; j < k; j++ {
			w := agreementWeight(i, j)
			poW += w * float64(counts[i][j]) / n
			peW += w * (rowSum[i] / n) * (colSum[j] / n)
		}
	}
	return (poW - peW) / (1 - peW)
}

func TestWeightedKappa_PerfectAgreement(t *testing.T) {
	a := []int{1, 2, 3, 1, 2, 3}
	b := []int{1, 2, 3, 1, 2, 3}
	for _, scheme := range []WeightScheme{Linear, Quadratic} {
		got, err := WeightedKappa(a, b, scheme)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if diff := got - 1.0; diff > 1e-12 || diff < -1e-12 {
			t.Errorf("scheme=%v: kappa = %v, want 1.0", scheme, got)
		}
	}
}

func TestWeightedKappa_LengthMismatch(t *testing.T) {
	_, err := WeightedKappa([]int{1, 2}, []int{1}, Linear)
	if err != ErrLengthMismatch {
		t.Errorf("err = %v, want ErrLengthMismatch", err)
	}
}
