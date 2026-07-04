package evt

import (
	"math"
	"testing"
)

// Hill estimator hand-derived on data = [1,2,4,8,16] (sorted desc:
// 16,8,4,2,1).
//
// k=3: threshold X_(4) = 2;
//
//	xi = (1/3)[ln(16/2)+ln(8/2)+ln(4/2)] = (1/3)[3ln2+2ln2+ln2] = 2 ln2.
//
// k=2: threshold X_(3) = 4;
//
//	xi = (1/2)[ln(16/4)+ln(8/4)] = (1/2)[2ln2+ln2] = 1.5 ln2.
func TestHillTailIndex_HandDerived(t *testing.T) {
	data := []float64{16, 2, 8, 1, 4} // unsorted input
	xi3, ok := HillTailIndex(data, 3)
	if !ok {
		t.Fatal("k=3 failed")
	}
	assertClose(t, "hill k=3", xi3, 2*math.Ln2, 1e-12)

	xi2, ok := HillTailIndex(data, 2)
	if !ok {
		t.Fatal("k=2 failed")
	}
	assertClose(t, "hill k=2", xi2, 1.5*math.Ln2, 1e-12)

	a3, ok := HillAlpha(data, 3)
	if !ok {
		t.Fatal("alpha failed")
	}
	assertClose(t, "hill alpha k=3", a3, 1/(2*math.Ln2), 1e-12)
}

func TestHillTailIndex_Guards(t *testing.T) {
	data := []float64{4, 3, 2, 1}
	if _, ok := HillTailIndex(data, 0); ok {
		t.Error("k<1 should fail")
	}
	if _, ok := HillTailIndex(data, 4); ok {
		t.Error("k>=n should fail")
	}
	if _, ok := HillTailIndex([]float64{-1, -2, -3}, 1); ok {
		t.Error("non-positive threshold should fail (log undefined)")
	}
	// Non-heavy (negative Hill) => HillAlpha not defined.
	if _, ok := HillAlpha([]float64{1.0, 1.0, 1.0, 1.0}, 2); ok {
		t.Error("zero Hill index should give alpha ok=false")
	}
}

// Recovery: for a Pareto tail P(X>x) = x^{-alpha}, Hill on the upper order
// statistics estimates xi = 1/alpha.  Generate exact Pareto quantiles and
// check.  (Embrechts-Kluppelberg-Mikosch 1997 §6.4.)
func TestHillTailIndex_ParetoRecovery(t *testing.T) {
	alpha := 2.5
	n := 2000
	data := make([]float64, n)
	for i := 0; i < n; i++ {
		u := (float64(i) + 0.5) / float64(n) // in (0,1)
		data[i] = math.Pow(1-u, -1.0/alpha)  // Pareto(alpha) quantile, x>=1
	}
	xi, ok := HillTailIndex(data, 300)
	if !ok {
		t.Fatal("fit failed")
	}
	assertClose(t, "hill xi ~ 1/alpha", xi, 1/alpha, 0.05)
}
