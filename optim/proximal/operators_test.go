package proximal

import (
	"math"
	"testing"
)

// =========================================================================
// L1 / L0
// =========================================================================

func TestProxL1_SoftThresholding(t *testing.T) {
	v := []float64{0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 0.0}
	out := make([]float64, len(v))
	ProxL1(v, 0.7, out)
	want := []float64{0.0, 0.0, 0.3, -0.3, 1.3, -1.3, 0.0}
	for i, w := range want {
		if math.Abs(out[i]-w) > 1e-12 {
			t.Errorf("ProxL1[%d] = %.12f, want %.12f", i, out[i], w)
		}
	}
}

func TestProxL1_InPlace(t *testing.T) {
	v := []float64{2.0, -2.0, 0.5}
	ProxL1(v, 1.0, v)
	want := []float64{1.0, -1.0, 0.0}
	for i, w := range want {
		if v[i] != w {
			t.Errorf("ProxL1 in-place [%d] = %v, want %v", i, v[i], w)
		}
	}
}

func TestProxL0_HardThresholding(t *testing.T) {
	v := []float64{0.5, 1.0, 1.5, 2.0, -1.5}
	out := make([]float64, len(v))
	ProxL0(v, 0.5, out) // threshold sqrt(2*0.5) = 1.0
	want := []float64{0.0, 0.0, 1.5, 2.0, -1.5}
	for i, w := range want {
		if out[i] != w {
			t.Errorf("ProxL0[%d] = %v, want %v", i, out[i], w)
		}
	}
}

// =========================================================================
// Squared L2
// =========================================================================

func TestProxSquaredL2_Shrinkage(t *testing.T) {
	v := []float64{1.0, -2.0, 3.0}
	out := make([]float64, 3)
	ProxSquaredL2(v, 1.0, out) // scale = 1/2
	want := []float64{0.5, -1.0, 1.5}
	for i, w := range want {
		if math.Abs(out[i]-w) > 1e-12 {
			t.Errorf("ProxSquaredL2[%d] = %v, want %v", i, out[i], w)
		}
	}
}

// =========================================================================
// Indicator projections
// =========================================================================

func TestProxNonNeg(t *testing.T) {
	v := []float64{-1.0, 0.0, 1.0, -2.5, 3.0}
	out := make([]float64, len(v))
	ProxNonNeg(v, 0.0, out)
	want := []float64{0.0, 0.0, 1.0, 0.0, 3.0}
	for i, w := range want {
		if out[i] != w {
			t.Errorf("ProxNonNeg[%d] = %v, want %v", i, out[i], w)
		}
	}
}

func TestProxBox(t *testing.T) {
	lo := []float64{0.0, -1.0, math.Inf(-1)}
	hi := []float64{1.0, 1.0, 5.0}
	prox := ProxBox(lo, hi)
	v := []float64{-0.5, 2.0, 10.0}
	out := make([]float64, 3)
	prox(v, 0.0, out)
	want := []float64{0.0, 1.0, 5.0}
	for i, w := range want {
		if out[i] != w {
			t.Errorf("ProxBox[%d] = %v, want %v", i, out[i], w)
		}
	}
}

func TestProxL2Ball_Inside(t *testing.T) {
	prox := ProxL2Ball(2.0)
	v := []float64{0.5, 0.5}
	out := make([]float64, 2)
	prox(v, 0.0, out)
	if out[0] != 0.5 || out[1] != 0.5 {
		t.Errorf("ProxL2Ball inside ball: got %v, want input unchanged", out)
	}
}

func TestProxL2Ball_Outside(t *testing.T) {
	prox := ProxL2Ball(1.0)
	v := []float64{3.0, 4.0} // norm = 5
	out := make([]float64, 2)
	prox(v, 0.0, out)
	// Projected to (3/5, 4/5).
	if math.Abs(out[0]-0.6) > 1e-12 || math.Abs(out[1]-0.8) > 1e-12 {
		t.Errorf("ProxL2Ball outside: got %v, want (0.6, 0.8)", out)
	}
}

func TestProxSimplex_KnownVector(t *testing.T) {
	// Wang & Carreira-Perpinan 2013 example.
	v := []float64{0.5, 0.0, 1.0, 1.5, -0.5}
	out := make([]float64, len(v))
	ProxSimplex(v, 0.0, out)
	var sum float64
	for _, x := range out {
		if x < -1e-12 {
			t.Errorf("ProxSimplex returned negative: %v", out)
			break
		}
		sum += x
	}
	if math.Abs(sum-1.0) > 1e-9 {
		t.Errorf("ProxSimplex sum = %v, want 1", sum)
	}
}

func TestProxSimplex_AlreadyOnSimplex(t *testing.T) {
	v := []float64{0.25, 0.25, 0.25, 0.25}
	out := make([]float64, len(v))
	ProxSimplex(v, 0.0, out)
	for i, x := range out {
		if math.Abs(x-0.25) > 1e-12 {
			t.Errorf("ProxSimplex of uniform[4]: out[%d]=%v, want 0.25", i, x)
		}
	}
}

// =========================================================================
// Linear
// =========================================================================

func TestProxLinear(t *testing.T) {
	c := []float64{1.0, -1.0, 2.0}
	prox := ProxLinear(c)
	v := []float64{5.0, 5.0, 5.0}
	out := make([]float64, 3)
	prox(v, 0.5, out)
	want := []float64{4.5, 5.5, 4.0}
	for i, w := range want {
		if math.Abs(out[i]-w) > 1e-12 {
			t.Errorf("ProxLinear[%d] = %v, want %v", i, out[i], w)
		}
	}
}
