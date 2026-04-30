package dcc

import (
	"math"
	"math/rand"
	"testing"
)

// =========================================================================
// Validation
// =========================================================================

func TestParams_Validate(t *testing.T) {
	cases := []struct {
		p    Params
		good bool
	}{
		{Params{Alpha: 0.05, Beta: 0.93, K: 2, Qbar: []float64{1, 0, 0, 1}}, true},
		{Params{Alpha: -0.01, Beta: 0.9, K: 2, Qbar: []float64{1, 0, 0, 1}}, false},
		{Params{Alpha: 0.5, Beta: 0.6, K: 2, Qbar: []float64{1, 0, 0, 1}}, false}, // sum >= 1
		{Params{Alpha: 0.05, Beta: 0.93, K: 2, Qbar: []float64{1, 0, 0}}, false},  // wrong dims
	}
	for i, c := range cases {
		err := c.p.Validate()
		if c.good && err != nil {
			t.Errorf("case %d: want valid, got %v", i, err)
		}
		if !c.good && err == nil {
			t.Errorf("case %d: want invalid, got nil", i)
		}
	}
}

// =========================================================================
// SampleQbar
// =========================================================================

func TestSampleQbar_Identity(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	const n, k = 5000, 2
	z := make([]float64, n*k)
	for i := 0; i < n; i++ {
		z[i*k] = rng.NormFloat64()
		z[i*k+1] = rng.NormFloat64()
	}
	out := make([]float64, k*k)
	if err := SampleQbar(z, n, k, out); err != nil {
		t.Fatal(err)
	}
	// Diagonal close to 1 (unit variance), off-diagonal close to 0.
	if math.Abs(out[0]-1.0) > 0.05 {
		t.Errorf("out[0,0] = %v, want close to 1", out[0])
	}
	if math.Abs(out[3]-1.0) > 0.05 {
		t.Errorf("out[1,1] = %v, want close to 1", out[3])
	}
	if math.Abs(out[1]) > 0.05 {
		t.Errorf("out[0,1] = %v, want close to 0", out[1])
	}
	if out[1] != out[2] {
		t.Errorf("Qbar not symmetric: [0,1]=%v [1,0]=%v", out[1], out[2])
	}
}

// =========================================================================
// Update closed-form
// =========================================================================

func TestUpdate_FirstStepFromIdentity(t *testing.T) {
	p := Params{Alpha: 0.1, Beta: 0.8, K: 2, Qbar: []float64{1, 0.3, 0.3, 1}}
	z := []float64{1.5, -0.5}
	Q := []float64{1, 0, 0, 1}
	qNew := make([]float64, 4)
	if err := p.Update(z, Q, qNew); err != nil {
		t.Fatal(err)
	}
	weight := 1.0 - p.Alpha - p.Beta // 0.1
	want := []float64{
		weight*1.0 + p.Alpha*1.5*1.5 + p.Beta*1.0,    // [0,0]
		weight*0.3 + p.Alpha*1.5*(-0.5) + p.Beta*0.0, // [0,1]
		weight*0.3 + p.Alpha*(-0.5)*1.5 + p.Beta*0.0, // [1,0]
		weight*1.0 + p.Alpha*0.5*0.5 + p.Beta*1.0,    // [1,1]
	}
	for i := range want {
		if math.Abs(qNew[i]-want[i]) > 1e-12 {
			t.Errorf("qNew[%d] = %v, want %v", i, qNew[i], want[i])
		}
	}
}

// =========================================================================
// CorrelationFromQ
// =========================================================================

func TestCorrelationFromQ_DiagonalsAreOne(t *testing.T) {
	Q := []float64{
		2.0, 1.0, 0.5,
		1.0, 4.0, -1.0,
		0.5, -1.0, 1.0,
	}
	R := make([]float64, 9)
	if err := CorrelationFromQ(Q, 3, R); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 3; i++ {
		if math.Abs(R[i*3+i]-1.0) > 1e-12 {
			t.Errorf("R[%d,%d] = %v, want 1", i, i, R[i*3+i])
		}
	}
	// Symmetric.
	if math.Abs(R[1]-R[3]) > 1e-12 || math.Abs(R[2]-R[6]) > 1e-12 {
		t.Errorf("R not symmetric: %v", R)
	}
	// Off-diagonals within [-1, 1].
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			r := R[i*3+j]
			if r < -1.0-1e-12 || r > 1.0+1e-12 {
				t.Errorf("R[%d,%d] = %v out of [-1,1]", i, j, r)
			}
		}
	}
}

// =========================================================================
// FilterSeries
// =========================================================================

func TestFilterSeries_RecoversCorrelation(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	const n, k = 2000, 2
	// Generate two correlated standardised normals: z2 = rho * z1 +
	// sqrt(1-rho^2) * eps, with rho = 0.6.  Across the whole sample the
	// realised correlation should be ~0.6 and DCC's R_t should converge
	// to the same.
	rho := 0.6
	z := make([]float64, n*k)
	for t := 0; t < n; t++ {
		z1 := rng.NormFloat64()
		eps := rng.NormFloat64()
		z[t*k] = z1
		z[t*k+1] = rho*z1 + math.Sqrt(1-rho*rho)*eps
	}

	qbar := make([]float64, k*k)
	if err := SampleQbar(z, n, k, qbar); err != nil {
		t.Fatal(err)
	}
	p := Params{Alpha: 0.05, Beta: 0.93, K: k, Qbar: qbar}
	rSeries := make([]float64, n*k*k)
	if err := p.FilterSeries(z, n, rSeries); err != nil {
		t.Fatal(err)
	}
	// Mean conditional correlation [0,1] across all t.
	var meanR float64
	for t := 0; t < n; t++ {
		meanR += rSeries[t*k*k+1] // R_t[0,1]
	}
	meanR /= float64(n)
	if math.Abs(meanR-rho) > 0.05 {
		t.Errorf("mean DCC R[0,1] = %v, want close to %v", meanR, rho)
	}
}

// =========================================================================
// Determinism
// =========================================================================

func TestUpdate_Deterministic(t *testing.T) {
	p := Params{Alpha: 0.1, Beta: 0.8, K: 2, Qbar: []float64{1, 0, 0, 1}}
	z := []float64{0.5, -0.5}
	Q := []float64{1, 0.2, 0.2, 1}
	a := make([]float64, 4)
	b := make([]float64, 4)
	_ = p.Update(z, Q, a)
	_ = p.Update(z, Q, b)
	for i := range a {
		if a[i] != b[i] {
			t.Errorf("non-deterministic at i=%d", i)
		}
	}
}

// =========================================================================
// Validation rejections
// =========================================================================

func TestUpdate_RejectsBadInputs(t *testing.T) {
	p := Params{Alpha: 0.1, Beta: 0.8, K: 2, Qbar: []float64{1, 0, 0, 1}}
	if err := p.Update([]float64{1}, make([]float64, 4), make([]float64, 4)); err == nil {
		t.Error("short z should error")
	}
	bad := Params{Alpha: 0.5, Beta: 0.6, K: 2, Qbar: []float64{1, 0, 0, 1}}
	if err := bad.Update([]float64{1, 1}, make([]float64, 4), make([]float64, 4)); err == nil {
		t.Error("non-stationary params should error")
	}
}
