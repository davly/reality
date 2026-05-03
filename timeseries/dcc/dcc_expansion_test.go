package dcc

import (
	"math"
	"math/rand"
	"testing"
)

// =========================================================================
// Expansion coverage for DCC. Pre-this: 7 tests (validate, sample-Qbar,
// 1-step Update, CorrelationFromQ basics, FilterSeries golden, determinism,
// rejects). Adds: NaN guards, alias handling, edge K, FilterSeries error
// paths, EngleDefault sanity, single-step parity with FilterSeries.
// =========================================================================

func TestEngleDefault_HasIndustryValues(t *testing.T) {
	p := EngleDefault()
	if p.Alpha != 0.05 {
		t.Errorf("Alpha=%v, want 0.05 (Engle 2002)", p.Alpha)
	}
	if p.Beta != 0.93 {
		t.Errorf("Beta=%v, want 0.93 (Engle 2002)", p.Beta)
	}
	// alpha+beta = 0.98 < 1.0 → stationary by construction.
	if p.Alpha+p.Beta >= 1.0 {
		t.Errorf("EngleDefault not stationary: %v + %v >= 1", p.Alpha, p.Beta)
	}
}

func TestParams_Validate_NaNAlpha(t *testing.T) {
	p := Params{Alpha: math.NaN(), Beta: 0.9, K: 2, Qbar: []float64{1, 0, 0, 1}}
	if err := p.Validate(); err == nil {
		t.Error("NaN alpha should error")
	}
}

func TestParams_Validate_NaNBeta(t *testing.T) {
	p := Params{Alpha: 0.05, Beta: math.NaN(), K: 2, Qbar: []float64{1, 0, 0, 1}}
	if err := p.Validate(); err == nil {
		t.Error("NaN beta should error")
	}
}

func TestParams_Validate_NegativeBeta(t *testing.T) {
	p := Params{Alpha: 0.05, Beta: -0.1, K: 2, Qbar: []float64{1, 0, 0, 1}}
	if err := p.Validate(); err == nil {
		t.Error("Negative beta should error")
	}
}

func TestParams_Validate_AlphaPlusBetaExactlyOne(t *testing.T) {
	// alpha+beta == 1 violates strict stationarity (>= 1 rejected).
	p := Params{Alpha: 0.5, Beta: 0.5, K: 2, Qbar: []float64{1, 0, 0, 1}}
	if err := p.Validate(); err == nil {
		t.Error("alpha+beta=1 should error")
	}
}

func TestParams_Validate_KZeroRejected(t *testing.T) {
	p := Params{Alpha: 0.05, Beta: 0.93, K: 0, Qbar: nil}
	if err := p.Validate(); err == nil {
		t.Error("K=0 should error")
	}
}

func TestParams_Validate_KOneAccepted(t *testing.T) {
	p := Params{Alpha: 0.05, Beta: 0.93, K: 1, Qbar: []float64{1.0}}
	if err := p.Validate(); err != nil {
		t.Errorf("K=1 should be accepted: %v", err)
	}
}

func TestSampleQbar_RejectsZeroN(t *testing.T) {
	if err := SampleQbar(nil, 0, 2, make([]float64, 4)); err == nil {
		t.Error("n=0 should error")
	}
}

func TestSampleQbar_RejectsZeroK(t *testing.T) {
	if err := SampleQbar(nil, 1, 0, nil); err == nil {
		t.Error("k=0 should error")
	}
}

func TestSampleQbar_RejectsZSeriesLengthMismatch(t *testing.T) {
	if err := SampleQbar(make([]float64, 3), 1, 2, make([]float64, 4)); err == nil {
		t.Error("len(z) != n*k should error")
	}
}

func TestSampleQbar_RejectsOutLengthMismatch(t *testing.T) {
	if err := SampleQbar(make([]float64, 4), 2, 2, make([]float64, 3)); err == nil {
		t.Error("len(out) != k*k should error")
	}
}

func TestSampleQbar_AllZeros_ReturnsZeroMatrix(t *testing.T) {
	zero := make([]float64, 10)
	out := []float64{1, 2, 3, 4} // pre-poison
	if err := SampleQbar(zero, 5, 2, out); err != nil {
		t.Fatal(err)
	}
	for i, v := range out {
		if v != 0 {
			t.Errorf("out[%d]=%v, want 0", i, v)
		}
	}
}

func TestSampleQbar_OverwritesPriorContents(t *testing.T) {
	// out is zeroed before accumulation — must not leak prior values.
	z := []float64{1, 0, 0, 1, 0, 0, 1, 0} // 4 obs, k=2
	prior := []float64{99, 99, 99, 99}
	if err := SampleQbar(z, 4, 2, prior); err != nil {
		t.Fatal(err)
	}
	// After overwrite, all entries should reflect z, not 99.
	if prior[0] == 99 || prior[3] == 99 {
		t.Errorf("output not zeroed before accumulation: %v", prior)
	}
}

func TestUpdate_AliasingQAndQOut(t *testing.T) {
	// Document: Q and qOut may alias (per godoc).
	p := Params{Alpha: 0.1, Beta: 0.8, K: 2, Qbar: []float64{1, 0.3, 0.3, 1}}
	z := []float64{1.0, -1.0}
	q := []float64{1, 0, 0, 1}
	qBackup := make([]float64, 4)
	copy(qBackup, q)

	// Reference computation with separate buffer.
	want := make([]float64, 4)
	if err := p.Update(z, qBackup, want); err != nil {
		t.Fatal(err)
	}

	// Now apply with Q == qOut (aliasing).
	if err := p.Update(z, q, q); err != nil {
		t.Fatal(err)
	}
	// Aliased path computes the same result element-by-element because
	// each output index touches only its own input slot.
	for i := range want {
		if math.Abs(q[i]-want[i]) > 1e-12 {
			t.Errorf("alias mismatch at i=%d: got %v want %v", i, q[i], want[i])
		}
	}
}

func TestUpdate_RejectsShortQ(t *testing.T) {
	p := Params{Alpha: 0.05, Beta: 0.9, K: 2, Qbar: []float64{1, 0, 0, 1}}
	if err := p.Update([]float64{1, 1}, []float64{1, 0, 0}, make([]float64, 4)); err == nil {
		t.Error("short Q should error")
	}
}

func TestUpdate_RejectsShortQOut(t *testing.T) {
	p := Params{Alpha: 0.05, Beta: 0.9, K: 2, Qbar: []float64{1, 0, 0, 1}}
	if err := p.Update([]float64{1, 1}, []float64{1, 0, 0, 1}, make([]float64, 3)); err == nil {
		t.Error("short qOut should error")
	}
}

func TestCorrelationFromQ_NegativeDiagonalRejected(t *testing.T) {
	Q := []float64{-1, 0.5, 0.5, 2.0}
	R := make([]float64, 4)
	if err := CorrelationFromQ(Q, 2, R); err == nil {
		t.Error("negative diagonal should error")
	}
}

func TestCorrelationFromQ_ZeroDiagonalRejected(t *testing.T) {
	Q := []float64{0, 0.5, 0.5, 1.0}
	R := make([]float64, 4)
	if err := CorrelationFromQ(Q, 2, R); err == nil {
		t.Error("zero diagonal should error")
	}
}

func TestCorrelationFromQ_AliasQAndR(t *testing.T) {
	// Q and rOut may alias.
	Q := []float64{4, 2, 2, 9}
	want := []float64{1, 2.0 / 6.0, 2.0 / 6.0, 1}
	if err := CorrelationFromQ(Q, 2, Q); err != nil {
		t.Fatal(err)
	}
	for i := range want {
		if math.Abs(Q[i]-want[i]) > 1e-12 {
			t.Errorf("alias mismatch i=%d: %v vs %v", i, Q[i], want[i])
		}
	}
}

func TestCorrelationFromQ_RejectsDimensionMismatch(t *testing.T) {
	if err := CorrelationFromQ(make([]float64, 3), 2, make([]float64, 4)); err == nil {
		t.Error("len(Q) != k*k should error")
	}
	if err := CorrelationFromQ(make([]float64, 4), 2, make([]float64, 3)); err == nil {
		t.Error("len(rOut) != k*k should error")
	}
}

func TestFilterSeries_RejectsBadParams(t *testing.T) {
	p := Params{Alpha: 0.5, Beta: 0.6, K: 2, Qbar: []float64{1, 0, 0, 1}}
	if err := p.FilterSeries(make([]float64, 4), 2, make([]float64, 8)); err == nil {
		t.Error("non-stationary params should error")
	}
}

func TestFilterSeries_RejectsZSeriesLengthMismatch(t *testing.T) {
	p := Params{Alpha: 0.05, Beta: 0.93, K: 2, Qbar: []float64{1, 0, 0, 1}}
	if err := p.FilterSeries(make([]float64, 3), 2, make([]float64, 8)); err == nil {
		t.Error("len(z) != n*k should error")
	}
}

func TestFilterSeries_RejectsRSeriesLengthMismatch(t *testing.T) {
	p := Params{Alpha: 0.05, Beta: 0.93, K: 2, Qbar: []float64{1, 0, 0, 1}}
	if err := p.FilterSeries(make([]float64, 4), 2, make([]float64, 7)); err == nil {
		t.Error("len(rSeries) != n*k*k should error")
	}
}

func TestFilterSeries_DiagonalsAlwaysOne(t *testing.T) {
	rng := rand.New(rand.NewSource(33))
	const n, k = 100, 3
	z := make([]float64, n*k)
	for i := range z {
		z[i] = rng.NormFloat64()
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
	// Every R_step[i, i] must be exactly 1.
	for step := 0; step < n; step++ {
		for i := 0; i < k; i++ {
			r := rSeries[step*k*k+i*k+i]
			if math.Abs(r-1.0) > 1e-12 {
				t.Errorf("R_t=%d[%d,%d] = %v, want 1", step, i, i, r)
			}
		}
	}
}

func TestFilterSeries_ParityWithSingleStepLoop(t *testing.T) {
	// FilterSeries should equal a manual loop of Update + CorrelationFromQ.
	rng := rand.New(rand.NewSource(44))
	const n, k = 30, 2
	z := make([]float64, n*k)
	for i := range z {
		z[i] = rng.NormFloat64()
	}
	p := Params{Alpha: 0.06, Beta: 0.92, K: k, Qbar: []float64{1, 0.4, 0.4, 1}}

	rRef := make([]float64, n*k*k)
	if err := p.FilterSeries(z, n, rRef); err != nil {
		t.Fatal(err)
	}

	rManual := make([]float64, n*k*k)
	Q := make([]float64, k*k)
	copy(Q, p.Qbar)
	qNew := make([]float64, k*k)
	for step := 0; step < n; step++ {
		zt := z[step*k : (step+1)*k]
		if err := p.Update(zt, Q, qNew); err != nil {
			t.Fatal(err)
		}
		if err := CorrelationFromQ(qNew, k, rManual[step*k*k:(step+1)*k*k]); err != nil {
			t.Fatal(err)
		}
		copy(Q, qNew)
	}

	for i := range rRef {
		if math.Abs(rRef[i]-rManual[i]) > 1e-12 {
			t.Errorf("FilterSeries vs manual loop differ at i=%d: %v vs %v", i, rRef[i], rManual[i])
		}
	}
}

func TestUpdate_KOne_ScalarCase(t *testing.T) {
	// K=1 reduces DCC to a scalar GARCH-like update.
	p := Params{Alpha: 0.1, Beta: 0.8, K: 1, Qbar: []float64{1.0}}
	z := []float64{2.0}
	Q := []float64{1.5}
	out := []float64{0}
	if err := p.Update(z, Q, out); err != nil {
		t.Fatal(err)
	}
	weight := 1.0 - 0.1 - 0.8
	want := weight*1.0 + 0.1*4.0 + 0.8*1.5
	if math.Abs(out[0]-want) > 1e-12 {
		t.Errorf("Update K=1 = %v, want %v", out[0], want)
	}
}

func TestCorrelationFromQ_KOne_ProducesUnity(t *testing.T) {
	Q := []float64{2.5}
	R := []float64{0}
	if err := CorrelationFromQ(Q, 1, R); err != nil {
		t.Fatal(err)
	}
	if math.Abs(R[0]-1.0) > 1e-12 {
		t.Errorf("CorrelationFromQ K=1 = %v, want 1", R[0])
	}
}

func TestUpdate_RejectsValidateFailure_NaNAlpha(t *testing.T) {
	p := Params{Alpha: math.NaN(), Beta: 0.5, K: 2, Qbar: []float64{1, 0, 0, 1}}
	if err := p.Update([]float64{1, 1}, []float64{1, 0, 0, 1}, make([]float64, 4)); err == nil {
		t.Error("NaN alpha should propagate from Validate")
	}
}
