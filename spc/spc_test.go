package spc

import (
	"errors"
	"math"
	"testing"
)

// All golden values below are hand-computed from the AIAG/SPC formulas (see the
// package doc comment). The DPMO/yield goldens were cross-checked against a
// reference implementation of Phi(z) = 0.5*erfc(-z/sqrt2):
//
//	centered Cp=1.0 (limits at +/-3 sigma): DPMO = 2699.796..., yield = 0.99730...
//	six sigma   Cp=2.0 (limits at +/-6 sigma): DPMO ~ 0.00197
//	off-center  mean=101 s=1 USL=103 LSL=97:  DPMO = 22781.803...
//
// A wrong formula (e.g. a 6 vs 3 mix-up in Cpk, the wrong sigma in Pp, or a sign
// flip in the DPMO tails) makes one of these asserts fail.

const tol = 1e-9

// ---------------------------------------------------------------------------
// Cp (standalone)
// ---------------------------------------------------------------------------

func TestCp(t *testing.T) {
	tests := []struct {
		name           string
		usl, lsl, sig  float64
		want           float64
		wantErr        error
	}{
		{"centered six-sigma spread", 103, 97, 1, 1.0, nil},
		{"six sigma (Cp=2)", 106, 94, 1, 2.0, nil},
		{"tight sigma doubles Cp", 103, 97, 0.5, 2.0, nil},
		{"AIAG floor halfwidth 3.99", 3.99, -3.99, 1, 1.33, nil},
		{"inverted spec errors", 97, 103, 1, 0, ErrSpecOrder},
		{"equal limits errors", 100, 100, 1, 0, ErrSpecOrder},
		{"zero sigma errors", 103, 97, 0, 0, ErrNonPositiveSigma},
		{"negative sigma errors", 103, 97, -1, 0, ErrNonPositiveSigma},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Cp(tt.usl, tt.lsl, tt.sig)
			if !errors.Is(err, tt.wantErr) {
				t.Fatalf("Cp err = %v, want %v", err, tt.wantErr)
			}
			if tt.wantErr == nil && math.Abs(got-tt.want) > tol {
				t.Errorf("Cp(%v,%v,%v) = %v, want %v", tt.usl, tt.lsl, tt.sig, got, tt.want)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Cpk (standalone) — centering penalty
// ---------------------------------------------------------------------------

func TestCpk(t *testing.T) {
	tests := []struct {
		name                string
		usl, lsl, mean, sig float64
		want                float64
		wantErr             error
	}{
		{"centered equals Cp", 103, 97, 100, 1, 1.0, nil},
		// mean=101 -> min(2,4)/3 = 2/3
		{"off-center lowers Cpk", 103, 97, 101, 1, 2.0 / 3.0, nil},
		// mean at USL -> (103-103)/3 = 0
		{"mean on USL gives 0", 103, 97, 103, 1, 0.0, nil},
		// mean beyond USL -> negative
		{"mean past USL negative", 103, 97, 104, 1, -1.0 / 3.0, nil},
		{"zero sigma errors", 103, 97, 100, 0, 0, ErrNonPositiveSigma},
		{"inverted spec errors", 97, 103, 100, 1, 0, ErrSpecOrder},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Cpk(tt.usl, tt.lsl, tt.mean, tt.sig)
			if !errors.Is(err, tt.wantErr) {
				t.Fatalf("Cpk err = %v, want %v", err, tt.wantErr)
			}
			if tt.wantErr == nil && math.Abs(got-tt.want) > tol {
				t.Errorf("Cpk(%v,%v,%v,%v) = %v, want %v",
					tt.usl, tt.lsl, tt.mean, tt.sig, got, tt.want)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// DPMO (standalone)
// ---------------------------------------------------------------------------

func TestDPMO(t *testing.T) {
	tests := []struct {
		name                string
		usl, lsl, mean, sig float64
		want                float64
		dpmoTol             float64
		wantErr             error
	}{
		// centered, +/-3 sigma to each limit -> two-sided 0.27% -> ~2700 DPMO
		{"centered Cpk=1 ~2700", 103, 97, 100, 1, 2699.796063, 1e-3, nil},
		// six sigma centered -> ~0.00197 DPMO
		{"six sigma tiny", 106, 94, 100, 1, 0.0019732, 1e-4, nil},
		// off-center mean=101 -> 22781.80
		{"off-center higher", 103, 97, 101, 1, 22781.803190, 1e-2, nil},
		{"zero sigma errors", 103, 97, 100, 0, 0, 0, ErrNonPositiveSigma},
		{"inverted spec errors", 97, 103, 100, 1, 0, 0, ErrSpecOrder},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := DPMO(tt.usl, tt.lsl, tt.mean, tt.sig)
			if !errors.Is(err, tt.wantErr) {
				t.Fatalf("DPMO err = %v, want %v", err, tt.wantErr)
			}
			if tt.wantErr == nil && math.Abs(got-tt.want) > tt.dpmoTol {
				t.Errorf("DPMO(%v,%v,%v,%v) = %v, want ~%v (tol %v)",
					tt.usl, tt.lsl, tt.mean, tt.sig, got, tt.want, tt.dpmoTol)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Compute — full result, golden values
// ---------------------------------------------------------------------------

func TestComputeCenteredGolden(t *testing.T) {
	// Centered process at Cp=1.0: mean=100, sigma=1, USL=103, LSL=97.
	// Cp = (103-97)/(6*1) = 1.0; centered so Cpk = Cp = 1.0.
	// No overall sigma supplied -> Pp = Cp, Ppk = Cpk.
	// SigmaLevel = 3*Cpk = 3.0. DPMO = 2699.796..., yield = 0.99730...
	r, err := Compute(Study{USL: 103, LSL: 97, Mean: 100, SigmaWithin: 1})
	if err != nil {
		t.Fatal(err)
	}
	checkClose(t, "Cp", r.Cp, 1.0, tol)
	checkClose(t, "Cpk", r.Cpk, 1.0, tol)
	checkClose(t, "Pp", r.Pp, 1.0, tol)
	checkClose(t, "Ppk", r.Ppk, 1.0, tol)
	checkClose(t, "SigmaLevel", r.SigmaLevel, 3.0, tol)
	checkClose(t, "DPMO", r.DPMO, 2699.796063260, 1e-3)
	checkClose(t, "ExpectedYield", r.ExpectedYield, 0.997300203937, 1e-9)
}

func TestComputeOffCenterGolden(t *testing.T) {
	// Off-center: mean=101, sigma=1, USL=103, LSL=97.
	// Cp unchanged = 1.0 (spec width / spread doesn't see centering).
	// Cpk = min(103-101, 101-97)/(3*1) = min(2,4)/3 = 2/3 = 0.66667 < Cp.
	// This is the discriminating "Cpk < Cp" case.
	r, err := Compute(Study{USL: 103, LSL: 97, Mean: 101, SigmaWithin: 1})
	if err != nil {
		t.Fatal(err)
	}
	checkClose(t, "Cp", r.Cp, 1.0, tol)
	checkClose(t, "Cpk", r.Cpk, 2.0/3.0, tol)
	if !(r.Cpk < r.Cp) {
		t.Errorf("expected Cpk (%v) < Cp (%v) for off-center process", r.Cpk, r.Cp)
	}
	checkClose(t, "SigmaLevel", r.SigmaLevel, 2.0, tol) // 3 * (2/3)
	checkClose(t, "DPMO", r.DPMO, 22781.803190012, 1e-2)
}

func TestComputeOverallSigmaDistinct(t *testing.T) {
	// Within sigma=1, overall sigma=2: Cp/Cpk use within, Pp/Ppk use overall.
	// mean=101 USL=103 LSL=97:
	//   Cp  = 6/(6*1) = 1.0;   Cpk = min(2,4)/(3*1) = 2/3
	//   Pp  = 6/(6*2) = 0.5;   Ppk = min(2,4)/(3*2) = 2/6 = 1/3
	r, err := Compute(Study{USL: 103, LSL: 97, Mean: 101, SigmaWithin: 1, SigmaOverall: 2})
	if err != nil {
		t.Fatal(err)
	}
	checkClose(t, "Cp", r.Cp, 1.0, tol)
	checkClose(t, "Cpk", r.Cpk, 2.0/3.0, tol)
	checkClose(t, "Pp", r.Pp, 0.5, tol)
	checkClose(t, "Ppk", r.Ppk, 1.0/3.0, tol)
	if !(r.Pp < r.Cp) {
		t.Errorf("worse overall sigma should give Pp (%v) < Cp (%v)", r.Pp, r.Cp)
	}
}

func TestComputeSixSigmaGolden(t *testing.T) {
	// Six-sigma centered: mean=100, sigma=1, USL=106, LSL=94.
	// Cp = 12/6 = 2.0; centered -> Cpk = 2.0; SigmaLevel = 6.0; DPMO ~ 0.00197.
	r, err := Compute(Study{USL: 106, LSL: 94, Mean: 100, SigmaWithin: 1})
	if err != nil {
		t.Fatal(err)
	}
	checkClose(t, "Cp", r.Cp, 2.0, tol)
	checkClose(t, "Cpk", r.Cpk, 2.0, tol)
	checkClose(t, "SigmaLevel", r.SigmaLevel, 6.0, tol)
	checkClose(t, "DPMO", r.DPMO, 0.0019732, 1e-4)
}

func TestComputeValidation(t *testing.T) {
	if _, err := Compute(Study{USL: 10, LSL: 9, Mean: 9.5, SigmaWithin: 0}); !errors.Is(err, ErrNonPositiveSigma) {
		t.Error("sigma 0 should error ErrNonPositiveSigma")
	}
	if _, err := Compute(Study{USL: 9, LSL: 10, Mean: 9.5, SigmaWithin: 0.1}); !errors.Is(err, ErrSpecOrder) {
		t.Error("USL<=LSL should error ErrSpecOrder")
	}
}

// ---------------------------------------------------------------------------
// Classify / ClassifyCpk — AIAG 1.33 floor boundary
// ---------------------------------------------------------------------------

func TestClassifyCpk(t *testing.T) {
	floor := AIAGCpkFloor // 1.33
	tests := []struct {
		name string
		cpk  float64
		want Rating
	}{
		{"well above floor", 1.67, Capable},
		{"exactly at floor (boundary)", 1.33, Capable},
		{"just below floor", 1.3299999, Marginal},
		{"capable-but-marginal", 1.10, Marginal},
		{"exactly 1.0 boundary", 1.0, Marginal},
		{"just below 1.0", 0.9999999, Incapable},
		{"incapable", 0.80, Incapable},
		{"negative (mean past spec)", -0.5, Incapable},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := ClassifyCpk(tt.cpk, floor); got != tt.want {
				t.Errorf("ClassifyCpk(%v, %v) = %v, want %v", tt.cpk, floor, got, tt.want)
			}
		})
	}
}

func TestClassifyResult(t *testing.T) {
	if got := Classify(Result{Cpk: 1.5}, AIAGCpkFloor); got != Capable {
		t.Errorf("Classify Cpk 1.5 -> %v, want CAPABLE", got)
	}
	if got := Classify(Result{Cpk: 1.2}, AIAGCpkFloor); got != Marginal {
		t.Errorf("Classify Cpk 1.2 -> %v, want MARGINAL", got)
	}
}

func TestRatingString(t *testing.T) {
	if Capable.String() != "CAPABLE" || Marginal.String() != "MARGINAL" || Incapable.String() != "INCAPABLE" {
		t.Errorf("Rating.String mismatch: %s/%s/%s", Capable, Marginal, Incapable)
	}
}

// ---------------------------------------------------------------------------
// PooledWithinSigma / OverallSigma — hand values
// ---------------------------------------------------------------------------

func TestPooledWithinSigma(t *testing.T) {
	// [[2,4],[6,8]]: each subgroup sample variance = 2 (diffs +/-1 -> ss=2, /(2-1)=2);
	// pooled var = (1*2 + 1*2)/(1+1) = 2; sigma = sqrt(2).
	s, err := PooledWithinSigma([][]float64{{2, 4}, {6, 8}})
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(s-math.Sqrt2) > tol {
		t.Errorf("pooled within sigma = %v, want sqrt(2) = %v", s, math.Sqrt2)
	}

	// Unequal subgroup sizes weight by (n-1).
	// [[10,12,14],[20,22]]: var1 = ss/(2) where ss=(−2)^2+0+2^2=8 -> 4; var2 = 2.
	// pooled = (2*4 + 1*2)/(2+1) = 10/3; sigma = sqrt(10/3).
	s, err = PooledWithinSigma([][]float64{{10, 12, 14}, {20, 22}})
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(s-math.Sqrt(10.0/3.0)) > tol {
		t.Errorf("unequal pooled sigma = %v, want sqrt(10/3) = %v", s, math.Sqrt(10.0/3.0))
	}

	// No subgroups errors.
	if _, err := PooledWithinSigma(nil); err == nil {
		t.Error("empty subgroups should error")
	}
	// A singleton subgroup errors (needs >= 2 points).
	if _, err := PooledWithinSigma([][]float64{{5}}); err == nil {
		t.Error("singleton subgroup should error")
	}
}

func TestOverallSigma(t *testing.T) {
	// [2,4,6,8]: mean 5, ss = 9+1+1+9 = 20, sample var = 20/3, sigma = sqrt(20/3).
	s, err := OverallSigma([]float64{2, 4, 6, 8})
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(s-math.Sqrt(20.0/3.0)) > tol {
		t.Errorf("overall sigma = %v, want sqrt(20/3) = %v", s, math.Sqrt(20.0/3.0))
	}

	// Identical points -> zero variance -> zero sigma.
	s, err = OverallSigma([]float64{7, 7, 7, 7})
	if err != nil {
		t.Fatal(err)
	}
	if s != 0 {
		t.Errorf("zero-variance sigma = %v, want 0", s)
	}

	// Single point errors.
	if _, err := OverallSigma([]float64{3}); err == nil {
		t.Error("single point should error")
	}
}

// TestPooledFeedsCompute is an end-to-end discriminating check: estimate sigmas
// from raw data, feed them into Compute, and confirm the indices come out as the
// pooled/overall sigmas predict.
func TestPooledFeedsCompute(t *testing.T) {
	subgroups := [][]float64{{2, 4}, {6, 8}} // pooled within sigma = sqrt(2)
	sw, err := PooledWithinSigma(subgroups)
	if err != nil {
		t.Fatal(err)
	}
	// Build a spec band centered on mean=5 with USL-LSL = 6*sqrt(2) so Cp = 1.0.
	half := 3 * math.Sqrt2
	r, err := Compute(Study{USL: 5 + half, LSL: 5 - half, Mean: 5, SigmaWithin: sw})
	if err != nil {
		t.Fatal(err)
	}
	checkClose(t, "Cp", r.Cp, 1.0, 1e-9)
	checkClose(t, "Cpk", r.Cpk, 1.0, 1e-9)
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

func checkClose(t *testing.T, name string, got, want, tolerance float64) {
	t.Helper()
	if math.Abs(got-want) > tolerance {
		t.Errorf("%s = %v, want %v (tol %v)", name, got, want, tolerance)
	}
}
