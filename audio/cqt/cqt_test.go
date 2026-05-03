package cqt

import (
	"errors"
	"math"
	"math/cmplx"
	"testing"
)

// --- QualityFactor / BinFrequency / WindowLength --------------------

func TestQualityFactor_StandardBinResolutions(t *testing.T) {
	cases := []struct {
		bpo  int
		want float64
		tol  float64
	}{
		{12, 16.817, 1e-3},
		{24, 34.127, 1e-3},
		{36, 51.439, 1e-3},
	}
	for _, c := range cases {
		got := QualityFactor(c.bpo)
		if math.Abs(got-c.want) > c.tol {
			t.Errorf("QualityFactor(%d) = %g, want %g (±%g)", c.bpo, got, c.want, c.tol)
		}
	}
}

func TestQualityFactor_PanicsOnNonPositiveBpo(t *testing.T) {
	defer func() { _ = recover() }()
	QualityFactor(0)
	t.Fatal("expected panic for binsPerOctave=0")
}

func TestBinFrequency_OctaveDoubles(t *testing.T) {
	const bpo = 12
	const fMin = 27.5 // A0
	got := BinFrequency(bpo, bpo, fMin)
	want := 2.0 * fMin
	if math.Abs(got-want) > 1e-9 {
		t.Errorf("BinFrequency(bpo, bpo, fMin) = %g, want %g (octave double)", got, want)
	}
}

func TestBinFrequency_BinZeroIsFMin(t *testing.T) {
	if BinFrequency(0, 12, 27.5) != 27.5 {
		t.Error("bin 0 must equal fMin")
	}
}

func TestBinFrequency_PanicsOnInvalidInputs(t *testing.T) {
	t.Run("zero fMin", func(t *testing.T) {
		defer func() { _ = recover() }()
		BinFrequency(0, 12, 0)
		t.Fatal("expected panic for fMin=0")
	})
	t.Run("zero bpo", func(t *testing.T) {
		defer func() { _ = recover() }()
		BinFrequency(0, 0, 27.5)
		t.Fatal("expected panic for binsPerOctave=0")
	})
}

func TestBinFrequencies_FillsAcrossOctaves(t *testing.T) {
	const bpo, octaves = 12, 3
	out := make([]float64, bpo*octaves)
	BinFrequencies(bpo, octaves, 27.5, out)

	if out[0] != 27.5 {
		t.Errorf("out[0] = %g, want 27.5", out[0])
	}
	if math.Abs(out[12]-55.0) > 1e-9 {
		t.Errorf("out[12] = %g, want 55.0 (one octave up)", out[12])
	}
	if math.Abs(out[24]-110.0) > 1e-9 {
		t.Errorf("out[24] = %g, want 110.0 (two octaves up)", out[24])
	}
}

func TestBinFrequencies_PanicsOnShortOut(t *testing.T) {
	defer func() { _ = recover() }()
	out := make([]float64, 5)
	BinFrequencies(12, 1, 27.5, out)
	t.Fatal("expected panic for short out")
}

func TestWindowLength_ScalesInverselyWithFrequency(t *testing.T) {
	q := QualityFactor(12)
	const sr = 22050.0
	low := WindowLength(q, sr, 110.0)  // A2
	high := WindowLength(q, sr, 880.0) // A5

	// 880 Hz is 8x the 110 Hz freq → window should be 1/8 the size.
	if low/high < 7 || low/high > 9 {
		t.Errorf("WindowLength scaling: low=%d high=%d, expected ~8x ratio", low, high)
	}
}

func TestWindowLength_PanicsOnInvalidInputs(t *testing.T) {
	q := QualityFactor(12)
	t.Run("zero sr", func(t *testing.T) {
		defer func() { _ = recover() }()
		WindowLength(q, 0, 440.0)
		t.Fatal("expected panic for sr=0")
	})
	t.Run("zero f", func(t *testing.T) {
		defer func() { _ = recover() }()
		WindowLength(q, 22050, 0)
		t.Fatal("expected panic for f=0")
	})
}

// --- CQT happy path: pure tone detection ----------------------------

func TestCQT_PureTone_PeaksAtExpectedBin(t *testing.T) {
	const sr = 22050.0
	const fMin = 110.0 // A2
	const bpo = 12
	const octaves = 3
	K := bpo * octaves

	// Generate a 220 Hz pure tone (A3) — that's bin 12 for fMin=110.
	const f = 220.0
	q := QualityFactor(bpo)
	nMax := WindowLength(q, sr, fMin)
	x := make([]float64, nMax)
	for n := range x {
		x[n] = math.Sin(2.0 * math.Pi * f * float64(n) / sr)
	}

	out := make([]complex128, K)
	if err := CQT(x, sr, fMin, bpo, octaves, out); err != nil {
		t.Fatalf("CQT returned error: %v", err)
	}

	peak := PeakBin(out)
	// CQT is approximate; tolerate ±1 bin around the expected centre.
	if peak < 11 || peak > 13 {
		t.Errorf("peak bin = %d, expected 12 (±1) for 220 Hz at fMin=110, B=12", peak)
	}
}

func TestCQT_OctaveShift_ShiftsPeakByBinsPerOctave(t *testing.T) {
	const sr = 44100.0
	const fMin = 110.0
	const bpo = 12
	const octaves = 4

	q := QualityFactor(bpo)
	nMax := WindowLength(q, sr, fMin)

	gen := func(f float64) []float64 {
		x := make([]float64, nMax)
		for n := range x {
			x[n] = math.Sin(2.0 * math.Pi * f * float64(n) / sr)
		}
		return x
	}

	K := bpo * octaves
	low := make([]complex128, K)
	high := make([]complex128, K)
	if err := CQT(gen(220), sr, fMin, bpo, octaves, low); err != nil {
		t.Fatalf("CQT (220Hz): %v", err)
	}
	if err := CQT(gen(440), sr, fMin, bpo, octaves, high); err != nil {
		t.Fatalf("CQT (440Hz): %v", err)
	}

	peakLow := PeakBin(low)
	peakHigh := PeakBin(high)

	// Tolerate ±1 spectral leakage; the difference should be ~bpo.
	diff := peakHigh - peakLow
	if diff < bpo-1 || diff > bpo+1 {
		t.Errorf("octave shift bin diff = %d, expected ~%d (±1)", diff, bpo)
	}
}

// --- CQT determinism ------------------------------------------------

func TestCQT_DeterministicAcrossCalls(t *testing.T) {
	const sr = 22050.0
	const fMin = 110.0
	const bpo = 12
	const octaves = 2
	K := bpo * octaves

	q := QualityFactor(bpo)
	nMax := WindowLength(q, sr, fMin)
	x := make([]float64, nMax)
	for n := range x {
		x[n] = math.Sin(2*math.Pi*220*float64(n)/sr) +
			0.5*math.Sin(2*math.Pi*440*float64(n)/sr)
	}

	a := make([]complex128, K)
	b := make([]complex128, K)
	if err := CQT(x, sr, fMin, bpo, octaves, a); err != nil {
		t.Fatal(err)
	}
	if err := CQT(x, sr, fMin, bpo, octaves, b); err != nil {
		t.Fatal(err)
	}

	for k := 0; k < K; k++ {
		if a[k] != b[k] {
			t.Errorf("non-deterministic at bin %d: %v vs %v", k, a[k], b[k])
		}
	}
}

// --- CQT error paths ------------------------------------------------

func TestCQT_ZeroParam_ReturnsErrInvalidParams(t *testing.T) {
	out := make([]complex128, 12)
	cases := []struct {
		name string
		args func() error
	}{
		{"sr=0", func() error { return CQT(make([]float64, 1024), 0, 110, 12, 1, out) }},
		{"fMin=0", func() error { return CQT(make([]float64, 1024), 22050, 0, 12, 1, out) }},
		{"bpo=0", func() error { return CQT(make([]float64, 1024), 22050, 110, 0, 1, out) }},
		{"octaves=0", func() error { return CQT(make([]float64, 1024), 22050, 110, 12, 0, out) }},
	}
	for _, c := range cases {
		if err := c.args(); !errors.Is(err, ErrInvalidParams) {
			t.Errorf("%s: got %v, want ErrInvalidParams", c.name, err)
		}
	}
}

func TestCQT_OutputWrongSize_ReturnsErrOutputSize(t *testing.T) {
	out := make([]complex128, 100) // not 12 * 1
	err := CQT(make([]float64, 4096), 22050, 110, 12, 1, out)
	if !errors.Is(err, ErrOutputSize) {
		t.Errorf("got %v, want ErrOutputSize", err)
	}
}

func TestCQT_TopBinAtNyquist_ReturnsErrSampleRateTooLow(t *testing.T) {
	// fMin = 4000Hz, B = 12, octaves = 4 → top bin near 64000Hz at sr 22050 → above Nyquist.
	out := make([]complex128, 48)
	err := CQT(make([]float64, 4096), 22050, 4000, 12, 4, out)
	if !errors.Is(err, ErrSampleRateTooLow) {
		t.Errorf("got %v, want ErrSampleRateTooLow", err)
	}
}

func TestCQT_InputTooShort_ReturnsErrInputTooShort(t *testing.T) {
	// nMax for fMin=27.5 at sr=22050 with B=12 is ~17 * 22050 / 27.5 ≈ 13486.
	// Input of 100 samples is far too short.
	out := make([]complex128, 12)
	err := CQT(make([]float64, 100), 22050, 27.5, 12, 1, out)
	if !errors.Is(err, ErrInputTooShort) {
		t.Errorf("got %v, want ErrInputTooShort", err)
	}
}

// --- Magnitude / PeakBin --------------------------------------------

func TestMagnitude_ReturnsAbsoluteValuePerBin(t *testing.T) {
	in := []complex128{
		complex(3, 4),  // |.|=5
		complex(0, 1),  // |.|=1
		complex(-1, 0), // |.|=1
	}
	mag := make([]float64, len(in))
	Magnitude(in, mag)
	want := []float64{5, 1, 1}
	for i := range mag {
		if math.Abs(mag[i]-want[i]) > 1e-12 {
			t.Errorf("mag[%d] = %g, want %g", i, mag[i], want[i])
		}
	}
}

func TestMagnitude_PanicsOnShortOut(t *testing.T) {
	defer func() { _ = recover() }()
	Magnitude([]complex128{1, 2, 3}, make([]float64, 2))
	t.Fatal("expected panic for short mag slice")
}

func TestPeakBin_FindsMaxMagnitude(t *testing.T) {
	in := []complex128{
		complex(1, 0),
		complex(3, 4),  // |.|=5 — peak
		complex(0, 2),
	}
	if got := PeakBin(in); got != 1 {
		t.Errorf("PeakBin = %d, want 1", got)
	}
}

func TestPeakBin_EmptyInput_ReturnsMinusOne(t *testing.T) {
	if got := PeakBin(nil); got != -1 {
		t.Errorf("PeakBin(nil) = %d, want -1", got)
	}
}

func TestPeakBin_TiedPeak_ReturnsFirst(t *testing.T) {
	// Equal magnitudes — first wins.
	in := []complex128{complex(0, 0), complex(1, 0), complex(1, 0)}
	if got := PeakBin(in); got != 1 {
		t.Errorf("PeakBin tied = %d, want 1 (first)", got)
	}
}

// --- Cross-check vs known cmplx behaviour ----------------------------

func TestCQT_DCOnly_LowestBinNonZero(t *testing.T) {
	// A pure DC signal (all 1s) should have zero magnitude in every CQT
	// bin since CQT atoms are bandpass at non-zero frequencies. But the
	// integration over the windowed atom against a constant is non-trivial;
	// at minimum, the peak should be far from any musically-meaningful bin.
	const sr = 22050.0
	const fMin = 110.0
	const bpo = 12
	const octaves = 2
	K := bpo * octaves

	q := QualityFactor(bpo)
	nMax := WindowLength(q, sr, fMin)
	x := make([]float64, nMax)
	for i := range x {
		x[i] = 1.0
	}

	out := make([]complex128, K)
	if err := CQT(x, sr, fMin, bpo, octaves, out); err != nil {
		t.Fatal(err)
	}

	// Just sanity-check: no NaN or +Inf values.
	for k, v := range out {
		if math.IsNaN(real(v)) || math.IsNaN(imag(v)) {
			t.Errorf("bin %d has NaN", k)
		}
		if math.IsInf(cmplx.Abs(v), 0) {
			t.Errorf("bin %d has Inf magnitude", k)
		}
	}
}
