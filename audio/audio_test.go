package audio

import (
	"math"
	"testing"

	"github.com/davly/reality/signal"
)

const tolerance = 1e-9
const epsilon = 1e-6

// ---------------------------------------------------------------------------
// Mel scale tests
// ---------------------------------------------------------------------------

func TestHzToMelKnownPoints(t *testing.T) {
	cases := []struct {
		hz, mel float64
	}{
		{0.0, 0.0},
		// Cross-checked against librosa.hz_to_mel(htk=True).
		// Note librosa uses 2595 * log10(1 + f/700) by default which
		// corresponds to 1127 * ln(1 + f/700) — same Slaney form.
		{700.0, 1127.0 * math.Log(2.0)},   // ~781.42 mel
		{1000.0, 1127.0 * math.Log1p(10.0/7.0)},
	}
	for _, c := range cases {
		got := HzToMel(c.hz)
		if math.Abs(got-c.mel) > tolerance {
			t.Errorf("HzToMel(%v) = %v, want %v", c.hz, got, c.mel)
		}
	}
}

func TestMelHzRoundTrip(t *testing.T) {
	for f := 0.0; f <= 8000.0; f += 137.0 {
		m := HzToMel(f)
		f2 := MelToHz(m)
		if math.Abs(f-f2) > 1e-9 {
			t.Errorf("MelToHz(HzToMel(%v)) = %v, drift %v", f, f2, math.Abs(f-f2))
		}
	}
}

func TestHzToMelMonotonic(t *testing.T) {
	prev := -1.0
	for f := 0.0; f <= 8000.0; f += 50.0 {
		m := HzToMel(f)
		if m < prev {
			t.Fatalf("HzToMel non-monotonic at f=%v: %v < %v", f, m, prev)
		}
		prev = m
	}
}

func TestMelFilterbankShape(t *testing.T) {
	const sr = 16000.0
	const nFFT = 512
	const numFilters = 26

	nBins := nFFT/2 + 1
	fb := make([]float64, numFilters*nBins)
	MelFilterbank(sr, nFFT, numFilters, 0, sr/2, fb)

	// Each filter should be triangular: rises, peaks, falls. Sum of all
	// weights in a filter should be > 0 (filter is non-trivial).
	for b := 0; b < numFilters; b++ {
		sum := 0.0
		peak := 0.0
		for k := 0; k < nBins; k++ {
			w := fb[b*nBins+k]
			if w < 0 {
				t.Errorf("filter %d bin %d has negative weight %v", b, k, w)
			}
			if w > peak {
				peak = w
			}
			sum += w
		}
		if peak <= 0 {
			t.Errorf("filter %d has zero peak", b)
		}
		// Triangular filters peak at most 1.0 in the unnormalised HTK
		// convention. Because filterbank centres are fractional bin
		// indices (Slaney 1998 §3.2 — preserves per-band area as
		// numFilters → ∞), the closest integer bin to a centre can
		// be up to 0.5 bins away, giving a sampled peak as low as
		// 1 - 0.5/min_band_width. For 26-band 0..8kHz over 257 bins,
		// the narrowest band is ~5 bins wide so peak >= 0.5 is sound.
		if peak < 0.5 || peak > 1.0+1e-12 {
			t.Errorf("filter %d peak = %v, expected in (0.5, 1.0]", b, peak)
		}
	}
}

func TestMelFilterbankPanics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on invalid frequency range")
		}
	}()
	out := make([]float64, 26*257)
	MelFilterbank(16000, 512, 26, 8000, 4000, out) // fMin > fMax
}

// ---------------------------------------------------------------------------
// MFCC tests
// ---------------------------------------------------------------------------

func TestMFCC_DCTOrthogonality(t *testing.T) {
	// DCT-II orthonormal on a single non-zero coefficient should round-trip
	// through DCT-III, but here we just sanity-check that constant log-energies
	// yield a single non-zero coefficient at k=0 (DC) with the expected
	// magnitude sqrt(M) * value.
	const M = 26
	const numCoeffs = 13
	logE := make([]float64, M)
	for i := range logE {
		logE[i] = 2.5
	}
	out := make([]float64, numCoeffs)
	MFCC(logE, numCoeffs, out)

	expected0 := math.Sqrt(float64(M)) * 2.5
	if math.Abs(out[0]-expected0) > 1e-9 {
		t.Errorf("MFCC DC coefficient = %v, want %v", out[0], expected0)
	}
	for k := 1; k < numCoeffs; k++ {
		if math.Abs(out[k]) > 1e-9 {
			t.Errorf("MFCC coefficient %d = %v, expected ~0 for constant input", k, out[k])
		}
	}
}

func TestMFCC_OrthonormalScaling(t *testing.T) {
	// L2 norm of orthonormal DCT-II output equals L2 norm of input.
	const M = 26
	logE := make([]float64, M)
	for i := range logE {
		logE[i] = math.Sin(float64(i)*0.3) + 0.1*float64(i%5)
	}
	out := make([]float64, M)
	MFCC(logE, M, out)

	inputNorm := 0.0
	for _, v := range logE {
		inputNorm += v * v
	}
	outputNorm := 0.0
	for _, v := range out {
		outputNorm += v * v
	}
	if math.Abs(inputNorm-outputNorm) > 1e-9 {
		t.Errorf("DCT-II orthonormality violation: input^2 = %v, output^2 = %v", inputNorm, outputNorm)
	}
}

// ---------------------------------------------------------------------------
// PowerSpectrum + ApplyFilterbank composition
// ---------------------------------------------------------------------------

func TestPowerSpectrumAndFilterbank_Composition(t *testing.T) {
	// Synthesise a sine wave at 1000 Hz, run through FFT → power → mel
	// filterbank, expect energy concentrated in the band containing 1000 Hz.
	const sr = 16000.0
	const nFFT = 512
	const numFilters = 26
	const targetHz = 1000.0

	frame := make([]float64, nFFT)
	imag := make([]float64, nFFT)
	for i := 0; i < nFFT; i++ {
		frame[i] = math.Sin(2 * math.Pi * targetHz * float64(i) / sr)
	}
	signal.FFT(frame, imag)

	power := make([]float64, nFFT/2+1)
	PowerSpectrum(frame, imag, power)

	fb := make([]float64, numFilters*(nFFT/2+1))
	MelFilterbank(sr, nFFT, numFilters, 0, sr/2, fb)

	melE := make([]float64, numFilters)
	ApplyFilterbank(power, fb, numFilters, nFFT/2+1, melE)

	// Identify the band with the highest energy and assert it contains
	// 1000 Hz. We compute band centre frequencies from the filterbank
	// construction.
	melLow := HzToMel(0)
	melHigh := HzToMel(sr / 2)
	bestBand := -1
	bestEnergy := 0.0
	for b := 0; b < numFilters; b++ {
		if melE[b] > bestEnergy {
			bestEnergy = melE[b]
			bestBand = b
		}
	}
	if bestBand < 0 {
		t.Fatal("no energy detected — synth signal may be wrong")
	}

	melCenter := melLow + float64(bestBand+1)*(melHigh-melLow)/float64(numFilters+1)
	hzCenter := MelToHz(melCenter)
	// Band centre should be within one filter spacing of 1000 Hz.
	bandWidth := MelToHz(melLow+(melHigh-melLow)/float64(numFilters+1)) - 0
	if math.Abs(hzCenter-targetHz) > bandWidth*1.5 {
		t.Errorf("strongest band centre %v Hz; expected within %v Hz of %v Hz",
			hzCenter, bandWidth*1.5, targetHz)
	}
}

// ---------------------------------------------------------------------------
// Fingerprint (Welford) tests
// ---------------------------------------------------------------------------

func TestFingerprint_KnownMeanVariance(t *testing.T) {
	// Hand-computed mean/variance for [1,2,3,4,5]: mean=3, var=2.5 (unbiased).
	fp := NewFingerprint(1)
	for _, x := range []float64{1, 2, 3, 4, 5} {
		UpdateFingerprint(&fp, []float64{x})
	}
	if math.Abs(fp.Mean[0]-3.0) > 1e-9 {
		t.Errorf("Welford mean = %v, want 3", fp.Mean[0])
	}
	v := make([]float64, 1)
	FingerprintVariance(&fp, v)
	if math.Abs(v[0]-2.5) > 1e-9 {
		t.Errorf("Welford variance = %v, want 2.5", v[0])
	}
}

func TestFingerprint_OrderInvariance(t *testing.T) {
	// Welford updates must be order-independent in the limit.
	in1 := []float64{5, 1, 4, 2, 3, 6, 7, 8, 9, 0}
	in2 := []float64{0, 9, 8, 7, 6, 3, 2, 4, 1, 5} // permutation of in1
	a := NewFingerprint(1)
	b := NewFingerprint(1)
	for _, x := range in1 {
		UpdateFingerprint(&a, []float64{x})
	}
	for _, x := range in2 {
		UpdateFingerprint(&b, []float64{x})
	}
	if math.Abs(a.Mean[0]-b.Mean[0]) > 1e-12 {
		t.Errorf("Welford order non-invariance in mean: %v vs %v", a.Mean[0], b.Mean[0])
	}
	if math.Abs(a.M2[0]-b.M2[0]) > 1e-9 {
		t.Errorf("Welford order non-invariance in M2: %v vs %v", a.M2[0], b.M2[0])
	}
}

func TestFingerprint_BestMatch(t *testing.T) {
	// Build two fingerprints with distinct centroids.
	fpA := NewFingerprint(2)
	fpB := NewFingerprint(2)
	for i := 0; i < 50; i++ {
		UpdateFingerprint(&fpA, []float64{0.0 + 0.01*float64(i%3), 0.0 + 0.01*float64(i%5)})
		UpdateFingerprint(&fpB, []float64{5.0 + 0.01*float64(i%3), 5.0 + 0.01*float64(i%5)})
	}
	fps := []Fingerprint{fpA, fpB}

	// Query near A.
	idx, d := BestMatch(fps, []float64{0.05, 0.05}, epsilon)
	if idx != 0 {
		t.Errorf("BestMatch near A returned %d (d=%v); expected 0", idx, d)
	}

	// Query near B.
	idx, d = BestMatch(fps, []float64{4.95, 5.05}, epsilon)
	if idx != 1 {
		t.Errorf("BestMatch near B returned %d (d=%v); expected 1", idx, d)
	}
}

func TestFingerprint_Merge(t *testing.T) {
	// Two halves of a stream merged should equal the full stream's fingerprint.
	full := NewFingerprint(2)
	half1 := NewFingerprint(2)
	half2 := NewFingerprint(2)

	xs := [][]float64{
		{1, 2}, {3, 1}, {5, 7}, {2, 4}, {8, 3}, {4, 6}, {6, 2}, {7, 5},
	}
	for _, x := range xs {
		UpdateFingerprint(&full, x)
	}
	for _, x := range xs[:4] {
		UpdateFingerprint(&half1, x)
	}
	for _, x := range xs[4:] {
		UpdateFingerprint(&half2, x)
	}

	merged := NewFingerprint(2)
	MergeFingerprints(&half1, &half2, &merged)

	if merged.N != full.N {
		t.Errorf("merged N=%d, full N=%d", merged.N, full.N)
	}
	for i := 0; i < 2; i++ {
		if math.Abs(merged.Mean[i]-full.Mean[i]) > 1e-12 {
			t.Errorf("merged.Mean[%d]=%v, full.Mean[%d]=%v", i, merged.Mean[i], i, full.Mean[i])
		}
		if math.Abs(merged.M2[i]-full.M2[i]) > 1e-9 {
			t.Errorf("merged.M2[%d]=%v, full.M2[%d]=%v", i, merged.M2[i], i, full.M2[i])
		}
	}
}

func TestFingerprint_EmptyMerge(t *testing.T) {
	a := NewFingerprint(2)
	b := NewFingerprint(2)
	out := NewFingerprint(2)
	UpdateFingerprint(&b, []float64{1, 2})
	UpdateFingerprint(&b, []float64{3, 4})

	MergeFingerprints(&a, &b, &out)
	if out.N != 2 {
		t.Errorf("merge with empty a: N=%d, expected 2", out.N)
	}
	if math.Abs(out.Mean[0]-2.0) > 1e-12 {
		t.Errorf("merge with empty a: Mean[0]=%v, expected 2", out.Mean[0])
	}
}

// ---------------------------------------------------------------------------
// Degradation tracker tests
// ---------------------------------------------------------------------------

func TestDegradation_StableBaseline_NoDrift(t *testing.T) {
	// Steady stream → window mean ~ baseline mean → z-score ~ 0.
	tr := NewDegradationTracker(8)
	for i := 0; i < 200; i++ {
		// jitter around 100 with ±2 noise
		x := 100.0 + 2.0*math.Sin(float64(i)*0.7)
		PushObservation(&tr, x)
	}
	z := ZScore(&tr)
	if math.Abs(z) > 1.0 {
		t.Errorf("steady baseline produced |z|=%v; expected near 0", z)
	}
}

func TestDegradation_StepShift_DetectsDrift(t *testing.T) {
	// Build a long stable baseline, then a sustained step shift in the
	// recent window. ZScore should be large and positive.
	tr := NewDegradationTracker(8)
	for i := 0; i < 500; i++ {
		x := 100.0 + 0.5*math.Sin(float64(i)*0.7)
		PushObservation(&tr, x)
	}
	for i := 0; i < 8; i++ {
		PushObservation(&tr, 110.0)
	}
	z := ZScore(&tr)
	// We expect a large positive z. Threshold is permissive because the
	// shift gets averaged with the existing baseline as we push (since
	// PushObservation also feeds the baseline).
	if z < 1.0 {
		t.Errorf("step shift produced z=%v; expected > 1.0", z)
	}
}

func TestDegradation_ResetWindowKeepsBaseline(t *testing.T) {
	tr := NewDegradationTracker(8)
	for i := 0; i < 100; i++ {
		PushObservation(&tr, 50.0+0.1*math.Sin(float64(i)))
	}
	priorN := tr.BaselineN
	priorMean := tr.BaselineMean
	ResetWindow(&tr)
	if tr.BaselineN != priorN || tr.BaselineMean != priorMean {
		t.Error("ResetWindow should not affect baseline")
	}
	if tr.WindowFill != 0 {
		t.Errorf("ResetWindow left WindowFill=%d", tr.WindowFill)
	}
}

func TestDegradation_ResetBaselineClearsAll(t *testing.T) {
	tr := NewDegradationTracker(8)
	for i := 0; i < 50; i++ {
		PushObservation(&tr, 50.0)
	}
	ResetBaseline(&tr)
	if tr.BaselineN != 0 {
		t.Errorf("ResetBaseline left BaselineN=%d", tr.BaselineN)
	}
	if tr.WindowFill != 0 {
		t.Errorf("ResetBaseline left WindowFill=%d", tr.WindowFill)
	}
}

func TestDegradation_PanicsOnSmallWindow(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on window size 1")
		}
	}()
	NewDegradationTracker(1)
}

// ---------------------------------------------------------------------------
// End-to-end: synthesise → fingerprint → degradation
// ---------------------------------------------------------------------------

// makeFrameMFCC composes the full per-frame pipeline used by mobile callers.
// Not in production code (callers control allocation themselves) — used here
// for test composition.
func makeFrameMFCC(t *testing.T, frame []float64, sr float64, numFilters, numCoeffs int, fb []float64) []float64 {
	t.Helper()
	nFFT := len(frame)

	imag := make([]float64, nFFT)
	signal.FFT(frame, imag)
	power := make([]float64, nFFT/2+1)
	PowerSpectrum(frame, imag, power)
	melE := make([]float64, numFilters)
	ApplyFilterbank(power, fb, numFilters, nFFT/2+1, melE)
	logE := make([]float64, numFilters)
	LogMelEnergies(melE, 1e-10, logE)
	mfcc := make([]float64, numCoeffs)
	MFCC(logE, numCoeffs, mfcc)
	return mfcc
}

func TestEndToEnd_FingerprintIdentifiesSamePitch(t *testing.T) {
	const sr = 16000.0
	const nFFT = 512
	const numFilters = 26
	const numCoeffs = 13

	fb := make([]float64, numFilters*(nFFT/2+1))
	MelFilterbank(sr, nFFT, numFilters, 0, sr/2, fb)

	// Build two fingerprints from frames at distinct pitches.
	fpA := NewFingerprint(numCoeffs)
	fpB := NewFingerprint(numCoeffs)
	hannWin := make([]float64, nFFT)
	signal.HannWindow(nFFT, hannWin)

	mkFrame := func(pitch float64, seed int) []float64 {
		f := make([]float64, nFFT)
		for i := 0; i < nFFT; i++ {
			// simple sine + small noise (deterministic seeded by index)
			noise := 0.001 * math.Sin(float64(i+seed)*1.7)
			f[i] = (math.Sin(2*math.Pi*pitch*float64(i)/sr) + noise) * hannWin[i]
		}
		return f
	}

	for k := 0; k < 30; k++ {
		fA := mkFrame(440, k)
		mfccA := makeFrameMFCC(t, fA, sr, numFilters, numCoeffs, fb)
		UpdateFingerprint(&fpA, mfccA)

		fB := mkFrame(880, k)
		mfccB := makeFrameMFCC(t, fB, sr, numFilters, numCoeffs, fb)
		UpdateFingerprint(&fpB, mfccB)
	}

	// Query: a fresh 440Hz frame should match fpA, an 880Hz fpB.
	queryA := mkFrame(440, 999)
	mfccQA := makeFrameMFCC(t, queryA, sr, numFilters, numCoeffs, fb)
	idx, _ := BestMatch([]Fingerprint{fpA, fpB}, mfccQA, epsilon)
	if idx != 0 {
		t.Errorf("440Hz query matched fingerprint %d, expected 0 (fpA)", idx)
	}

	queryB := mkFrame(880, 998)
	mfccQB := makeFrameMFCC(t, queryB, sr, numFilters, numCoeffs, fb)
	idx, _ = BestMatch([]Fingerprint{fpA, fpB}, mfccQB, epsilon)
	if idx != 1 {
		t.Errorf("880Hz query matched fingerprint %d, expected 1 (fpB)", idx)
	}
}
