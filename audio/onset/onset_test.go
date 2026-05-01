package onset

import (
	"math"
	"testing"

	"github.com/davly/reality/audio/spectrogram"
	"github.com/davly/reality/signal"
)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// makePercussiveTrain synthesises an audio signal with `numOnsets`
// equally-spaced onsets. Each onset is a short exponentially-decaying
// burst of high-frequency content. Returns the buffer and the
// onset-sample positions.
func makePercussiveTrain(numOnsets, samplesPerOnset, sr int, decayConst float64) ([]float64, []int) {
	totalSamples := numOnsets * samplesPerOnset
	out := make([]float64, totalSamples)
	positions := make([]int, numOnsets)
	for k := 0; k < numOnsets; k++ {
		onsetIdx := k * samplesPerOnset
		positions[k] = onsetIdx
		// Exponentially-decaying burst of mid-frequency content (so it
		// shows up across many bins, robust to FFT window placement).
		for i := 0; i < samplesPerOnset; i++ {
			t := float64(i)
			env := math.Exp(-t / decayConst)
			// Multi-sine: 1 kHz + 3 kHz so spectral flux finds it.
			out[onsetIdx+i] = env * (math.Sin(2*math.Pi*1000.0*t/float64(sr)) +
				0.5*math.Sin(2*math.Pi*3000.0*t/float64(sr)))
		}
	}
	return out, positions
}

// computeSTFT helper using Hann window.
func computeSTFT(samples []float64, frameSize, hopSize int) [][]complex128 {
	window := make([]float64, frameSize)
	signal.HannWindow(frameSize, window)
	return spectrogram.Compute(samples, frameSize, hopSize, window)
}

// makeSinusoid returns a pure sine of given freq.
func makeSinusoid(n int, freq, sr float64) []float64 {
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		out[i] = math.Sin(2 * math.Pi * freq * float64(i) / sr)
	}
	return out
}

// countNear counts how many `wantSamples` have at least one detected
// frame within ±toleranceFrames of (wantSample / hopSize).
func countNear(detected []int, wantSamples []int, hopSize, toleranceFrames int) int {
	matched := 0
	for _, w := range wantSamples {
		wantFrame := w / hopSize
		for _, d := range detected {
			if d-wantFrame <= toleranceFrames && wantFrame-d <= toleranceFrames {
				matched++
				break
			}
		}
	}
	return matched
}

// ---------------------------------------------------------------------------
// Energy onset
// ---------------------------------------------------------------------------

func TestEnergyOnset_FindsFourPercussiveOnsets(t *testing.T) {
	sr := 16000
	frameSize := 1024
	hopSize := 256
	samples, positions := makePercussiveTrain(4, 16000, sr, 800)

	got := EnergyOnset(samples, frameSize, hopSize)
	matched := countNear(got, positions, hopSize, 4)
	if matched < 3 {
		t.Errorf("EnergyOnset matched only %d/4 onsets; got=%v positions=%v", matched, got, positions)
	}
}

func TestEnergyOnset_PureToneNoOnsets(t *testing.T) {
	sr := 16000
	samples := makeSinusoid(sr, 440, float64(sr))
	got := EnergyOnset(samples, 1024, 256)
	// A pure stationary tone should produce no onsets (energy is
	// essentially constant across frames).
	if len(got) > 1 {
		t.Errorf("EnergyOnset on pure tone returned %d onsets, expected 0-1; got=%v", len(got), got)
	}
}

func TestEnergyOnset_PanicsOnInvalid(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on invalid hopSize")
		}
	}()
	EnergyOnset([]float64{1, 2, 3, 4, 5, 6, 7, 8}, 4, 0)
}

// ---------------------------------------------------------------------------
// Spectral-flux onset
// ---------------------------------------------------------------------------

func TestSpectralFluxOnset_FindsFourOnsets(t *testing.T) {
	sr := 16000
	frameSize := 1024
	hopSize := 256
	samples, positions := makePercussiveTrain(4, 16000, sr, 800)

	stft := computeSTFT(samples, frameSize, hopSize)
	got := SpectralFluxOnset(stft)
	matched := countNear(got, positions, hopSize, 4)
	if matched < 3 {
		t.Errorf("SpectralFluxOnset matched only %d/4 onsets; got=%v positions=%v", matched, got, positions)
	}
}

func TestSpectralFluxOnset_PureToneNoOnsets(t *testing.T) {
	sr := 16000.0
	samples := makeSinusoid(8192, 440, sr)
	stft := computeSTFT(samples, 1024, 256)
	got := SpectralFluxOnset(stft)
	if len(got) > 1 {
		t.Errorf("SpectralFluxOnset on pure tone returned %d onsets, expected 0-1; got=%v", len(got), got)
	}
}

func TestSpectralFluxOnset_PanicsOnTooShort(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on len(stft) < 2")
		}
	}()
	SpectralFluxOnset([][]complex128{{1 + 0i}})
}

func TestSpectralFluxStrength_FirstFrameZero(t *testing.T) {
	stft := [][]complex128{{1, 1, 1}, {2, 2, 2}, {3, 3, 3}}
	sf := SpectralFluxStrength(stft)
	if sf[0] != 0 {
		t.Errorf("SF[0] should be 0 by convention, got %v", sf[0])
	}
	// SF[1] should equal Σ_k (|stft[1][k]| - |stft[0][k]|) = 3
	if math.Abs(sf[1]-3.0) > 1e-12 {
		t.Errorf("SF[1] = %v, expected 3.0", sf[1])
	}
}

// ---------------------------------------------------------------------------
// Complex-domain onset
// ---------------------------------------------------------------------------

func TestComplexDomainOnset_FindsFourOnsets(t *testing.T) {
	sr := 16000
	frameSize := 1024
	hopSize := 256
	samples, positions := makePercussiveTrain(4, 16000, sr, 800)

	stft := computeSTFT(samples, frameSize, hopSize)
	got := ComplexDomainOnset(stft)
	matched := countNear(got, positions, hopSize, 4)
	if matched < 3 {
		t.Errorf("ComplexDomainOnset matched only %d/4 onsets; got=%v positions=%v", matched, got, positions)
	}
}

func TestComplexDomainOnset_PureToneNoOnsets(t *testing.T) {
	sr := 16000.0
	samples := makeSinusoid(8192, 440, sr)
	stft := computeSTFT(samples, 1024, 256)
	got := ComplexDomainOnset(stft)
	if len(got) > 2 {
		t.Errorf("ComplexDomainOnset on pure tone returned %d onsets, expected ≤2; got=%v", len(got), got)
	}
}

func TestComplexDomainOnset_PanicsOnTooShort(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on len(stft) < 3")
		}
	}()
	stft := [][]complex128{{1 + 0i}, {1 + 0i}}
	ComplexDomainOnset(stft)
}

func TestWrapPhase_RangeReduction(t *testing.T) {
	cases := []struct {
		in   float64
		want float64
	}{
		{0, 0},
		{math.Pi, math.Pi},
		{-math.Pi, -math.Pi},
		{3 * math.Pi, math.Pi},
		{-3 * math.Pi, -math.Pi},
	}
	for _, c := range cases {
		got := wrapPhase(c.in)
		if math.Abs(got-c.want) > 1e-12 {
			t.Errorf("wrapPhase(%v) = %v, want %v", c.in, got, c.want)
		}
	}
}

// ---------------------------------------------------------------------------
// SuperFlux
// ---------------------------------------------------------------------------

func TestSuperFlux_FindsFourOnsets(t *testing.T) {
	sr := 16000
	frameSize := 1024
	hopSize := 256
	samples, positions := makePercussiveTrain(4, 16000, sr, 800)

	stft := computeSTFT(samples, frameSize, hopSize)
	got := SuperFlux(stft, 3)
	matched := countNear(got, positions, hopSize, 4)
	if matched < 3 {
		t.Errorf("SuperFlux matched only %d/4 onsets; got=%v positions=%v", matched, got, positions)
	}
}

func TestSuperFlux_VibratoSuppressesFalsePositives(t *testing.T) {
	// A vibrato signal — frequency-modulated tone with no real onsets.
	// Vanilla flux would flag every sweep as an onset; SuperFlux with
	// max-filter should produce far fewer.
	sr := 16000.0
	N := 16384
	samples := make([]float64, N)
	for i := 0; i < N; i++ {
		t := float64(i) / sr
		// 5 Hz vibrato around 440 Hz.
		f := 440.0 + 30.0*math.Sin(2*math.Pi*5*t)
		samples[i] = math.Sin(2 * math.Pi * f * t)
	}
	stft := computeSTFT(samples, 1024, 256)
	got := SuperFlux(stft, 3)
	if len(got) > 5 {
		t.Errorf("SuperFlux on vibrato signal returned %d false onsets, max-filter not suppressing", len(got))
	}
}

func TestSuperFlux_PanicsOnNegativeFilter(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on maxFilterFrames < 0")
		}
	}()
	stft := [][]complex128{{1 + 0i}, {1 + 0i}}
	SuperFlux(stft, -1)
}

// ---------------------------------------------------------------------------
// Peak picking
// ---------------------------------------------------------------------------

func TestPickPeaks_FindsLocalMaxima(t *testing.T) {
	strengths := []float64{0, 1, 0, 2, 0, 3, 0}
	got := PickPeaks(strengths, 0.5, 1)
	want := []int{1, 3, 5}
	if len(got) != len(want) {
		t.Fatalf("got %v peaks, want %v", got, want)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Errorf("peak %d: got %v, want %v", i, got[i], want[i])
		}
	}
}

func TestPickPeaks_RespectsMinSpacing(t *testing.T) {
	// Peaks at 1, 2, 3 with minSpacing=2 should yield only 1, 3.
	strengths := []float64{0, 5, 4, 3, 0}
	got := PickPeaks(strengths, 0.1, 2)
	if len(got) > 2 {
		t.Errorf("got %d peaks (too many — minSpacing not enforced): %v", len(got), got)
	}
}

func TestPickPeaks_BelowThresholdSkipped(t *testing.T) {
	strengths := []float64{0, 5, 0, 0.5, 0}
	got := PickPeaks(strengths, 1.0, 1)
	if len(got) != 1 || got[0] != 1 {
		t.Errorf("expected only peak at index 1, got %v", got)
	}
}

func TestPickPeaks_PanicsOnNaNThreshold(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on NaN threshold")
		}
	}()
	PickPeaks([]float64{1, 2, 1}, math.NaN(), 0)
}

func TestPickPeaksAdaptive_ScalesWithSignalLevel(t *testing.T) {
	// Two signals — one with peaks at amp 1.0, one with peaks at amp
	// 100.0. Both should produce the same number of peaks because the
	// adaptive threshold scales with mean+σ.
	low := []float64{0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0}
	high := make([]float64, len(low))
	for i := range low {
		high[i] = 100 * low[i]
	}
	gLow := PickPeaksAdaptive(low, 1.0, 1)
	gHigh := PickPeaksAdaptive(high, 1.0, 1)
	if len(gLow) != len(gHigh) {
		t.Errorf("adaptive threshold not scale-invariant: low=%d high=%d", len(gLow), len(gHigh))
	}
	if len(gLow) < 3 {
		t.Errorf("expected ≥3 peaks at evenly-spaced positions, got %v", gLow)
	}
}

func TestPickPeaks_EmptyInput(t *testing.T) {
	got := PickPeaks(nil, 0.5, 1)
	if len(got) != 0 {
		t.Errorf("expected empty result for empty input, got %v", got)
	}
}
