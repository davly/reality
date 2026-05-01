package vibration

import (
	"math"
	"testing"

	"github.com/davly/reality/audio"
	"github.com/davly/reality/signal"
)

// makeMechFrame synthesises a frame containing a fundamental + harmonics
// at the given amplitudes — a stand-in for "machine running cleanly"
// (high harmonic content) or "bearing wear" (low harmonic content +
// broadband noise). Deterministic given seed.
func makeMechFrame(fundamental, sr float64, n int, harmonicAmps []float64, broadbandNoise float64, seed int) []float64 {
	f := make([]float64, n)
	for i := 0; i < n; i++ {
		v := 0.0
		for h, amp := range harmonicAmps {
			v += amp * math.Sin(2*math.Pi*fundamental*float64(h+1)*float64(i)/sr)
		}
		// Deterministic broadband noise via interleaved sines (cheap PRNG).
		v += broadbandNoise * math.Sin(float64(i+seed*37)*1.7)
		v += broadbandNoise * 0.5 * math.Sin(float64(i+seed*101)*0.31)
		f[i] = v
	}
	return f
}

// ---------------------------------------------------------------------------
// FundamentalHz
// ---------------------------------------------------------------------------

func TestFundamentalHz_DetectsCleanFundamental(t *testing.T) {
	sr := 16000.0
	n := 2048
	frame := makeMechFrame(120, sr, n, []float64{1.0, 0.5, 0.25}, 0.05, 0)
	imag := make([]float64, n)
	hz := FundamentalHz(frame, imag, sr, 50, 1000)

	// FFT bin width = 16000 / 2048 = ~7.8 Hz. We expect detected
	// fundamental within one bin width of 120 Hz.
	if math.Abs(hz-120) > 8.0 {
		t.Errorf("FundamentalHz returned %v, expected ~120", hz)
	}
}

func TestFundamentalHz_DetectsHigherFundamental(t *testing.T) {
	// Higher fundamental — verify peak-bin detection across the search
	// band rather than locking onto a low bin.
	sr := 16000.0
	n := 2048
	frame := makeMechFrame(440, sr, n, []float64{1.0, 0.4}, 0.05, 7)
	imag := make([]float64, n)
	hz := FundamentalHz(frame, imag, sr, 50, 2000)

	if math.Abs(hz-440) > 8.0 {
		t.Errorf("FundamentalHz returned %v, expected ~440", hz)
	}
}

func TestFundamentalHz_SilentFrameReturnsZero(t *testing.T) {
	// Pure-zero frame has no peak; FundamentalHz should report 0
	// rather than the bin-1 frequency by accident.
	sr := 16000.0
	n := 1024
	frame := make([]float64, n)
	imag := make([]float64, n)

	hz := FundamentalHz(frame, imag, sr, 50, 1000)
	if hz != 0 {
		t.Errorf("silent frame produced FundamentalHz=%v, expected 0", hz)
	}
}

func TestFundamentalHz_PanicsOnUnequalSlices(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on unequal slice lengths")
		}
	}()
	frame := make([]float64, 1024)
	imag := make([]float64, 1023)
	FundamentalHz(frame, imag, 16000, 50, 1000)
}

// ---------------------------------------------------------------------------
// HarmonicEnergyRatio
// ---------------------------------------------------------------------------

func TestHarmonicEnergyRatio_HighForCleanSignal(t *testing.T) {
	// Clean signal with strong harmonics → high HER.
	sr := 16000.0
	n := 2048
	frame := makeMechFrame(120, sr, n, []float64{1.0, 0.7, 0.4, 0.2}, 0.02, 0)
	imag := make([]float64, n)
	power := make([]float64, n/2+1)

	frameCopy := make([]float64, n)
	copy(frameCopy, frame)
	signal.FFT(frameCopy, imag)
	audio.PowerSpectrum(frameCopy, imag, power)

	ratio := HarmonicEnergyRatio(power, sr, 120, 5.0, 50, 1000, 5)
	if ratio < 0.5 {
		t.Errorf("clean harmonic signal got HER=%.3f, expected >= 0.5", ratio)
	}
	if ratio > 1.0 {
		t.Errorf("HER=%.3f exceeded 1.0 — clamp violated", ratio)
	}
}

func TestHarmonicEnergyRatio_LowForNoisySignal(t *testing.T) {
	sr := 16000.0
	n := 2048
	// Pure broadband noise — no harmonics. HER should be small.
	frame := makeMechFrame(120, sr, n, []float64{0.0}, 1.0, 42)
	imag := make([]float64, n)
	power := make([]float64, n/2+1)
	frameCopy := make([]float64, n)
	copy(frameCopy, frame)
	signal.FFT(frameCopy, imag)
	audio.PowerSpectrum(frameCopy, imag, power)

	ratio := HarmonicEnergyRatio(power, sr, 120, 5.0, 50, 1000, 5)
	if ratio > 0.3 {
		t.Errorf("noisy signal got HER=%.3f, expected < 0.3", ratio)
	}
}

func TestHarmonicEnergyRatio_ZeroFundamentalReturnsZero(t *testing.T) {
	// Defensive: callers that haven't yet detected a fundamental pass
	// 0; we must return 0 (not divide by zero).
	power := make([]float64, 1025)
	for i := range power {
		power[i] = 1.0
	}
	if r := HarmonicEnergyRatio(power, 16000, 0, 5.0, 50, 1000, 5); r != 0 {
		t.Errorf("HER with fundamental=0 returned %v, expected 0", r)
	}
	if r := HarmonicEnergyRatio(power, 16000, -120, 5.0, 50, 1000, 5); r != 0 {
		t.Errorf("HER with negative fundamental returned %v, expected 0", r)
	}
}

func TestHarmonicEnergyRatio_SilentSpectrumReturnsZero(t *testing.T) {
	// All-zero power → total energy 0 → HER = 0 (not NaN).
	power := make([]float64, 1025)
	r := HarmonicEnergyRatio(power, 16000, 120, 5.0, 50, 1000, 5)
	if r != 0 {
		t.Errorf("silent spectrum produced HER=%v, expected 0", r)
	}
	if math.IsNaN(r) {
		t.Error("silent spectrum produced NaN — division-by-zero leaked")
	}
}

func TestHarmonicEnergyRatio_ClampedAtOne(t *testing.T) {
	// Construct a power spectrum where harmonic bands cover the full
	// [fMin, fMax] range; the ratio should be exactly 1.0, not >1.0
	// (which would indicate the safety clamp is missing).
	sr := 16000.0
	n := 2048
	nBins := n/2 + 1
	power := make([]float64, nBins)
	binWidth := sr / float64(2*(nBins-1))
	binMin := int(math.Ceil(50 / binWidth))
	binMax := int(math.Floor(1000 / binWidth))
	for k := binMin; k <= binMax; k++ {
		power[k] = 1.0
	}

	// fundamental small + bandwidth wide enough that harmonic bands
	// cover the entire [50, 1000] range.
	r := HarmonicEnergyRatio(power, sr, 50, 2000, 50, 1000, 20)
	if r > 1.0+1e-12 {
		t.Errorf("HER=%v exceeded 1.0 — clamp violated", r)
	}
}

// ---------------------------------------------------------------------------
// Composition: end-to-end vibration analysis
// ---------------------------------------------------------------------------

func TestVibration_EndToEnd_DetectAndScore(t *testing.T) {
	// Composition test — extract fundamental, then score harmonic ratio
	// using the same frame's power spectrum. This is the canonical
	// Dipstick-forge composition.
	sr := 16000.0
	n := 2048
	frame := makeMechFrame(180, sr, n, []float64{1.0, 0.6, 0.3}, 0.05, 11)

	// Step 1: fundamental detection (FFT + argmax).
	frameForFFT := make([]float64, n)
	copy(frameForFFT, frame)
	imag := make([]float64, n)
	hz := FundamentalHz(frameForFFT, imag, sr, 50, 1000)
	if math.Abs(hz-180) > 8.0 {
		t.Fatalf("FundamentalHz returned %v, expected ~180", hz)
	}

	// Step 2: power spectrum + HER. frameForFFT now contains FFT
	// output, so reuse it.
	power := make([]float64, n/2+1)
	audio.PowerSpectrum(frameForFFT, imag, power)
	ratio := HarmonicEnergyRatio(power, sr, hz, 5.0, 50, 1000, 5)
	if ratio < 0.3 {
		t.Errorf("end-to-end HER=%v, expected >= 0.3 for clean harmonic signal", ratio)
	}
	if ratio > 1.0+1e-12 {
		t.Errorf("end-to-end HER=%v exceeded 1.0 — clamp violated", ratio)
	}
}
