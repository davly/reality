package pitch

import (
	"math"
	"testing"

	"github.com/davly/reality/audio"
	"github.com/davly/reality/signal"
)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func makeSinusoid(n int, freq, sr float64, phase float64) []float64 {
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		out[i] = math.Sin(2*math.Pi*freq*float64(i)/sr + phase)
	}
	return out
}

func makeHarmonicMix(n int, freqs []float64, sr float64) []float64 {
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		v := 0.0
		for _, f := range freqs {
			v += math.Sin(2 * math.Pi * f * float64(i) / sr)
		}
		out[i] = v / float64(len(freqs))
	}
	return out
}

// makeNoisyTone returns a `freq` Hz signal with substantial broadband
// noise. Deterministic. Used to verify that pitch detectors flag
// reduced confidence (YIN aperiodicity, MPM clarity drop) on noisy
// inputs. The noise amplitude is set to dominate the spectrum
// (signal:noise ratio in the time-domain max-amplitude sense ≈ 1:3 —
// well below 0 dB SNR).
func makeNoisyTone(n int, freq, sr float64, seed int) []float64 {
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		s := math.Sin(2 * math.Pi * freq * float64(i) / sr)
		// Deterministic broadband-ish noise at high amplitude — 3x the
		// signal — so YIN's d'(τ) at the true period is dominated by
		// the noise rather than the signal periodicity.
		nz := 3.0 * (math.Sin(float64(i+seed*13)*1.7) +
			0.7*math.Sin(float64(i+seed*97)*0.31) +
			0.5*math.Sin(float64(i+seed*53)*5.3) +
			0.4*math.Sin(float64(i+seed*71)*2.9))
		out[i] = s + nz
	}
	return out
}

// computePowerSpectrumPow2 computes the power spectrum of a real-valued
// frame at a power-of-2 length (zero-pads if necessary).
func computePowerSpectrumPow2(samples []float64, n int) []float64 {
	r := make([]float64, n)
	im := make([]float64, n)
	copy(r, samples)
	signal.FFT(r, im)
	power := make([]float64, n/2+1)
	audio.PowerSpectrum(r, im, power)
	return power
}

// nextPow2 returns the smallest power of 2 >= x.
func nextPow2(x int) int {
	n := 1
	for n < x {
		n *= 2
	}
	return n
}

// ---------------------------------------------------------------------------
// AutocorrelationPitch
// ---------------------------------------------------------------------------

func TestAutocorrelationPitch_Detects440Hz(t *testing.T) {
	// ACF returns sampleRate / integer_lag, so the precision is bounded
	// by the integer-period quantisation. At sr=16000 and f=440 Hz, the
	// nearest two integer lags are 36 and 37 → 444.4 Hz and 432.4 Hz.
	// We assert ±5 Hz, which is the achievable bound without parabolic
	// interpolation.
	sr := 16000.0
	frame := makeSinusoid(2048, 440, sr, 0)
	got := AutocorrelationPitch(frame, sr, 80, 1000)
	if math.Abs(got-440) > 5.0 {
		t.Errorf("AutocorrelationPitch on 440 Hz returned %v, want 440 ± 5", got)
	}
}

func TestAutocorrelationPitch_DetectsFundamentalNotHarmonic(t *testing.T) {
	// 440 Hz fundamental + 880 Hz harmonic. ACF must return 440, not 880.
	sr := 16000.0
	frame := makeHarmonicMix(2048, []float64{440, 880}, sr)
	got := AutocorrelationPitch(frame, sr, 80, 2000)
	if math.Abs(got-440) > 5.0 {
		t.Errorf("AutocorrelationPitch returned %v, want 440 (the fundamental, not 880)", got)
	}
}

func TestAutocorrelationPitch_SilentReturnsZero(t *testing.T) {
	sr := 16000.0
	got := AutocorrelationPitch(make([]float64, 2048), sr, 80, 1000)
	if got != 0 {
		t.Errorf("silent frame returned %v, want 0", got)
	}
}

func TestAutocorrelationPitch_PanicsOnInvalidRange(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on fMin >= fMax")
		}
	}()
	AutocorrelationPitch([]float64{1, 2, 3}, 16000, 1000, 100)
}

func TestAutocorrelationPitch_PanicsOnZeroSampleRate(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on sampleRate <= 0")
		}
	}()
	AutocorrelationPitch([]float64{1, 2, 3}, 0, 100, 1000)
}

// ---------------------------------------------------------------------------
// YIN
// ---------------------------------------------------------------------------

func TestYin_Detects440Hz(t *testing.T) {
	sr := 16000.0
	frame := makeSinusoid(2048, 440, sr, 0)
	got, ap := Yin(frame, sr, 0.10, 80, 1000)
	if math.Abs(got-440) > 1.0 {
		t.Errorf("YIN on 440 Hz returned %v, want 440 ± 1", got)
	}
	if ap > 0.1 {
		t.Errorf("YIN aperiodicity %v on clean tone, want < 0.1", ap)
	}
}

func TestYin_DetectsFundamentalNotHarmonic(t *testing.T) {
	sr := 16000.0
	frame := makeHarmonicMix(2048, []float64{440, 880}, sr)
	got, _ := Yin(frame, sr, 0.10, 80, 2000)
	if math.Abs(got-440) > 5.0 {
		t.Errorf("YIN returned %v, want 440 (the fundamental, not 880)", got)
	}
}

func TestYin_HighAperiodicityForNoisyTone(t *testing.T) {
	// Heavy broadband noise drowning a 440 Hz tone — YIN should flag
	// elevated aperiodicity (d'(τ*) >> clean-tone level). For our
	// clean-tone test above we observed aperiodicity < 0.1; a noisy
	// tone should push that to >= 0.15. Calibrating the absolute
	// threshold against the chosen noise generator is a pragmatic
	// trade-off — the YIN spec does not pin an absolute aperiodicity
	// scale.
	sr := 16000.0
	frame := makeNoisyTone(2048, 440, sr, 7)
	_, ap := Yin(frame, sr, 0.10, 80, 1000)
	if ap < 0.15 {
		t.Errorf("YIN aperiodicity for noisy tone is %v, want >= 0.15 (clean-tone level was < 0.1)", ap)
	}
}

func TestYin_SilentReturnsZeroAperiodicityOne(t *testing.T) {
	sr := 16000.0
	got, ap := Yin(make([]float64, 2048), sr, 0.10, 80, 1000)
	if got != 0 {
		t.Errorf("silent frame YIN returned pitch %v, want 0", got)
	}
	if ap != 1.0 {
		t.Errorf("silent frame YIN returned aperiodicity %v, want 1.0", ap)
	}
}

func TestYin_PanicsOnInvalidThreshold(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on threshold > 1")
		}
	}()
	Yin(make([]float64, 1024), 16000, 1.5, 80, 1000)
}

// ---------------------------------------------------------------------------
// McLeodPitchMethod
// ---------------------------------------------------------------------------

func TestMcLeod_Detects440Hz(t *testing.T) {
	sr := 16000.0
	frame := makeSinusoid(2048, 440, sr, 0)
	got, clarity := McLeodPitchMethod(frame, sr, 80, 1000)
	if math.Abs(got-440) > 1.0 {
		t.Errorf("McLeod on 440 Hz returned %v, want 440 ± 1", got)
	}
	if clarity < 0.85 {
		t.Errorf("McLeod clarity %v on clean tone, want >= 0.85", clarity)
	}
}

func TestMcLeod_DetectsFundamentalNotHarmonic(t *testing.T) {
	sr := 16000.0
	frame := makeHarmonicMix(2048, []float64{440, 880}, sr)
	got, _ := McLeodPitchMethod(frame, sr, 80, 2000)
	if math.Abs(got-440) > 5.0 {
		t.Errorf("McLeod returned %v, want 440 (the fundamental, not 880)", got)
	}
}

func TestMcLeod_SilentReturnsZero(t *testing.T) {
	sr := 16000.0
	got, _ := McLeodPitchMethod(make([]float64, 2048), sr, 80, 1000)
	if got != 0 {
		t.Errorf("silent frame returned %v, want 0", got)
	}
}

func TestMcLeod_PanicsOnInvalidRange(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on fMin >= fMax")
		}
	}()
	McLeodPitchMethod([]float64{1, 2, 3}, 16000, 1000, 100)
}

// ---------------------------------------------------------------------------
// SubharmonicSummation
// ---------------------------------------------------------------------------

func TestSubharmonicSummation_Detects440Hz(t *testing.T) {
	sr := 16000.0
	n := 2048
	frame := makeSinusoid(n, 440, sr, 0)
	power := computePowerSpectrumPow2(frame, n)
	got := SubharmonicSummation(power, sr, 100, 800, 5)
	if math.Abs(got-440) > 10.0 {
		t.Errorf("SHS on 440 Hz returned %v, want 440 ± 10 (1 Hz grid)", got)
	}
}

func TestSubharmonicSummation_DetectsFundamentalFromHarmonics(t *testing.T) {
	// Mix of 440, 880, 1320 Hz. SHS should return 440, even though
	// the strongest individual bin might be at any of the harmonics.
	sr := 16000.0
	n := nextPow2(4096)
	frame := makeHarmonicMix(n, []float64{440, 880, 1320}, sr)
	power := computePowerSpectrumPow2(frame, n)
	got := SubharmonicSummation(power, sr, 100, 2000, 5)
	if math.Abs(got-440) > 10.0 {
		t.Errorf("SHS on 440+880+1320 returned %v, want 440 (the fundamental)", got)
	}
}

func TestSubharmonicSummation_SilentReturnsZero(t *testing.T) {
	sr := 16000.0
	power := make([]float64, 1025)
	got := SubharmonicSummation(power, sr, 100, 800, 5)
	if got != 0 {
		t.Errorf("silent spectrum returned %v, want 0", got)
	}
}

func TestSubharmonicSummation_PanicsOnInvalidHarmonics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on harmonics < 1")
		}
	}()
	power := make([]float64, 1025)
	SubharmonicSummation(power, 16000, 100, 800, 0)
}

func TestSubharmonicSummation_PanicsOnShortSpectrum(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on spectrum length < 2")
		}
	}()
	SubharmonicSummation([]float64{1}, 16000, 100, 800, 5)
}

// ---------------------------------------------------------------------------
// Cross-algorithm consistency
// ---------------------------------------------------------------------------

func TestPitchTrio_AllAgreeOnCleanTone(t *testing.T) {
	// All four algorithms should agree on a clean 440 Hz tone within
	// reasonable precision tolerances.
	sr := 16000.0
	n := 2048
	frame := makeSinusoid(n, 440, sr, 0)

	acf := AutocorrelationPitch(frame, sr, 80, 1000)
	yin, _ := Yin(frame, sr, 0.10, 80, 1000)
	mpm, _ := McLeodPitchMethod(frame, sr, 80, 1000)
	power := computePowerSpectrumPow2(frame, n)
	shs := SubharmonicSummation(power, sr, 100, 800, 5)

	t.Logf("ACF=%.2f  YIN=%.2f  MPM=%.2f  SHS=%.2f", acf, yin, mpm, shs)

	for _, val := range []float64{acf, yin, mpm, shs} {
		if math.Abs(val-440) > 10.0 {
			t.Errorf("pitch estimate %v deviates by > 10 Hz from 440", val)
		}
	}
}
