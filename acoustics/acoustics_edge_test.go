package acoustics

import (
	"math"
	"testing"
)

// ═══════════════════════════════════════════════════════════════════════════
// SoundSpeed — additional edge cases
// ═══════════════════════════════════════════════════════════════════════════

func TestSoundSpeed_ProportionalToSqrtT(t *testing.T) {
	// Speed should scale as sqrt(T)
	c1 := SoundSpeed(1.4, 8.314, 100, 0.029)
	c2 := SoundSpeed(1.4, 8.314, 400, 0.029)
	// c2/c1 should be sqrt(400/100) = 2
	ratio := c2 / c1
	assertClose(t, "sqrt-T proportionality", ratio, 2.0, 1e-10)
}

func TestSoundSpeed_InverselyProportionalToSqrtM(t *testing.T) {
	c1 := SoundSpeed(1.4, 8.314, 300, 0.004) // light gas
	c2 := SoundSpeed(1.4, 8.314, 300, 0.016) // heavier gas
	// c1/c2 should be sqrt(0.016/0.004) = 2
	ratio := c1 / c2
	assertClose(t, "sqrt-M inverse", ratio, 2.0, 1e-10)
}

func TestSoundSpeed_NegativeTempReturnsNaN(t *testing.T) {
	c := SoundSpeed(1.4, 8.314, -100, 0.029)
	if !math.IsNaN(c) {
		t.Errorf("expected NaN for negative T, got %v", c)
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// SoundIntensity — additional edge cases
// ═══════════════════════════════════════════════════════════════════════════

func TestSoundIntensity_LargeDistance(t *testing.T) {
	I := SoundIntensity(1.0, 1e6)
	if I <= 0 || math.IsInf(I, 0) {
		t.Errorf("expected small positive value at large distance, got %v", I)
	}
}

func TestSoundIntensity_NegativePower(t *testing.T) {
	// Mathematically valid but physically unusual
	I := SoundIntensity(-1.0, 1.0)
	if I >= 0 {
		t.Errorf("expected negative for negative power, got %v", I)
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// DecibelSPL — additional edge cases
// ═══════════════════════════════════════════════════════════════════════════

func TestDecibelSPL_ZeroPressure(t *testing.T) {
	dB := DecibelSPL(0, 20e-6)
	if !math.IsInf(dB, -1) {
		t.Errorf("expected -Inf for p=0, got %v", dB)
	}
}

func TestDecibelSPL_TenX(t *testing.T) {
	// 10x pressure → +20 dB
	dB1 := DecibelSPL(1.0, 20e-6)
	dB2 := DecibelSPL(10.0, 20e-6)
	assertClose(t, "10x pressure", dB2-dB1, 20.0, 1e-10)
}

func TestDecibelSPL_SubReference(t *testing.T) {
	// Below reference: negative dB
	dB := DecibelSPL(10e-6, 20e-6)
	if dB >= 0 {
		t.Errorf("expected negative dB below reference, got %v", dB)
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// DecibelFromIntensity — additional edge cases
// ═══════════════════════════════════════════════════════════════════════════

func TestDecibelFromIntensity_TenX(t *testing.T) {
	// 10x intensity → +10 dB
	dB1 := DecibelFromIntensity(1e-6, 1e-12)
	dB2 := DecibelFromIntensity(1e-5, 1e-12)
	assertClose(t, "10x intensity", dB2-dB1, 10.0, 1e-10)
}

func TestDecibelFromIntensity_ZeroIntensity(t *testing.T) {
	dB := DecibelFromIntensity(0, 1e-12)
	if !math.IsInf(dB, -1) {
		t.Errorf("expected -Inf for I=0, got %v", dB)
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// SabineRT60 — additional edge cases
// ═══════════════════════════════════════════════════════════════════════════

func TestSabineRT60_ProportionalToVolume(t *testing.T) {
	t1 := SabineRT60(100, 20)
	t2 := SabineRT60(200, 20)
	assertClose(t, "volume proportionality", t2, 2*t1, 1e-12)
}

func TestSabineRT60_InverseToAbsorption(t *testing.T) {
	t1 := SabineRT60(100, 20)
	t2 := SabineRT60(100, 40)
	assertClose(t, "absorption inverse", t2, t1/2, 1e-12)
}

// ═══════════════════════════════════════════════════════════════════════════
// DopplerShift — additional edge cases
// ═══════════════════════════════════════════════════════════════════════════

func TestDopplerShift_SourceAtSoundSpeed(t *testing.T) {
	// When source moves at speed of sound away (vs = c), denominator = 2c
	f := DopplerShift(1000, 343, 0, 343)
	assertClose(t, "source at c", f, 500.0, 1e-10)
}

func TestDopplerShift_ZeroFrequency(t *testing.T) {
	assertClose(t, "zero freq", DopplerShift(0, 10, 0, 343), 0.0, 1e-15)
}

func TestDopplerShift_NegativeSourceVelocity(t *testing.T) {
	// Source moving toward receiver (vs negative) → higher frequency
	f := DopplerShift(1000, -100, 0, 343)
	if f <= 1000 {
		t.Errorf("expected frequency > 1000 for approaching source, got %v", f)
	}
}

func TestDopplerShift_BothMovingSameDirection(t *testing.T) {
	// Source and receiver moving in same direction at same speed → no shift
	f := DopplerShift(1000, 50, -50, 343)
	// (343 + (-50)) / (343 + 50) = 293/393 ≈ 0.745... times f0
	expected := 1000.0 * 293.0 / 393.0
	assertClose(t, "same direction", f, expected, 1e-6)
}

// ═══════════════════════════════════════════════════════════════════════════
// ResonantFrequency — additional edge cases
// ═══════════════════════════════════════════════════════════════════════════

func TestResonantFrequency_HighHarmonic(t *testing.T) {
	f10 := ResonantFrequency(1.0, 10, 343)
	f1 := ResonantFrequency(1.0, 1, 343)
	assertClose(t, "10th harmonic ratio", f10, 10*f1, 1e-10)
}

func TestResonantFrequency_InverseToLength(t *testing.T) {
	f1 := ResonantFrequency(1.0, 1, 343)
	f2 := ResonantFrequency(2.0, 1, 343)
	assertClose(t, "half frequency for double length", f2, f1/2.0, 1e-10)
}

// ═══════════════════════════════════════════════════════════════════════════
// WaveLength — additional edge cases
// ═══════════════════════════════════════════════════════════════════════════

func TestWaveLength_InverseToFrequency(t *testing.T) {
	l1 := WaveLength(100, 343)
	l2 := WaveLength(200, 343)
	assertClose(t, "half wavelength for double freq", l2, l1/2.0, 1e-14)
}

func TestWaveLength_LargeFrequency(t *testing.T) {
	l := WaveLength(1e9, 3e8) // 1 GHz radio wave in vacuum
	assertClose(t, "1GHz wavelength", l, 0.3, 1e-14)
}

// ═══════════════════════════════════════════════════════════════════════════
// AWeighting — additional edge cases
// ═══════════════════════════════════════════════════════════════════════════

func TestAWeighting_VeryLowFreq(t *testing.T) {
	a := AWeighting(20)
	if a > -40 {
		t.Errorf("expected heavy attenuation at 20 Hz, got %v dB", a)
	}
}

func TestAWeighting_2kHz(t *testing.T) {
	// 2 kHz should be near peak (small boost)
	a := AWeighting(2000)
	if a < -1 || a > 2 {
		t.Errorf("A-weighting at 2 kHz out of expected range: got %v", a)
	}
}

func TestAWeighting_VeryHighFreq(t *testing.T) {
	a := AWeighting(20000)
	// Should be significantly attenuated
	if a > 0 {
		t.Errorf("expected attenuation at 20 kHz, got %v dB", a)
	}
}

func TestAWeighting_SymmetricallyDecreasing(t *testing.T) {
	// Below 1kHz, increases; above ~4kHz, decreases
	a1k := AWeighting(1000)
	a10k := AWeighting(10000)
	a100 := AWeighting(100)
	// Both 100 Hz and 10 kHz should be below 1 kHz
	if a100 >= a1k {
		t.Errorf("100 Hz should be below 1 kHz: %v vs %v", a100, a1k)
	}
	if a10k >= a1k {
		t.Errorf("10 kHz should be below 1 kHz: %v vs %v", a10k, a1k)
	}
}
