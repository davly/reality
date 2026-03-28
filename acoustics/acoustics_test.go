package acoustics

import (
	"math"
	"testing"

	"github.com/davly/reality/testutil"
)

// ═══════════════════════════════════════════════════════════════════════════
// Golden-file tests — shared test vectors across Go, Python, C++, C#
// ═══════════════════════════════════════════════════════════════════════════

func TestGolden_SoundSpeed(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/acoustics/sound_speed.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			gamma := testutil.InputFloat64(t, tc, "gamma")
			R := testutil.InputFloat64(t, tc, "R")
			T := testutil.InputFloat64(t, tc, "T")
			M := testutil.InputFloat64(t, tc, "M")
			got := SoundSpeed(gamma, R, T, M)
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

func TestGolden_DecibelSPL(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/acoustics/decibel_spl.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			p := testutil.InputFloat64(t, tc, "p")
			pRef := testutil.InputFloat64(t, tc, "pRef")
			got := DecibelSPL(p, pRef)
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

func TestGolden_DopplerShift(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/acoustics/doppler.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			f0 := testutil.InputFloat64(t, tc, "f0")
			vs := testutil.InputFloat64(t, tc, "vs")
			vr := testutil.InputFloat64(t, tc, "vr")
			c := testutil.InputFloat64(t, tc, "c")
			got := DopplerShift(f0, vs, vr, c)
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

func TestGolden_AWeighting(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/acoustics/a_weighting.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			f := testutil.InputFloat64(t, tc, "f")
			got := AWeighting(f)
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — Sound Speed
// ═══════════════════════════════════════════════════════════════════════════

func TestSoundSpeed_AirSTP(t *testing.T) {
	// Air at 20°C: expect ~343 m/s
	c := SoundSpeed(1.4, 8.314462618, 293.15, 0.02896)
	if c < 342 || c > 344 {
		t.Errorf("air sound speed out of range: got %v", c)
	}
}

func TestSoundSpeed_Helium(t *testing.T) {
	// Helium at 300K: expect ~1017 m/s
	c := SoundSpeed(1.66, 8.314462618, 300, 0.004003)
	if c < 1015 || c > 1020 {
		t.Errorf("helium sound speed out of range: got %v", c)
	}
}

func TestSoundSpeed_ZeroTemp(t *testing.T) {
	assertClose(t, "c-T=0", SoundSpeed(1.4, 8.314462618, 0, 0.02896), 0.0, 0)
}

func TestSoundSpeed_UnitValues(t *testing.T) {
	assertClose(t, "c-unit", SoundSpeed(1, 1, 1, 1), 1.0, 1e-15)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — Sound Intensity
// ═══════════════════════════════════════════════════════════════════════════

func TestSoundIntensity_UnitSource(t *testing.T) {
	// P=1W at r=1m: I = 1/(4*pi)
	assertClose(t, "I-unit", SoundIntensity(1, 1), 1.0/(4.0*math.Pi), 1e-15)
}

func TestSoundIntensity_InverseSquare(t *testing.T) {
	// Doubling distance quarters intensity
	I1 := SoundIntensity(100, 5)
	I2 := SoundIntensity(100, 10)
	assertClose(t, "I-inv-sq", I2, I1/4.0, 1e-12)
}

func TestSoundIntensity_ZeroDistance(t *testing.T) {
	got := SoundIntensity(1, 0)
	if !math.IsInf(got, 1) {
		t.Errorf("expected +Inf for r=0, got %v", got)
	}
}

func TestSoundIntensity_ZeroPower(t *testing.T) {
	assertClose(t, "I-zero-P", SoundIntensity(0, 10), 0.0, 0)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — Decibels
// ═══════════════════════════════════════════════════════════════════════════

func TestDecibelSPL_RefPressure(t *testing.T) {
	// p = pRef -> 0 dB
	assertClose(t, "dB-ref", DecibelSPL(20e-6, 20e-6), 0.0, 1e-12)
}

func TestDecibelSPL_1Pa(t *testing.T) {
	// 1 Pa -> ~94 dB
	dB := DecibelSPL(1.0, 20e-6)
	if dB < 93 || dB > 95 {
		t.Errorf("1 Pa SPL out of range: got %v", dB)
	}
}

func TestDecibelSPL_Doubling(t *testing.T) {
	// Doubling pressure adds ~6.02 dB
	dB1 := DecibelSPL(1.0, 20e-6)
	dB2 := DecibelSPL(2.0, 20e-6)
	assertClose(t, "dB-double", dB2-dB1, 20*math.Log10(2), 1e-10)
}

func TestDecibelFromIntensity_Reference(t *testing.T) {
	// I = IRef -> 0 dB
	assertClose(t, "dBI-ref", DecibelFromIntensity(1e-12, 1e-12), 0.0, 1e-12)
}

func TestDecibelFromIntensity_SixtyDB(t *testing.T) {
	// I = 1e-6, IRef = 1e-12 -> 60 dB
	assertClose(t, "dBI-60", DecibelFromIntensity(1e-6, 1e-12), 60.0, 1e-10)
}

func TestDecibelFromIntensity_DoublingIntensity(t *testing.T) {
	// Doubling intensity adds ~3.01 dB
	dB1 := DecibelFromIntensity(1e-6, 1e-12)
	dB2 := DecibelFromIntensity(2e-6, 1e-12)
	assertClose(t, "dBI-double", dB2-dB1, 10*math.Log10(2), 1e-10)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — Sabine RT60
// ═══════════════════════════════════════════════════════════════════════════

func TestSabineRT60_Known(t *testing.T) {
	// V=500 m^3, A=50 m^2 -> T60 = 0.161*500/50 = 1.61 s
	assertClose(t, "sabine-known", SabineRT60(500, 50), 1.61, 1e-12)
}

func TestSabineRT60_SmallRoom(t *testing.T) {
	// V=100, A=20 -> T60 = 0.805 s
	assertClose(t, "sabine-small", SabineRT60(100, 20), 0.805, 1e-12)
}

func TestSabineRT60_ZeroAbsorption(t *testing.T) {
	got := SabineRT60(100, 0)
	if !math.IsInf(got, 1) {
		t.Errorf("expected +Inf for zero absorption, got %v", got)
	}
}

func TestSabineRT60_ZeroVolume(t *testing.T) {
	assertClose(t, "sabine-V=0", SabineRT60(0, 50), 0.0, 0)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — Doppler Effect
// ═══════════════════════════════════════════════════════════════════════════

func TestDopplerShift_Approaching(t *testing.T) {
	// Source approaching at 30 m/s: frequency increases
	f := DopplerShift(700, -30, 0, 343)
	if f <= 700 {
		t.Errorf("expected higher frequency for approaching source, got %v", f)
	}
}

func TestDopplerShift_Receding(t *testing.T) {
	// Source receding at 30 m/s: frequency decreases
	f := DopplerShift(700, 30, 0, 343)
	if f >= 700 {
		t.Errorf("expected lower frequency for receding source, got %v", f)
	}
}

func TestDopplerShift_BothAtRest(t *testing.T) {
	assertClose(t, "doppler-rest", DopplerShift(1000, 0, 0, 343), 1000.0, 1e-12)
}

func TestDopplerShift_ReceiverApproaching(t *testing.T) {
	// Receiver moving toward source at 20 m/s
	f := DopplerShift(500, 0, 20, 343)
	expected := 500.0 * (343.0 + 20.0) / 343.0
	assertClose(t, "doppler-vr", f, expected, 1e-10)
}

func TestDopplerShift_SymmetryCheck(t *testing.T) {
	// Source approaching at v should give different result than receiver
	// approaching at v (asymmetry is a real feature of the Doppler effect)
	fSource := DopplerShift(1000, -30, 0, 343)
	fReceiver := DopplerShift(1000, 0, 30, 343)
	if math.Abs(fSource-fReceiver) < 0.1 {
		t.Errorf("source and receiver Doppler should differ: fSource=%v, fReceiver=%v",
			fSource, fReceiver)
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — Resonance & Waves
// ═══════════════════════════════════════════════════════════════════════════

func TestResonantFrequency_Fundamental(t *testing.T) {
	// L=1m, n=1, c=343 -> f = 343/(2*1) = 171.5 Hz
	assertClose(t, "res-fund", ResonantFrequency(1.0, 1, 343), 171.5, 1e-12)
}

func TestResonantFrequency_ThirdHarmonic(t *testing.T) {
	// L=0.5m, n=3, c=343 -> f = 3*343/(2*0.5) = 1029 Hz
	assertClose(t, "res-3rd", ResonantFrequency(0.5, 3, 343), 1029.0, 1e-12)
}

func TestResonantFrequency_Proportionality(t *testing.T) {
	// Harmonics are integer multiples of fundamental
	f1 := ResonantFrequency(2.0, 1, 343)
	f3 := ResonantFrequency(2.0, 3, 343)
	assertClose(t, "res-harmonic", f3, 3*f1, 1e-12)
}

func TestWaveLength_MiddleA(t *testing.T) {
	// A4 (440 Hz) in air: lambda = 343/440 ~ 0.78 m
	assertClose(t, "wl-440", WaveLength(440, 343), 343.0/440.0, 1e-14)
}

func TestWaveLength_1kHz(t *testing.T) {
	assertClose(t, "wl-1k", WaveLength(1000, 343), 0.343, 1e-14)
}

func TestWaveLength_ZeroFrequency(t *testing.T) {
	got := WaveLength(0, 343)
	if !math.IsInf(got, 1) {
		t.Errorf("expected +Inf for f=0, got %v", got)
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — A-Weighting
// ═══════════════════════════════════════════════════════════════════════════

func TestAWeighting_1kHz(t *testing.T) {
	// 1 kHz is the reference; should be ~0 dB
	a := AWeighting(1000)
	assertClose(t, "A-1kHz", a, 0.0, 0.01)
}

func TestAWeighting_100Hz(t *testing.T) {
	// 100 Hz should be significantly attenuated (~-19 dB)
	a := AWeighting(100)
	if a > -15 || a < -25 {
		t.Errorf("A-weighting at 100 Hz out of expected range: got %v", a)
	}
}

func TestAWeighting_4kHz(t *testing.T) {
	// 4 kHz should have slight boost (~+1 dB)
	a := AWeighting(4000)
	if a < 0 || a > 2 {
		t.Errorf("A-weighting at 4 kHz out of expected range: got %v", a)
	}
}

func TestAWeighting_MonotonicLowFreq(t *testing.T) {
	// A-weighting should increase from 50 Hz to 1000 Hz
	a50 := AWeighting(50)
	a100 := AWeighting(100)
	a500 := AWeighting(500)
	a1k := AWeighting(1000)
	if !(a50 < a100 && a100 < a500 && a500 < a1k) {
		t.Errorf("A-weighting should increase in low freq range: 50=%v 100=%v 500=%v 1k=%v",
			a50, a100, a500, a1k)
	}
}

func TestAWeighting_HighFreqRolloff(t *testing.T) {
	// A-weighting should roll off at very high frequencies
	a10k := AWeighting(10000)
	a20k := AWeighting(20000)
	if a20k >= a10k {
		t.Errorf("expected rolloff: A(20kHz)=%v should be < A(10kHz)=%v", a20k, a10k)
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

func assertClose(t *testing.T, label string, got, want, tol float64) {
	t.Helper()
	if math.Abs(got-want) > tol {
		t.Errorf("%s: got %v, want %v (tol %v)", label, got, want, tol)
	}
}
