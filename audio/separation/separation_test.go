package separation

import (
	"math"
	"math/cmplx"
	"testing"

	"github.com/davly/reality/signal"
)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func makeSinusoid(n int, freq, sr, amp, phase float64) []float64 {
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		out[i] = amp * math.Sin(2*math.Pi*freq*float64(i)/sr+phase)
	}
	return out
}

// makeWhiteish returns a frame of deterministic broadband-ish noise via
// interleaved high-frequency sines (cheap PRNG-free pattern).
func makeWhiteish(n, seed int, amp float64) []float64 {
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		v := math.Sin(float64(i+seed*13)*1.7) +
			0.5*math.Sin(float64(i+seed*97)*0.31) +
			0.25*math.Sin(float64(i+seed*53)*5.3)
		out[i] = amp * v
	}
	return out
}

// fftReal returns the FFT of a real input as a complex slice.
func fftReal(x []float64) []complex128 {
	n := len(x)
	r := make([]float64, n)
	im := make([]float64, n)
	copy(r, x)
	signal.FFT(r, im)
	out := make([]complex128, n)
	for i := 0; i < n; i++ {
		out[i] = complex(r[i], im[i])
	}
	return out
}

// magnitudeMean returns the mean of the magnitudes across a complex slice.
func magnitudeMean(c []complex128) float64 {
	if len(c) == 0 {
		return 0
	}
	s := 0.0
	for i := 0; i < len(c); i++ {
		s += cmplx.Abs(c[i])
	}
	return s / float64(len(c))
}

// ---------------------------------------------------------------------------
// Spectral subtraction
// ---------------------------------------------------------------------------

func TestSubtractSpectrum_ReducesStationaryNoise(t *testing.T) {
	// Build a noise-only frame and a "signal+noise" frame at the same
	// sample rate. Spectral subtraction should reduce the magnitude of
	// the bins where signal is absent.
	n := 1024
	sr := 16000.0

	noiseTime := makeWhiteish(n, 42, 0.5)
	signalTime := makeSinusoid(n, 440, sr, 1.0, 0)
	mixedTime := make([]float64, n)
	for i := 0; i < n; i++ {
		mixedTime[i] = signalTime[i] + noiseTime[i]
	}

	noiseFFT := fftReal(noiseTime)
	mixedFFT := fftReal(mixedTime)

	clean := SubtractSpectrum(mixedFFT, noiseFFT)
	if len(clean) != len(mixedFFT) {
		t.Fatalf("output length mismatch: got %d want %d", len(clean), len(mixedFFT))
	}

	// The mean magnitude after subtraction should be lower than before
	// (we removed the noise floor in bins without signal).
	beforeMean := magnitudeMean(mixedFFT)
	afterMean := magnitudeMean(clean)
	if afterMean >= beforeMean {
		t.Errorf("spectral subtraction did not reduce mean magnitude: before=%.4f after=%.4f", beforeMean, afterMean)
	}

	// The signal-bin (around k = freq*N/sr = 440*1024/16000 ≈ 28) should
	// retain non-trivial magnitude.
	signalBin := int(440 * float64(n) / sr)
	cleanMag := cmplx.Abs(clean[signalBin])
	mixedMag := cmplx.Abs(mixedFFT[signalBin])
	// At least 30% of original energy should survive at the signal bin.
	if cleanMag < 0.3*mixedMag {
		t.Errorf("signal bin attenuated too aggressively: %.4f -> %.4f", mixedMag, cleanMag)
	}
}

func TestSubtractSpectrum_SpectralFloorPreserved(t *testing.T) {
	// Subtracting a spectrum from itself should leave at least β=1%
	// of the original magnitude per bin (the floor — prevents
	// musical-noise artefacts).
	in := []complex128{
		complex(1.0, 0),
		complex(2.0, 0.5),
		complex(0.5, -0.3),
	}
	out := SubtractSpectrum(in, in)
	for k := 0; k < len(in); k++ {
		want := math.Sqrt(0.01) * cmplx.Abs(in[k]) // sqrt(beta * |X|^2)
		got := cmplx.Abs(out[k])
		if math.Abs(got-want) > 1e-9 {
			t.Errorf("bin %d: floor magnitude got %v want %v", k, got, want)
		}
	}
}

func TestSubtractSpectrum_PhasePreserved(t *testing.T) {
	// The phase of the noisy observation must be preserved after
	// magnitude subtraction. Boll 1979 + Wang & Lim 1982.
	in := []complex128{complex(2.0, 0)}     // phase 0
	noise := []complex128{complex(1.0, 0)}  // half magnitude
	out := SubtractSpectrum(in, noise)
	if math.Abs(cmplx.Phase(out[0])-cmplx.Phase(in[0])) > 1e-12 {
		t.Errorf("phase not preserved: in=%v out=%v", cmplx.Phase(in[0]), cmplx.Phase(out[0]))
	}

	// Test with non-trivial phase too.
	in2 := []complex128{complex(1.0, 1.0)} // phase π/4
	noise2 := []complex128{complex(0.3, 0)}
	out2 := SubtractSpectrum(in2, noise2)
	if math.Abs(cmplx.Phase(out2[0])-cmplx.Phase(in2[0])) > 1e-9 {
		t.Errorf("phase not preserved (non-trivial): in=%v out=%v", cmplx.Phase(in2[0]), cmplx.Phase(out2[0]))
	}
}

func TestSubtractSpectrum_LengthMismatchPanics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on length mismatch")
		}
	}()
	SubtractSpectrum([]complex128{1}, []complex128{1, 2})
}

func TestEstimateNoiseSpectrum_Averages(t *testing.T) {
	// Three silent frames with magnitudes 1, 2, 3 should average to 2.
	frames := [][]complex128{
		{complex(1, 0), complex(1, 0)},
		{complex(2, 0), complex(2, 0)},
		{complex(3, 0), complex(3, 0)},
	}
	out := make([]complex128, 2)
	EstimateNoiseSpectrum(frames, 2, out)
	if math.Abs(real(out[0])-2.0) > 1e-12 || math.Abs(real(out[1])-2.0) > 1e-12 {
		t.Errorf("noise estimate wrong: %v", out)
	}
}

// ---------------------------------------------------------------------------
// Wiener filter
// ---------------------------------------------------------------------------

func TestWienerFilter_AttenuatesNoiseBins(t *testing.T) {
	// At a signal bin (|X|²=10, |N|²=1): SNR = 9 → G = 9/10 = 0.9 → |Ŝ| = 0.9·|X|.
	// At a noise-only bin (|X|² = |N|² = 1): SNR = 0 → G = 0 → |Ŝ| = 0.
	in := []complex128{complex(math.Sqrt(10), 0), complex(1, 0)}
	noise := []complex128{complex(1, 0), complex(1, 0)}
	out := WienerFilter(in, noise)

	// Signal bin: gain = 0.9; |Ŝ| = 0.9 · sqrt(10).
	want0 := 0.9 * math.Sqrt(10)
	got0 := cmplx.Abs(out[0])
	if math.Abs(got0-want0) > 1e-9 {
		t.Errorf("signal-bin attenuation wrong: got %v want %v", got0, want0)
	}
	// Noise bin: gain = 0; |Ŝ| = 0.
	if cmplx.Abs(out[1]) > 1e-12 {
		t.Errorf("noise-bin should be zeroed, got %v", out[1])
	}
}

func TestWienerFilter_ZeroNoisePassesThrough(t *testing.T) {
	// |N|²=0: gain = 1; output equals input.
	in := []complex128{complex(2, 1), complex(3, -2)}
	noise := []complex128{complex(0, 0), complex(0, 0)}
	out := WienerFilter(in, noise)
	for k := 0; k < len(in); k++ {
		if math.Abs(real(out[k])-real(in[k])) > 1e-12 || math.Abs(imag(out[k])-imag(in[k])) > 1e-12 {
			t.Errorf("bin %d: zero-noise passthrough failed: in=%v out=%v", k, in[k], out[k])
		}
	}
}

func TestWienerFilter_NoiseDominatesGivesZero(t *testing.T) {
	// |X|² < |N|²: SNR truncated to 0 → gain = 0 → output = 0.
	in := []complex128{complex(0.5, 0)}
	noise := []complex128{complex(2.0, 0)}
	out := WienerFilter(in, noise)
	if cmplx.Abs(out[0]) > 1e-12 {
		t.Errorf("noise-dominated bin should be zeroed, got %v", out[0])
	}
}

// ---------------------------------------------------------------------------
// FastICA
// ---------------------------------------------------------------------------

// pearsonCorr returns Pearson correlation in [-1, 1].
func pearsonCorr(a, b []float64) float64 {
	n := len(a)
	if n != len(b) || n < 2 {
		return 0
	}
	var ma, mb float64
	for i := 0; i < n; i++ {
		ma += a[i]
		mb += b[i]
	}
	ma /= float64(n)
	mb /= float64(n)
	var num, da, db float64
	for i := 0; i < n; i++ {
		ax := a[i] - ma
		bx := b[i] - mb
		num += ax * bx
		da += ax * ax
		db += bx * bx
	}
	denom := math.Sqrt(da * db)
	if denom == 0 {
		return 0
	}
	return num / denom
}

func TestFastICA_RecoversTwoSources(t *testing.T) {
	// Two independent (non-Gaussian) sources mixed by a 2x2 matrix —
	// FastICA should recover sources up to permutation and sign.
	T := 2000

	// Source 1: square wave (highly non-Gaussian, super-Gaussian kurt < 0).
	s1 := make([]float64, T)
	for t := 0; t < T; t++ {
		if math.Mod(float64(t)/40, 2) >= 1 {
			s1[t] = 1.0
		} else {
			s1[t] = -1.0
		}
	}
	// Source 2: sinusoid at unrelated frequency.
	s2 := makeSinusoid(T, 7, 100, 1.0, 0)

	// Mix: x = A · s where A = [[1, 0.6], [0.4, 1]]
	x1 := make([]float64, T)
	x2 := make([]float64, T)
	for t := 0; t < T; t++ {
		x1[t] = 1.0*s1[t] + 0.6*s2[t]
		x2[t] = 0.4*s1[t] + 1.0*s2[t]
	}

	obs := [][]float64{x1, x2}
	rec := FastICA(obs, 200)
	if len(rec) != 2 {
		t.Fatalf("expected 2 recovered sources, got %d", len(rec))
	}

	// Match each recovered source to its best original (allowing sign flip
	// and permutation).
	bestMatch := func(rec []float64) float64 {
		c1 := math.Abs(pearsonCorr(rec, s1))
		c2 := math.Abs(pearsonCorr(rec, s2))
		if c1 > c2 {
			return c1
		}
		return c2
	}
	c1 := bestMatch(rec[0])
	c2 := bestMatch(rec[1])
	t.Logf("recovered |corr| max: rec0=%.4f rec1=%.4f", c1, c2)
	if c1 < 0.85 {
		t.Errorf("rec[0] poor recovery: |corr| = %.4f want >= 0.85", c1)
	}
	if c2 < 0.85 {
		t.Errorf("rec[1] poor recovery: |corr| = %.4f want >= 0.85", c2)
	}
	// And the two outputs should each track DIFFERENT originals.
	c11 := math.Abs(pearsonCorr(rec[0], s1))
	c12 := math.Abs(pearsonCorr(rec[0], s2))
	c21 := math.Abs(pearsonCorr(rec[1], s1))
	c22 := math.Abs(pearsonCorr(rec[1], s2))
	// rec[0] best matches whichever has higher c1{1,2}; rec[1] should
	// best-match the OTHER source.
	if (c11 > c12) == (c21 > c22) {
		t.Errorf("rec[0] and rec[1] both matched the same source: c11=%.3f c12=%.3f c21=%.3f c22=%.3f", c11, c12, c21, c22)
	}
}

func TestFastICA_ShapeAndDeterminism(t *testing.T) {
	// Same input → same output (deterministic algorithm given LCG seed).
	T := 400
	x1 := makeSinusoid(T, 5, 100, 1.0, 0)
	x2 := makeSinusoid(T, 11, 100, 0.5, 0)
	obs := [][]float64{x1, x2}

	rec1 := FastICA(obs, 100)
	rec2 := FastICA(obs, 100)
	if len(rec1) != 2 || len(rec1[0]) != T {
		t.Fatalf("shape wrong: rows=%d cols=%d", len(rec1), len(rec1[0]))
	}
	for i := 0; i < 2; i++ {
		for tt := 0; tt < T; tt++ {
			if math.Abs(rec1[i][tt]-rec2[i][tt]) > 1e-9 {
				t.Errorf("non-deterministic at row %d, col %d: %v vs %v", i, tt, rec1[i][tt], rec2[i][tt])
				return
			}
		}
	}
}

func TestFastICA_PanicsOnEmpty(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on empty observations")
		}
	}()
	FastICA([][]float64{}, 100)
}

// ---------------------------------------------------------------------------
// NMF
// ---------------------------------------------------------------------------

func TestDecompose_RecoversBasis(t *testing.T) {
	// Construct a synthetic spectrogram where the basis is known: two
	// distinct spectral templates (rows) that activate at different times.
	F := 12
	T := 60
	rank := 2

	// Template 0: peaks at bin 2 and 5
	// Template 1: peaks at bin 8 and 10
	tmpl0 := make([]float64, F)
	tmpl1 := make([]float64, F)
	tmpl0[2] = 1.0
	tmpl0[5] = 0.5
	tmpl1[8] = 1.0
	tmpl1[10] = 0.5

	// Activations: tmpl0 active during first half, tmpl1 active during second half.
	V := make([][]float64, F)
	for f := 0; f < F; f++ {
		V[f] = make([]float64, T)
		for tt := 0; tt < T; tt++ {
			a0 := 0.0
			a1 := 0.0
			if tt < T/2 {
				a0 = 1.0
			} else {
				a1 = 1.0
			}
			V[f][tt] = a0*tmpl0[f] + a1*tmpl1[f]
		}
	}

	W, H := Decompose(V, rank, 200)
	if len(W) != F || len(W[0]) != rank {
		t.Fatalf("W shape wrong: %dx%d", len(W), len(W[0]))
	}
	if len(H) != rank || len(H[0]) != T {
		t.Fatalf("H shape wrong: %dx%d", len(H), len(H[0]))
	}

	// Frobenius reconstruction error should be small relative to ‖V‖.
	recon := Reconstruct(W, H)
	err := FrobeniusError(V, recon)

	// Compute ‖V‖_F.
	vNorm := 0.0
	for f := 0; f < F; f++ {
		for tt := 0; tt < T; tt++ {
			vNorm += V[f][tt] * V[f][tt]
		}
	}
	vNorm = math.Sqrt(vNorm)

	relErr := err / vNorm
	t.Logf("NMF relative reconstruction error: %.4f", relErr)
	if relErr > 0.20 {
		t.Errorf("reconstruction error too high: %.4f", relErr)
	}

	// Each H row should be near-zero for half the time steps and non-zero
	// for the other half (validating that the basis discovered matches
	// the synthesis structure, up to permutation).
	earlyMass := []float64{0, 0}
	lateMass := []float64{0, 0}
	for r := 0; r < rank; r++ {
		for tt := 0; tt < T; tt++ {
			if tt < T/2 {
				earlyMass[r] += H[r][tt]
			} else {
				lateMass[r] += H[r][tt]
			}
		}
	}
	// At least one row should have early-dominant activation, and the
	// other should have late-dominant activation.
	earlyDominant0 := earlyMass[0] > lateMass[0]*1.5
	earlyDominant1 := earlyMass[1] > lateMass[1]*1.5
	lateDominant0 := lateMass[0] > earlyMass[0]*1.5
	lateDominant1 := lateMass[1] > earlyMass[1]*1.5
	hasEarly := earlyDominant0 || earlyDominant1
	hasLate := lateDominant0 || lateDominant1
	if !hasEarly || !hasLate {
		t.Errorf("no temporal split discovered: earlyMass=%v lateMass=%v", earlyMass, lateMass)
	}
}

func TestDecompose_FrobeniusErrorMonotone(t *testing.T) {
	// Verify that more iterations don't INCREASE Frobenius error
	// (Lee & Seung Theorem 1: monotonic decrease).
	F := 8
	T := 12
	V := make([][]float64, F)
	for f := 0; f < F; f++ {
		V[f] = make([]float64, T)
		for tt := 0; tt < T; tt++ {
			V[f][tt] = float64((f*tt+1)%3 + 1)
		}
	}

	W1, H1 := Decompose(V, 3, 10)
	W2, H2 := Decompose(V, 3, 100)
	err1 := FrobeniusError(V, Reconstruct(W1, H1))
	err2 := FrobeniusError(V, Reconstruct(W2, H2))
	t.Logf("err@10=%v err@100=%v", err1, err2)
	if err2 > err1+1e-6 {
		t.Errorf("error increased with more iterations: %v -> %v", err1, err2)
	}
}

func TestDecompose_NonNegativeOutput(t *testing.T) {
	// Multiplicative updates preserve non-negativity. Verify directly.
	F := 6
	T := 8
	V := make([][]float64, F)
	for f := 0; f < F; f++ {
		V[f] = make([]float64, T)
		for tt := 0; tt < T; tt++ {
			V[f][tt] = float64((f+tt)%4) + 0.1
		}
	}
	W, H := Decompose(V, 2, 50)
	for f := 0; f < F; f++ {
		for r := 0; r < 2; r++ {
			if W[f][r] < 0 {
				t.Errorf("W[%d][%d] = %v < 0", f, r, W[f][r])
			}
		}
	}
	for r := 0; r < 2; r++ {
		for tt := 0; tt < T; tt++ {
			if H[r][tt] < 0 {
				t.Errorf("H[%d][%d] = %v < 0", r, tt, H[r][tt])
			}
		}
	}
}

func TestDecompose_PanicsOnNegativeInput(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on negative input")
		}
	}()
	V := [][]float64{{1, 2}, {3, -1}}
	Decompose(V, 1, 10)
}

// ---------------------------------------------------------------------------
// VAD
// ---------------------------------------------------------------------------

func TestIsVoiced_DetectsVoicedSilent(t *testing.T) {
	// Loud sinusoid: voiced.
	loud := makeSinusoid(512, 440, 16000, 1.0, 0)
	if !IsVoiced(loud, 0.01) {
		t.Errorf("loud sinusoid not detected as voiced; energy=%v", FrameEnergy(loud))
	}

	// Pure-zero frame: unvoiced.
	silent := make([]float64, 512)
	if IsVoiced(silent, 0.01) {
		t.Errorf("silent frame detected as voiced")
	}

	// Quiet sinusoid below threshold: unvoiced.
	quiet := makeSinusoid(512, 440, 16000, 0.01, 0)
	if IsVoiced(quiet, 0.01) {
		t.Errorf("quiet sinusoid (energy=%v) misclassified as voiced", FrameEnergy(quiet))
	}
}

func TestFrameEnergy_KnownValues(t *testing.T) {
	// E = (1/N) · Σ x[n]². For x = [1, 0, -1, 0]: E = 0.5.
	frame := []float64{1, 0, -1, 0}
	got := FrameEnergy(frame)
	want := 0.5
	if math.Abs(got-want) > 1e-12 {
		t.Errorf("FrameEnergy got %v want %v", got, want)
	}
}

func TestZeroCrossingRate_HighForNoise(t *testing.T) {
	// Constant non-zero frame: ZCR = 0 (no crossings).
	steady := makeSinusoid(512, 100, 16000, 1.0, 0)
	zcrSteady := ZeroCrossingRate(steady)

	// High-frequency frame: ZCR should be much higher than steady.
	highFreq := makeSinusoid(512, 5000, 16000, 1.0, 0)
	zcrHigh := ZeroCrossingRate(highFreq)
	t.Logf("ZCR steady (100Hz)=%.3f highFreq (5kHz)=%.3f", zcrSteady, zcrHigh)
	if zcrHigh <= zcrSteady*5 {
		t.Errorf("ZCR not significantly higher for high-freq: steady=%v high=%v", zcrSteady, zcrHigh)
	}
}

func TestIsVoicedAdaptive_TracksNoiseFloor(t *testing.T) {
	// Frame energy 1.0; noise energy 0.01; margin 6 dB → threshold = 0.04.
	// Frame at 1.0 should pass; frame at 0.01 should not.
	loud := []float64{1, 1, 1, 1}
	quiet := []float64{0.05, -0.05, 0.05, -0.05}
	if !IsVoicedAdaptive(loud, 0.01, 6) {
		t.Errorf("loud frame missed by adaptive VAD")
	}
	if IsVoicedAdaptive(quiet, 0.01, 6) {
		t.Errorf("quiet frame falsely detected by adaptive VAD; energy=%v", FrameEnergy(quiet))
	}
}

func TestIsVoiced_PanicsOnEmpty(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on empty frame")
		}
	}()
	IsVoiced(nil, 0.01)
}
