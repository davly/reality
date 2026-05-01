package spectrogram

import (
	"bytes"
	"image/png"
	"math"
	"math/cmplx"
	"testing"

	"github.com/davly/reality/signal"
)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func makeSinusoid(n int, freq, sr, amp float64) []float64 {
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		out[i] = amp * math.Sin(2*math.Pi*freq*float64(i)/sr)
	}
	return out
}

// ---------------------------------------------------------------------------
// STFT
// ---------------------------------------------------------------------------

func TestCompute_ShapeAndSpectralPeak(t *testing.T) {
	// 1 second of 1 kHz sine at 16 kHz, frameSize 512, hop 256.
	sr := 16000.0
	totalSamples := 16000
	samples := makeSinusoid(totalSamples, 1000, sr, 1.0)
	frameSize := 512
	hopSize := 256
	window := make([]float64, frameSize)
	signal.HannWindow(frameSize, window)

	stft := Compute(samples, frameSize, hopSize, window)

	// Expected number of frames.
	expectedFrames := (totalSamples + hopSize - 1) / hopSize
	if len(stft) != expectedFrames {
		t.Errorf("frame count: got %d want %d", len(stft), expectedFrames)
	}
	if len(stft[0]) != frameSize {
		t.Errorf("frame length: got %d want %d", len(stft[0]), frameSize)
	}

	// Peak bin in a representative middle frame should be near
	// freq*N/sr = 1000 * 512 / 16000 = 32.
	mid := len(stft) / 2
	peakBin := 0
	peakMag := 0.0
	for k := 1; k < frameSize/2; k++ {
		m := cmplx.Abs(stft[mid][k])
		if m > peakMag {
			peakMag = m
			peakBin = k
		}
	}
	expectedBin := int(1000 * float64(frameSize) / sr)
	if absInt(peakBin-expectedBin) > 1 {
		t.Errorf("peak bin: got %d want ~%d", peakBin, expectedBin)
	}
}

func TestCompute_RoundTrip(t *testing.T) {
	// Forward STFT + inverse OLA should reconstruct the input
	// (modulo edge effects / small windowing residual).
	sr := 16000.0
	frameSize := 256
	hopSize := 64 // 75% overlap — Hann is COLA at hop=frameSize/4.
	N := 4096
	samples := makeSinusoid(N, 800, sr, 1.0)
	window := make([]float64, frameSize)
	signal.HannWindow(frameSize, window)

	stft := Compute(samples, frameSize, hopSize, window)
	recon := Inverse(stft, frameSize, hopSize, window)

	// Check the central region (skip first / last frameSize samples
	// to avoid OLA edge attenuation).
	maxErr := 0.0
	for n := frameSize; n < N-frameSize; n++ {
		d := math.Abs(recon[n] - samples[n])
		if d > maxErr {
			maxErr = d
		}
	}
	t.Logf("STFT round-trip max error: %.3e", maxErr)
	if maxErr > 1e-6 {
		t.Errorf("STFT round-trip max error too large: %.3e", maxErr)
	}
}

func TestCompute_PanicsOnInvalidWindow(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on window length mismatch")
		}
	}()
	samples := []float64{1, 2, 3, 4}
	window := make([]float64, 8) // wrong size for frameSize=4
	Compute(samples, 4, 4, window)
}

func TestCompute_PanicsOnZeroFrameSize(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on frameSize < 2")
		}
	}()
	Compute([]float64{1}, 1, 1, []float64{1})
}

// ---------------------------------------------------------------------------
// Magnitude / LogMagnitude / PowerSpectrum
// ---------------------------------------------------------------------------

func TestMagnitude_KnownValues(t *testing.T) {
	stft := [][]complex128{
		{complex(3, 4), complex(5, 12), complex(0, 0)},
		{complex(8, 6), complex(0, 1), complex(2, 0)},
	}
	mag := Magnitude(stft)
	wants := [][]float64{
		{5, 13, 0},
		{10, 1, 2},
	}
	for tt := 0; tt < 2; tt++ {
		for f := 0; f < 3; f++ {
			if math.Abs(mag[tt][f]-wants[tt][f]) > 1e-12 {
				t.Errorf("Magnitude[%d][%d] = %v want %v", tt, f, mag[tt][f], wants[tt][f])
			}
		}
	}
}

func TestPowerSpectrum_EqualsMagSquared(t *testing.T) {
	stft := [][]complex128{
		{complex(3, 4), complex(0, 1)},
	}
	mag := Magnitude(stft)
	pwr := PowerSpectrum(stft)
	for f := 0; f < 2; f++ {
		want := mag[0][f] * mag[0][f]
		if math.Abs(pwr[0][f]-want) > 1e-12 {
			t.Errorf("PowerSpectrum[0][%d] = %v want %v", f, pwr[0][f], want)
		}
	}
}

func TestLogMagnitude_DbScale(t *testing.T) {
	// |X| = 10 → 20*log10(10) = 20 dB.
	stft := [][]complex128{{complex(10, 0)}}
	got := LogMagnitude(stft)[0][0]
	want := 20.0
	if math.Abs(got-want) > 1e-9 {
		t.Errorf("LogMagnitude got %v want %v", got, want)
	}
}

func TestHalfSpectrum_HalfLength(t *testing.T) {
	stft := [][]complex128{make([]complex128, 8)}
	half := HalfSpectrum(stft)
	if len(half[0]) != 5 {
		t.Errorf("HalfSpectrum length: got %d want 5", len(half[0]))
	}
}

// ---------------------------------------------------------------------------
// MelSpectrogram
// ---------------------------------------------------------------------------

func TestMelSpectrogram_NonNegative(t *testing.T) {
	// Mel-band energies are sum of (non-negative weights) × (non-negative
	// power) — must be non-negative.
	sr := 16000.0
	frameSize := 512
	hopSize := 256
	N := 8000
	samples := makeSinusoid(N, 1000, sr, 1.0)
	window := make([]float64, frameSize)
	signal.HannWindow(frameSize, window)

	stft := Compute(samples, frameSize, hopSize, window)
	mel := MelSpectrogram(stft, sr, frameSize, 26, 0, sr/2)

	if len(mel) != len(stft) {
		t.Fatalf("frame count: got %d want %d", len(mel), len(stft))
	}
	if len(mel[0]) != 26 {
		t.Fatalf("filter count: got %d want 26", len(mel[0]))
	}
	for tt := 0; tt < len(mel); tt++ {
		for b := 0; b < 26; b++ {
			if mel[tt][b] < 0 {
				t.Errorf("mel[%d][%d] = %v < 0", tt, b, mel[tt][b])
				return
			}
		}
	}
}

func TestMelSpectrogram_PeakBandFollowsSignal(t *testing.T) {
	// 4 kHz sine should produce its mel-energy peak near the
	// corresponding mel band.
	sr := 16000.0
	frameSize := 512
	hopSize := 256
	N := 8000
	samples := makeSinusoid(N, 4000, sr, 1.0)
	window := make([]float64, frameSize)
	signal.HannWindow(frameSize, window)

	stft := Compute(samples, frameSize, hopSize, window)
	mel := MelSpectrogram(stft, sr, frameSize, 40, 0, sr/2)

	// In a representative middle frame, find the peak band.
	mid := len(mel) / 2
	peakBand := 0
	peakEnergy := 0.0
	for b := 0; b < 40; b++ {
		if mel[mid][b] > peakEnergy {
			peakEnergy = mel[mid][b]
			peakBand = b
		}
	}
	// The peak should be in the upper half of the bands (4 kHz is
	// near the upper end of the mel range from 0 to 8 kHz).
	if peakBand < 20 {
		t.Errorf("peak band %d too low for 4 kHz sine; expected >= 20", peakBand)
	}
}

func TestLogMelSpectrogram_FloorAtNegInf(t *testing.T) {
	sr := 16000.0
	frameSize := 256
	hopSize := 128
	samples := make([]float64, 2048) // silence
	window := make([]float64, frameSize)
	signal.HannWindow(frameSize, window)

	stft := Compute(samples, frameSize, hopSize, window)
	logMel := LogMelSpectrogram(stft, sr, frameSize, 26, 0, sr/2)

	// All values should be log(floor=1e-10) = -23.026 (natural log).
	want := math.Log(1e-10)
	for tt := 0; tt < len(logMel); tt++ {
		for b := 0; b < 26; b++ {
			if math.Abs(logMel[tt][b]-want) > 1e-9 {
				t.Errorf("logMel[%d][%d] = %v, want floor %v", tt, b, logMel[tt][b], want)
				return
			}
		}
	}
}

// ---------------------------------------------------------------------------
// Colourmap
// ---------------------------------------------------------------------------

func TestColourmap_EndpointsAndMonotonicLuminance(t *testing.T) {
	// Each colourmap should map t=0 to its LUT[0] and t=1 to LUT[15].
	for name, cmap := range map[string]ColourmapFunc{
		"Plasma":  Plasma,
		"Magma":   Magma,
		"Viridis": Viridis,
		"Inferno": Inferno,
	} {
		r0, g0, b0 := cmap(0)
		r1, g1, b1 := cmap(1)
		l0 := 0.2126*float64(r0) + 0.7152*float64(g0) + 0.0722*float64(b0)
		l1 := 0.2126*float64(r1) + 0.7152*float64(g1) + 0.0722*float64(b1)
		// Sequential perceptual maps: t=1 should be brighter than t=0.
		if l1 <= l0 {
			t.Errorf("%s: brightness(1)=%v <= brightness(0)=%v", name, l1, l0)
		}
	}
}

func TestColourmap_ClampsOutOfRange(t *testing.T) {
	r1, g1, b1 := Viridis(-1.0)
	r2, g2, b2 := Viridis(0.0)
	if r1 != r2 || g1 != g2 || b1 != b2 {
		t.Errorf("negative input not clamped: %v vs %v", []uint8{r1, g1, b1}, []uint8{r2, g2, b2})
	}
	r3, g3, b3 := Viridis(2.0)
	r4, g4, b4 := Viridis(1.0)
	if r3 != r4 || g3 != g4 || b3 != b4 {
		t.Errorf("over-range input not clamped: %v vs %v", []uint8{r3, g3, b3}, []uint8{r4, g4, b4})
	}
}

// ---------------------------------------------------------------------------
// ToHeatmap (PNG output)
// ---------------------------------------------------------------------------

func TestToHeatmap_ProducesValidPNG(t *testing.T) {
	// Build a small mel-spectrogram and render it.
	sr := 16000.0
	frameSize := 256
	hopSize := 128
	N := 4096
	samples := makeSinusoid(N, 1500, sr, 1.0)
	window := make([]float64, frameSize)
	signal.HannWindow(frameSize, window)

	stft := Compute(samples, frameSize, hopSize, window)
	mel := MelSpectrogram(stft, sr, frameSize, 40, 0, sr/2)

	pngBytes := ToHeatmap(mel, 320, 240)
	if len(pngBytes) < 100 {
		t.Fatalf("PNG too small: %d bytes", len(pngBytes))
	}

	// Decode it back to confirm validity.
	img, err := png.Decode(bytes.NewReader(pngBytes))
	if err != nil {
		t.Fatalf("png.Decode failed: %v", err)
	}
	bounds := img.Bounds()
	if bounds.Dx() != 320 || bounds.Dy() != 240 {
		t.Errorf("image dimensions: got %dx%d want 320x240", bounds.Dx(), bounds.Dy())
	}
}

func TestToHeatmap_DifferentColourmaps(t *testing.T) {
	matrix := [][]float64{
		{0, 0.25, 0.5},
		{0.5, 0.75, 1.0},
	}
	for name, cmap := range map[string]ColourmapFunc{
		"Plasma":  Plasma,
		"Magma":   Magma,
		"Viridis": Viridis,
		"Inferno": Inferno,
	} {
		pngBytes := ToHeatmapWith(matrix, 64, 48, cmap)
		_, err := png.Decode(bytes.NewReader(pngBytes))
		if err != nil {
			t.Errorf("%s: png.Decode failed: %v", name, err)
		}
	}
}

func TestToHeatmap_ConstantInputDoesNotPanic(t *testing.T) {
	// All-equal matrix: span = 0; should produce a valid (uniform) PNG.
	matrix := [][]float64{
		{5, 5, 5},
		{5, 5, 5},
	}
	pngBytes := ToHeatmap(matrix, 32, 24)
	_, err := png.Decode(bytes.NewReader(pngBytes))
	if err != nil {
		t.Errorf("constant input PNG decode failed: %v", err)
	}
}

func TestNormaliseTo01_KnownValues(t *testing.T) {
	matrix := [][]float64{
		{1, 2, 3},
		{4, 5, 6},
	}
	norm := NormaliseTo01(matrix)
	wants := [][]float64{
		{0.0, 0.2, 0.4},
		{0.6, 0.8, 1.0},
	}
	for tt := 0; tt < 2; tt++ {
		for f := 0; f < 3; f++ {
			if math.Abs(norm[tt][f]-wants[tt][f]) > 1e-12 {
				t.Errorf("Normalise[%d][%d] = %v want %v", tt, f, norm[tt][f], wants[tt][f])
			}
		}
	}
}

func TestNormaliseTo01_ConstantInput(t *testing.T) {
	matrix := [][]float64{{7, 7}, {7, 7}}
	norm := NormaliseTo01(matrix)
	for tt := 0; tt < 2; tt++ {
		for f := 0; f < 2; f++ {
			if norm[tt][f] != 0 {
				t.Errorf("constant input should produce zeros, got %v at [%d][%d]", norm[tt][f], tt, f)
			}
		}
	}
}

func TestToHeatmap_PanicsOnZeroDim(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on zero width")
		}
	}()
	matrix := [][]float64{{1, 2}}
	ToHeatmap(matrix, 0, 100)
}

// absInt returns |a|.
func absInt(a int) int {
	if a < 0 {
		return -a
	}
	return a
}
