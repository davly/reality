package signal

import (
	"math"
	"testing"

	"github.com/davly/reality/testutil"
)

// ═══════════════════════════════════════════════════════════════════════════
// Golden-file tests — shared test vectors across Go, Python, C++, C#
// ═══════════════════════════════════════════════════════════════════════════

func TestGolden_FFT(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/signal/fft.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			real := testutil.InputFloat64Slice(t, tc, "real")
			imag := testutil.InputFloat64Slice(t, tc, "imag")

			// Make copies since FFT is in-place.
			realCopy := make([]float64, len(real))
			imagCopy := make([]float64, len(imag))
			copy(realCopy, real)
			copy(imagCopy, imag)

			FFT(realCopy, imagCopy)

			// Expected is a flat array: [real0, imag0, real1, imag1, ...]
			expected := testutil.InputFloat64Slice(t, tc, "expected_interleaved")
			if len(expected) != 2*len(realCopy) {
				t.Fatalf("expected_interleaved must have length 2*N, got %d", len(expected))
			}

			tol := tc.Tolerance
			for i := 0; i < len(realCopy); i++ {
				if math.Abs(realCopy[i]-expected[2*i]) > tol {
					t.Errorf("real[%d]: got %v, want %v (tol %v)", i, realCopy[i], expected[2*i], tol)
				}
				if math.Abs(imagCopy[i]-expected[2*i+1]) > tol {
					t.Errorf("imag[%d]: got %v, want %v (tol %v)", i, imagCopy[i], expected[2*i+1], tol)
				}
			}
		})
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — FFT
// ═══════════════════════════════════════════════════════════════════════════

func TestFFT_DCSignal(t *testing.T) {
	// All-ones signal: FFT should give [N, 0, 0, ...] in real, all zeros in imag.
	n := 8
	real := make([]float64, n)
	imag := make([]float64, n)
	for i := range real {
		real[i] = 1.0
	}
	FFT(real, imag)
	assertClose(t, "DC-real[0]", real[0], float64(n), 1e-12)
	for i := 1; i < n; i++ {
		assertClose(t, "DC-real-rest", real[i], 0.0, 1e-12)
	}
	for i := 0; i < n; i++ {
		assertClose(t, "DC-imag", imag[i], 0.0, 1e-12)
	}
}

func TestFFT_Impulse(t *testing.T) {
	// Delta function: FFT should give all ones in real, all zeros in imag.
	n := 8
	real := make([]float64, n)
	imag := make([]float64, n)
	real[0] = 1.0
	FFT(real, imag)
	for i := 0; i < n; i++ {
		assertClose(t, "impulse-real", real[i], 1.0, 1e-12)
		assertClose(t, "impulse-imag", imag[i], 0.0, 1e-12)
	}
}

func TestFFT_Sinusoid_Peak(t *testing.T) {
	// Pure sinusoid at frequency k=1 in an 8-point FFT.
	// x[n] = cos(2*pi*1*n/8)
	n := 8
	real := make([]float64, n)
	imag := make([]float64, n)
	for i := 0; i < n; i++ {
		real[i] = math.Cos(2.0 * math.Pi * float64(i) / float64(n))
	}
	FFT(real, imag)
	// Peak at bin 1 and bin N-1 (conjugate symmetric).
	// Bin 1 real should be N/2 = 4, bin 7 real should be N/2 = 4.
	assertClose(t, "sin-peak-bin1-real", real[1], float64(n)/2, 1e-10)
	assertClose(t, "sin-peak-bin7-real", real[n-1], float64(n)/2, 1e-10)
	// All other bins should be ~0.
	for _, idx := range []int{0, 2, 3, 4, 5, 6} {
		assertClose(t, "sin-zero-bin", real[idx], 0.0, 1e-10)
	}
}

func TestFFT_Sinusoid_Bin3(t *testing.T) {
	// Pure sinusoid at frequency k=3 in a 16-point FFT.
	n := 16
	real := make([]float64, n)
	imag := make([]float64, n)
	for i := 0; i < n; i++ {
		real[i] = math.Cos(2.0 * math.Pi * 3.0 * float64(i) / float64(n))
	}
	FFT(real, imag)
	// Peaks at bin 3 and bin 13.
	assertClose(t, "bin3-peak-real", real[3], float64(n)/2, 1e-10)
	assertClose(t, "bin13-peak-real", real[13], float64(n)/2, 1e-10)
	// Bin 0 should be ~0.
	assertClose(t, "bin0-zero", real[0], 0.0, 1e-10)
}

func TestFFT_Length1(t *testing.T) {
	real := []float64{42.0}
	imag := []float64{0.0}
	FFT(real, imag)
	assertClose(t, "len1-real", real[0], 42.0, 0)
	assertClose(t, "len1-imag", imag[0], 0.0, 0)
}

func TestFFT_Length2(t *testing.T) {
	real := []float64{1.0, 2.0}
	imag := []float64{0.0, 0.0}
	FFT(real, imag)
	// DFT of [1, 2]: X[0] = 3, X[1] = -1
	assertClose(t, "len2-real0", real[0], 3.0, 1e-15)
	assertClose(t, "len2-real1", real[1], -1.0, 1e-15)
	assertClose(t, "len2-imag0", imag[0], 0.0, 1e-15)
	assertClose(t, "len2-imag1", imag[1], 0.0, 1e-15)
}

func TestFFT_PanicNotPow2(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for non-power-of-2 length")
		}
	}()
	FFT(make([]float64, 6), make([]float64, 6))
}

func TestFFT_PanicLengthMismatch(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for length mismatch")
		}
	}()
	FFT(make([]float64, 8), make([]float64, 4))
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — IFFT
// ═══════════════════════════════════════════════════════════════════════════

func TestIFFT_Roundtrip(t *testing.T) {
	// FFT then IFFT should recover the original signal.
	n := 16
	original := make([]float64, n)
	for i := range original {
		original[i] = float64(i*i) - float64(n)*0.5 // some non-trivial signal
	}

	real := make([]float64, n)
	imag := make([]float64, n)
	copy(real, original)

	FFT(real, imag)
	IFFT(real, imag)

	for i := 0; i < n; i++ {
		assertClose(t, "roundtrip-real", real[i], original[i], 1e-10)
		assertClose(t, "roundtrip-imag", imag[i], 0.0, 1e-10)
	}
}

func TestIFFT_RoundtripComplex(t *testing.T) {
	// Roundtrip with complex input.
	n := 8
	origReal := []float64{1, 2, 3, 4, 5, 6, 7, 8}
	origImag := []float64{8, 7, 6, 5, 4, 3, 2, 1}

	real := make([]float64, n)
	imag := make([]float64, n)
	copy(real, origReal)
	copy(imag, origImag)

	FFT(real, imag)
	IFFT(real, imag)

	for i := 0; i < n; i++ {
		assertClose(t, "complex-roundtrip-real", real[i], origReal[i], 1e-10)
		assertClose(t, "complex-roundtrip-imag", imag[i], origImag[i], 1e-10)
	}
}

func TestIFFT_DC(t *testing.T) {
	// IFFT of [N, 0, 0, ...] should be all ones.
	n := 8
	real := make([]float64, n)
	imag := make([]float64, n)
	real[0] = float64(n)
	IFFT(real, imag)
	for i := 0; i < n; i++ {
		assertClose(t, "ifft-dc", real[i], 1.0, 1e-12)
		assertClose(t, "ifft-dc-imag", imag[i], 0.0, 1e-12)
	}
}

func TestIFFT_PanicNotPow2(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for non-power-of-2 length")
		}
	}()
	IFFT(make([]float64, 5), make([]float64, 5))
}

func TestIFFT_PanicLengthMismatch(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for length mismatch")
		}
	}()
	IFFT(make([]float64, 8), make([]float64, 4))
}

func TestIFFT_Length1(t *testing.T) {
	real := []float64{42.0}
	imag := []float64{7.0}
	IFFT(real, imag)
	assertClose(t, "ifft-len1-real", real[0], 42.0, 0)
	assertClose(t, "ifft-len1-imag", imag[0], 7.0, 0)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — PowerSpectrum
// ═══════════════════════════════════════════════════════════════════════════

func TestPowerSpectrum_Impulse(t *testing.T) {
	n := 8
	real := make([]float64, n)
	imag := make([]float64, n)
	real[0] = 1.0
	out := make([]float64, n/2+1)
	PowerSpectrum(real, imag, out)
	// Impulse → flat spectrum. All bins should have power = 1.
	for i := 0; i <= n/2; i++ {
		assertClose(t, "ps-impulse", out[i], 1.0, 1e-12)
	}
}

func TestPowerSpectrum_DC(t *testing.T) {
	n := 8
	real := make([]float64, n)
	imag := make([]float64, n)
	for i := range real {
		real[i] = 1.0
	}
	out := make([]float64, n/2+1)
	PowerSpectrum(real, imag, out)
	// DC signal → all power at bin 0 (value = N^2 = 64).
	assertClose(t, "ps-dc-bin0", out[0], float64(n*n), 1e-10)
	for i := 1; i <= n/2; i++ {
		assertClose(t, "ps-dc-rest", out[i], 0.0, 1e-10)
	}
}

func TestPowerSpectrum_PanicNotPow2(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic")
		}
	}()
	PowerSpectrum(make([]float64, 6), make([]float64, 6), make([]float64, 4))
}

func TestPowerSpectrum_PanicOutTooShort(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for short out")
		}
	}()
	PowerSpectrum(make([]float64, 8), make([]float64, 8), make([]float64, 3))
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — FFTFrequencies
// ═══════════════════════════════════════════════════════════════════════════

func TestFFTFrequencies_8pt_44100(t *testing.T) {
	n := 8
	sr := 44100.0
	out := make([]float64, n/2+1)
	FFTFrequencies(n, sr, out)
	// freq[k] = k * sr / n
	for k := 0; k <= n/2; k++ {
		expected := float64(k) * sr / float64(n)
		assertClose(t, "fft-freq", out[k], expected, 1e-10)
	}
}

func TestFFTFrequencies_Nyquist(t *testing.T) {
	n := 1024
	sr := 48000.0
	out := make([]float64, n/2+1)
	FFTFrequencies(n, sr, out)
	// Last bin should be Nyquist frequency = sr/2.
	assertClose(t, "nyquist", out[n/2], sr/2, 1e-10)
}

func TestFFTFrequencies_PanicNotPow2(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic")
		}
	}()
	FFTFrequencies(10, 44100.0, make([]float64, 6))
}

func TestFFTFrequencies_PanicOutTooShort(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic")
		}
	}()
	FFTFrequencies(8, 44100.0, make([]float64, 3))
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — Convolve
// ═══════════════════════════════════════════════════════════════════════════

func TestConvolve_ImpulseResponse(t *testing.T) {
	// Convolving with a delta function should return the original signal.
	signal := []float64{1, 2, 3, 4, 5}
	kernel := []float64{1} // delta
	out := make([]float64, len(signal)+len(kernel)-1)
	Convolve(signal, kernel, out)
	assertSliceClose(t, "conv-impulse", out, signal, 1e-15)
}

func TestConvolve_BoxFilter(t *testing.T) {
	// Convolving [1,1,1,1] with [0.5, 0.5]
	signal := []float64{1, 1, 1, 1}
	kernel := []float64{0.5, 0.5}
	out := make([]float64, 5)
	Convolve(signal, kernel, out)
	assertSliceClose(t, "conv-box", out, []float64{0.5, 1.0, 1.0, 1.0, 0.5}, 1e-15)
}

func TestConvolve_KnownValues(t *testing.T) {
	// [1, 2, 3] * [4, 5] = [4, 13, 22, 15]
	signal := []float64{1, 2, 3}
	kernel := []float64{4, 5}
	out := make([]float64, 4)
	Convolve(signal, kernel, out)
	assertSliceClose(t, "conv-known", out, []float64{4, 13, 22, 15}, 1e-15)
}

func TestConvolve_Symmetric(t *testing.T) {
	// Convolution is commutative.
	a := []float64{1, 2, 3}
	b := []float64{4, 5, 6}
	out1 := make([]float64, 5)
	out2 := make([]float64, 5)
	Convolve(a, b, out1)
	Convolve(b, a, out2)
	assertSliceClose(t, "conv-commutative", out1, out2, 1e-15)
}

func TestConvolve_SingleElements(t *testing.T) {
	out := make([]float64, 1)
	Convolve([]float64{3.0}, []float64{7.0}, out)
	assertClose(t, "conv-single", out[0], 21.0, 1e-15)
}

func TestConvolve_PanicEmptySignal(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for empty signal")
		}
	}()
	Convolve([]float64{}, []float64{1}, make([]float64, 1))
}

func TestConvolve_PanicEmptyKernel(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for empty kernel")
		}
	}()
	Convolve([]float64{1}, []float64{}, make([]float64, 1))
}

func TestConvolve_PanicOutTooShort(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for short out")
		}
	}()
	Convolve([]float64{1, 2}, []float64{3, 4}, make([]float64, 2))
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — MovingAverage
// ═══════════════════════════════════════════════════════════════════════════

func TestMovingAverage_WindowSize1(t *testing.T) {
	signal := []float64{1, 2, 3, 4, 5}
	out := make([]float64, 5)
	MovingAverage(signal, 1, out)
	assertSliceClose(t, "ma-win1", out, signal, 1e-15)
}

func TestMovingAverage_WindowSize3(t *testing.T) {
	signal := []float64{1, 2, 3, 4, 5}
	out := make([]float64, 5)
	MovingAverage(signal, 3, out)
	// Centered window of 3:
	// i=0: avg(1, 2) = 1.5 (partial: left edge)
	// i=1: avg(1, 2, 3) = 2.0
	// i=2: avg(2, 3, 4) = 3.0
	// i=3: avg(3, 4, 5) = 4.0
	// i=4: avg(4, 5) = 4.5 (partial: right edge)
	assertSliceClose(t, "ma-win3", out, []float64{1.5, 2.0, 3.0, 4.0, 4.5}, 1e-15)
}

func TestMovingAverage_ConstantSignal(t *testing.T) {
	signal := []float64{5, 5, 5, 5, 5}
	out := make([]float64, 5)
	MovingAverage(signal, 3, out)
	assertSliceClose(t, "ma-const", out, []float64{5, 5, 5, 5, 5}, 1e-15)
}

func TestMovingAverage_PanicEmpty(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for empty signal")
		}
	}()
	MovingAverage([]float64{}, 3, make([]float64, 0))
}

func TestMovingAverage_PanicWindowSize0(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for windowSize 0")
		}
	}()
	MovingAverage([]float64{1}, 0, make([]float64, 1))
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — ExponentialMovingAverage
// ═══════════════════════════════════════════════════════════════════════════

func TestEMA_ConstantSignal(t *testing.T) {
	// EMA of a constant signal should always return that constant.
	signal := []float64{7, 7, 7, 7, 7}
	out := make([]float64, 5)
	ExponentialMovingAverage(signal, 0.3, out)
	assertSliceClose(t, "ema-const", out, []float64{7, 7, 7, 7, 7}, 1e-15)
}

func TestEMA_Alpha1(t *testing.T) {
	// alpha=1 means no smoothing — output equals input.
	signal := []float64{1, 5, 3, 8, 2}
	out := make([]float64, 5)
	ExponentialMovingAverage(signal, 1.0, out)
	assertSliceClose(t, "ema-alpha1", out, signal, 1e-15)
}

func TestEMA_StepResponse(t *testing.T) {
	// Step from 0 to 1.
	signal := []float64{0, 1, 1, 1, 1}
	alpha := 0.5
	out := make([]float64, 5)
	ExponentialMovingAverage(signal, alpha, out)
	// out[0] = 0
	// out[1] = 0.5*1 + 0.5*0 = 0.5
	// out[2] = 0.5*1 + 0.5*0.5 = 0.75
	// out[3] = 0.5*1 + 0.5*0.75 = 0.875
	// out[4] = 0.5*1 + 0.5*0.875 = 0.9375
	assertSliceClose(t, "ema-step", out, []float64{0, 0.5, 0.75, 0.875, 0.9375}, 1e-15)
}

func TestEMA_SmallAlpha(t *testing.T) {
	// Very small alpha means heavy smoothing.
	signal := []float64{10, 0, 0, 0, 0}
	out := make([]float64, 5)
	ExponentialMovingAverage(signal, 0.1, out)
	// out[0] = 10
	// out[1] = 0.1*0 + 0.9*10 = 9
	// out[2] = 0.1*0 + 0.9*9 = 8.1
	// out[3] = 0.1*0 + 0.9*8.1 = 7.29
	// out[4] = 0.1*0 + 0.9*7.29 = 6.561
	assertSliceClose(t, "ema-small-alpha", out, []float64{10, 9, 8.1, 7.29, 6.561}, 1e-12)
}

func TestEMA_PanicAlphaZero(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for alpha=0")
		}
	}()
	ExponentialMovingAverage([]float64{1}, 0.0, make([]float64, 1))
}

func TestEMA_PanicAlphaNegative(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for alpha<0")
		}
	}()
	ExponentialMovingAverage([]float64{1}, -0.1, make([]float64, 1))
}

func TestEMA_PanicAlphaGT1(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for alpha>1")
		}
	}()
	ExponentialMovingAverage([]float64{1}, 1.1, make([]float64, 1))
}

func TestEMA_PanicEmpty(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for empty signal")
		}
	}()
	ExponentialMovingAverage([]float64{}, 0.5, make([]float64, 0))
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — MedianFilter
// ═══════════════════════════════════════════════════════════════════════════

func TestMedianFilter_WindowSize1(t *testing.T) {
	signal := []float64{5, 3, 8, 1, 9}
	out := make([]float64, 5)
	MedianFilter(signal, 1, out)
	assertSliceClose(t, "median-win1", out, signal, 1e-15)
}

func TestMedianFilter_WindowSize3(t *testing.T) {
	signal := []float64{1, 100, 2, 3, 200}
	out := make([]float64, 5)
	MedianFilter(signal, 3, out)
	// i=0: median(1, 100) = 50.5 (even count → average of middle two)
	// i=1: median(1, 100, 2) = sorted [1,2,100] → 2
	// i=2: median(100, 2, 3) = sorted [2,3,100] → 3
	// i=3: median(2, 3, 200) = sorted [2,3,200] → 3
	// i=4: median(3, 200) = 101.5 (even count)
	assertSliceClose(t, "median-win3", out, []float64{50.5, 2, 3, 3, 101.5}, 1e-15)
}

func TestMedianFilter_SpikeRemoval(t *testing.T) {
	// Classic spike removal: one outlier in smooth data.
	signal := []float64{1, 1, 1, 100, 1, 1, 1}
	out := make([]float64, 7)
	MedianFilter(signal, 3, out)
	// The spike at index 3 should be removed.
	// i=3: median(1, 100, 1) = sorted [1,1,100] → 1
	assertClose(t, "median-spike", out[3], 1.0, 1e-15)
}

func TestMedianFilter_PanicEmpty(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for empty signal")
		}
	}()
	MedianFilter([]float64{}, 3, make([]float64, 0))
}

func TestMedianFilter_PanicWindowSize0(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for windowSize 0")
		}
	}()
	MedianFilter([]float64{1}, 0, make([]float64, 1))
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — Window functions
// ═══════════════════════════════════════════════════════════════════════════

func TestHannWindow_Endpoints(t *testing.T) {
	n := 64
	out := make([]float64, n)
	HannWindow(n, out)
	assertClose(t, "hann-start", out[0], 0.0, 1e-15)
	assertClose(t, "hann-end", out[n-1], 0.0, 1e-15)
}

func TestHannWindow_Symmetry(t *testing.T) {
	n := 64
	out := make([]float64, n)
	HannWindow(n, out)
	for i := 0; i < n/2; i++ {
		assertClose(t, "hann-sym", out[i], out[n-1-i], 1e-15)
	}
}

func TestHannWindow_Peak(t *testing.T) {
	n := 65 // odd length for exact center
	out := make([]float64, n)
	HannWindow(n, out)
	// Center should be 1.0.
	assertClose(t, "hann-peak", out[32], 1.0, 1e-15)
}

func TestHannWindow_Length1(t *testing.T) {
	out := make([]float64, 1)
	HannWindow(1, out)
	assertClose(t, "hann-len1", out[0], 1.0, 0)
}

func TestHannWindow_PanicN0(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for n=0")
		}
	}()
	HannWindow(0, make([]float64, 0))
}

func TestHammingWindow_Endpoints(t *testing.T) {
	n := 64
	out := make([]float64, n)
	HammingWindow(n, out)
	// Hamming endpoints should be 0.08.
	assertClose(t, "hamming-start", out[0], 0.08, 1e-15)
	assertClose(t, "hamming-end", out[n-1], 0.08, 1e-15)
}

func TestHammingWindow_Symmetry(t *testing.T) {
	n := 64
	out := make([]float64, n)
	HammingWindow(n, out)
	for i := 0; i < n/2; i++ {
		assertClose(t, "hamming-sym", out[i], out[n-1-i], 1e-14)
	}
}

func TestHammingWindow_Peak(t *testing.T) {
	n := 65
	out := make([]float64, n)
	HammingWindow(n, out)
	// Center should be 1.0.
	assertClose(t, "hamming-peak", out[32], 1.0, 1e-15)
}

func TestHammingWindow_Length1(t *testing.T) {
	out := make([]float64, 1)
	HammingWindow(1, out)
	assertClose(t, "hamming-len1", out[0], 1.0, 0)
}

func TestBlackmanWindow_Endpoints(t *testing.T) {
	n := 64
	out := make([]float64, n)
	BlackmanWindow(n, out)
	// Blackman endpoints should be ~0.
	assertClose(t, "blackman-start", out[0], 0.0, 1e-15)
	assertClose(t, "blackman-end", out[n-1], 0.0, 1e-15)
}

func TestBlackmanWindow_Symmetry(t *testing.T) {
	n := 64
	out := make([]float64, n)
	BlackmanWindow(n, out)
	for i := 0; i < n/2; i++ {
		assertClose(t, "blackman-sym", out[i], out[n-1-i], 1e-14)
	}
}

func TestBlackmanWindow_Peak(t *testing.T) {
	n := 65
	out := make([]float64, n)
	BlackmanWindow(n, out)
	// Center should be 1.0.
	assertClose(t, "blackman-peak", out[32], 1.0, 1e-15)
}

func TestBlackmanWindow_Length1(t *testing.T) {
	out := make([]float64, 1)
	BlackmanWindow(1, out)
	assertClose(t, "blackman-len1", out[0], 1.0, 0)
}

func TestBlackmanWindow_PanicN0(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for n=0")
		}
	}()
	BlackmanWindow(0, make([]float64, 0))
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — ApplyWindow
// ═══════════════════════════════════════════════════════════════════════════

func TestApplyWindow_Identity(t *testing.T) {
	// Window of all ones should pass signal through.
	sig := []float64{1, 2, 3, 4}
	win := []float64{1, 1, 1, 1}
	out := make([]float64, 4)
	ApplyWindow(sig, win, out)
	assertSliceClose(t, "apply-identity", out, sig, 1e-15)
}

func TestApplyWindow_Zero(t *testing.T) {
	sig := []float64{1, 2, 3, 4}
	win := []float64{0, 0, 0, 0}
	out := make([]float64, 4)
	ApplyWindow(sig, win, out)
	assertSliceClose(t, "apply-zero", out, []float64{0, 0, 0, 0}, 0)
}

func TestApplyWindow_HannIntegration(t *testing.T) {
	// Apply a Hann window to a signal and verify endpoints are zeroed.
	n := 8
	sig := make([]float64, n)
	for i := range sig {
		sig[i] = 1.0
	}
	win := make([]float64, n)
	HannWindow(n, win)
	out := make([]float64, n)
	ApplyWindow(sig, win, out)
	assertClose(t, "apply-hann-start", out[0], 0.0, 1e-15)
	assertClose(t, "apply-hann-end", out[n-1], 0.0, 1e-15)
}

func TestApplyWindow_PanicLengthMismatch(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for length mismatch")
		}
	}()
	ApplyWindow([]float64{1, 2}, []float64{1}, make([]float64, 2))
}

// ═══════════════════════════════════════════════════════════════════════════
// Integration tests — FFT + window + power spectrum pipeline
// ═══════════════════════════════════════════════════════════════════════════

func TestPipeline_WindowedFFT(t *testing.T) {
	// Generate a sinusoid, window it, FFT, verify peak location.
	n := 256
	sr := 1000.0 // 1 kHz sample rate
	freq := 100.0 // 100 Hz signal

	sig := make([]float64, n)
	for i := range sig {
		sig[i] = math.Sin(2.0 * math.Pi * freq * float64(i) / sr)
	}

	win := make([]float64, n)
	HannWindow(n, win)

	windowed := make([]float64, n)
	ApplyWindow(sig, win, windowed)

	real := make([]float64, n)
	imag := make([]float64, n)
	copy(real, windowed)

	FFT(real, imag)

	// Find the bin with maximum magnitude.
	maxMag := 0.0
	maxBin := 0
	for k := 0; k <= n/2; k++ {
		mag := real[k]*real[k] + imag[k]*imag[k]
		if mag > maxMag {
			maxMag = mag
			maxBin = k
		}
	}

	// Expected bin for 100 Hz at sr=1000, n=256: bin = 100 * 256 / 1000 = 25.6 → bin 26
	expectedBin := int(math.Round(freq * float64(n) / sr))
	if maxBin != expectedBin {
		t.Errorf("peak at bin %d, expected bin %d (freq %v Hz)", maxBin, expectedBin, freq)
	}
}

func TestPipeline_PowerSpectrumFrequencies(t *testing.T) {
	// Verify FFTFrequencies + PowerSpectrum give consistent results.
	n := 64
	sr := 8000.0
	real := make([]float64, n)
	imag := make([]float64, n)

	// Inject a pure tone at 1000 Hz.
	for i := 0; i < n; i++ {
		real[i] = math.Cos(2.0 * math.Pi * 1000.0 * float64(i) / sr)
	}

	out := make([]float64, n/2+1)
	PowerSpectrum(real, imag, out)

	freqs := make([]float64, n/2+1)
	FFTFrequencies(n, sr, freqs)

	// Find the peak frequency.
	maxPow := 0.0
	maxIdx := 0
	for k := 0; k <= n/2; k++ {
		if out[k] > maxPow {
			maxPow = out[k]
			maxIdx = k
		}
	}

	peakFreq := freqs[maxIdx]
	assertClose(t, "pipeline-peak-freq", peakFreq, 1000.0, 1e-6)
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

func assertSliceClose(t *testing.T, label string, got, want []float64, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length mismatch: got %d, want %d", label, len(got), len(want))
	}
	for i := range got {
		if math.Abs(got[i]-want[i]) > tol {
			t.Errorf("%s[%d]: got %v, want %v (tol %v)", label, i, got[i], want[i], tol)
		}
	}
}
