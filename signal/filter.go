package signal

import "sort"

// Convolve computes the linear convolution of signal and kernel, writing the
// result into out. The output length is len(signal) + len(kernel) - 1.
//
// This is the direct (naive) O(N*M) algorithm, which is optimal for short
// kernels typical in real-time filtering. For long kernels, FFT-based
// convolution is preferred (compose FFT + multiply + IFFT from this package).
//
// out must have length >= len(signal) + len(kernel) - 1.
// Panics if signal or kernel is empty, or out is too short.
// Zero heap allocations.
//
// Definition: (f * g)[n] = sum_{m} f[m] * g[n-m]
//
// Consumers: Pistachio (audio FIR), Oracle (smoothing kernels).
func Convolve(signal, kernel, out []float64) {
	sLen := len(signal)
	kLen := len(kernel)
	if sLen == 0 || kLen == 0 {
		panic("signal.Convolve: signal and kernel must be non-empty")
	}
	outLen := sLen + kLen - 1
	if len(out) < outLen {
		panic("signal.Convolve: out must have length >= len(signal)+len(kernel)-1")
	}

	// Zero out the output region.
	for i := 0; i < outLen; i++ {
		out[i] = 0
	}

	for i := 0; i < sLen; i++ {
		for j := 0; j < kLen; j++ {
			out[i+j] += signal[i] * kernel[j]
		}
	}
}

// MovingAverage computes the simple moving average of the signal with the
// given window size. The output has the same length as the signal, using
// a centered window. For positions near the edges where the full window
// is unavailable, a truncated window is used (partial averaging).
//
// out must have length >= len(signal).
// windowSize must be a positive odd number for symmetric centering.
// If windowSize is even, it is treated as-is (asymmetric left-leaning).
// Panics if signal is empty, windowSize < 1, or out is too short.
// Zero heap allocations — uses running sum.
//
// Consumers: Oracle (trend extraction), Sentinel (metric smoothing).
func MovingAverage(signal []float64, windowSize int, out []float64) {
	n := len(signal)
	if n == 0 {
		panic("signal.MovingAverage: signal must be non-empty")
	}
	if windowSize < 1 {
		panic("signal.MovingAverage: windowSize must be >= 1")
	}
	if len(out) < n {
		panic("signal.MovingAverage: out must have length >= len(signal)")
	}

	half := (windowSize - 1) / 2

	for i := 0; i < n; i++ {
		lo := i - half
		hi := lo + windowSize - 1
		if lo < 0 {
			lo = 0
		}
		if hi >= n {
			hi = n - 1
		}
		sum := 0.0
		count := hi - lo + 1
		for j := lo; j <= hi; j++ {
			sum += signal[j]
		}
		out[i] = sum / float64(count)
	}
}

// ExponentialMovingAverage computes the EMA of the signal with smoothing
// factor alpha. The first output value equals the first input value.
// Subsequent values: EMA[i] = alpha * signal[i] + (1 - alpha) * EMA[i-1].
//
// alpha must be in (0, 1]. Panics if out of range.
// out must have length >= len(signal).
// Panics if signal is empty or out is too short.
// Zero heap allocations.
//
// Consumers: Oracle (adaptive trend), Sentinel (alert smoothing),
// RubberDuck (price EMA).
func ExponentialMovingAverage(signal []float64, alpha float64, out []float64) {
	n := len(signal)
	if n == 0 {
		panic("signal.ExponentialMovingAverage: signal must be non-empty")
	}
	if alpha <= 0 || alpha > 1 {
		panic("signal.ExponentialMovingAverage: alpha must be in (0, 1]")
	}
	if len(out) < n {
		panic("signal.ExponentialMovingAverage: out must have length >= len(signal)")
	}

	out[0] = signal[0]
	oneMinusAlpha := 1.0 - alpha
	for i := 1; i < n; i++ {
		out[i] = alpha*signal[i] + oneMinusAlpha*out[i-1]
	}
}

// MedianFilter applies a running median filter with the given window size.
// Uses a centered window. Edge handling mirrors MovingAverage: truncated
// windows at boundaries.
//
// out must have length >= len(signal).
// windowSize must be >= 1.
// Panics if signal is empty, windowSize < 1, or out is too short.
//
// Note: This implementation copies window elements into a stack-allocated
// scratch buffer for sorting. For windowSize <= 63, the scratch fits in
// a stack array (zero heap allocation). For larger windows, a heap
// allocation occurs for the sort buffer.
//
// Consumers: Sentinel (spike removal), Oracle (robust trend).
func MedianFilter(signal []float64, windowSize int, out []float64) {
	n := len(signal)
	if n == 0 {
		panic("signal.MedianFilter: signal must be non-empty")
	}
	if windowSize < 1 {
		panic("signal.MedianFilter: windowSize must be >= 1")
	}
	if len(out) < n {
		panic("signal.MedianFilter: out must have length >= len(signal)")
	}

	half := (windowSize - 1) / 2

	// Use a fixed-size stack buffer for small windows to avoid allocation.
	var stackBuf [64]float64

	for i := 0; i < n; i++ {
		lo := i - half
		hi := lo + windowSize - 1
		if lo < 0 {
			lo = 0
		}
		if hi >= n {
			hi = n - 1
		}
		count := hi - lo + 1

		var buf []float64
		if count <= len(stackBuf) {
			buf = stackBuf[:count]
		} else {
			buf = make([]float64, count)
		}

		copy(buf, signal[lo:lo+count])
		sort.Float64s(buf)

		if count%2 == 1 {
			out[i] = buf[count/2]
		} else {
			out[i] = (buf[count/2-1] + buf[count/2]) / 2.0
		}
	}
}
