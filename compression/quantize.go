package compression

import "math"

// ScalarQuantize maps continuous float64 values to discrete integer levels
// using uniform (linear) quantization. The data range is divided into
// equal-width bins, and each value is assigned to the nearest bin index.
//
// Parameters:
//   - data:   input float64 values to quantize
//   - levels: number of quantization levels (must be >= 1)
//   - out:    pre-allocated output slice (must have len >= len(data));
//     receives the quantized bin indices in [0, levels-1]
//
// Returns:
//   - min:  the minimum value of data (reconstruction base)
//   - step: the quantization step size (reconstruction scale)
//
// Formula:
//
//	step = (max - min) / (levels - 1)              [if levels > 1]
//	out[i] = clamp(round((data[i] - min) / step), 0, levels-1)
//
// When levels == 1, all values map to bin 0 and step = 0.
// When all data values are identical, step = 0 and all map to bin 0.
//
// Valid range: data may contain any finite float64; levels >= 1
// Precision: quantization error <= step/2 per element
// Reference: Gray, Neuhoff (1998) "Quantization", IEEE Trans. Info. Theory
func ScalarQuantize(data []float64, levels int, out []int) (min, step float64) {
	if len(data) == 0 || levels < 1 {
		return 0, 0
	}

	// Find data range.
	min = data[0]
	max := data[0]
	for _, v := range data[1:] {
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}

	// Compute step size.
	if levels <= 1 || max == min {
		step = 0
		for i := range data {
			if i < len(out) {
				out[i] = 0
			}
		}
		return min, step
	}

	step = (max - min) / float64(levels-1)

	for i, v := range data {
		if i >= len(out) {
			break
		}
		q := int(math.Round((v - min) / step))
		if q < 0 {
			q = 0
		}
		if q >= levels {
			q = levels - 1
		}
		out[i] = q
	}
	return min, step
}

// ScalarDequantize reconstructs approximate float64 values from quantized
// bin indices. This is the inverse of ScalarQuantize.
//
// Parameters:
//   - quantized: bin indices produced by ScalarQuantize
//   - min:       the minimum returned by ScalarQuantize
//   - step:      the step returned by ScalarQuantize
//   - out:       pre-allocated output slice (must have len >= len(quantized));
//     receives the reconstructed float64 values
//
// Formula: out[i] = min + quantized[i] * step
//
// Valid range: quantized values >= 0; step >= 0
// Precision: reconstructed values may differ from originals by up to step/2
// Reference: inverse of uniform scalar quantization
func ScalarDequantize(quantized []int, min, step float64, out []float64) {
	for i, q := range quantized {
		if i >= len(out) {
			break
		}
		out[i] = min + float64(q)*step
	}
}
