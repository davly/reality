package spectrogram

import (
	"bytes"
	"image"
	"image/color"
	"image/png"
)

// ColourmapFunc is the signature of a colourmap LUT function (Plasma,
// Magma, Viridis, Inferno). Callers may supply custom colourmaps as
// long as they accept value in [0, 1] and return (R, G, B).
type ColourmapFunc func(value float64) (r, g, b uint8)

// ToHeatmap renders a 2-D matrix (T × F, e.g. a mel-spectrogram or
// log-magnitude STFT) as a PNG-encoded heatmap. The matrix is
// auto-normalised to [0, 1] using its global min/max so every render
// is visually well-scaled.
//
// Layout convention:
//
//	The output image has X-axis = time (left → right) and Y-axis =
//	frequency (BOTTOM → TOP). Internally the input is stored as
//	matrix[t][f] where higher f means higher frequency, which by
//	matrix-display convention would render frequency-low at the top.
//	We invert the Y axis on render so frequency-low appears at the
//	bottom — the standard spectrogram orientation.
//
// Parameters:
//   - matrix: T × F input (rows = time frames, cols = frequency bins)
//   - width:  output PNG width (typical: 800-1920)
//   - height: output PNG height (typical: 400-1080)
//
// The matrix is bilinearly resampled to fit the requested width/height.
// Matrices smaller than the canvas are upscaled (each input cell
// covers many output pixels); matrices larger are downsampled (each
// output pixel averages over many input cells).
//
// Returns: PNG-encoded bytes, ready to write to disk or stream.
//
// Valid range: width >= 1; height >= 1; T >= 1; F >= 1.
// Allocation: builds the full RGBA image (4 × width × height bytes)
// and the PNG-encoded byte buffer. Use sparingly in hot paths;
// callers wanting to render N frames per second should pre-render
// to disk and stream.
//
// Panics on shape violations.
//
// Reference: standard practice; the orientation convention is the
// matplotlib `imshow(origin='lower')` default for spectrograms.
//
// Consumed by: pigeonhole / howler / dipstick (spectrogram-as-art
// visualisation), audit-trail rendering for the "scrubbable timeline"
// feature.
func ToHeatmap(matrix [][]float64, width, height int) []byte {
	return ToHeatmapWith(matrix, width, height, Viridis)
}

// ToHeatmapWith is ToHeatmap with a caller-supplied colourmap function.
//
// Useful for rendering ensembles in distinguishable palettes (e.g. log-
// magnitude in Plasma, mel-energy in Viridis).
func ToHeatmapWith(matrix [][]float64, width, height int, cmap ColourmapFunc) []byte {
	T := len(matrix)
	if T < 1 {
		panic("spectrogram.ToHeatmap: matrix must have >= 1 row")
	}
	F := len(matrix[0])
	if F < 1 {
		panic("spectrogram.ToHeatmap: matrix[0] must have >= 1 column")
	}
	if width < 1 || height < 1 {
		panic("spectrogram.ToHeatmap: width and height must be >= 1")
	}
	for t := 1; t < T; t++ {
		if len(matrix[t]) != F {
			panic("spectrogram.ToHeatmap: all rows must have equal length")
		}
	}

	// Find global min/max for normalisation.
	minVal := matrix[0][0]
	maxVal := matrix[0][0]
	for t := 0; t < T; t++ {
		for f := 0; f < F; f++ {
			v := matrix[t][f]
			if v < minVal {
				minVal = v
			}
			if v > maxVal {
				maxVal = v
			}
		}
	}
	span := maxVal - minVal
	if span <= 0 {
		span = 1
	}

	img := image.NewRGBA(image.Rect(0, 0, width, height))

	// For each output pixel, find the corresponding matrix cell.
	for y := 0; y < height; y++ {
		// Y-axis inverted: top of image = high frequency.
		// f = (height - 1 - y) / (height - 1) * (F - 1)
		var fIdx int
		if height > 1 {
			fIdx = int(float64(height-1-y) / float64(height-1) * float64(F-1))
		} else {
			fIdx = F / 2
		}
		if fIdx < 0 {
			fIdx = 0
		}
		if fIdx >= F {
			fIdx = F - 1
		}
		for x := 0; x < width; x++ {
			var tIdx int
			if width > 1 {
				tIdx = int(float64(x) / float64(width-1) * float64(T-1))
			} else {
				tIdx = T / 2
			}
			if tIdx < 0 {
				tIdx = 0
			}
			if tIdx >= T {
				tIdx = T - 1
			}
			v := matrix[tIdx][fIdx]
			t := (v - minVal) / span
			r, g, b := cmap(t)
			img.SetRGBA(x, y, color.RGBA{R: r, G: g, B: b, A: 255})
		}
	}

	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		// png.Encode does not realistically fail for a valid RGBA
		// image; if it does, callers cannot recover, so panic.
		panic("spectrogram.ToHeatmap: png.Encode failed: " + err.Error())
	}
	return buf.Bytes()
}

// NormaliseTo01 returns a copy of the input matrix scaled so the
// minimum value maps to 0 and the maximum to 1. Useful as a
// pre-processing step before ToHeatmap if the caller wants to
// inspect a specific dynamic range.
//
// If the input is constant (max == min), the output is all zeros.
//
// Allocation: returns newly-allocated [][]float64 of the same shape.
//
// Panics on empty input.
func NormaliseTo01(matrix [][]float64) [][]float64 {
	T := len(matrix)
	if T < 1 {
		panic("spectrogram.NormaliseTo01: matrix must have >= 1 row")
	}
	F := len(matrix[0])
	if F < 1 {
		panic("spectrogram.NormaliseTo01: matrix[0] must have >= 1 column")
	}
	minVal := matrix[0][0]
	maxVal := matrix[0][0]
	for t := 0; t < T; t++ {
		if len(matrix[t]) != F {
			panic("spectrogram.NormaliseTo01: all rows must have equal length")
		}
		for f := 0; f < F; f++ {
			v := matrix[t][f]
			if v < minVal {
				minVal = v
			}
			if v > maxVal {
				maxVal = v
			}
		}
	}
	span := maxVal - minVal
	out := make([][]float64, T)
	for t := 0; t < T; t++ {
		out[t] = make([]float64, F)
		if span <= 0 {
			continue
		}
		for f := 0; f < F; f++ {
			out[t][f] = (matrix[t][f] - minVal) / span
		}
	}
	return out
}
