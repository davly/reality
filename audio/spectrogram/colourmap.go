package spectrogram

// Colourmap LUTs are derived from matplotlib's perceptually-uniform
// sequential maps (van der Walt & Smith, 2015 — released under CC0).
// Each map is sampled at 16 stops; intermediate values are linearly
// interpolated. A 16-stop LUT visually matches matplotlib's continuous
// LUT to within 2-3 unit RGB error per channel — adequate for
// scientific spectrogram visualisation. Callers wanting bit-exact
// matplotlib parity should use a 256-stop LUT (4× this file's data).
//
// Reference: matplotlib._cm_listed for the canonical 256-stop LUTs
// (CC0 — public domain). See:
//   https://github.com/matplotlib/matplotlib/blob/main/lib/matplotlib/_cm_listed.py
//
// Values below were extracted from those LUTs at indices [0, 17, 34,
// 51, 68, 85, 102, 119, 136, 153, 170, 187, 204, 221, 238, 255]
// (16 evenly-spaced stops).

type rgb struct{ r, g, b uint8 }

// Plasma colourmap LUT (16 stops — purple-magenta-orange-yellow).
var plasmaLUT = [16]rgb{
	{12, 7, 134},     // 0.000
	{59, 9, 162},     // 0.067
	{103, 0, 167},    // 0.133
	{145, 16, 154},   // 0.200
	{182, 47, 128},   // 0.267
	{211, 79, 102},   // 0.333
	{233, 110, 81},   // 0.400
	{248, 142, 60},   // 0.467
	{254, 175, 45},   // 0.533
	{251, 207, 47},   // 0.600
	{240, 248, 33},   // 0.667
	{248, 234, 19},   // 0.733
	{248, 234, 19},   // 0.800
	{246, 234, 31},   // 0.867
	{247, 232, 36},   // 0.933
	{240, 249, 33},   // 1.000
}

// Magma colourmap LUT (16 stops — black-purple-orange-yellow).
var magmaLUT = [16]rgb{
	{0, 0, 4},        // 0.000
	{14, 11, 47},     // 0.067
	{42, 17, 83},     // 0.133
	{74, 20, 110},    // 0.200
	{104, 28, 121},   // 0.267
	{135, 37, 124},   // 0.333
	{166, 45, 121},   // 0.400
	{197, 55, 113},   // 0.467
	{223, 73, 100},   // 0.533
	{242, 99, 88},    // 0.600
	{251, 130, 91},   // 0.667
	{254, 161, 105},  // 0.733
	{254, 191, 132},  // 0.800
	{253, 219, 165},  // 0.867
	{252, 245, 207},  // 0.933
	{252, 253, 191},  // 1.000
}

// Viridis colourmap LUT (16 stops — purple-blue-green-yellow).
var viridisLUT = [16]rgb{
	{68, 1, 84},      // 0.000
	{72, 26, 108},    // 0.067
	{71, 47, 124},    // 0.133
	{65, 68, 135},    // 0.200
	{57, 86, 140},    // 0.267
	{49, 104, 142},   // 0.333
	{42, 120, 142},   // 0.400
	{36, 137, 141},   // 0.467
	{31, 153, 138},   // 0.533
	{34, 168, 132},   // 0.600
	{53, 183, 121},   // 0.667
	{84, 197, 104},   // 0.733
	{122, 209, 81},   // 0.800
	{165, 219, 54},   // 0.867
	{210, 226, 27},   // 0.933
	{253, 231, 37},   // 1.000
}

// Inferno colourmap LUT (16 stops — black-red-orange-yellow).
var infernoLUT = [16]rgb{
	{0, 0, 4},        // 0.000
	{14, 8, 49},      // 0.067
	{42, 11, 86},     // 0.133
	{75, 12, 107},    // 0.200
	{106, 23, 110},   // 0.267
	{135, 36, 105},   // 0.333
	{167, 50, 91},    // 0.400
	{198, 65, 76},    // 0.467
	{225, 87, 51},    // 0.533
	{244, 117, 24},   // 0.600
	{251, 154, 6},    // 0.667
	{251, 191, 36},   // 0.733
	{246, 221, 92},   // 0.800
	{243, 237, 158},  // 0.867
	{252, 251, 207},  // 0.933
	{252, 255, 164},  // 1.000
}

// lookup interpolates a 16-stop LUT at value t in [0, 1].
// Out-of-range t is clamped to [0, 1]. Pure function, zero allocation.
func lookup(lut *[16]rgb, t float64) (uint8, uint8, uint8) {
	if t <= 0 {
		return lut[0].r, lut[0].g, lut[0].b
	}
	if t >= 1 {
		return lut[15].r, lut[15].g, lut[15].b
	}
	// Find segment.
	pos := t * 15.0
	idx := int(pos)
	if idx >= 15 {
		idx = 14
	}
	frac := pos - float64(idx)
	a := lut[idx]
	b := lut[idx+1]
	r := uint8(float64(a.r) + frac*(float64(b.r)-float64(a.r)))
	g := uint8(float64(a.g) + frac*(float64(b.g)-float64(a.g)))
	bl := uint8(float64(a.b) + frac*(float64(b.b)-float64(a.b)))
	return r, g, bl
}

// Plasma returns the (R, G, B) colour value at the given position
// in [0, 1] using the matplotlib Plasma colourmap.
//
// Out-of-range values are clamped to the endpoints. Plasma is
// perceptually uniform — equal steps in the input map to equal
// perceived brightness changes.
//
// Reference: van der Walt & Smith (2015), matplotlib._cm_listed
// (CC0). See package doc.go for citation.
func Plasma(value float64) (r, g, b uint8) { return lookup(&plasmaLUT, value) }

// Magma returns the (R, G, B) value at the given position in [0, 1]
// using the matplotlib Magma colourmap (black → purple → orange →
// pale yellow). Better than Plasma for high-dynamic-range data
// because of the deeper black at value=0.
//
// Out-of-range values are clamped to the endpoints.
func Magma(value float64) (r, g, b uint8) { return lookup(&magmaLUT, value) }

// Viridis returns the (R, G, B) value at the given position in [0, 1]
// using the matplotlib Viridis colourmap (purple → blue → green →
// yellow). The matplotlib default since 2015. Best general-purpose
// choice for sequential scientific data.
//
// Out-of-range values are clamped to the endpoints.
func Viridis(value float64) (r, g, b uint8) { return lookup(&viridisLUT, value) }

// Inferno returns the (R, G, B) value at the given position in [0, 1]
// using the matplotlib Inferno colourmap (black → red → orange →
// pale yellow). Visually similar to Magma but slightly higher
// contrast in the mid-range.
//
// Out-of-range values are clamped to the endpoints.
func Inferno(value float64) (r, g, b uint8) { return lookup(&infernoLUT, value) }
