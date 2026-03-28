// Package color provides color science functions: color space conversions
// (sRGB, linear RGB, CIE XYZ, CIELAB, HSV), color difference metrics
// (DeltaE76, CIEDE2000), chromatic adaptation (Bradford), and spectral
// computations (blackbody radiation, tone mapping).
//
// All functions use fixed-size returns (not slices) for stack allocation.
// Zero external dependencies.
package color

import "math"

// ---------------------------------------------------------------------------
// sRGB <-> Linear RGB
// ---------------------------------------------------------------------------

// SRGBToLinear converts a single sRGB channel value to linear RGB using
// the piecewise sRGB transfer function (gamma decode).
//
// Formula:
//
//	if srgb <= 0.04045: linear = srgb / 12.92
//	else:               linear = ((srgb + 0.055) / 1.055) ^ 2.4
//
// Valid range: srgb in [0, 1]
// Precision: exact to float64 precision
// Reference: IEC 61966-2-1:1999 (sRGB standard)
func SRGBToLinear(srgb float64) float64 {
	if srgb <= 0.04045 {
		return srgb / 12.92
	}
	return math.Pow((srgb+0.055)/1.055, 2.4)
}

// LinearToSRGB converts a single linear RGB channel value to sRGB using
// the piecewise sRGB transfer function (gamma encode).
//
// Formula:
//
//	if linear <= 0.0031308: srgb = linear * 12.92
//	else:                   srgb = 1.055 * linear^(1/2.4) - 0.055
//
// Valid range: linear in [0, 1]
// Precision: exact to float64 precision
// Reference: IEC 61966-2-1:1999 (sRGB standard)
func LinearToSRGB(linear float64) float64 {
	if linear <= 0.0031308 {
		return linear * 12.92
	}
	return 1.055*math.Pow(linear, 1.0/2.4) - 0.055
}

// ---------------------------------------------------------------------------
// Linear RGB <-> CIE XYZ (using sRGB/D65 primaries)
// ---------------------------------------------------------------------------

// LinearRGBToXYZ converts linear RGB (sRGB primaries) to CIE XYZ using the
// standard sRGB-to-XYZ matrix (D65 illuminant).
//
// Formula: [X,Y,Z] = M * [R,G,B] where M is the sRGB-to-XYZ matrix.
// Valid range: r, g, b in [0, 1]
// Precision: limited by float64 matrix multiplication
// Reference: IEC 61966-2-1:1999, Annex A
func LinearRGBToXYZ(r, g, b float64) (X, Y, Z float64) {
	// sRGB to XYZ matrix (D65 reference white)
	X = 0.4124564*r + 0.3575761*g + 0.1804375*b
	Y = 0.2126729*r + 0.7151522*g + 0.0721750*b
	Z = 0.0193339*r + 0.1191920*g + 0.9503041*b
	return
}

// XYZToLinearRGB converts CIE XYZ to linear RGB (sRGB primaries) using the
// inverse of the sRGB-to-XYZ matrix (D65 illuminant).
//
// Formula: [R,G,B] = M^{-1} * [X,Y,Z]
// Valid range: values may go outside [0,1] for out-of-gamut colors
// Precision: limited by float64 matrix multiplication
// Reference: IEC 61966-2-1:1999, Annex A
func XYZToLinearRGB(X, Y, Z float64) (r, g, b float64) {
	// XYZ to sRGB matrix (inverse of above)
	r = 3.2404542*X - 1.5371385*Y - 0.4985314*Z
	g = -0.9692660*X + 1.8760108*Y + 0.0415560*Z
	b = 0.0556434*X - 0.2040259*Y + 1.0572252*Z
	return
}

// ---------------------------------------------------------------------------
// CIE XYZ <-> CIELAB
// ---------------------------------------------------------------------------

// labF is the forward CIELAB nonlinear function.
// f(t) = t^(1/3)           if t > (6/29)^3
// f(t) = (t/(3*(6/29)^2)) + 4/29  otherwise
func labF(t float64) float64 {
	const delta = 6.0 / 29.0
	if t > delta*delta*delta {
		return math.Cbrt(t)
	}
	return t/(3*delta*delta) + 4.0/29.0
}

// labFInv is the inverse CIELAB nonlinear function.
func labFInv(t float64) float64 {
	const delta = 6.0 / 29.0
	if t > delta {
		return t * t * t
	}
	return 3 * delta * delta * (t - 4.0/29.0)
}

// XYZToLab converts CIE XYZ to CIELAB given reference white point (Xn, Yn, Zn).
//
// Formula:
//
//	L* = 116 * f(Y/Yn) - 16
//	a* = 500 * (f(X/Xn) - f(Y/Yn))
//	b* = 200 * (f(Y/Yn) - f(Z/Zn))
//
// where f(t) = t^(1/3) for t > (6/29)^3, otherwise linear approximation.
// Valid range: X,Y,Z >= 0; Xn,Yn,Zn > 0
// Output range: L* in [0, 100], a* and b* approximately [-128, 127]
// Precision: limited by float64 cube root
// Reference: CIE 15:2004 (Colorimetry, 3rd ed.)
func XYZToLab(X, Y, Z, Xn, Yn, Zn float64) (L, a, bStar float64) {
	fY := labF(Y / Yn)
	L = 116*fY - 16
	a = 500 * (labF(X/Xn) - fY)
	bStar = 200 * (fY - labF(Z/Zn))
	return
}

// LabToXYZ converts CIELAB to CIE XYZ given reference white point (Xn, Yn, Zn).
//
// Formula: inverse of XYZToLab.
// Valid range: L in [0, 100]; a, b any real
// Precision: limited by float64 cube
// Reference: CIE 15:2004
func LabToXYZ(L, a, bStar, Xn, Yn, Zn float64) (X, Y, Z float64) {
	fY := (L + 16) / 116
	fX := a/500 + fY
	fZ := fY - bStar/200

	X = Xn * labFInv(fX)
	Y = Yn * labFInv(fY)
	Z = Zn * labFInv(fZ)
	return
}

// ---------------------------------------------------------------------------
// RGB <-> HSV
// ---------------------------------------------------------------------------

// RGBToHSV converts RGB [0,1] to HSV where H is in [0,360), S in [0,1], V in [0,1].
// Returns H=0 for achromatic colors (when S=0).
//
// Formula: standard max/min decomposition.
// Valid range: r, g, b in [0, 1]
// Precision: exact to float64
// Reference: Smith, A.R. (1978) "Color Gamut Transform Pairs"
func RGBToHSV(r, g, b float64) (h, s, v float64) {
	maxC := math.Max(r, math.Max(g, b))
	minC := math.Min(r, math.Min(g, b))
	delta := maxC - minC

	v = maxC

	if maxC == 0 {
		return 0, 0, v
	}
	s = delta / maxC

	if delta == 0 {
		return 0, s, v
	}

	switch maxC {
	case r:
		h = 60 * math.Mod((g-b)/delta, 6)
	case g:
		h = 60 * ((b-r)/delta + 2)
	case b:
		h = 60 * ((r-g)/delta + 4)
	}
	if h < 0 {
		h += 360
	}

	return
}

// HSVToRGB converts HSV (H in [0,360), S in [0,1], V in [0,1]) to RGB [0,1].
//
// Formula: standard sector-based conversion.
// Valid range: h in [0, 360), s in [0, 1], v in [0, 1]
// Precision: exact to float64
// Reference: Smith, A.R. (1978) "Color Gamut Transform Pairs"
func HSVToRGB(h, s, v float64) (r, g, b float64) {
	c := v * s
	hPrime := h / 60
	x := c * (1 - math.Abs(math.Mod(hPrime, 2)-1))
	m := v - c

	switch {
	case hPrime < 1:
		r, g, b = c, x, 0
	case hPrime < 2:
		r, g, b = x, c, 0
	case hPrime < 3:
		r, g, b = 0, c, x
	case hPrime < 4:
		r, g, b = 0, x, c
	case hPrime < 5:
		r, g, b = x, 0, c
	default:
		r, g, b = c, 0, x
	}

	r += m
	g += m
	b += m
	return
}
