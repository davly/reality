package color

import "math"

// ---------------------------------------------------------------------------
// Color Difference Metrics
// ---------------------------------------------------------------------------

// DeltaE76 computes the CIE76 color difference between two colors in
// CIELAB space. This is simply the Euclidean distance in Lab.
//
// Formula: sqrt((L1-L2)^2 + (a1-a2)^2 + (b1-b2)^2)
// Valid range: any Lab values
// Output range: [0, +inf)
// Precision: exact to float64 sqrt precision
// Reference: CIE 15:2004 (Colorimetry, 3rd ed.)
func DeltaE76(L1, a1, b1, L2, a2, b2 float64) float64 {
	dL := L1 - L2
	da := a1 - a2
	db := b1 - b2
	return math.Sqrt(dL*dL + da*da + db*db)
}

// DeltaE2000 computes the CIEDE2000 color difference between two colors
// in CIELAB space. This is the state-of-the-art perceptual color difference
// metric, accounting for lightness, chroma, and hue weighting, plus
// interaction terms.
//
// The full CIEDE2000 formula includes:
//   - Chroma-dependent a' adjustment (rotation in a*b* plane)
//   - Lightness, chroma, and hue difference terms
//   - Weighting functions SL, SC, SH
//   - Rotation term RT for the blue region
//
// Valid range: any Lab values
// Output range: [0, +inf)
// Precision: limited by float64 trigonometric functions; typically 1e-10
// Reference: Sharma, Wu, Dalal (2005) "The CIEDE2000 Color-Difference Formula:
// Implementation Notes, Supplementary Test Data, and Mathematical Observations"
// Color Research and Application, 30(1), pp.21-30.
func DeltaE2000(L1, a1, b1, L2, a2, b2 float64) float64 {
	// Step 1: Calculate C'ab and h'ab
	cab1 := math.Sqrt(a1*a1 + b1*b1)
	cab2 := math.Sqrt(a2*a2 + b2*b2)
	cabAvg := (cab1 + cab2) / 2.0

	cabAvg7 := math.Pow(cabAvg, 7)
	g := 0.5 * (1 - math.Sqrt(cabAvg7/(cabAvg7+6103515625))) // 25^7 = 6103515625

	a1p := a1 * (1 + g)
	a2p := a2 * (1 + g)

	cp1 := math.Sqrt(a1p*a1p + b1*b1)
	cp2 := math.Sqrt(a2p*a2p + b2*b2)

	hp1 := hueAngle(a1p, b1)
	hp2 := hueAngle(a2p, b2)

	// Step 2: Calculate delta L', delta C', delta H'
	dLp := L2 - L1
	dCp := cp2 - cp1

	dhp := 0.0
	if cp1*cp2 != 0 {
		dhp = hp2 - hp1
		if dhp > 180 {
			dhp -= 360
		} else if dhp < -180 {
			dhp += 360
		}
	}
	dHp := 2 * math.Sqrt(cp1*cp2) * math.Sin(deg2rad(dhp/2))

	// Step 3: Calculate weighting functions
	lAvg := (L1 + L2) / 2
	cpAvg := (cp1 + cp2) / 2

	hpAvg := 0.0
	if cp1*cp2 != 0 {
		hpAvg = hp1 + hp2
		if math.Abs(hp1-hp2) > 180 {
			if hp1+hp2 < 360 {
				hpAvg += 360
			} else {
				hpAvg -= 360
			}
		}
		hpAvg /= 2
	}

	t := 1 -
		0.17*math.Cos(deg2rad(hpAvg-30)) +
		0.24*math.Cos(deg2rad(2*hpAvg)) +
		0.32*math.Cos(deg2rad(3*hpAvg+6)) -
		0.20*math.Cos(deg2rad(4*hpAvg-63))

	lAvg50 := (lAvg - 50) * (lAvg - 50)
	sl := 1 + 0.015*lAvg50/math.Sqrt(20+lAvg50)
	sc := 1 + 0.045*cpAvg
	sh := 1 + 0.015*cpAvg*t

	cpAvg7 := math.Pow(cpAvg, 7)
	rc := 2 * math.Sqrt(cpAvg7/(cpAvg7+6103515625))

	dTheta := 30 * math.Exp(-((hpAvg-275)/25)*((hpAvg-275)/25))
	rt := -math.Sin(deg2rad(2*dTheta)) * rc

	// Step 4: Calculate total CIEDE2000
	termL := dLp / sl
	termC := dCp / sc
	termH := dHp / sh

	return math.Sqrt(termL*termL + termC*termC + termH*termH + rt*termC*termH)
}

// hueAngle computes the hue angle in degrees from a' and b values.
// Returns 0 if both a' and b are zero.
func hueAngle(ap, b float64) float64 {
	if ap == 0 && b == 0 {
		return 0
	}
	h := rad2deg(math.Atan2(b, ap))
	if h < 0 {
		h += 360
	}
	return h
}

// deg2rad converts degrees to radians.
func deg2rad(d float64) float64 {
	return d * math.Pi / 180
}

// rad2deg converts radians to degrees.
func rad2deg(r float64) float64 {
	return r * 180 / math.Pi
}
