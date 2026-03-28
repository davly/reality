package color

// ---------------------------------------------------------------------------
// Chromatic Adaptation
// ---------------------------------------------------------------------------

// BradfordAdapt performs chromatic adaptation using the Bradford transform.
// It converts a color from a source white point to a destination white point.
//
// The source and destination white points are specified as (x, y) chromaticity
// coordinates (CIE 1931). Common white points:
//   - D50: (0.3457, 0.3585)
//   - D65: (0.3127, 0.3290)
//   - A:   (0.4476, 0.4074)
//
// The Bradford matrix is the standard cone response matrix used in ICC
// profiles and modern color management systems.
//
// Formula:
//  1. Convert source and destination white points from xy to XYZ.
//  2. Compute cone response ratios via Bradford matrix.
//  3. Build the adaptation matrix M = Minv * diag(ratios) * M.
//  4. Apply M to the input XYZ.
//
// Valid range: X, Y, Z >= 0; white point y > 0
// Precision: limited by float64 matrix operations
// Reference: Lam, K.M. (1985); Lindbloom, B. (2003) "Chromatic Adaptation"
func BradfordAdapt(X, Y, Z, srcWPx, srcWPy, dstWPx, dstWPy float64) (Xa, Ya, Za float64) {
	// Bradford matrix (cone response)
	//  0.8951000  0.2664000 -0.1614000
	// -0.7502000  1.7135000  0.0367000
	//  0.0389000 -0.0685000  1.0296000

	// Convert xy chromaticity to XYZ (assuming Y=1 for white point)
	srcX := srcWPx / srcWPy
	srcY := 1.0
	srcZ := (1 - srcWPx - srcWPy) / srcWPy

	dstX := dstWPx / dstWPy
	dstY := 1.0
	dstZ := (1 - dstWPx - dstWPy) / dstWPy

	// Compute source cone responses
	srcRho := 0.8951000*srcX + 0.2664000*srcY + -0.1614000*srcZ
	srcGam := -0.7502000*srcX + 1.7135000*srcY + 0.0367000*srcZ
	srcBet := 0.0389000*srcX + -0.0685000*srcY + 1.0296000*srcZ

	// Compute destination cone responses
	dstRho := 0.8951000*dstX + 0.2664000*dstY + -0.1614000*dstZ
	dstGam := -0.7502000*dstX + 1.7135000*dstY + 0.0367000*dstZ
	dstBet := 0.0389000*dstX + -0.0685000*dstY + 1.0296000*dstZ

	// Cone response ratios
	ratRho := dstRho / srcRho
	ratGam := dstGam / srcGam
	ratBet := dstBet / srcBet

	// Build adaptation matrix: M^{-1} * diag(ratios) * M
	// Bradford inverse:
	//  0.9869929 -0.1470543  0.1599627
	//  0.4323053  0.5183603  0.0492912
	// -0.0085287  0.0400428  0.9684867

	// First: compute M * [X, Y, Z] (cone response of input)
	rho := 0.8951000*X + 0.2664000*Y + -0.1614000*Z
	gam := -0.7502000*X + 1.7135000*Y + 0.0367000*Z
	bet := 0.0389000*X + -0.0685000*Y + 1.0296000*Z

	// Scale by ratios
	rho *= ratRho
	gam *= ratGam
	bet *= ratBet

	// Apply inverse Bradford: M^{-1} * [adapted cone]
	Xa = 0.9869929*rho + -0.1470543*gam + 0.1599627*bet
	Ya = 0.4323053*rho + 0.5183603*gam + 0.0492912*bet
	Za = -0.0085287*rho + 0.0400428*gam + 0.9684867*bet

	return
}
