package pitch

// McLeodPitchMethod estimates the fundamental frequency of a frame
// via the McLeod Pitch Method (MPM, 2005). The modern alternative to
// YIN — uses a normalised square-difference function and parabolic-
// interpolated peak picking. Particularly strong on instrumental notes
// with rich harmonic content where YIN's threshold-based dip selection
// can lock onto the wrong sub-harmonic.
//
// Algorithm (McLeod & Wyvill 2005 §2):
//
//	Step 1: Square-difference function (similar to YIN's d(τ))
//	  m(τ) = Σ_n (frame[n]² + frame[n+τ]²)        (squared-sum normaliser)
//	  r(τ) = Σ_n frame[n] · frame[n+τ]            (autocorrelation)
//	  n(τ) = 2·r(τ) / m(τ)                         (normalised SDF; in [-1, 1])
//
//	Step 2: Find the highest peak above threshold
//	  k = 0.93                                     (McLeod's recommended threshold)
//	  globalMax = max_τ n(τ)                       (in search range)
//	  τ* = first τ where n(τ) is a local maximum AND n(τ) >= k · globalMax
//	  if no τ found: τ* = argmax_τ n(τ)
//
//	Step 3: Parabolic interpolation
//	  τ_refined = τ* + 0.5 · (n(τ*-1) - n(τ*+1)) / (n(τ*-1) - 2·n(τ*) + n(τ*+1))
//
//	Step 4: f_0 = sampleRate / τ_refined
//	Step 5: clarity = n(τ*)
//
// Edge cases:
//   - silent frame: returns (0, 0)
//   - no peak found: returns (sampleRate / argmax_τ n(τ),
//     n(argmax_τ)) — uses the global peak instead
//
// Parameters:
//   - frame:      audio frame samples
//   - sampleRate: Hz, > 0
//   - fMin, fMax: search range (Hz). Must satisfy 0 < fMin < fMax,
//     fMax < sampleRate/2.
//
// Returns: (pitch in Hz, clarity in [-1, 1+]). Clarity > 0.85
// indicates a strongly periodic signal; < 0.5 is unreliable.
//
// Valid range: len(frame) >= 2 · ceil(sampleRate / fMin).
// Precision: sub-sample via parabolic interpolation; pitch error
// typically < 0.5% for clean tonal input.
//
// Allocation: O(τ_max) scratch — single slice. Returns two float64.
//
// Speculative: McLeod's original threshold k = 0.93 is on the
// aggressive side for some signals. Callers wanting greater recall
// can lower k by post-processing the clarity output (e.g., accept
// pitches with clarity > 0.85). The k constant is hard-coded here
// for fidelity to the published recipe.
//
// Reference: McLeod, P. & Wyvill, G. (2005). "A smarter way to find
// pitch." Proc. International Computer Music Conference (ICMC),
// 138-141.
//
// Consumed by: pigeonhole (instrumental-tone bird-song fundamental
// detection), dipstick (clean-spectral-content machine resonance).
func McLeodPitchMethod(frame []float64, sampleRate, fMin, fMax float64) (float64, float64) {
	if sampleRate <= 0 {
		panic("pitch.McLeodPitchMethod: sampleRate must be > 0")
	}
	if fMin <= 0 || fMax <= fMin {
		panic("pitch.McLeodPitchMethod: must satisfy 0 < fMin < fMax")
	}
	N := len(frame)
	if N < 2 {
		return 0, 0
	}

	maxAbs := 0.0
	for i := 0; i < N; i++ {
		a := frame[i]
		if a < 0 {
			a = -a
		}
		if a > maxAbs {
			maxAbs = a
		}
	}
	if maxAbs == 0 {
		return 0, 0
	}

	tauMin := int(sampleRate / fMax)
	if tauMin < 1 {
		tauMin = 1
	}
	tauMax := int(sampleRate / fMin)
	if tauMax >= N {
		tauMax = N - 1
	}
	if tauMin >= tauMax {
		return 0, 0
	}

	// Compute n(τ) over [0, tauMax].
	nsdf := make([]float64, tauMax+1)
	for tau := 0; tau <= tauMax; tau++ {
		var r, m float64
		for n := 0; n < N-tau; n++ {
			a := frame[n]
			b := frame[n+tau]
			r += a * b
			m += a*a + b*b
		}
		if m > 0 {
			nsdf[tau] = 2 * r / m
		}
	}

	// Find local maxima above kThreshold · globalMax.
	const kThreshold = 0.93
	globalMax := 0.0
	for tau := tauMin; tau <= tauMax; tau++ {
		if nsdf[tau] > globalMax {
			globalMax = nsdf[tau]
		}
	}
	if globalMax <= 0 {
		return 0, 0
	}

	tauStar := -1
	for tau := tauMin + 1; tau < tauMax; tau++ {
		if nsdf[tau] > nsdf[tau-1] && nsdf[tau] > nsdf[tau+1] && nsdf[tau] >= kThreshold*globalMax {
			tauStar = tau
			break
		}
	}
	// Fallback to global argmax in range.
	if tauStar < 0 {
		tauStar = tauMin
		for tau := tauMin + 1; tau <= tauMax; tau++ {
			if nsdf[tau] > nsdf[tauStar] {
				tauStar = tau
			}
		}
	}

	clarity := nsdf[tauStar]

	// Parabolic interpolation.
	tauRefined := float64(tauStar)
	if tauStar > 0 && tauStar < tauMax {
		y0 := nsdf[tauStar-1]
		y1 := nsdf[tauStar]
		y2 := nsdf[tauStar+1]
		denom := y0 - 2*y1 + y2
		if denom != 0 {
			tauRefined = float64(tauStar) + 0.5*(y0-y2)/denom
			if tauRefined < float64(tauStar)-1 {
				tauRefined = float64(tauStar) - 1
			}
			if tauRefined > float64(tauStar)+1 {
				tauRefined = float64(tauStar) + 1
			}
		}
	}
	if tauRefined < 1 {
		return 0, clarity
	}

	return sampleRate / tauRefined, clarity
}
