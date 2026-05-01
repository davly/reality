package pitch

// Yin estimates the fundamental frequency of a frame via the YIN
// algorithm (de Cheveigné & Kawahara 2002) — the de-facto modern
// monophonic pitch detector. Reports both pitch and aperiodicity (a
// confidence/quality indicator).
//
// Algorithm (de Cheveigné & Kawahara 2002, the canonical 6-step recipe):
//
//	Step 1: Difference function
//	  d(τ) = Σ_{n=0..N/2-1} (frame[n] - frame[n+τ])²
//
//	Step 2: Cumulative-mean-normalised difference function
//	  d'(τ) = 1                 if τ == 0
//	        = d(τ) / ((1/τ) · Σ_{j=1..τ} d(j))   otherwise
//
//	Step 3: Absolute-threshold dip selection
//	  τ* = smallest τ in [τ_min, τ_max] with d'(τ) < threshold AND
//	       d'(τ) is a local minimum (d'(τ-1) > d'(τ) < d'(τ+1))
//	  if no τ found: τ* = argmin_τ d'(τ)
//
//	Step 4: Parabolic interpolation around τ* for sub-sample precision
//	  τ_refined = τ* - 0.5 · (d'(τ*-1) - d'(τ*+1)) / (d'(τ*-1) + d'(τ*+1) - 2·d'(τ*))
//	            (skip if denominator is zero)
//
//	Step 5: f_0 = sampleRate / τ_refined
//	Step 6: aperiodicity = d'(τ*)   (lower = more periodic)
//
// Edge cases:
//   - silent frame: returns (0, 1) — no pitch, full aperiodicity
//   - no dip found: returns (sampleRate / argmin_τ d'(τ),
//     d'(argmin_τ)) — uses the global minimum instead
//
// Parameters:
//   - frame:      audio frame samples
//   - sampleRate: Hz, > 0
//   - threshold:  YIN absolute threshold for dip selection. Typical:
//     0.10 (de Cheveigné's default; aggressive). Higher values (0.15,
//     0.20) trade precision for recall. Must be in (0, 1].
//   - fMin, fMax: search range (Hz). Must satisfy 0 < fMin < fMax,
//     fMax < sampleRate/2.
//
// Returns: (pitch in Hz, aperiodicity in [0, 1+]). Pitch == 0 indicates
// silent input; aperiodicity > 0.5 indicates a noisy / unreliable
// estimate.
//
// Valid range: len(frame) >= 2 · ceil(sampleRate / fMin); shorter
// frames produce unreliable results. threshold in (0, 1].
// Precision: sub-sample via parabolic interpolation; pitch error
// typically < 0.5% for clean tonal input.
//
// Allocation: O(N/2) scratch for d(τ) and d'(τ) — single pair of
// pre-allocated slices. Returns two float64.
//
// Reference: de Cheveigné, A. & Kawahara, H. (2002). "YIN, a
// fundamental frequency estimator for speech and music." J. Acoust.
// Soc. Am. 111(4), 1917-1930.
//
// Consumed by: pigeonhole (bird-song pitch tracking with confidence),
// howler (vocalisation pitch with reliability gate).
func Yin(frame []float64, sampleRate, threshold, fMin, fMax float64) (float64, float64) {
	if sampleRate <= 0 {
		panic("pitch.Yin: sampleRate must be > 0")
	}
	if threshold <= 0 || threshold > 1 {
		panic("pitch.Yin: threshold must be in (0, 1]")
	}
	if fMin <= 0 || fMax <= fMin {
		panic("pitch.Yin: must satisfy 0 < fMin < fMax")
	}
	N := len(frame)
	if N < 2 {
		return 0, 1
	}

	// Silent frame check.
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
		return 0, 1
	}

	// YIN uses N/2 as the maximum lag.
	W := N / 2
	if W < 2 {
		W = N - 1
	}

	tauMin := int(sampleRate / fMax)
	if tauMin < 1 {
		tauMin = 1
	}
	tauMax := int(sampleRate / fMin)
	if tauMax >= W {
		tauMax = W - 1
	}
	if tauMin >= tauMax {
		return 0, 1
	}

	// Step 1: difference function d(τ).
	d := make([]float64, W)
	for tau := 1; tau < W; tau++ {
		s := 0.0
		for n := 0; n < W; n++ {
			diff := frame[n] - frame[n+tau]
			s += diff * diff
		}
		d[tau] = s
	}

	// Step 2: cumulative-mean-normalised difference d'(τ).
	dPrime := make([]float64, W)
	dPrime[0] = 1
	cumSum := 0.0
	for tau := 1; tau < W; tau++ {
		cumSum += d[tau]
		if cumSum > 0 {
			dPrime[tau] = d[tau] / (cumSum / float64(tau))
		} else {
			dPrime[tau] = 1
		}
	}

	// Step 3: absolute-threshold dip selection. Look for the first τ in
	// [tauMin, tauMax] where d'(τ) < threshold AND d'(τ) is a local
	// minimum.
	tauStar := -1
	for tau := tauMin; tau <= tauMax; tau++ {
		if dPrime[tau] < threshold {
			// Walk to local-minimum position.
			for tau+1 <= tauMax && dPrime[tau+1] < dPrime[tau] {
				tau++
			}
			tauStar = tau
			break
		}
	}
	// Fallback: if no dip found below threshold, use global min in range.
	if tauStar < 0 {
		minVal := dPrime[tauMin]
		tauStar = tauMin
		for tau := tauMin + 1; tau <= tauMax; tau++ {
			if dPrime[tau] < minVal {
				minVal = dPrime[tau]
				tauStar = tau
			}
		}
	}

	aperiodicity := dPrime[tauStar]

	// Step 4: parabolic interpolation.
	tauRefined := float64(tauStar)
	if tauStar > 0 && tauStar < W-1 {
		y0 := dPrime[tauStar-1]
		y1 := dPrime[tauStar]
		y2 := dPrime[tauStar+1]
		denom := y0 + y2 - 2*y1
		if denom != 0 {
			tauRefined = float64(tauStar) + 0.5*(y0-y2)/denom
			// Clamp the refinement to within ±1 of tauStar to defend
			// against unstable parabolic fits at near-flat dips.
			if tauRefined < float64(tauStar)-1 {
				tauRefined = float64(tauStar) - 1
			}
			if tauRefined > float64(tauStar)+1 {
				tauRefined = float64(tauStar) + 1
			}
		}
	}
	if tauRefined < 1 {
		return 0, aperiodicity
	}

	return sampleRate / tauRefined, aperiodicity
}
