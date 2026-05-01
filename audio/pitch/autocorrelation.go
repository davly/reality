package pitch

// AutocorrelationPitch estimates the fundamental frequency of a frame
// via classic autocorrelation peak picking. The lag with the highest
// autocorrelation value in the period range [τ_min, τ_max] = [sr/fMax,
// sr/fMin] gives the period T₀ in samples; the fundamental is
// sampleRate / T₀.
//
// Algorithm (Roads 1996 §10):
//
//	r(τ) = Σ_{n=0..N-τ-1} frame[n] · frame[n+τ]
//	τ_min = floor(sampleRate / fMax)
//	τ_max = ceil(sampleRate / fMin)
//	τ* = argmax_{τ_min <= τ <= τ_max} r(τ)
//	f_0 = sampleRate / τ*
//
// Edge cases:
//   - silent frame (all zeros): returns 0
//   - τ_max >= len(frame): clamped to len(frame) - 1
//   - τ_min < 1: clamped to 1
//
// Parameters:
//   - frame:      audio frame samples (real-valued)
//   - sampleRate: Hz, > 0
//   - fMin, fMax: search range (Hz). Must satisfy 0 < fMin < fMax,
//     fMax < sampleRate/2.
//
// Returns: estimated fundamental in Hz, or 0 if no peak found.
//
// Valid range: len(frame) >= ceil(sampleRate/fMin) + 1 to make sense
// (we need at least one full period to autocorrelate); a more
// permissive valid range is len(frame) >= 2.
// Precision: pitch is integer-period quantised — best precision is
// sampleRate / τ². For sub-sample precision callers should apply
// parabolic interpolation on the peak (not implemented here — the
// canonical use is the Pigeonhole-style relative-shift tracker).
//
// Allocation: O(τ_max) scratch for the ACF — single pre-allocated
// slice. Returns a single float64.
//
// Reference: Roads, C. (1996). "The Computer Music Tutorial." MIT
// Press §10 (autocorrelation pitch detection); standard DSP textbook
// material going back to Rabiner 1977 "On the use of autocorrelation
// analysis for pitch detection."
//
// Consumed by: pigeonhole (cheap-fast pitch tracking on cleaned bird
// calls), howler (vocalisation pitch), pitch.Yin (delegates ACF
// computation pattern to here).
func AutocorrelationPitch(frame []float64, sampleRate, fMin, fMax float64) float64 {
	if sampleRate <= 0 {
		panic("pitch.AutocorrelationPitch: sampleRate must be > 0")
	}
	if fMin <= 0 || fMax <= fMin {
		panic("pitch.AutocorrelationPitch: must satisfy 0 < fMin < fMax")
	}
	N := len(frame)
	if N < 2 {
		return 0
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
		return 0
	}

	// Check silent frame.
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
		return 0
	}

	bestLag := -1
	bestACF := 0.0
	for tau := tauMin; tau <= tauMax; tau++ {
		s := 0.0
		for n := 0; n < N-tau; n++ {
			s += frame[n] * frame[n+tau]
		}
		if s > bestACF {
			bestACF = s
			bestLag = tau
		}
	}
	if bestLag < 1 {
		return 0
	}
	return sampleRate / float64(bestLag)
}
