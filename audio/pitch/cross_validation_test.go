package pitch

import (
	"math"
	"testing"
)

// This file saturates R132 MUTUAL-CROSS-VALIDATION-IN-PARITY-TEST for the
// reality/audio/pitch package. The package ships FOUR independent
// fundamental-frequency estimators that solve the SAME problem from
// different mathematical bases:
//
//   - AutocorrelationPitch  — time-domain ACF peak (Roads 1996 §10)
//   - Yin                   — cumulative-mean-normalised difference dip
//                             (de Cheveigné & Kawahara 2002)
//   - McLeodPitchMethod     — normalised square-difference function peak
//                             (McLeod & Wyvill 2005)
//   - SubharmonicSummation  — frequency-domain harmonic-evidence summation
//                             (Hermes 1988)
//
// R132 saturation requires THREE distinct cross-validation legs:
//
//  1. Multi-fundamental sweep agreement — all four detectors must agree
//     on a battery of clean tones spanning the typical [100, 800] Hz
//     audible-speech / mechanical-resonance band.
//  2. Harmonic-mixture fundamental locking — all four detectors must
//     return the fundamental (not a harmonic) on a battery of
//     harmonic-mix inputs. The four algorithms each have a distinct
//     failure mode (ACF locks onto the half-period; YIN locks onto a
//     sub-harmonic; MPM has the clarity-threshold corner; SHS has the
//     bin-quantisation corner) so mutual agreement is meaningful.
//  3. Silent-frame zero-divergence — all four detectors must return
//     pitch 0 on a silent input. This is the simplest of the three
//     legs but it is the canonical Mirror-Problem trap: a detector
//     that returns bin-1 frequency on silence (by accident) would be
//     caught by no individual unit test that doesn't check this exact
//     case across all four implementations as a single set.
//
// Plus an R85 same-class audit (per
// `feedback_mirror_problem_same_class_audit.md` — when one case is
// named, audit the same-class claim in the same scope before fixing):
//
//   - All four detectors share the contract "0 < fMin < fMax,
//     sampleRate > 0". A violation in any of the four MUST behave the
//     same way (the doc-comments agree on this). This test asserts
//     the contract is uniformly enforced.
//
// Tolerance discipline:
//   - 1e-12 tolerance is the right band for closed-form-vs-autodiff
//     legs (R131); pitch detectors have bin-quantisation precision
//     bounded by sampleRate / period_in_samples and so require a
//     relative tolerance commensurate with that physical limit.
//   - ACF without parabolic interpolation has precision ~sampleRate /
//     τ² so we use the documented ±5 Hz / ±10 Hz tolerances from the
//     individual unit tests as the shared cross-validation bound. The
//     R132 leg adds the SET-LEVEL agreement check (max - min < bound)
//     on top of the per-detector accuracy checks already present in
//     pitch_test.go.
//
// Saturation evidence for the canonical pkg/canonical promotion ledger
// (R132 — first promoted 2026-05-06 at 3/3, saturated at scale):
//
//   - timeseries/garch/autodiff_test.go (analytic gradient × autodiff)
//   - audio/onset/cross_validation_test.go (3-detector percussive train)
//   - changepoint/infogeo_test.go (BOCPD × posterior parity)
//   - audio/parity_test.go (golden-file cross-substrate oracle)
//   - prob/copula/autodiff_test.go (Clayton log-PDF × analytic)
//   - infogeo/autodiff_test.go (KL gradient × q-p closed form)
//   - timeseries/garch/garch_test.go (ForecastVariance closed-form)
//   - optim/proximal_consumer_test.go (FBS / FISTA / ADMM agreement)
//   - This file (four pitch detectors on shared synthetic battery —
//     fundamental sweep + harmonic-mix battery + silent-frame
//     zero-divergence + R85 same-class audit).
//
// Per `feedback_mirror_problem_arc.md`: the saturation count is the
// number of DISTINCT mathematical-domain consumer sites, not the
// number of test functions. This file is the ninth saturation site
// for R132 and the second within reality/audio (onset was first).

// ---------------------------------------------------------------------------
// Constants shared across the cross-validation suite.
// ---------------------------------------------------------------------------

// Cross-detector tolerance for the pitch SET (max - min agreement bound).
// Set by the floor-precision of the WORST-precision detector in the
// agreement set — that is ACF and SHS, both bin-quantised. At sr=16000
// and 440 Hz, ACF returns sampleRate / round(sr/f) = 444.4 Hz (one bin off)
// and SHS has 1 Hz grid; their disagreement on a clean tone is ≤ 10 Hz.
// We use 15 Hz here so the bound holds across the 100-800 Hz sweep.
const crossDetectorToleranceHz = 15.0

// ---------------------------------------------------------------------------
// Leg 1 — fundamental sweep agreement
// ---------------------------------------------------------------------------

// TestR132_FourPitchDetectors_AgreeOnFundamentalSweep is leg 1 of R132
// saturation for the pitch package. Sweep clean tones from 100 Hz to
// 800 Hz at 100 Hz steps and assert all four detectors agree on each
// tone within crossDetectorToleranceHz (set-level disagreement bound).
//
// Per-detector accuracy is already pinned by pitch_test.go; this test
// adds the cross-detector set-level agreement that no individual
// per-detector test can catch. A regression that drifts ONE detector
// while keeping its individual accuracy test green would be caught
// here.
func TestR132_FourPitchDetectors_AgreeOnFundamentalSweep(t *testing.T) {
	const (
		sr       = 16000.0
		nFrame   = 2048
		shsGridN = 4096 // SHS needs higher resolution to hit 1-Hz grid cleanly.
	)
	// Eight sweep points from low-bass speech (100 Hz) through alto-vocal
	// (220 Hz, 330 Hz, 440 Hz), brass (550 Hz), violin-E (660 Hz), and
	// soprano (770 Hz, ~G5).
	fundamentals := []float64{100, 220, 330, 440, 550, 660, 770, 800}

	for _, f0 := range fundamentals {
		t.Run("sweep_"+ftoa(f0), func(t *testing.T) {
			frame := makeSinusoid(nFrame, f0, sr, 0)

			acf := AutocorrelationPitch(frame, sr, 80, 1000)
			yin, _ := Yin(frame, sr, 0.10, 80, 1000)
			mpm, _ := McLeodPitchMethod(frame, sr, 80, 1000)

			// SHS needs a separate longer frame for the 1-Hz-grid scan to
			// resolve cleanly. Build a fresh frame at shsGridN.
			shsFrame := makeSinusoid(shsGridN, f0, sr, 0)
			power := computePowerSpectrumPow2(shsFrame, shsGridN)
			shs := SubharmonicSummation(power, sr, 80, 1000, 5)

			values := []float64{acf, yin, mpm, shs}
			labels := []string{"ACF", "YIN", "MPM", "SHS"}

			// Per-detector accuracy floor — each must be within
			// crossDetectorToleranceHz of the true f0.
			for i, v := range values {
				if math.Abs(v-f0) > crossDetectorToleranceHz {
					t.Errorf("%s on %v Hz returned %v (|diff|=%v > %v)",
						labels[i], f0, v, math.Abs(v-f0), crossDetectorToleranceHz)
				}
			}

			// Set-level agreement: max - min of the four detector outputs
			// MUST be within crossDetectorToleranceHz. This is the R132
			// mutual-cross-validation assertion proper.
			minV, maxV := values[0], values[0]
			for _, v := range values[1:] {
				if v < minV {
					minV = v
				}
				if v > maxV {
					maxV = v
				}
			}
			if maxV-minV > crossDetectorToleranceHz {
				t.Errorf("R132 disagreement on %v Hz: ACF=%v YIN=%v MPM=%v SHS=%v range=%v > %v",
					f0, acf, yin, mpm, shs, maxV-minV, crossDetectorToleranceHz)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Leg 2 — harmonic-mixture fundamental locking
// ---------------------------------------------------------------------------

// TestR132_FourPitchDetectors_AgreeOnHarmonicMixtures is leg 2 of R132
// saturation. Each algorithm has a distinct failure mode on harmonic
// mixtures:
//
//   - ACF can lock onto the half-period when the second harmonic is
//     unusually strong (Roads 1996 §10, well-documented failure).
//   - YIN can lock onto a sub-harmonic when the cumulative-mean-
//     normalised difference function has a deeper dip at 2·τ₀ than at
//     τ₀ (the classic "octave-error" of YIN; de Cheveigné & Kawahara
//     2002 §IV.A).
//   - MPM's k=0.93 threshold can reject the fundamental in favour of
//     the higher harmonic in noisy mixtures (McLeod & Wyvill 2005
//     §III.B; mitigated by the local-maximum requirement).
//   - SHS specifically EXISTS to handle the missing-fundamental case
//     and so it is the most robust of the four on harmonic mixes —
//     but its 1 Hz grid means it can drift by ≤1 Hz between calls.
//
// Mutual cross-validation: when all four detectors agree, that is
// stronger evidence than any single detector's correctness claim.
// This test asserts agreement on a battery of harmonic mixes where
// each individual detector COULD plausibly fail without the cross-
// validation guard.
func TestR132_FourPitchDetectors_AgreeOnHarmonicMixtures(t *testing.T) {
	const (
		sr     = 16000.0
		nFrame = 4096 // longer frame so SHS resolution suffices.
	)
	cases := []struct {
		name        string
		fundamental float64
		mix         []float64
	}{
		{"f0_220_with_2nd", 220, []float64{220, 440}},
		{"f0_220_with_3rd_5th", 220, []float64{220, 660, 1100}},
		{"f0_330_strong_3rd", 330, []float64{330, 990}},
		{"f0_440_full_harmonic_stack", 440, []float64{440, 880, 1320, 1760}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			frame := makeHarmonicMix(nFrame, tc.mix, sr)

			acf := AutocorrelationPitch(frame, sr, 80, 2000)
			yin, _ := Yin(frame, sr, 0.10, 80, 2000)
			mpm, _ := McLeodPitchMethod(frame, sr, 80, 2000)
			power := computePowerSpectrumPow2(frame, nFrame)
			shs := SubharmonicSummation(power, sr, 80, 2000, 5)

			values := []float64{acf, yin, mpm, shs}
			labels := []string{"ACF", "YIN", "MPM", "SHS"}

			// Each detector must return the fundamental (not a harmonic).
			// Octave-error tolerance: each individual detector's output
			// MUST be within crossDetectorToleranceHz of tc.fundamental.
			// A detector that returns 2*f0 or 3*f0 would fail this check.
			for i, v := range values {
				if math.Abs(v-tc.fundamental) > crossDetectorToleranceHz {
					t.Errorf("%s on mix %v returned %v (octave-error or drift; |diff|=%v > %v)",
						labels[i], tc.mix, v, math.Abs(v-tc.fundamental), crossDetectorToleranceHz)
				}
			}

			// Set-level agreement.
			minV, maxV := values[0], values[0]
			for _, v := range values[1:] {
				if v < minV {
					minV = v
				}
				if v > maxV {
					maxV = v
				}
			}
			if maxV-minV > crossDetectorToleranceHz {
				t.Errorf("R132 disagreement on mix %v (expected f0=%v): ACF=%v YIN=%v MPM=%v SHS=%v range=%v > %v",
					tc.mix, tc.fundamental, acf, yin, mpm, shs, maxV-minV, crossDetectorToleranceHz)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Leg 3 — silent-frame zero-divergence (Mirror-Problem-canonical case)
// ---------------------------------------------------------------------------

// TestR132_AllPitchDetectors_AgreeOnSilenceReturnsZero is leg 3 of R132
// saturation. The simplest of the three legs but the canonical
// Mirror-Problem trap: a detector that accidentally returns bin-1
// frequency or the smallest searchable period as a "pitch" on a
// silent input would not be caught by any per-detector unit test that
// asserts pitch=0 on silence — because each unit test only checks ITS
// detector.
//
// Cross-validation: all four detectors share the silent-input contract.
// A regression in any one of them would be caught here and only here.
//
// Bitwise: silent input must produce EXACTLY 0.0, not 1e-17 noise.
// This is achievable because all four implementations have explicit
// silent-frame guards (verified by reading their source); the test
// pins that contract.
func TestR132_AllPitchDetectors_AgreeOnSilenceReturnsZero(t *testing.T) {
	const (
		sr     = 16000.0
		nFrame = 2048
	)
	frame := make([]float64, nFrame)
	power := make([]float64, nFrame/2+1)

	t.Run("ACF", func(t *testing.T) {
		got := AutocorrelationPitch(frame, sr, 80, 1000)
		if got != 0 {
			t.Errorf("ACF on silence returned %v (bitwise-not-zero); want exactly 0", got)
		}
	})
	t.Run("YIN", func(t *testing.T) {
		got, ap := Yin(frame, sr, 0.10, 80, 1000)
		if got != 0 {
			t.Errorf("YIN on silence returned pitch %v; want exactly 0", got)
		}
		// YIN's silent-frame aperiodicity contract: 1.0 (full aperiodicity).
		if ap != 1.0 {
			t.Errorf("YIN on silence returned aperiodicity %v; want exactly 1.0", ap)
		}
	})
	t.Run("MPM", func(t *testing.T) {
		got, clarity := McLeodPitchMethod(frame, sr, 80, 1000)
		if got != 0 {
			t.Errorf("MPM on silence returned pitch %v; want exactly 0", got)
		}
		// MPM's silent-frame clarity contract: 0 (no peak found).
		if clarity != 0 {
			t.Errorf("MPM on silence returned clarity %v; want exactly 0", clarity)
		}
	})
	t.Run("SHS", func(t *testing.T) {
		got := SubharmonicSummation(power, sr, 100, 800, 5)
		if got != 0 {
			t.Errorf("SHS on silence returned %v; want exactly 0", got)
		}
	})
}

// ---------------------------------------------------------------------------
// Leg 4 — R85 same-class audit on the bad-range contract
// ---------------------------------------------------------------------------

// TestR85_AllPitchDetectors_PanicOnInvertedRange is the R85 same-class
// audit (per `feedback_mirror_problem_same_class_audit.md`). When one
// case is named, audit the same-class claim in the same scope before
// fixing.
//
// All four detectors share the validity contract "0 < fMin < fMax". A
// per-detector test already pins the contract on ACF and YIN; this
// test audits THE SAME CLASS across ALL FOUR detectors as a single
// set, ensuring no detector silently allows the inverted range while
// the others reject it. A regression that loosens the contract on a
// single detector would be caught here.
//
// Why this is R85-flavoured rather than R132-flavoured: R85 is the
// owned-primitives rule — "what counts as a pitch-detector primitive
// shares an enforced common contract". Here the common contract is
// the bad-input-rejection panic. This test pins the SET membership of
// "pitch detector that panics on inverted range" to "all four", which
// is the auditable structural claim.
func TestR85_AllPitchDetectors_PanicOnInvertedRange(t *testing.T) {
	t.Run("ACF", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Error("ACF did not panic on fMin >= fMax — contract loosening regression")
			}
		}()
		AutocorrelationPitch([]float64{1, 2, 3, 4}, 16000, 1000, 100)
	})
	t.Run("YIN", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Error("YIN did not panic on fMin >= fMax — contract loosening regression")
			}
		}()
		Yin([]float64{1, 2, 3, 4}, 16000, 0.10, 1000, 100)
	})
	t.Run("MPM", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Error("MPM did not panic on fMin >= fMax — contract loosening regression")
			}
		}()
		McLeodPitchMethod([]float64{1, 2, 3, 4}, 16000, 1000, 100)
	})
	t.Run("SHS", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Error("SHS did not panic on fMin >= fMax — contract loosening regression")
			}
		}()
		// SHS needs a non-empty spectrum to reach the range check; the
		// length-2 spectrum is the minimum valid input.
		SubharmonicSummation([]float64{1, 1}, 16000, 1000, 100, 5)
	})
}

// TestR85_AllPitchDetectors_PanicOnZeroSampleRate is the second leg of
// the R85 same-class audit. The "sampleRate > 0" contract is shared
// across all four detectors per their doc-comments. A detector that
// silently divided-by-zero or returned NaN on sr=0 would be a
// Mirror-Problem at the package-contract level.
func TestR85_AllPitchDetectors_PanicOnZeroSampleRate(t *testing.T) {
	t.Run("ACF", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Error("ACF did not panic on sampleRate=0 — contract loosening regression")
			}
		}()
		AutocorrelationPitch([]float64{1, 2, 3, 4}, 0, 100, 1000)
	})
	t.Run("YIN", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Error("YIN did not panic on sampleRate=0 — contract loosening regression")
			}
		}()
		Yin([]float64{1, 2, 3, 4}, 0, 0.10, 100, 1000)
	})
	t.Run("MPM", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Error("MPM did not panic on sampleRate=0 — contract loosening regression")
			}
		}()
		McLeodPitchMethod([]float64{1, 2, 3, 4}, 0, 100, 1000)
	})
	t.Run("SHS", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Error("SHS did not panic on sampleRate=0 — contract loosening regression")
			}
		}()
		SubharmonicSummation([]float64{1, 1}, 0, 100, 1000, 5)
	})
}

// ---------------------------------------------------------------------------
// Leg 5 — confidence-signal directional agreement (bonus R132 leg)
// ---------------------------------------------------------------------------

// TestR132_PitchTrio_ConfidenceMonotonicity asserts that YIN's
// aperiodicity and MPM's clarity move in OPPOSITE directions as input
// quality degrades (more noise → aperiodicity rises AND clarity drops).
//
// This is a R132 mutual-cross-validation leg on the SECONDARY outputs
// of the pitch detectors (not the pitch itself, but the confidence
// signal). YIN and MPM are independently-derived algorithms; their
// confidence signals computed from different mathematical bases
// (cumulative-mean-normalised difference vs normalised square-
// difference) should AGREE on the directional verdict "this input is
// noisy" even though their absolute scales differ.
//
// Per the existing pitch_test.go: clean 440 Hz tone produces YIN
// aperiodicity < 0.1 and MPM clarity > 0.85. Noisy 440 Hz tone (3x
// noise per the makeNoisyTone helper) produces YIN aperiodicity >=
// 0.15. This test pins the dual-directional pattern:
//
//   - clean tone: aperiodicity_clean < aperiodicity_noisy AND
//                 clarity_clean > clarity_noisy.
//
// A regression in either confidence signal alone would still be
// directional; a regression in BOTH simultaneously (the harder-to-
// catch case) would be detected here because the two signals are
// mathematically independent.
func TestR132_PitchTrio_ConfidenceMonotonicity(t *testing.T) {
	const (
		sr     = 16000.0
		nFrame = 2048
		f0     = 440.0
	)
	cleanFrame := makeSinusoid(nFrame, f0, sr, 0)
	noisyFrame := makeNoisyTone(nFrame, f0, sr, 7)

	_, apClean := Yin(cleanFrame, sr, 0.10, 80, 1000)
	_, apNoisy := Yin(noisyFrame, sr, 0.10, 80, 1000)
	_, clarityClean := McLeodPitchMethod(cleanFrame, sr, 80, 1000)
	_, clarityNoisy := McLeodPitchMethod(noisyFrame, sr, 80, 1000)

	// YIN aperiodicity must RISE under noise.
	if !(apNoisy > apClean) {
		t.Errorf("YIN aperiodicity did not rise under noise: clean=%v noisy=%v", apClean, apNoisy)
	}

	// MPM clarity must DROP under noise.
	if !(clarityClean > clarityNoisy) {
		t.Errorf("MPM clarity did not drop under noise: clean=%v noisy=%v", clarityClean, clarityNoisy)
	}

	// The directional verdicts agree: BOTH confidence signals flag the
	// noisy frame as lower-quality than the clean frame. This is the
	// R132 mutual-cross-validation assertion on the confidence signals.
	// Without this, a regression that swapped YIN's aperiodicity sign
	// convention (lower = more periodic → higher = more periodic) would
	// only be caught by the absolute-threshold test in pitch_test.go,
	// which depends on calibration; here the directional test is
	// calibration-free.
	if !(apClean < apNoisy && clarityClean > clarityNoisy) {
		t.Errorf("R132 confidence-signal directional disagreement: YIN(%v→%v) MPM(%v→%v)",
			apClean, apNoisy, clarityClean, clarityNoisy)
	}
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

// ftoa formats a float for use as a subtest name. Avoids the "/" that
// some Hz values would produce under %g, which would break Go's
// subtest naming.
func ftoa(v float64) string {
	// Integer-valued for our sweep; render as integer.
	iv := int(v)
	if float64(iv) == v {
		return itoa(iv)
	}
	// Fallback (not currently hit by our sweep).
	const base = 1000
	return itoa(int(v)) + "_" + itoa(int(v*base)%base)
}

func itoa(v int) string {
	if v == 0 {
		return "0"
	}
	neg := v < 0
	if neg {
		v = -v
	}
	var buf [16]byte
	i := len(buf)
	for v > 0 {
		i--
		buf[i] = byte('0' + v%10)
		v /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
}
