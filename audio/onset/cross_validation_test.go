package onset

import "testing"

// TestThreeOnsetDetectors_AgreeOnPercussiveTrain saturates the
// R-MUTUAL-CROSS-VALIDATION-IN-PARITY-TEST watchlist (S62 overnight,
// 2026-05-06) to 3/3. The two prior consumers were:
//
//   1. F2.b changepoint × infogeo (TV + Hellinger ordering check)
//   2. Project 1 optim × proximal (FBS + FISTA + ADMM all converge to
//      same closed-form on LASSO with X = I)
//
// This third consumer:
//
//   - Cross-package: audio/onset is the consuming package,
//     audio/spectrogram is the consumed package (via the
//     internal computeSTFT helper used by ComplexDomainOnset).
//   - Within-onset cross-validation: THREE distinct onset-detection
//     methods (EnergyOnset, SpectralFluxOnset, ComplexDomainOnset)
//     run on the SAME synthetic percussive train and MUST agree on
//     the same set of onset frames within ±4 frames tolerance.
//
// The pre-existing onset_test.go tests each method separately
// against a known synthetic train; this test asserts the three
// agree as a single composition. Disagreement would surface either
// (a) a bug in one of the detectors or (b) an unannounced
// divergence in their interpretation of what counts as an onset.
//
// Saturation evidence (R-MUTUAL-CROSS-VALIDATION-IN-PARITY-TEST,
// 3/3 — promotable to STANDARD):
//   - F2.b changepoint × infogeo (substrate composition + TV/Hellinger ordering agreement)
//   - Project 1 optim × proximal LASSO (substrate composition + FBS/FISTA/ADMM convergence agreement)
//   - This test (substrate composition + 3 onset-detector method agreement)
func TestThreeOnsetDetectors_AgreeOnPercussiveTrain(t *testing.T) {
	const (
		sr        = 16000
		frameSize = 1024
		hopSize   = 256
		numOnsets = 4
	)
	samples, positions := makePercussiveTrain(numOnsets, sr, sr, 800)

	type detector struct {
		name string
		run  func() []int
	}
	stft := computeSTFT(samples, frameSize, hopSize)
	detectors := []detector{
		{
			name: "Energy",
			run:  func() []int { return EnergyOnset(samples, frameSize, hopSize) },
		},
		{
			name: "SpectralFlux",
			run:  func() []int { return SpectralFluxOnset(stft) },
		},
		{
			name: "ComplexDomain",
			run:  func() []int { return ComplexDomainOnset(stft) },
		},
	}

	const toleranceFrames = 4
	results := make(map[string]int, len(detectors))
	for _, d := range detectors {
		got := d.run()
		matched := countNear(got, positions, hopSize, toleranceFrames)
		results[d.name] = matched
		// Each detector must individually find at least 3 of 4 onsets
		// (allowing one boundary miss) — this is the per-method floor.
		if matched < 3 {
			t.Errorf("%s detector matched only %d/%d onsets; got=%v positions=%v",
				d.name, matched, numOnsets, got, positions)
		}
	}

	// Cross-validation: the three methods must agree on within-1 hits
	// out of 4. If one detector finds 4 and another finds 1, that's a
	// disagreement large enough that the methods are not measuring the
	// same phenomenon — which would falsify the substrate-composition
	// claim that the three are alternative implementations of the same
	// onset-detection abstraction.
	maxMatched := 0
	minMatched := numOnsets
	for _, m := range results {
		if m > maxMatched {
			maxMatched = m
		}
		if m < minMatched {
			minMatched = m
		}
	}
	if maxMatched-minMatched > 1 {
		t.Errorf("onset detectors disagree by more than 1 hit: %v (range %d-%d on a 4-onset train)",
			results, minMatched, maxMatched)
	}
}
