package onset

// EnergyOnset detects onsets by tracking the rise in short-time
// frame energy. Returns the indices (in frame units, 0-based) at
// which an onset is detected.
//
// Algorithm (Schloss 1985; Klapuri 1999 §2):
//
//	For frame t = 0, 1, 2, ...:
//	  E[t] = (1/frameSize) · Σ samples[t·hopSize + i]²    (i = 0..frameSize-1)
//	D[t] = max(0, E[t] - E[t-1])      (half-wave-rectified rise)
//	threshold = mean(D) + 1.5·stdev(D)
//	onsets = { t : D[t] > threshold and D[t] is local max in a 3-frame window }
//
// The mean+1.5σ adaptive threshold scales naturally with signal
// level: louder recordings produce larger D values but the threshold
// rises proportionally, so detection sensitivity is constant in
// relative terms.
//
// Parameters:
//   - samples:   real-valued audio buffer
//   - frameSize: per-frame window length (samples). Typical: 1024.
//   - hopSize:   advance between frames (samples). Typical: frameSize/4.
//
// Returns: slice of frame indices where onsets occur. Empty if no
// rise exceeds the adaptive threshold.
//
// Valid range: len(samples) >= frameSize; frameSize >= 1; 1 <= hopSize.
// Precision: limited by float64 sum-of-squares accumulation (~1e-12
// per frame for typical inputs).
//
// Reference: Schloss, W. A. (1985). "On the automatic transcription
// of percussive music." PhD thesis, Stanford CCRMA. Klapuri, A.
// (1999). "Sound onset detection by applying psychoacoustic knowledge."
// ICASSP 1999.
//
// Allocation: returns newly-allocated []int. Per-frame energies are
// computed in a single pre-allocated scratch.
//
// Consumed by: pigeonhole (call onset rough localisation),
// dipstick (impulsive-event detection in mechanical signals).
func EnergyOnset(samples []float64, frameSize, hopSize int) []int {
	if frameSize < 1 {
		panic("onset.EnergyOnset: frameSize must be >= 1")
	}
	if hopSize < 1 {
		panic("onset.EnergyOnset: hopSize must be >= 1")
	}
	if len(samples) < frameSize {
		return nil
	}

	numFrames := (len(samples)-frameSize)/hopSize + 1
	if numFrames < 2 {
		return nil
	}

	energies := make([]float64, numFrames)
	maxE := 0.0
	for t := 0; t < numFrames; t++ {
		start := t * hopSize
		s := 0.0
		for i := 0; i < frameSize; i++ {
			x := samples[start+i]
			s += x * x
		}
		energies[t] = s / float64(frameSize)
		if energies[t] > maxE {
			maxE = energies[t]
		}
	}

	// Half-wave-rectified differences D[t].
	D := make([]float64, numFrames)
	for t := 1; t < numFrames; t++ {
		d := energies[t] - energies[t-1]
		if d < 0 {
			d = 0
		}
		D[t] = d
	}

	// Picks must additionally clear an absolute floor of 10% of the peak
	// energy — defends against false positives on near-stationary signals
	// where mean+kσ collapses to a tiny threshold and amplifies tiny
	// FFT-leakage / quantisation variations into spurious "onsets".
	picks := PickPeaksAdaptive(D, 1.5, 1)
	floor := 0.10 * maxE
	out := picks[:0]
	for _, p := range picks {
		if D[p] >= floor {
			out = append(out, p)
		}
	}
	return out
}
