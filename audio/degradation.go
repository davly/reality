package audio

import "math"

// DegradationTracker tracks slow drift of a feature value across many
// observations using Welford running statistics, and detects when a
// recent window of values has deviated from the established baseline
// by more than a configurable z-score threshold.
//
// Used by:
//   - dipstick: per-machine fundamental-frequency drift (bearing wear)
//   - howler: per-pet night-vocalisation-frequency drift (cognitive decline)
//   - pigeonhole: per-individual phrase-pattern drift (returning bird ageing)
//
// Architecture: SEPARATE "baseline" and "recent window" statistics. The
// baseline accumulates over the lifetime of the entity; the recent
// window covers the last N observations. The forge ASSESS step compares
// recent-window mean to baseline mean and fires the escape hatch if
// the z-score |recent_mean - baseline_mean| / sigma_baseline exceeds
// a threshold.
//
// This is the standard EWMA-control-chart structure adapted to discrete
// observation streams (Western Electric / Shewhart 1924; Hunter 1986
// EWMA refinements).
//
// Fields are exported for serialisation; callers must NOT modify N
// without the corresponding Mean/M2 update — use UpdateBaseline / etc.
type DegradationTracker struct {
	BaselineN    int     // total observations contributing to baseline
	BaselineMean float64 // long-term mean
	BaselineM2   float64 // long-term sum of squared deviations

	WindowSize int       // number of recent observations tracked
	Window     []float64 // ring buffer of recent observations
	WindowHead int       // ring-buffer write index
	WindowFill int       // number of valid entries in window (0..WindowSize)
}

// NewDegradationTracker allocates a tracker with a recent-window of size W.
// W must be >= 2 (variance over a 1-element window is undefined).
func NewDegradationTracker(W int) DegradationTracker {
	if W < 2 {
		panic("audio.NewDegradationTracker: window size must be >= 2")
	}
	return DegradationTracker{
		WindowSize: W,
		Window:     make([]float64, W),
	}
}

// UpdateBaseline contributes a single observation to the long-term
// baseline only. Useful when the caller has determined (e.g. by
// service-event flag) that the observation is "normal" and should
// shape the reference distribution.
//
// Zero allocation.
func UpdateBaseline(t *DegradationTracker, x float64) {
	t.BaselineN++
	delta := x - t.BaselineMean
	t.BaselineMean += delta / float64(t.BaselineN)
	delta2 := x - t.BaselineMean
	t.BaselineM2 += delta * delta2
}

// PushObservation pushes one observation into both the baseline AND the
// recent window. Most consumers use this; reserve UpdateBaseline for
// special "trusted-good" observations.
//
// Zero allocation.
func PushObservation(t *DegradationTracker, x float64) {
	UpdateBaseline(t, x)

	t.Window[t.WindowHead] = x
	t.WindowHead = (t.WindowHead + 1) % t.WindowSize
	if t.WindowFill < t.WindowSize {
		t.WindowFill++
	}
}

// BaselineStdDev returns the unbiased baseline standard deviation.
// Returns 0 if BaselineN < 2.
func BaselineStdDev(t *DegradationTracker) float64 {
	if t.BaselineN < 2 {
		return 0.0
	}
	v := t.BaselineM2 / float64(t.BaselineN-1)
	if v < 0 {
		v = 0
	}
	return math.Sqrt(v)
}

// WindowMean returns the arithmetic mean of the current window.
// Returns 0 if window is empty.
func WindowMean(t *DegradationTracker) float64 {
	if t.WindowFill == 0 {
		return 0.0
	}
	s := 0.0
	for i := 0; i < t.WindowFill; i++ {
		s += t.Window[i]
	}
	return s / float64(t.WindowFill)
}

// ZScore returns the z-score of the current window mean against the
// baseline distribution.
//
// Formula: z = (window_mean - baseline_mean) / baseline_stddev
//
// Returns 0 if BaselineN < 2 (no baseline variance) or window empty.
//
// SIGN CONVENTION: positive z-score means the window has drifted
// HIGHER than baseline. Consumers interpret per their domain:
//   - dipstick: fundamental-frequency UP can mean bearing wear
//     (different harmonic profile); fundamental-frequency DOWN can mean
//     drive belt slip. Sign matters.
//   - howler: night-vocalisation count UP suggests cognitive decline;
//     DOWN may indicate sedation or lethargy.
//   - pigeonhole: phrase-rate DOWN may indicate ageing or molt; UP may
//     indicate breeding-season onset.
//
// Threshold convention: |z| >= 2 fires the forge escape hatch by
// default (corresponds to ~5% two-tailed false-positive rate under
// gaussian assumption); |z| >= 3 corresponds to ~0.3% (~control-chart
// 3-sigma).
func ZScore(t *DegradationTracker) float64 {
	if t.WindowFill == 0 {
		return 0.0
	}
	sigma := BaselineStdDev(t)
	if sigma == 0 {
		return 0.0
	}
	return (WindowMean(t) - t.BaselineMean) / sigma
}

// ResetWindow clears the recent window without affecting the baseline.
// Used when the caller has confirmed a service event (e.g. dipstick
// "bearings replaced") that invalidates the recent-window narrative
// without affecting the long-term machine identity.
func ResetWindow(t *DegradationTracker) {
	t.WindowHead = 0
	t.WindowFill = 0
	for i := range t.Window {
		t.Window[i] = 0
	}
}

// ResetBaseline clears the baseline and the window. Used when a service
// event fundamentally re-baselines the entity (dipstick: motor
// replaced; howler: pet's primary diet changed).
func ResetBaseline(t *DegradationTracker) {
	t.BaselineN = 0
	t.BaselineMean = 0
	t.BaselineM2 = 0
	ResetWindow(t)
}
