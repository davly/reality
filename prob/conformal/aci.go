package conformal

import (
	"errors"
	"math"
)

// Adaptive Conformal Inference (ACI), Gibbs & Candès (2021), "Adaptive
// Conformal Inference Under Distribution Shift" (arXiv:2106.00170).
//
// Split conformal gives 1-alpha coverage only under exchangeability; under
// distribution shift it silently loses coverage. ACI fixes this online without
// any model of the shift: it treats the miscoverage level as a control variable
// and nudges it after every observation,
//
//	alpha_{t+1} = alpha_t + gamma * ( alpha - err_t ),
//
// where err_t = 1 if the realised value fell outside the level-alpha_t set and 0
// otherwise. Because alpha_t stays bounded, the cumulative-error telescopes and
// the long-run miscoverage is pinned to the target regardless of the shift:
//
//	| (1/T) sum_t err_t  -  alpha |  <=  ( alpha_1 + gamma ) / ( gamma * T ).
//
// gamma trades adaptation speed against steady-state variance: larger gamma
// reacts faster to a shift but lets the realised level swing more.
//
// ACI is agnostic to the base predictor — it only adapts the LEVEL. Pair it with
// any conformal threshold method (e.g. SplitQuantile at Level()). ACIStream wires
// it to a rolling SplitQuantile for a self-contained streaming predictor.

// ACI is the online level controller. It holds an unclamped internal level
// alpha_t (so it retains integral memory when a set is pushed to the [0,1] edge)
// and exposes the clamped, usable Level(). Not safe for concurrent use.
type ACI struct {
	alpha  float64 // target miscoverage
	gamma  float64 // step size
	alphaT float64 // current (unclamped) level
	t      int
}

// NewACI constructs a controller for target miscoverage alpha in (0,1) and step
// size gamma > 0. alpha_1 is initialised to the target.
func NewACI(alpha, gamma float64) (*ACI, error) {
	if !(alpha > 0 && alpha < 1) || math.IsNaN(alpha) {
		return nil, ErrInvalidAlpha
	}
	if !(gamma > 0) || math.IsInf(gamma, 0) || math.IsNaN(gamma) {
		return nil, errors.New("conformal: ACI gamma must be positive and finite")
	}
	return &ACI{alpha: alpha, gamma: gamma, alphaT: alpha}, nil
}

// Level returns the current usable miscoverage level, clamped to [0,1]. A clamp
// to 0 means "cover everything this step"; a clamp to 1 means "empty set".
func (a *ACI) Level() float64 {
	if a.alphaT < 0 {
		return 0
	}
	if a.alphaT > 1 {
		return 1
	}
	return a.alphaT
}

// RawLevel returns the unclamped internal level alpha_t (for inspection/tests).
func (a *ACI) RawLevel() float64 { return a.alphaT }

// Update applies the ACI recursion given whether the most recent realised value
// was miscovered by the level-Level() set, and returns the new Level().
func (a *ACI) Update(miscovered bool) float64 {
	var err float64
	if miscovered {
		err = 1
	}
	a.alphaT += a.gamma * (a.alpha - err)
	a.t++
	return a.Level()
}

// Step returns the number of Update calls so far.
func (a *ACI) Step() int { return a.t }

// ACIStream is a self-contained streaming conformal predictor: a rolling window
// of the most recent nonconformity scores supplies SplitQuantile, and an ACI
// controller adapts the level so coverage holds under drift. Scores must be
// non-negative and finite (e.g. absolute residuals). Not safe for concurrent use.
type ACIStream struct {
	aci    *ACI
	window int
	scores []float64 // most-recent-last ring (bounded by window)
}

// NewACIStream constructs a streaming predictor with target alpha, step gamma,
// and a rolling calibration window of the given size (> 0).
func NewACIStream(alpha, gamma float64, window int) (*ACIStream, error) {
	if window < 1 {
		return nil, errors.New("conformal: ACIStream window must be >= 1")
	}
	aci, err := NewACI(alpha, gamma)
	if err != nil {
		return nil, err
	}
	return &ACIStream{aci: aci, window: window, scores: make([]float64, 0, window)}, nil
}

// Threshold returns the current conformal threshold at the controller's level
// from the scores seen so far: +Inf when the level has been driven to 0 (cover
// everything) or the window cannot yet support the quantile, and a finite score
// otherwise. A value is covered iff its score <= Threshold.
func (s *ACIStream) Threshold() (float64, error) {
	lvl := s.aci.Level()
	if lvl <= 0 || len(s.scores) == 0 {
		return math.Inf(1), nil
	}
	if lvl >= 1 {
		return math.Inf(-1), nil // empty set: nothing is covered
	}
	return SplitQuantile(s.scores, lvl)
}

// Observe processes one nonconformity score: it computes the current threshold,
// determines coverage, updates the ACI level on the miscoverage indicator, then
// adds the score to the rolling window. Returns the threshold used and whether
// the score was covered. Scores must be non-negative and finite.
func (s *ACIStream) Observe(score float64) (threshold float64, covered bool, err error) {
	if math.IsNaN(score) || math.IsInf(score, 0) || score < 0 {
		return 0, false, ErrInvalidScore
	}
	threshold, err = s.Threshold()
	if err != nil {
		return 0, false, err
	}
	covered = score <= threshold
	s.aci.Update(!covered)
	s.push(score)
	return threshold, covered, nil
}

// Level exposes the controller's current clamped level.
func (s *ACIStream) Level() float64 { return s.aci.Level() }

// Step exposes the number of observations processed.
func (s *ACIStream) Step() int { return s.aci.Step() }

func (s *ACIStream) push(score float64) {
	if len(s.scores) < s.window {
		s.scores = append(s.scores, score)
		return
	}
	// Drop oldest (index 0), shift left, append. Window sizes here are small.
	copy(s.scores, s.scores[1:])
	s.scores[len(s.scores)-1] = score
}
