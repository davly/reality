package prob

import (
	"errors"
	"sort"
)

// Venn-Abers predictors (Vovk & Petej 2014): distribution-free, automatically
// CALIBRATED probability estimates. Where a raw score or a single isotonic fit
// gives a point probability with no calibration guarantee, a Venn-Abers predictor
// outputs a multiprobability interval [p0, p1] that is perfectly calibrated under
// exchangeability — and the interval width is an honest uncertainty signal. It is
// the calibration twin of conformal prediction, built on the package's existing
// IsotonicRegression (Pool-Adjacent-Violators).
//
// Given calibration pairs (score_i, label_i in {0,1}) and a test score s, the
// predictor fits isotonic regression twice on the calibration set augmented with
// the test point labelled 0 and labelled 1, reading off p0 and p1 at the test
// score. Always p0 <= p1; the calibrated point estimate is p1/(1-p0+p1).

// VennAbers holds a fitted calibration set, sorted by score so each prediction
// inserts the test point without re-sorting (O(n) per prediction). Build with
// NewVennAbers.
type VennAbers struct {
	cps []CalibrationPoint // calibration points sorted ascending by X (=score)
}

// NewVennAbers validates the calibration pairs and stores them sorted by score.
// scores and labels must be the same non-zero length and every label must be 0
// or 1.
func NewVennAbers(scores, labels []float64) (*VennAbers, error) {
	if len(scores) == 0 {
		return nil, errors.New("prob: VennAbers requires at least one calibration point")
	}
	if len(scores) != len(labels) {
		return nil, errors.New("prob: VennAbers scores and labels must have equal length")
	}
	for _, y := range labels {
		if y != 0 && y != 1 {
			return nil, errors.New("prob: VennAbers labels must be 0 or 1")
		}
	}
	cps := make([]CalibrationPoint, len(scores))
	for i := range scores {
		cps[i] = CalibrationPoint{X: scores[i], Y: labels[i]}
	}
	sort.Slice(cps, func(i, j int) bool { return cps[i].X < cps[j].X })
	return &VennAbers{cps: cps}, nil
}

// Predict returns the multiprobability interval [p0, p1] for a test score s. The
// interval is calibrated; its width reflects how much the calibration data
// constrains the probability at s.
func (v *VennAbers) Predict(s float64) (p0, p1 float64) {
	p0 = v.fitAt(s, 0)
	p1 = v.fitAt(s, 1)
	if p0 > p1 { // guard against numerical noise; theoretically p0 <= p1
		p0, p1 = p1, p0
	}
	return p0, p1
}

// PredictPoint returns the calibrated point probability p1/(1-p0+p1) (Vovk's
// minimax-optimal merge of the multiprobability into a single number).
func (v *VennAbers) PredictPoint(s float64) float64 {
	p0, p1 := v.Predict(s)
	return VennAbersPoint(p0, p1)
}

// VennAbersPoint merges a multiprobability [p0, p1] into the calibrated point
// estimate p1 / (1 - p0 + p1), which always lies in [0, 1] for 0 <= p0 <= p1 <= 1.
func VennAbersPoint(p0, p1 float64) float64 {
	den := 1 - p0 + p1
	if den == 0 {
		return 0
	}
	return p1 / den
}

// fitAt inserts (s, label) into the pre-sorted calibration set at its score
// position, runs isotonic regression, and returns the fitted value at the test
// point. No per-call sort — O(n) insert + O(n) isotonic.
func (v *VennAbers) fitAt(s, label float64) float64 {
	n := len(v.cps)
	// Insertion index: first calibration point with X > s (test point sorts after
	// equal-scored calibration points, matching the previous stable-sort order).
	idx := sort.Search(n, func(i int) bool { return v.cps[i].X > s })
	merged := make([]CalibrationPoint, 0, n+1)
	merged = append(merged, v.cps[:idx]...)
	merged = append(merged, CalibrationPoint{X: s, Y: label})
	merged = append(merged, v.cps[idx:]...)
	return IsotonicRegression(merged)[idx].Y
}
