package slo

import (
	"errors"
	"math"
)

// ErrEmptyPolicy is returned when a Policy has no windows to evaluate.
var ErrEmptyPolicy = errors.New("slo: policy has no windows")

// ErrObservationCount is returned when the number of observations passed to
// Evaluate does not match the number of windows in the policy.
var ErrObservationCount = errors.New("slo: observation count must match window count")

// Window is one severity tier of a multiwindow, multi-burn-rate alert: a long
// window and its short confirmation window, both of which must exceed the same
// burn-rate threshold for the tier to fire.
//
// Long and Short are durations in the SAME unit as the Policy period (seconds,
// hours, ...). BurnRateThreshold is the tier's BR* — normally derived with
// ThresholdBurnRate from a target budget fraction, the window, and the period.
type Window struct {
	// Long is the long (primary) window duration.
	Long float64
	// Short is the short (confirmation) window duration, typically Long/12.
	Short float64
	// BurnRateThreshold is the burn-rate threshold BR* this tier fires at.
	BurnRateThreshold float64
	// Severity is a free-form label (e.g. "page", "ticket") carried through to
	// the result for the caller's convenience; it does not affect any math.
	Severity string
}

// Observation is the measured burn rate over a Window's long and short windows,
// as computed by the caller (e.g. via BurnRate over each window's consumption).
type Observation struct {
	// LongBurnRate is the burn rate measured over the tier's long window.
	LongBurnRate float64
	// ShortBurnRate is the burn rate measured over the tier's short window.
	ShortBurnRate float64
}

// Fires reports whether this tier's multiwindow condition is met: BOTH the long
// and short measured burn rates are at or above the threshold. Requiring the
// short window as well suppresses alerts on spikes that have already recovered
// and shortens the reset time.
func (w Window) Fires(obs Observation) bool {
	return obs.LongBurnRate >= w.BurnRateThreshold &&
		obs.ShortBurnRate >= w.BurnRateThreshold
}

// Policy is a multiwindow, multi-burn-rate SLO alerting policy: an SLO target
// over a period, with an ordered list of severity Windows (most severe /
// fastest-burn first by convention).
type Policy struct {
	// SLO is the objective target in [0, 1), e.g. 0.999.
	SLO float64
	// Period is the SLO rolling period, in the same time unit as the windows.
	Period float64
	// Windows are the severity tiers, evaluated in order.
	Windows []Window
}

// Result is the outcome of evaluating a Policy against per-window observations.
type Result struct {
	// Fire is true if any tier's multiwindow condition is met.
	Fire bool
	// TierIndex is the index into Policy.Windows of the FIRST (most severe, by
	// the caller's ordering) firing tier, or -1 if none fired.
	TierIndex int
	// Severity is the firing tier's Severity label, or "" if none fired.
	Severity string
	// DetectionTime is how long into the incident the firing tier's LONG window
	// took to reach threshold, given the observed long-window burn rate:
	// Long * BurnRateThreshold / observedLongBurnRate. NaN if none fired.
	DetectionTime float64
	// BudgetSpentAtDetection is the fraction of the total error budget consumed
	// at the moment of detection. By construction this equals the tier's design
	// fraction f = BurnRateThreshold * Long / Period. NaN if none fired.
	BudgetSpentAtDetection float64
}

// Evaluate applies the policy to one Observation per window (aligned by index)
// and returns whether an alert fires, which tier, and the detection-time and
// budget-spent figures for that tier. Tiers are checked in slice order and the
// first firing tier wins, so callers should order Windows most-severe first.
//
// Evaluate returns ErrEmptyPolicy if the policy has no windows, ErrObservationCount
// if len(observations) != len(Windows), and ErrSLORange if the SLO is invalid.
func (p Policy) Evaluate(observations []Observation) (Result, error) {
	if len(p.Windows) == 0 {
		return Result{}, ErrEmptyPolicy
	}
	if len(observations) != len(p.Windows) {
		return Result{}, ErrObservationCount
	}
	if !(p.SLO >= 0 && p.SLO < 1) {
		return Result{}, ErrSLORange
	}
	if p.Period <= 0 {
		return Result{}, ErrNonPositivePeriod
	}

	res := Result{
		Fire:                   false,
		TierIndex:              -1,
		DetectionTime:          math.NaN(),
		BudgetSpentAtDetection: math.NaN(),
	}

	for i, w := range p.Windows {
		obs := observations[i]
		if !w.Fires(obs) {
			continue
		}
		res.Fire = true
		res.TierIndex = i
		res.Severity = w.Severity
		// Detection time from the long window at the observed long burn rate.
		dt, err := DetectionTime(w.Long, w.BurnRateThreshold, obs.LongBurnRate)
		if err != nil {
			return Result{}, err
		}
		res.DetectionTime = dt
		// Budget consumed at detection is the tier's design fraction f, which is
		// exactly BudgetFractionConsumed at the threshold burn rate over the long
		// window.
		bf, err := BudgetFractionConsumed(w.BurnRateThreshold, w.Long, p.Period)
		if err != nil {
			return Result{}, err
		}
		res.BudgetSpentAtDetection = bf
		break
	}
	return res, nil
}

// RecommendedTier names one row of the SRE Workbook's recommended multiwindow
// parameter table for a 30-day SLO.
type RecommendedTier struct {
	// BudgetFraction is the fraction of the total budget consumed at detection.
	BudgetFraction float64
	// LongWindow is the long window duration (in the caller's chosen time unit).
	LongWindow float64
	// Severity is the Workbook's suggested severity for this tier.
	Severity string
}

// RecommendedWindows returns the SRE Workbook Chapter 5 recommended three-tier
// multiwindow configuration for the given SLO period, in the period's own time
// unit. The tiers are (fraction of budget, long window as a fraction of a 30-day
// period, severity):
//
//	2%  in 1h  (period/720)  -> page
//	5%  in 6h  (period/120)  -> page
//	10% in 3d  (period/10)   -> ticket
//
// Each tier's threshold is f*P/w and its short window is w/12, so for a literal
// 30-day period the thresholds come out to the Workbook's 14.4, 6, and 1. The
// windows scale with the period so the same table applies to any period length.
// Returns an error for a non-positive period.
func RecommendedWindows(period float64) ([]Window, error) {
	if period <= 0 {
		return nil, ErrNonPositivePeriod
	}
	tiers := []RecommendedTier{
		{BudgetFraction: 0.02, LongWindow: period / 720, Severity: "page"},   // 1h of 30d
		{BudgetFraction: 0.05, LongWindow: period / 120, Severity: "page"},   // 6h of 30d
		{BudgetFraction: 0.10, LongWindow: period / 10, Severity: "ticket"},  // 3d of 30d
	}
	windows := make([]Window, len(tiers))
	for i, t := range tiers {
		br, err := ThresholdBurnRate(t.BudgetFraction, t.LongWindow, period)
		if err != nil {
			return nil, err
		}
		sw, err := ShortWindow(t.LongWindow)
		if err != nil {
			return nil, err
		}
		windows[i] = Window{
			Long:              t.LongWindow,
			Short:             sw,
			BurnRateThreshold: br,
			Severity:          t.Severity,
		}
	}
	return windows, nil
}
