// Package slo computes the multiwindow, multi-burn-rate SLO alerting algebra
// from the Google SRE Workbook, Chapter 5 ("Alerting on SLOs"), as closed-form
// deterministic arithmetic over primitive floats.
//
// A Service Level Objective (SLO) sets a target success ratio (e.g. 99.9%) over
// a rolling period (e.g. 30 days). The complement, 1 - SLO, is the ERROR BUDGET:
// the fraction of events that may fail before the objective is missed. The BURN
// RATE is how fast, relative to the SLO, the service is spending that budget:
//
//	burn rate 1  spends the WHOLE budget in exactly one SLO period
//	burn rate 2  spends it in half a period
//	burn rate 10 spends it in a tenth of a period, and so on.
//
// Formally, if a service consumes a fraction c of its TOTAL error budget during
// a trailing window of length w within an SLO period P, its burn rate over that
// window is
//
//	BR = c / (w / P)                                            (normalisation)
//
// so that BR = 1 exactly when c = w/P (spending the budget linearly across the
// period). Equivalently, from the raw observed failure ratio r over the window,
//
//	BR = r / (1 - SLO)                                          (rate form)
//
// # Multiwindow, multi-burn-rate alerting
//
// A good SLO alert fires FAST for outages that would exhaust the budget in
// minutes and SLOWLY (but still fires) for slow leaks that would take days. The
// SRE Workbook derives, for an alert that should trigger once a fraction f of the
// total budget has been consumed within a window w, the THRESHOLD burn rate
//
//	BR* = f * P / w                                             (threshold)
//
// and its exact inverse, the budget fraction consumed at burn rate BR over w,
//
//	f  = BR * w / P                                             (budget consumed)
//
// The two are algebraic inverses: ThresholdBurnRate and BudgetFractionConsumed
// round-trip to machine precision. Given an alert fires when the trailing-window
// burn rate reaches BR*, and the actual (constant) burn rate during an incident
// is BR_actual >= BR*, the trailing-window average ramps linearly, so the alert
// DETECTS the incident after
//
//	t_detect = w * BR* / BR_actual   (<= w; +Inf if BR_actual < BR*)
//
// and, symmetrically, after the incident stops the alert RESETS after
//
//	t_reset  = w * (1 - BR* / BR_actual)
//
// A subtle property falls out: the fraction of budget consumed AT the moment of
// detection equals the design fraction f regardless of BR_actual, because the
// alert is by construction the "f consumed in w" test.
//
// To keep alerts from firing on a spike that has already recovered and to
// shorten the reset time, the Workbook pairs each long window with a SHORT
// confirmation window equal to one twelfth of it (the "1/12 rule"); a tier fires
// only when BOTH windows exceed BR*.
//
// Finally, the time until the budget is fully exhausted at the current burn rate
// is the linear projection
//
//	t_exhaust = remaining_budget_fraction * P / BR   (+Inf if BR <= 0)
//
// # Recommended configuration
//
// The Workbook's recommended 30-day, three-tier policy is reproduced by these
// formulas exactly (see the RecommendedWindows table and the golden vectors in
// testdata/slo/): 2% of budget in 1h -> BR* 14.4 (page), 5% in 6h -> BR* 6
// (page), 10% in 3d -> BR* 1 (ticket), each with a long/12 short window.
//
// # Honesty
//
// These are informational SIGNALS derived from published arithmetic, not an
// autonomous paging or remediation decision — the choice of SLO, budget
// fractions, and windows, and the action taken on an alert, belong to the
// service owner. Domain violations (SLO outside [0,1), non-positive window or
// period) return a sentinel error rather than a silent wrong number. The finite
// mathematical limits "never detects" and "never exhausts" are reported honestly
// as +Inf, not as errors, because they are real answers.
//
// All functions operate over primitive float64 inputs and use only the Go
// standard library (math), preserving reality's zero-dependency law.
//
// References:
//   - Beyer, B. et al. (eds.), "The Site Reliability Workbook" (Google, O'Reilly
//     2018), Chapter 5, "Alerting on SLOs" — burn-rate definition, the
//     multiwindow multi-burn-rate method, the recommended 30-day parameter
//     table, and the 1/12 short-window rule.
//   - Beyer, B. et al. (eds.), "Site Reliability Engineering" (Google, O'Reilly
//     2016), Chapter 4, "Service Level Objectives" — error-budget definition.
package slo

import (
	"errors"
	"math"
)

var (
	// ErrSLORange is returned when the SLO target is outside [0, 1).
	ErrSLORange = errors.New("slo: SLO target must be in [0, 1)")
	// ErrNonPositiveWindow is returned when a window duration is <= 0.
	ErrNonPositiveWindow = errors.New("slo: window must be positive")
	// ErrNonPositivePeriod is returned when the SLO period is <= 0.
	ErrNonPositivePeriod = errors.New("slo: period must be positive")
	// ErrBudgetFractionRange is returned when a budget fraction is outside [0, 1].
	ErrBudgetFractionRange = errors.New("slo: budget fraction must be in [0, 1]")
	// ErrNegativeBurnRate is returned when a burn rate is negative.
	ErrNegativeBurnRate = errors.New("slo: burn rate must be non-negative")
)

// ShortWindowDivisor is the SRE Workbook "1/12 rule": the short confirmation
// window is the long window divided by twelve. It is exported so callers can see
// and, if they must, override the ratio; ShortWindow uses it.
const ShortWindowDivisor = 12.0

// ErrorBudget returns the error budget of an SLO target: the fraction of events
// that may fail while still meeting the objective, i.e. 1 - slo. For a 99.9% SLO
// (slo = 0.999) the budget is 0.001. Returns ErrSLORange if slo is not in [0,1).
func ErrorBudget(slo float64) (float64, error) {
	if !(slo >= 0 && slo < 1) {
		return math.NaN(), ErrSLORange
	}
	return 1 - slo, nil
}

// BurnRate normalises the fraction of the TOTAL error budget consumed during a
// trailing window into a period-relative burn rate:
//
//	BR = consumedFraction / (window / period)
//
// A burn rate of 1 means the budget is being spent exactly fast enough to run
// out at the end of one period. consumedFraction is a fraction of the whole
// budget in [0, 1] (values above 1 mean the window already overspent the entire
// budget and are permitted). Returns an error for a non-positive window or
// period, or a consumedFraction below 0.
func BurnRate(consumedFraction, window, period float64) (float64, error) {
	if window <= 0 {
		return math.NaN(), ErrNonPositiveWindow
	}
	if period <= 0 {
		return math.NaN(), ErrNonPositivePeriod
	}
	if consumedFraction < 0 {
		return math.NaN(), ErrBudgetFractionRange
	}
	return consumedFraction * period / window, nil
}

// BurnRateFromErrorRate converts an observed failure ratio r (bad events / total
// events over some window) into a burn rate relative to the SLO:
//
//	BR = r / (1 - slo)
//
// A service at exactly its SLO error rate burns at 1. Returns ErrSLORange if slo
// is not in [0,1) and ErrBudgetFractionRange if errorRate is outside [0,1].
func BurnRateFromErrorRate(errorRate, slo float64) (float64, error) {
	if !(slo >= 0 && slo < 1) {
		return math.NaN(), ErrSLORange
	}
	if !(errorRate >= 0 && errorRate <= 1) {
		return math.NaN(), ErrBudgetFractionRange
	}
	return errorRate / (1 - slo), nil
}

// ThresholdBurnRate derives the burn-rate alert threshold BR* for an alert that
// should fire once a fraction f of the TOTAL error budget has been consumed
// within a window w of an SLO period P:
//
//	BR* = f * P / w
//
// For example f=0.02 (2%), w=1h, P=30d gives BR*=14.4 — the Workbook's headline
// fast-burn page. Returns an error for f outside [0,1], or non-positive window or
// period.
func ThresholdBurnRate(budgetFraction, window, period float64) (float64, error) {
	if !(budgetFraction >= 0 && budgetFraction <= 1) {
		return math.NaN(), ErrBudgetFractionRange
	}
	if window <= 0 {
		return math.NaN(), ErrNonPositiveWindow
	}
	if period <= 0 {
		return math.NaN(), ErrNonPositivePeriod
	}
	return budgetFraction * period / window, nil
}

// BudgetFractionConsumed is the exact inverse of ThresholdBurnRate: the fraction
// of the total error budget consumed by sustaining burn rate BR over a window w
// of period P:
//
//	f = BR * w / P
//
// ThresholdBurnRate(BudgetFractionConsumed(BR, w, P), w, P) == BR to machine
// precision. Returns an error for a negative burn rate or non-positive window or
// period.
func BudgetFractionConsumed(burnRate, window, period float64) (float64, error) {
	if burnRate < 0 {
		return math.NaN(), ErrNegativeBurnRate
	}
	if window <= 0 {
		return math.NaN(), ErrNonPositiveWindow
	}
	if period <= 0 {
		return math.NaN(), ErrNonPositivePeriod
	}
	return burnRate * window / period, nil
}

// DetectionTime returns how long after an incident begins a burn-rate alert on a
// trailing window of length w fires, given the threshold burn rate BR* and the
// actual (assumed constant) burn rate BR_actual during the incident. The
// trailing-window average ramps up linearly, reaching BR* after
//
//	t_detect = w * BR* / BR_actual
//
// which is at most w (when BR_actual == BR*, the window must fill completely) and
// shrinks as the incident burns hotter. If BR_actual < BR* the window never
// reaches the threshold and the honest answer is +Inf ("never detects"); the same
// holds for BR_actual <= 0. Returns an error for a non-positive window, negative
// threshold, or negative actual burn rate.
func DetectionTime(window, thresholdBurnRate, actualBurnRate float64) (float64, error) {
	if window <= 0 {
		return math.NaN(), ErrNonPositiveWindow
	}
	if thresholdBurnRate < 0 || actualBurnRate < 0 {
		return math.NaN(), ErrNegativeBurnRate
	}
	if actualBurnRate < thresholdBurnRate || actualBurnRate == 0 {
		return math.Inf(1), nil
	}
	return window * thresholdBurnRate / actualBurnRate, nil
}

// ResetTime returns how long after an incident STOPS a burn-rate alert on a
// trailing window of length w keeps firing before it clears, given the threshold
// BR* and the actual burn rate BR_actual that preceded the recovery (with the
// window assumed full). The trailing-window average decays linearly from
// BR_actual to 0, crossing back below BR* after
//
//	t_reset = w * (1 - BR* / BR_actual)
//
// This is why a long window has a long, sticky reset time and why the Workbook
// pairs it with a short confirmation window: the short window's much smaller w
// bounds the reset. If BR_actual <= BR* the alert was not firing to begin with,
// so the reset time is 0. Returns an error for a non-positive window, negative
// threshold, or negative actual burn rate.
func ResetTime(window, thresholdBurnRate, actualBurnRate float64) (float64, error) {
	if window <= 0 {
		return math.NaN(), ErrNonPositiveWindow
	}
	if thresholdBurnRate < 0 || actualBurnRate < 0 {
		return math.NaN(), ErrNegativeBurnRate
	}
	if actualBurnRate <= thresholdBurnRate {
		return 0, nil
	}
	return window * (1 - thresholdBurnRate/actualBurnRate), nil
}

// TimeToExhaustion projects how long until the error budget is fully spent at the
// current burn rate:
//
//	t_exhaust = remainingBudgetFraction * P / BR
//
// remainingBudgetFraction is the fraction of the TOTAL budget still available in
// [0, 1] (1 = untouched, 0 = already exhausted). The result is in the same time
// units as the period P. A non-positive burn rate never exhausts the budget, so
// the honest answer is +Inf. Returns an error for a remaining fraction outside
// [0,1] or a non-positive period.
func TimeToExhaustion(remainingBudgetFraction, burnRate, period float64) (float64, error) {
	if !(remainingBudgetFraction >= 0 && remainingBudgetFraction <= 1) {
		return math.NaN(), ErrBudgetFractionRange
	}
	if period <= 0 {
		return math.NaN(), ErrNonPositivePeriod
	}
	if burnRate <= 0 {
		return math.Inf(1), nil
	}
	return remainingBudgetFraction * period / burnRate, nil
}

// ShortWindow returns the SRE Workbook short confirmation window for a given long
// window using the 1/12 rule (long / ShortWindowDivisor). A 1h long window yields
// a 5-minute short window, 6h yields 30 minutes, 3d yields 6h. Returns an error
// for a non-positive long window.
func ShortWindow(longWindow float64) (float64, error) {
	if longWindow <= 0 {
		return math.NaN(), ErrNonPositiveWindow
	}
	return longWindow / ShortWindowDivisor, nil
}
