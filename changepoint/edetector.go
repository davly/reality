package changepoint

import (
	"errors"
	"math"
)

// E-detectors: nonparametric, anytime-valid sequential change detection.
//
// This file implements the e-detector framework of Shin, Ramdas & Rinaldo
// (2024), "E-detectors: a nonparametric framework for sequential change
// detection" (arXiv:2203.03532), built on the betting e-values of
// Waudby-Smith & Ramdas (2023, JRSS-B). It is the frequentist, false-alarm-
// controlled companion to this package's Bayesian BOCPD: where BOCPD reports a
// run-length posterior, an e-detector raises a single alarm with a hard,
// assumption-light false-alarm guarantee that holds at every stopping time.
//
// # Why "anytime-valid"
//
// An e-value for a null H0 is a non-negative statistic e with E_{H0}[e] <= 1.
// The running product of (conditionally) valid e-values is a test martingale
// (EProcess); by Ville's inequality, for any threshold 1/alpha,
//
//	P_{H0}( sup_t  E_t >= 1/alpha ) <= alpha.
//
// So thresholding the e-process at 1/alpha gives a level-alpha test that may be
// peeked at after every observation without inflating the error — no fixed
// sample size, no exchangeability or Gaussianity assumption beyond a bounded
// observation range. The CUSUM-style EDetector resets the process so it detects
// a change at an *unknown* time while controlling the average run length to
// false alarm (ARL >= 1/alpha).
//
// State per observation is O(1) (a single log-accumulator), unlike BOCPD's
// O(R_max) run-length vector.

// EValueFunc maps an observation to an e-value for a fixed simple/one-sided
// null. The contract the detectors rely on is:
//
//	e(x) >= 0          for every reachable x, and
//	E_{H0}[ e(X) ] <= 1   under the null.
//
// BettingEValue constructs a valid EValueFunc for a bounded-support mean null.
// Callers may supply their own (e.g. a likelihood ratio) provided the contract
// holds; a function that violates E_{H0}[e] <= 1 voids the false-alarm
// guarantee.
type EValueFunc func(x float64) float64

// BettingEValue returns a betting e-value for the null H0: E[X] = mu0, valid
// for observations X in [lo, hi]. Each factor is
//
//	e(x) = 1 + lambda * ( norm(x) - norm(mu0) ),   norm(z) = (z-lo)/(hi-lo),
//
// which satisfies E_{H0}[e(X)] = 1 exactly (any X with the null mean) and
// e(x) >= 0 for all x in [lo, hi] when lambda lies in the admissible range
//
//	-1/(1-m0)  <=  lambda  <=  1/m0,   m0 = norm(mu0).
//
// A positive lambda bets on an upward mean shift (powerful against X > mu0); a
// negative lambda bets downward. |lambda| trades early power against a larger
// drawdown if the bet is wrong. Returns an error if the arguments are
// non-finite, lo >= hi, mu0 is not strictly inside (lo, hi), or lambda is
// outside the admissible range (which would allow a negative e-value).
func BettingEValue(mu0, lambda, lo, hi float64) (EValueFunc, error) {
	for _, v := range []float64{mu0, lambda, lo, hi} {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return nil, errors.New("changepoint: BettingEValue requires finite arguments")
		}
	}
	if !(lo < hi) {
		return nil, errors.New("changepoint: BettingEValue requires lo < hi")
	}
	if !(mu0 > lo && mu0 < hi) {
		return nil, errors.New("changepoint: BettingEValue requires lo < mu0 < hi")
	}
	width := hi - lo
	m0 := (mu0 - lo) / width
	// Admissible bet range keeping e(x) >= 0 over x in [lo, hi].
	lambdaMax := 1.0 / m0
	lambdaMin := -1.0 / (1.0 - m0)
	if lambda < lambdaMin || lambda > lambdaMax {
		return nil, errors.New("changepoint: BettingEValue lambda outside admissible range [-1/(1-m0), 1/m0]")
	}
	return func(x float64) float64 {
		xn := (x - lo) / width
		// Clamp to the support so an out-of-range observation cannot produce a
		// negative e-value and silently void the guarantee.
		if xn < 0 {
			xn = 0
		} else if xn > 1 {
			xn = 1
		}
		e := 1.0 + lambda*(xn-m0)
		if e < 0 {
			e = 0
		}
		return e
	}, nil
}

// logEValue returns the natural log of e, mapping a non-positive (collapsed)
// e-value to -Inf so the accumulators handle martingale death cleanly.
func logEValue(e float64) float64 {
	if e <= 0 {
		return math.Inf(-1)
	}
	return math.Log(e)
}

// thresholdLog converts a false-alarm level alpha in (0,1) to the log-threshold
// log(1/alpha) that the log-statistic is compared against.
func thresholdLog(alpha float64) (float64, error) {
	if !(alpha > 0 && alpha < 1) {
		return 0, errors.New("changepoint: alpha must be in (0, 1)")
	}
	return -math.Log(alpha), nil
}

// EProcess is a non-resetting test martingale E_t = prod_{i<=t} e(x_i). It tests
// the fixed null over the whole stream: by Ville's inequality the probability it
// ever crosses 1/alpha under H0 is at most alpha. Use it to ask "has the stream
// deviated from the null at all", with continuous monitoring.
//
// Value is tracked in the log domain to avoid overflow; E_t can grow
// exponentially. Not safe for concurrent use.
type EProcess struct {
	ev   EValueFunc
	logE float64 // log E_t; starts at 0 (E_0 = 1)
	t    int
}

// NewEProcess constructs a test-martingale e-process from a valid e-value
// function. Returns an error if ev is nil.
func NewEProcess(ev EValueFunc) (*EProcess, error) {
	if ev == nil {
		return nil, errors.New("changepoint: NewEProcess requires a non-nil EValueFunc")
	}
	return &EProcess{ev: ev}, nil
}

// Update folds in one observation and returns the current value E_t. Once E_t
// reaches 0 (a factor was 0) it stays 0: the martingale is dead.
func (p *EProcess) Update(x float64) float64 {
	p.t++
	p.logE += logEValue(p.ev(x))
	return p.Value()
}

// Value returns the current e-process value E_t = exp(logE). Returns +Inf if it
// has overflowed the float64 range (still a valid "fired" signal).
func (p *EProcess) Value() float64 { return math.Exp(p.logE) }

// LogValue returns log E_t directly (overflow-safe).
func (p *EProcess) LogValue() float64 { return p.logE }

// Fired reports whether E_t has reached the level-alpha threshold 1/alpha, i.e.
// the null is rejected at anytime-valid level alpha. Returns an error if alpha
// is not in (0,1).
func (p *EProcess) Fired(alpha float64) (bool, error) {
	thr, err := thresholdLog(alpha)
	if err != nil {
		return false, err
	}
	return p.logE >= thr, nil
}

// Step returns the number of observations folded in so far.
func (p *EProcess) Step() int { return p.t }

// EDetector is a CUSUM-style e-detector for a change at an unknown time. It runs
// the recursion
//
//	W_t = max(1, W_{t-1}) * e(x_t),   W_0 = 1,
//
// (in the log domain logW_t = max(0, logW_{t-1}) + log e(x_t)), and raises an
// alarm the first time W_t >= 1/alpha. The max(1, .) reset starts a fresh
// e-process whenever the accumulated evidence falls back below 1, so the
// statistic tracks the most recent run of anomalous observations rather than the
// whole history. Under the null the alarm time tau = inf{t : W_t >= 1/alpha}
// satisfies the average-run-length bound E_inf[tau] >= 1/alpha.
//
// State is O(1). Not safe for concurrent use.
type EDetector struct {
	ev       EValueFunc
	logW     float64 // log W_t; starts at 0 (W_0 = 1)
	t        int
	fireTime int // 1-based step of first alarm; 0 means not yet fired
}

// NewEDetector constructs a CUSUM e-detector from a valid e-value function.
// Returns an error if ev is nil.
func NewEDetector(ev EValueFunc) (*EDetector, error) {
	if ev == nil {
		return nil, errors.New("changepoint: NewEDetector requires a non-nil EValueFunc")
	}
	return &EDetector{ev: ev}, nil
}

// Update folds in one observation and returns the current statistic W_t. The
// reset max(1, W_{t-1}) is applied before multiplying by the new e-value.
func (d *EDetector) Update(x float64) float64 {
	d.t++
	base := d.logW
	if base < 0 {
		base = 0 // max(1, W_{t-1}) in the log domain
	}
	d.logW = base + logEValue(d.ev(x))
	if d.logW < 0 {
		// Floor at log W = 0 is NOT applied here: the reset happens on the next
		// step via max(0, .). We keep the true (possibly negative) logW so a
		// single weak observation can be recovered from. But guard -Inf so a
		// dead step resets cleanly next round.
		if math.IsInf(d.logW, -1) {
			d.logW = math.Inf(-1)
		}
	}
	return d.Value()
}

// Value returns the current statistic W_t = exp(logW).
func (d *EDetector) Value() float64 { return math.Exp(d.logW) }

// LogValue returns log W_t directly (overflow-safe).
func (d *EDetector) LogValue() float64 { return d.logW }

// Fired reports whether the detector has crossed the 1/alpha threshold at the
// current step. Returns an error if alpha is not in (0,1).
func (d *EDetector) Fired(alpha float64) (bool, error) {
	thr, err := thresholdLog(alpha)
	if err != nil {
		return false, err
	}
	if d.logW >= thr && d.fireTime == 0 {
		d.fireTime = d.t
	}
	return d.logW >= thr, nil
}

// FireTime returns the 1-based step at which the detector first crossed the
// threshold passed to Fired, or 0 if it has not fired. FireTime is only updated
// by calls to Fired, so call Fired once per step to track the first alarm.
func (d *EDetector) FireTime() int { return d.fireTime }

// Step returns the number of observations folded in so far.
func (d *EDetector) Step() int { return d.t }
