package retrymath

import "math"

// This file holds the CLOSED-FORM delay moments of the jitter families. For a
// draw u ~ Uniform[0,1) the per-attempt delay of every family EXCEPT
// decorrelated is an affine image a + s*u of the capped-exponential term
// e(n) = min(cap, base*factor^n), hence uniform on an interval [lo, hi] with
//
//	mean      = (lo + hi) / 2
//	variance  = (hi - lo)^2 / 12          (variance of a uniform)
//	quantile(q) = lo + q*(hi - lo)         (the inverse CDF at level q)
//
// The interval endpoints per family (all as multiples of e = e(n)):
//
//	None            : [e,          e]          (degenerate; variance 0)
//	Full            : [0,          e]
//	Equal           : [e/2,        e]
//	Multiplicative  : [0.5 e,      1.5 e]
//	Symmetric(fr)   : [(1-fr) e,   (1+fr) e]
//	ReduceOnly(fr)  : [(1-fr) e,   e]
//
// Decorrelated jitter is a stochastic recurrence with NO simple closed form; it
// gets rigorous bounds (DecorrelatedUncappedMean) rather than an exact moment,
// and its schedule is validated by golden-file replay of a fixed u-sequence.
//
// Reference: any standard result for the continuous uniform distribution —
// e.g. Casella & Berger, "Statistical Inference," 2nd ed., §3.3.

// Family enumerates the jitter families with affine (hence uniform) per-attempt
// delay distributions. It exists so ExpectedDelay / DelayVariance / DelayQuantile
// / ExpectedTotalDelay can be written once over the endpoint algebra above.
// Decorrelated is intentionally NOT a member because it has no closed-form
// moment (use DecorrelatedUncappedMean for its bound).
type Family int

const (
	// None is the deterministic capped-exponential schedule (no jitter).
	None Family = iota
	// Full is Full jitter: delay ~ U(0, e).
	Full
	// Equal is Equal jitter: delay ~ U(e/2, e).
	Equal
	// Multiplicative is the [0.5,1.5) multiplicative band: delay ~ U(0.5e, 1.5e).
	Multiplicative
)

// interval returns the [lo, hi] delay endpoints for a fixed-shape family (None,
// Full, Equal, Multiplicative) at the given attempt, as absolute delays.
func (fam Family) interval(base, cap, factor float64, attempt int) (lo, hi float64) {
	e := CappedExponentialTerm(base, cap, factor, attempt)
	switch fam {
	case None:
		return e, e
	case Full:
		return 0, e
	case Equal:
		return e / 2, e
	case Multiplicative:
		return 0.5 * e, 1.5 * e
	default:
		panic("retrymath: unknown Family")
	}
}

// ExpectedDelay returns the mean per-attempt delay E[delay_n] for a fixed-shape
// family at attempt n. Mean of a uniform on [lo, hi] is (lo+hi)/2.
func ExpectedDelay(fam Family, base, cap, factor float64, attempt int) float64 {
	lo, hi := fam.interval(base, cap, factor, attempt)
	return (lo + hi) / 2
}

// DelayVariance returns the variance of the per-attempt delay for a fixed-shape
// family at attempt n. Variance of a uniform on [lo, hi] is (hi-lo)^2/12.
func DelayVariance(fam Family, base, cap, factor float64, attempt int) float64 {
	lo, hi := fam.interval(base, cap, factor, attempt)
	w := hi - lo
	return w * w / 12
}

// DelayQuantile returns the q-quantile (inverse CDF at level q ∈ [0,1]) of the
// per-attempt delay for a fixed-shape family at attempt n: lo + q*(hi-lo).
// q is clamped to [0,1]; e.g. DelayQuantile(Full, ..., 0.99) is the p99 wait.
func DelayQuantile(fam Family, base, cap, factor float64, attempt int, q float64) float64 {
	if q < 0 {
		q = 0
	}
	if q > 1 {
		q = 1
	}
	lo, hi := fam.interval(base, cap, factor, attempt)
	return lo + q*(hi-lo)
}

// ExpectedSymmetricDelay returns the mean per-attempt delay of the Symmetric(fr)
// family: the band is centred on e(n) so the mean is exactly e(n), independent
// of fr. (Provided separately because Symmetric carries the extra fraction
// parameter and so is not a bare Family member.)
func ExpectedSymmetricDelay(base, cap, factor float64, attempt int) float64 {
	return CappedExponentialTerm(base, cap, factor, attempt)
}

// ExpectedReduceOnlyDelay returns the mean per-attempt delay of the
// ReduceOnly(fr) family: U((1-fr)e, e) has mean e*(1 - fr/2).
func ExpectedReduceOnlyDelay(base, cap, factor, fraction float64, attempt int) float64 {
	if fraction < 0 || fraction > 1 {
		panic("retrymath: fraction must be in [0,1]")
	}
	return CappedExponentialTerm(base, cap, factor, attempt) * (1 - fraction/2)
}

// ExpectedTotalDelay returns the expected SUM of per-attempt delays across
// attempts n = 0 .. k-1 for a fixed-shape family — the mean total time a caller
// spends waiting if it exhausts all k backoffs. Because expectation is linear
// this is exactly the sum of the per-attempt means, capping included.
//
// k is the number of backoff intervals (retries). k <= 0 returns 0.
func ExpectedTotalDelay(fam Family, base, cap, factor float64, k int) float64 {
	total := 0.0
	for n := 0; n < k; n++ {
		total += ExpectedDelay(fam, base, cap, factor, n)
	}
	return total
}

// DecorrelatedUncappedMean returns the exact mean of the UNCAPPED decorrelated
// recurrence after `attempt` steps, seeded at prev == base:
//
//	m_0 = base,   m_{n+1} = E[U(base, 3 m_n)] = base/2 + (3/2) m_n
//
// Solving the linear recurrence gives the closed form
//
//	m_n = base * (2 * 1.5^n - 1).
//
// This is a rigorous UPPER BOUND on the mean of the CAPPED decorrelated schedule
// (capping can only lower a delay), and together with the trivial lower bound
// mean >= base (every capped draw is >= base while cap >= base) it brackets the
// true mean, which has no elementary closed form. Use it to size a cap or to
// bound expected total wait; do NOT treat it as the exact capped mean once
// 1.5^n * 2 * base approaches the cap.
//
// Panics if base <= 0. Negative attempt is treated as 0.
func DecorrelatedUncappedMean(base float64, attempt int) float64 {
	if base <= 0 {
		panic("retrymath: base must be positive")
	}
	if attempt < 0 {
		attempt = 0
	}
	return base * (2*math.Pow(1.5, float64(attempt)) - 1)
}
