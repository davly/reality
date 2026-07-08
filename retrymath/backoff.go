// Package retrymath provides the PURE calculus of exponential-backoff-with-jitter
// retry policies: deterministic schedule generators, closed-form delay moments,
// and retry-storm load-amplification math that composes with reality/queue.
//
// # Scope: the math, never the loop
//
// This package deliberately holds ONLY the pure, deterministic, allocation-free
// arithmetic of a retry policy. The impure parts of a real retry loop — sleeping,
// context cancellation, RNG sourcing, error classification — stay in each
// service. A backoff schedule is a pure function of the attempt index and a
// single injected uniform draw u ∈ [0,1); this package computes exactly that,
// which makes every schedule reproducible and golden-file testable across
// languages. The caller owns time.After / context / crypto-rand; it passes the
// draw in and receives a delay out.
//
// All delays are dimensionless float64 "time units" (seconds by convention).
// The package never allocates and never touches the clock.
//
// # The jitter families
//
// Given attempt index n (0-based: n == 0 is the first backoff after the initial
// try), a base delay b > 0, a growth factor f >= 1 (2 by convention), and a cap
// c > 0, define the CAPPED exponential term
//
//	e(n) = min(c, b * f^n)
//
// The jitter families all draw a single u ∈ [0,1) and map e(n) to a delay:
//
//	None          : delay = e(n)                        (deterministic)
//	Full          : delay = u * e(n)                    ∈ [0, e(n))
//	Equal         : delay = e(n)/2 + u * e(n)/2         ∈ [e(n)/2, e(n))
//	Decorrelated  : delay = min(c, b + u*(3*prev - b))  (stateful recurrence)
//
// Full / Equal / Decorrelated are the three families named in Marc Brooker's
// canonical AWS analysis (see references). Full jitter minimises contention and
// completion-time variance; Equal jitter trades a little contention for a
// guaranteed minimum wait; Decorrelated jitter self-clocks off the previous
// sleep and is competitive with Full while never starving.
//
// Three additional families encode the exact conventions of the flagship
// hand-rolled copies this package consolidates, so those call sites become
// deterministic under test without changing their observable distribution:
//
//	Multiplicative(0.5) : delay = e(n) * (0.5 + u)          ∈ [0.5 e(n), 1.5 e(n))   (sentinel, conduit, aicore donor)
//	Symmetric(fr)       : delay = e(n) * (1 + fr*(2u-1))    ∈ [(1-fr)e(n), (1+fr)e(n))  (nexus ±JitterFraction)
//	ReduceOnly(fr)      : delay = e(n) * (1 - fr*u)         ∈ [(1-fr)e(n), e(n))     (switchyard reduce-by-fraction)
//
// # Why this exists in Reality
//
// Exponential backoff with jitter was independently reimplemented at least six
// times across the infrastructure fleet, each with a mutually inconsistent
// jitter convention and several sourcing crypto/rand inline so the schedules
// were impossible to unit test:
//
//   - sentinel/internal/resilient/retry.go:70   — multiplicative [0.5,1.5), crypto/rand inline
//   - conduit/internal/resilient/retry.go:62     — separate copy, same [0.5,1.5) convention
//   - aicore/resilient/retry.go:32               — the donor of the two above, also impure
//   - nexus/src/api/internal/ai/retry.go:53      — symmetric ±JitterFraction·(2u-1)
//   - switchyard/internal/retry/policy.go:30     — reduce-by-fraction, injectable RNG
//   - phantom/internal/ai/resilient.go           — jitterless (1<<attempt)*500ms
//
// None of them could answer the design questions a retry policy actually raises:
// expected total delay before giving up, delay quantiles, load amplification
// under partial failure, or whether the policy destabilises the very service it
// protects. This package factors out ONLY the pure calculus (the impure sleep
// loop stays put), exactly the EWMoments precedent where the online
// mean/variance recurrence was hoisted out of three flagship reinventions
// (see timeseries/ewvar.go "Why this exists"). Centralising the schedule math
// makes all six call sites goldenly testable against one another and unlocks the
// amplification / stability analysis in amplification.go.
//
// # References
//
//   - Brooker, Marc. "Exponential Backoff And Jitter." AWS Architecture Blog,
//     2015-03-04. Defines the Full, Equal, and Decorrelated jitter families and
//     shows Full/Decorrelated minimise contention and completion time.
//   - Bronson, N., Aghayev, A., Charapko, A., Zhu, T. "Metastable Failures in
//     Distributed Systems." HotOS 2021. Motivates the retry-storm amplification
//     and stability guard in amplification.go.
//   - AWS SDK "adaptive"/"standard" retry modes; the same capped-exponential +
//     jitter families are the industry default.
package retrymath

import "math"

// CappedExponentialTerm returns e(n) = min(cap, base * factor^n), the
// deterministic capped-exponential delay for attempt index n BEFORE any jitter
// is applied. It is the shared spine of every jitter family in this package.
//
// Parameters:
//   - base:    the initial delay b > 0 (the delay at attempt 0 pre-cap).
//   - cap:     the maximum per-attempt delay c > 0. Pass math.Inf(1) for "no cap".
//   - factor:  the growth factor f >= 1 (2 by convention; f == 1 is a constant
//     schedule).
//   - attempt: the 0-based attempt index n. Negative n is treated as 0 (matching
//     the switchyard call site), so the first backoff is always base.
//
// Precision: exact for integer factor and small n (the only transcendental is
// math.Pow(factor, n), evaluated exactly for factor == 2). Documented tolerance
// in the golden files is 1e-12.
//
// Panics if base <= 0, cap <= 0, or factor < 1 — these are domain errors, not
// runtime conditions the caller should paper over with a default.
func CappedExponentialTerm(base, cap, factor float64, attempt int) float64 {
	validateSchedule(base, cap, factor)
	if attempt < 0 {
		attempt = 0
	}
	term := base * math.Pow(factor, float64(attempt))
	if term > cap {
		return cap
	}
	return term
}

// FullJitter returns the Full-jitter delay for attempt n: u * e(n), uniformly
// distributed on [0, e(n)). This is Brooker's recommended default: it minimises
// both server contention and completion-time variance.
//
// u must lie in [0,1); it is the caller's single injected uniform draw. The
// result is a pure linear function of u — out-of-range u yields a proportionally
// out-of-range delay (a documented precondition, not a silently clamped one).
func FullJitter(base, cap, factor float64, attempt int, u float64) float64 {
	return u * CappedExponentialTerm(base, cap, factor, attempt)
}

// EqualJitter returns the Equal-jitter delay for attempt n:
// e(n)/2 + u*e(n)/2, uniformly distributed on [e(n)/2, e(n)). It guarantees a
// minimum wait of half the capped-exponential term while still spreading load.
//
// u must lie in [0,1). See FullJitter for the u contract.
func EqualJitter(base, cap, factor float64, attempt int, u float64) float64 {
	half := CappedExponentialTerm(base, cap, factor, attempt) / 2
	return half + u*half
}

// DecorrelatedJitter advances the decorrelated-jitter recurrence one step:
//
//	next = min(cap, base + u*(3*prev - base))   == min(cap, U(base, 3*prev))
//
// where prev is the PREVIOUS sleep (seed the recurrence with prev == base for
// the first step). Unlike the attempt-indexed families this is stateful: the
// next delay is clocked off the last one, not off n, which is what lets it grow
// smoothly toward the cap while never dropping below base. Because it is a
// stochastic recurrence it has NO simple closed-form moments (see moments.go for
// the uncapped-mean bound); its schedule is validated by golden-file replay of a
// fixed u-sequence instead.
//
// u must lie in [0,1). Panics on the same domain errors as
// CappedExponentialTerm (base <= 0, cap <= 0); factor is not used here.
func DecorrelatedJitter(base, cap, prev, u float64) float64 {
	if base <= 0 {
		panic("retrymath: base must be positive")
	}
	if cap <= 0 {
		panic("retrymath: cap must be positive")
	}
	// U(base, 3*prev): lower endpoint base, upper endpoint 3*prev. When
	// 3*prev < base (only possible if a caller seeds prev < base/3) the interval
	// inverts; guard by flooring the span at 0 so the draw never dips below base.
	span := 3*prev - base
	if span < 0 {
		span = 0
	}
	next := base + u*span
	if next > cap {
		return cap
	}
	return next
}

// MultiplicativeJitter returns e(n) * (0.5 + u), uniformly distributed on
// [0.5*e(n), 1.5*e(n)). This is the exact convention of the sentinel/conduit/
// aicore hand-rolled copies (jitter := 0.5 + rand; delay := e(n) * jitter).
//
// u must lie in [0,1). See FullJitter for the u contract.
func MultiplicativeJitter(base, cap, factor float64, attempt int, u float64) float64 {
	return CappedExponentialTerm(base, cap, factor, attempt) * (0.5 + u)
}

// SymmetricJitter returns e(n) * (1 + fraction*(2u-1)), uniformly distributed on
// [(1-fraction)*e(n), (1+fraction)*e(n)). This is the nexus convention
// (jitter := backoff * fraction * (2u-1); delay := backoff + jitter).
//
// fraction is the half-width of the symmetric band, in [0,1]; fraction == 0 is
// deterministic. u must lie in [0,1). Panics if fraction is outside [0,1].
func SymmetricJitter(base, cap, factor, fraction float64, attempt int, u float64) float64 {
	if fraction < 0 || fraction > 1 {
		panic("retrymath: fraction must be in [0,1]")
	}
	return CappedExponentialTerm(base, cap, factor, attempt) * (1 + fraction*(2*u-1))
}

// ReduceOnlyJitter returns e(n) * (1 - fraction*u), uniformly distributed on
// [(1-fraction)*e(n), e(n)). This is the switchyard convention (reduce the
// capped delay by up to a fraction — jitter only ever shortens the wait).
//
// fraction is the maximum reduction, in [0,1]; fraction == 0 is deterministic.
// u must lie in [0,1). Panics if fraction is outside [0,1].
func ReduceOnlyJitter(base, cap, factor, fraction float64, attempt int, u float64) float64 {
	if fraction < 0 || fraction > 1 {
		panic("retrymath: fraction must be in [0,1]")
	}
	return CappedExponentialTerm(base, cap, factor, attempt) * (1 - fraction*u)
}

// validateSchedule enforces the shared domain preconditions of the
// attempt-indexed families.
func validateSchedule(base, cap, factor float64) {
	if base <= 0 {
		panic("retrymath: base must be positive")
	}
	if cap <= 0 {
		panic("retrymath: cap must be positive")
	}
	if factor < 1 {
		panic("retrymath: factor must be >= 1 (backoff must not shrink)")
	}
}
