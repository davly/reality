package retrymath

import "math"

// This file holds the retry-storm LOAD-AMPLIFICATION calculus and the stability
// predicate that composes with reality/queue.
//
// # The amplification model
//
// A client issues one logical request but retries up to k total attempts, each
// attempt failing independently with probability p ∈ [0,1] (a per-attempt
// failure/timeout rate). Under "retry until success or k attempts", attempt i
// (1-based) is made iff attempts 1..i-1 all failed, which happens with
// probability p^(i-1). The EXPECTED number of attempts actually sent to the
// server per logical request is therefore
//
//	A(p,k) = sum_{i=0}^{k-1} p^i = (1 - p^k) / (1 - p)     (p != 1)
//	A(1,k) = k
//
// A(p,k) is the load AMPLIFICATION factor: at p=0 it is 1 (no retries fire), and
// it climbs toward k as p -> 1. If the offered logical arrival rate is lambda,
// the server actually sees an effective rate lambda_eff = lambda * A(p,k). This
// is the mechanism behind metastable retry storms: a latency blip raises p,
// which raises lambda_eff, which raises latency further.
//
// References:
//   - Standard finite-geometric-series identity.
//   - Bronson et al., "Metastable Failures in Distributed Systems," HotOS 2021.

// AmplificationFactor returns A(p,k) = (1 - p^k)/(1 - p), the expected number of
// server-visible attempts per logical request when each of up to k attempts
// fails independently with probability p. Equivalently, the factor by which a
// retry policy multiplies offered load.
//
//   - p is the per-attempt failure probability, in [0,1].
//   - k is the maximum number of attempts, k >= 1 (k == 1 means no retries, A == 1).
//
// The p -> 1 limit is handled exactly (A(1,k) = k). Panics if p is outside [0,1]
// or k < 1.
//
// Precision: the only transcendental is math.Pow(p, k); documented golden
// tolerance 1e-12.
func AmplificationFactor(p float64, k int) float64 {
	if p < 0 || p > 1 {
		panic("retrymath: p must be in [0,1]")
	}
	if k < 1 {
		panic("retrymath: k must be >= 1")
	}
	if p == 1 {
		return float64(k)
	}
	return (1 - math.Pow(p, float64(k))) / (1 - p)
}

// ExpectedAttempts is an alias for AmplificationFactor with the attempt-count
// reading: it returns the expected number of attempts a single logical request
// consumes under the same model. (Amplification factor and expected attempts are
// numerically identical; the two names document the two uses — load scaling vs
// per-request cost.)
func ExpectedAttempts(p float64, k int) float64 {
	return AmplificationFactor(p, k)
}

// EffectiveArrivalRate returns lambda_eff = lambda * A(p,k), the request rate a
// downstream service actually sees once retries are folded into an offered
// logical arrival rate lambda. This is the value a caller must feed into a
// queueing model (e.g. reality/queue.MM1) instead of the raw observed rate —
// under partial failure the raw rate systematically understates offered load.
//
// lambda must be >= 0. Panics on invalid p / k (via AmplificationFactor).
func EffectiveArrivalRate(lambda, p float64, k int) float64 {
	if lambda < 0 {
		panic("retrymath: lambda must be >= 0")
	}
	return lambda * AmplificationFactor(p, k)
}

// EffectiveUtilization returns the retry-amplified utilization
// rho_eff = lambda_eff / (servers * mu), where lambda_eff = lambda * A(p,k).
// It is the drop-in replacement for the naive rho = lambda/(c*mu) that a
// capacity check computes when it ignores retries. rho_eff >= 1 means the
// service cannot keep up with the retry-inflated load — the metastable-storm
// regime.
//
//   - lambda: offered logical arrival rate (>= 0).
//   - mu:     per-server service rate (> 0).
//   - servers: number of servers c (>= 1).
//   - p, k:   the retry policy's per-attempt failure prob and max attempts.
//
// Panics if mu <= 0 or servers < 1 (or on invalid p/k/lambda).
func EffectiveUtilization(lambda, mu float64, servers int, p float64, k int) float64 {
	if mu <= 0 {
		panic("retrymath: mu must be positive")
	}
	if servers < 1 {
		panic("retrymath: servers must be >= 1")
	}
	lambdaEff := EffectiveArrivalRate(lambda, p, k)
	return lambdaEff / (float64(servers) * mu)
}

// StableUnderRetries reports whether a service stays below saturation once retry
// amplification is accounted for: it returns true iff rho_eff < threshold, where
// rho_eff is EffectiveUtilization and threshold is the caller's saturation bar
// (e.g. sentinel's SaturationThreshold = 0.85). This is the metastable-failure
// guard: a naive check on the un-amplified rate can report "Healthy" while the
// retry-inflated load is already unstable.
//
// threshold must be in (0,1]. Panics on invalid threshold or on the same domain
// errors as EffectiveUtilization.
func StableUnderRetries(lambda, mu float64, servers int, p float64, k int, threshold float64) bool {
	if threshold <= 0 || threshold > 1 {
		panic("retrymath: threshold must be in (0,1]")
	}
	return EffectiveUtilization(lambda, mu, servers, p, k) < threshold
}
