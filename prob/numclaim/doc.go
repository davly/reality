// Package numclaim implements deterministic numeric-claim consistency: given a
// number a model (or any narrator) STATED in prose and the set of ground-truth
// numbers it was supposed to be grounded in, decide whether the stated number
// is EQUIVALENT to one of the truths under a small, explicit set of
// value-preserving transforms — rounding, percent<->fraction scaling, and
// exact/tolerance equality.
//
// # Why this exists in Reality
//
// portfolio-intelligence (a live £-revenue advice surface) ships an advice gate
// that checks the LANGUAGE of a model narrative but not its ARITHMETIC. Its own
// LIMITS.md §2 (lines 42-50) admits the hole verbatim:
//
//	"A model could state a number wrongly in prose ('your fees are £520' when
//	 the table says £52.60) ... the next rung is a deterministic
//	 number-consistency check (extract numerals from model prose, match against
//	 the JSON) — designed but not built in v1."
//
// The hard part of that rung is not extraction (a regex, which stays app-side)
// — it is numeric EQUIVALENCE: £52.60 vs 52.6 vs "about £53", 0.004 vs "0.4%",
// 7,500 vs 7500 must all MATCH, while the order-of-magnitude slip £520 vs
// £52.60 (or the decimal shift 526.0 vs 52.60) must NOT. That is
// language-neutral value math, so it belongs in Reality with one golden-file
// corpus feeding every consumer, rather than being re-implemented ad hoc in
// each. The estate already proves the duplication risk: the efficient-compute
// loop's V1 invented-value check hand-rolls a weaker `amounts_equal`
// (abs-diff < 0.01, no percent or rounding classes).
//
// # Consumers (this package wires none of them — golden files are the contract)
//
//   - portfolio-intelligence intelligence/gate.py — a third gate family
//     alongside directive/prediction: an unmatched prose numeral is a gate
//     violation, driving the same fail-closed corrective-retry-then-template
//     pipeline. Python port validated against this package's golden files.
//   - tools/verified-compute V1 invented-value check — replaces its ad hoc
//     amounts_equal with a port of the same equivalence classes.
//
// # API surface
//
//   - NumericEquivalent(claim, truth float64, opts Options) bool — does the
//     stated `claim` equal `truth` under the enabled transforms?
//   - ClaimConsistency(claims, truths []float64, opts Options) []Verdict — for
//     each claim, the matching truth + the transform used, or (unmatched) the
//     nearest-miss truth. The per-claim verdict a gate audits.
//   - Options / DefaultOptions() — MaxRoundDP, PercentScale, Tolerance.
//
// All functions are pure, deterministic, allocation-light, and use only the Go
// standard library ("math"). No parsing, no I/O: extraction of numerals from
// prose is the caller's job; this package operates on float64 inputs only.
//
// # Equivalence classes and their preference order
//
// For a claim c and truth t, a match is sought in this fixed order (the FIRST
// hit wins, so the reported transform is always the cheapest explanation):
//
//  1. exact:               |c - t| <= Tolerance
//  2. percent<->fraction:  |c/100 - t| <= Tolerance (percent_to_fraction) or
//     |c*100 - t| <= Tolerance (fraction_to_percent), only if PercentScale
//  3. rounded:             |round(t, dp) - c'| <= Tolerance for some
//     dp in [0, MaxRoundDP], where c' is c under the identity scale first,
//     then each percent scale (the claim is a ROUNDED rendering of the truth).
//     dp is scanned ascending, so the coarsest rounding that explains the
//     claim is reported.
//
// The ×10 / ÷10 error class is deliberately NOT a transform, so 526.0 vs 52.60
// and £520 vs £52.60 are correctly reported as mismatches — that is the whole
// point of the LIMITS.md example.
//
// # Precision
//
//   - round(x, dp) uses math.Round (round half away from zero) after scaling by
//     10^dp; dp <= 0 rounds to the nearest integer. Base-2 float
//     representation means values like 0.1 are not exact, so comparisons use
//     the absolute Tolerance (default 1e-9), never == on floats.
//   - Non-finite inputs (NaN, ±Inf) never match anything: NumericEquivalent
//     returns false and ClaimConsistency reports unmatched.
//   - Tolerance is an ABSOLUTE bound on the compared (possibly scaled) values.
//     A negative Tolerance is clamped to 0.
//
// # Golden-file provenance
//
// The golden vectors in testdata/ are worked from the LIMITS.md example and its
// adversarial near-misses (52.60 vs 526.0, £520 vs £52.60, 52.61 vs 52.60),
// plus the percent<->fraction cases the portfolio-intelligence narrative
// actually produces (weights 0.45 stated as "45%", expense ratios 0.0003 stated
// as "0.03%"). They are a language-neutral CONTRACT over the deterministic
// classification fields (matched, transform, roundDP, matched/nearest truth);
// only the Go implementation ships in this repository. RelError is an advisory
// derived float and is validated by the Go unit tests, not the shared golden
// contract.
package numclaim
