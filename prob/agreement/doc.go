// Package agreement implements chance-corrected inter-rater agreement
// statistics: Cohen's kappa, Cohen's weighted kappa (linear + quadratic),
// Fleiss' kappa, and Krippendorff's alpha (nominal, ordinal, interval).
//
// # Why this exists in Reality
//
// The load-bearing decision number for post-Fable judge substitution in the
// efficient-compute direction memo — "kappa 0.88" — was computed by a
// hand-rolled 2x2 Cohen's kappa in a throwaway one-off JS script
// (reviews/EFFICIENT_COMPUTE_GOLD_2026-07-03/work/judge_agreement.js), and
// that memo commits to recomputing it on every future corpus replay. There
// was no kappa/alpha implementation anywhere in Reality or aicore — exactly
// the scattered-ad-hoc-math failure mode Reality exists to eliminate
// (ARCHITECTURE.md). The judge data is also graded/ordinal ("mean |grade
// diff| 0.45"), which a binary kappa throws away entirely; weighted kappa
// and Krippendorff's alpha (whose ordinal/interval difference functions
// natively use graded distance) are the statistically correct tools.
//
// Consumer: EFFICIENT_COMPUTE_GOLD judge-substitution pipeline and its
// monthly corpus-replay cron (reviews/EFFICIENT_COMPUTE_GOLD_2026-07-03/
// DIRECTION_MEMO.md); secondarily quorum's cross-model convergence scoring,
// whose missing/degenerate-value exclusion (quorum/aggregate.py) matches
// Krippendorff's native missing-data handling.
//
// This package does not wire either consumer — it is the Go-canonical
// reference the JS/Python replay scripts (and any future consumer in any
// language) validate against, per Reality's golden-file cross-language
// model (CONTEXT.md): Go is canonical, other substrates check their own
// arithmetic against the same worked examples.
//
// # API surface
//
//   - CohenKappa(a, b []int) — two-rater nominal chance-corrected agreement.
//   - WeightedKappa(a, b []int, scheme WeightScheme) — two-rater ORDINAL
//     agreement, crediting near-miss disagreement less than a full miss
//     (Linear or Quadratic rank-distance weights).
//   - FleissKappa(ratings [][]int) — nominal agreement among >2 raters.
//   - KrippendorffAlpha(data [][]float64, metric Metric) — any number of
//     raters, any of Nominal/Ordinal/Interval metrics, native missing-data
//     handling (NaN entries).
//
// All four are pure, deterministic, allocation-light closed-form
// computations over integer/float64 slices — no I/O, zero non-stdlib
// dependencies (only "errors", "math", "sort").
//
// # Golden-file provenance
//
// Every function's test suite reproduces a published worked example and
// asserts against it, rather than against a value only this package has
// ever produced:
//
//   - CohenKappa: Cohen (1960), the 50-grant-proposal 2x2 example as
//     reproduced in Wikipedia's "Cohen's kappa" article (kappa = 0.40).
//   - WeightedKappa: the k=2 reduction identity (both Linear and Quadratic
//     weights collapse to unweighted kappa when there are only 2
//     categories, so the Cohen (1960) example above must reproduce exactly
//     under either scheme), plus a hand-derived 3-category example
//     cross-checked by two independent, algebraically-equivalent formula
//     routes (Cohen 1968).
//   - FleissKappa: Fleiss (1971) Table 1, the 10-patient/14-psychiatrist/
//     5-diagnosis example, as reproduced in Wikipedia's "Fleiss' kappa"
//     article (kappa = 0.210).
//   - KrippendorffAlpha: Krippendorff (2011), "Computing Krippendorff's
//     Alpha-Reliability", the 4-observer/12-unit worked example with
//     missing data (nominal = 0.743, ordinal = 0.815, interval = 0.849),
//     plus its smaller 2-observer nominal (0.692) and binary (0.095)
//     examples.
//
// # References
//
//   - Cohen, J. (1960). A coefficient of agreement for nominal scales.
//     Educational and Psychological Measurement 20(1): 37-46.
//   - Cohen, J. (1968). Weighted kappa: Nominal scale agreement with
//     provision for scaled disagreement or partial credit. Psychological
//     Bulletin 70(4): 213-220.
//   - Fleiss, J. L. (1971). Measuring nominal scale agreement among many
//     raters. Psychological Bulletin 76(5): 378-382.
//   - Krippendorff, K. (2011). Computing Krippendorff's Alpha-Reliability.
//     https://www.asc.upenn.edu/sites/default/files/2021-03/Computing%20Krippendorff's%20Alpha-Reliability.pdf
//   - Hayes, A. F., & Krippendorff, K. (2007). Answering the call for a
//     standard reliability measure for coding data. Communication Methods
//     and Measures 1: 77-89.
package agreement
