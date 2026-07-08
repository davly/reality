// Package agreement implements paired-design inference over the shared
// repeated-measures shape common to every eval harness in the estate (same
// items, two arms/judges/subjects): chance-corrected inter-rater agreement
// statistics — Cohen's kappa, Cohen's weighted kappa (linear + quadratic),
// Fleiss' kappa, and Krippendorff's alpha (nominal, ordinal, interval) — plus
// the two paired-difference hypothesis tests that operate on the same paired
// data: McNemar's exact/mid-p test for binary matched pairs, and the exact
// paired permutation (randomization) test for numeric matched pairs.
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
// Paired-difference hypothesis tests (do the two arms/judges DIFFER, the
// question kappa/alpha cannot answer):
//
//   - McNemarExact(b, c int) / McNemarMidP(b, c int) — exact conditional and
//     mid-p two-sided tests of marginal homogeneity for binary matched pairs,
//     given the two discordant cell counts (the only informative pairs).
//   - DiscordantCounts(x, y []int) — derives (b, c) from raw paired 0/1
//     slices (e.g. per-item correct/incorrect for two subjects).
//   - PairedPermutationTest(x, y []float64) — exact two-sided paired
//     permutation test by complete deterministic enumeration of all 2^n
//     sign-flips (no RNG, no seed); the assumption-free counterpart to the
//     paired t-test.
//
// All are pure, deterministic, allocation-light computations over
// integer/float64 slices — no I/O, zero non-stdlib dependencies (only
// "errors", "math", "math/big", "sort"). McNemarExact forms its dyadic
// tail probability with exact big-integer/rational arithmetic (single final
// float64 rounding, no 2^-n underflow); PairedPermutationTest enumerates the
// full null distribution exactly.
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
//   - McNemarExact / McNemarMidP: Fagerland, Lydersen & Laake (2013), the
//     airway hyper-responsiveness matched-pairs data, discordant counts
//     (1, 7): exact = 0.0703125, mid-p = 0.0390625.
//   - PairedPermutationTest: Darwin's Zea mays data as used by Fisher (1935),
//     15 paired height differences (sum = 314): 863 of 32768 arrangements
//     reach the observed one tail, so two-sided p = 1726/32768 = 0.052674
//     (also Ernst 2004); plus a fully hand-enumerable (1,2,3) case, p = 0.25.
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
//   - McNemar, Q. (1947). Note on the sampling error of the difference
//     between correlated proportions or percentages. Psychometrika 12(2):
//     153-157.
//   - Fagerland, M. W., Lydersen, S., & Laake, P. (2013). The McNemar test
//     for binary matched-pairs data: mid-p and asymptotic are better than
//     exact conditional. BMC Medical Research Methodology 13: 91.
//   - Fisher, R. A. (1935). The Design of Experiments. Oliver & Boyd. §21.
//   - Ernst, M. D. (2004). Permutation methods: a basis for exact inference.
//     Statistical Science 19(4): 676-685.
package agreement
