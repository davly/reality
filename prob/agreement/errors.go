package agreement

import "errors"

// ErrLengthMismatch is returned when two raters' rating slices differ in
// length — every agreement statistic requires both raters to have judged
// the same set of units, positionally aligned.
var ErrLengthMismatch = errors.New("agreement: paired rating slices must have equal length")

// ErrEmptyInput is returned when a ratings slice or ratings matrix has zero
// units to score.
var ErrEmptyInput = errors.New("agreement: ratings input must be non-empty")

// ErrTooFewRaters is returned when fewer than 2 raters are supplied to a
// statistic that requires at least 2 (FleissKappa, KrippendorffAlpha).
var ErrTooFewRaters = errors.New("agreement: at least 2 raters are required")

// ErrRaggedRows is returned when FleissKappa's ratings matrix has a row
// whose category counts do not sum to the same fixed rater-panel size as
// every other row, or when KrippendorffAlpha's data matrix has raters with
// differing numbers of unit columns.
var ErrRaggedRows = errors.New("agreement: ratings rows must be rectangular and sum to a fixed rater count per subject")

// ErrNegativeCount is returned when a FleissKappa rating count is negative.
var ErrNegativeCount = errors.New("agreement: rating counts must be non-negative")

// ErrDegenerateChanceAgreement is returned when chance-expected agreement
// is 1, or the weighted-disagreement denominator is 0 — the kappa/alpha
// ratio is mathematically undefined (0/0) because there is no variability
// across categories or ranks left to correct for chance against.
var ErrDegenerateChanceAgreement = errors.New("agreement: expected chance agreement is degenerate (kappa/alpha undefined)")
