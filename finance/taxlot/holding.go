package taxlot

// Holding-period term classification — the short-term / long-term boundary of
// IRC §1222, computed as pure civil-date arithmetic.
//
// # The statutory rule (and the off-by-one it exists to prevent)
//
// A capital gain or loss is LONG-TERM only if the asset was held for MORE THAN
// one year (IRC §1222(3),(4): "more than 1 year"). The holding period is
// counted by the IRS with two rules that together produce a one-year-and-a-day
// boundary:
//
//   - The holding period BEGINS on the day AFTER the trade date of acquisition
//     (the acquisition day itself is excluded).
//   - The day of disposition IS included in the holding period.
//
// (IRS Publication 550, "Holding Period"; Rev. Rul. 66-7.)
//
// Consequently an asset acquired on date A becomes long-term on the disposal
// date A + 1 year + 1 day, NOT on the one-year anniversary A + 1 year. Selling
// exactly on the anniversary is SHORT-TERM. IRS Pub 544 gives the worked
// boundary: property bought on Feb 5 and sold the following Feb 5 was held
// exactly one year — short-term; sold Feb 6 — long-term.
//
// This is precisely the boundary a downstream tax engine got wrong (an ST/LT
// off-by-one pinned as "correct" by two of its own unit tests, per
// reviews/RD_RISK_MATH_REVIEW_2026-06-25/02_TAX_MATH_FINDINGS.md). A single
// self-consistent codebase cannot catch this class of error by testing itself;
// the statutory boundary must come from an external, legally-pinned reference.
// That is what this kernel is.

// Term is a capital-gains holding-period classification.
type Term int

const (
	// ShortTerm is a holding period of one year or less (taxed as ordinary
	// income in the US). IRC §1222(1),(2).
	ShortTerm Term = iota
	// LongTerm is a holding period of MORE THAN one year (preferential US
	// capital-gains rates). IRC §1222(3),(4).
	LongTerm
)

// String renders the term as "short-term" or "long-term".
func (t Term) String() string {
	if t == LongTerm {
		return "long-term"
	}
	return "short-term"
}

// IsLongTerm reports whether an asset acquired on acq and disposed of on
// disposal was held long enough to be long-term under IRC §1222 — i.e. strictly
// MORE than one year, applying the day-after-acquisition start rule.
//
// Equivalent formulation: the asset is long-term iff the disposal date is
// strictly after the one-year anniversary of the acquisition date. Selling on
// the anniversary itself (one year to the day) is short-term.
//
// The anniversary is computed with the end-of-month clamp of Date.AddYears, so
// a Feb-29 acquisition anniversary falls on Feb 28 of a common year (its last
// valid February day) — leap-safe.
//
// A disposal strictly before the acquisition date is nonsensical; the function
// does not error, it simply returns false (short-term), because a negative
// holding period cannot be long-term.
func IsLongTerm(acq, disposal Date) bool {
	anniversary := acq.AddYears(1)
	return disposal.After(anniversary)
}

// Classify returns LongTerm or ShortTerm for the given acquisition and
// disposal dates, per IsLongTerm.
func Classify(acq, disposal Date) Term {
	if IsLongTerm(acq, disposal) {
		return LongTerm
	}
	return ShortTerm
}
