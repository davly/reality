// Package taxlot implements a statutory tax-lot kernel: the proportional
// (matched-share) wash-sale loss disallowance and basis adjustment of IRC
// §1091, holding-period tacking under IRC §1223(3), and the short-term /
// long-term holding-period boundary of IRC §1222 — all as pure, deterministic,
// zero-dependency integer-and-date arithmetic.
//
// # Why this exists in Reality
//
// A downstream tax engine (RubberDuck's WashSaleDetector, surfaced through its
// get_tax_report tool) treats every wash sale as all-or-nothing: it disallows
// the entire realised loss even when fewer replacement shares are bought than
// were sold, whereas IRC §1091 disallows the loss only on the matched shares
// and leaves the rest deductible. Its basis-adjustment fields are read into
// the tax report but set by no production code — the apportionment math is
// genuinely unbuilt (reviews/RD_RISK_MATH_REVIEW_2026-06-25/
// 02_TAX_MATH_FINDINGS.md, item B3).
//
// The same review demonstrated a deeper failure mode: that codebase
// institutionalised a WRONG short-term/long-term boundary that two of its own
// unit tests pinned as correct. A single self-consistent codebase cannot catch
// this class of error by testing against itself. Wrong tax math is real HMRC /
// IRS liability.
//
// The countermeasure is an external, legally-pinned reference. Reality is
// Go-canonical; consumers in any language (RubberDuck is C#) validate their
// own arithmetic against the same golden vectors. Every golden vector in this
// package is transcribed from an IRS PUBLISHED worked example (Pub 550, Pub
// 544) — the answers are legally fixed, so the fixtures are an arbiter the
// consumer's own test suite provably cannot be.
//
// # API surface
//
//   - ApplyWashSale(sale, replacements) — proportional IRC §1091 disallowance
//     with per-lot basis increase and holding-period tacking.
//   - Classify(acq, disposal) / IsLongTerm(acq, disposal) — IRC §1222 ST/LT
//     boundary (strictly more than one year, day-after-acquisition rule).
//   - Date — a pure civil-date type (Hinnant epoch-day algorithms) with
//     AddDays / AddYears / DaysUntil, leap-safe.
//
// All computation is exact: money is integer minor units (Cents), dates are
// (year, month, day) triples, and there is no floating point anywhere. Only
// the standard library "errors" and "sort" packages are imported.
//
// # Scope
//
// This kernel covers the US federal rules that have a verified unbuilt
// consumer today. The UK share-matching ladder (same-day / 30-day
// bed-and-breakfast / Section 104 pool average-cost) named in the originating
// spec is deliberately OUT OF SCOPE for this increment — it has no verified
// consumer in the estate today (per the item's own verification caveat) and
// would be a separate deterministic state machine. It is left as a documented
// follow-up rather than built speculatively.
//
// # References
//
//   - IRC §1091 (wash sales), §1091(d) (basis of replacement stock).
//   - IRC §1222 (short-term / long-term definitions), §1223(3) (tacked
//     holding period of wash-sale replacement stock).
//   - Treas. Reg. §1.1091-1(c) (matching replacement shares in order of
//     acquisition).
//   - IRS Publication 550, "Investment Income and Expenses", section "Wash
//     Sales" (partial wash-sale worked example).
//   - IRS Publication 544, "Sales and Other Dispositions of Assets", section
//     "Holding Period" (the Feb-5 / Feb-6 short/long-term boundary example).
//   - Rev. Rul. 66-7 (holding period begins the day after acquisition).
//   - Howard Hinnant, "chrono-Compatible Low-Level Date Algorithms" (2013) —
//     the days_from_civil / civil_from_days calendar kernel.
package taxlot
