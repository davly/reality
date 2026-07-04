package taxlot

import "sort"

// Wash-sale kernel — proportional (matched-share) loss disallowance, basis
// adjustment, and holding-period tacking under IRC §1091 and its regulations.
//
// # The statutory rule
//
// A wash sale occurs when a taxpayer sells stock or securities at a loss and,
// within a 61-day window (30 days before through 30 days after the sale),
// acquires substantially identical stock or securities. The realised loss is
// then DISALLOWED for the current year (IRC §1091(a)).
//
// The apportionment rule this package exists to encode: when FEWER
// replacement shares are acquired than were sold at a loss, the loss is
// disallowed only PROPORTIONALLY — the loss attributable to the matched
// (replaced) shares is disallowed, and the loss on the excess unreplaced
// shares remains deductible (IRC §1091(a), second sentence; IRS Pub 550,
// "Wash Sales", the "more or less stock bought than sold" case).
//
//	matched shares      = min(shares sold, replacement shares acquired)
//	disallowed loss     = total loss × (matched shares / shares sold)
//	deductible loss     = total loss − disallowed loss
//
// The disallowed loss is not lost forever: it is added to the cost basis of
// the replacement shares (IRC §1091(d)), and the replacement shares inherit
// ("tack") the holding period of the shares sold (IRC §1223(3)). Together
// these defer the loss and preserve any long-term character.
//
// When there is more than one replacement lot, the matched shares are assigned
// to lots in the ORDER OF THEIR ACQUISITION (Reg. §1.1091-1(c)), and the
// disallowed loss follows the matched shares onto those lots.
//
// # Why this is a Reality kernel and not an app function
//
// The consuming tax engine (RubberDuck's WashSaleDetector) currently disallows
// the ENTIRE loss with no apportionment and reads a basis-adjustment field
// that no production code ever sets — the apportionment math is genuinely
// unbuilt (reviews/RD_RISK_MATH_REVIEW_2026-06-25/02_TAX_MATH_FINDINGS.md,
// item B3). The same review proved that codebase institutionalised a wrong
// tax answer that two of its own tests pinned as correct. The countermeasure
// is an external, legally-pinned reference: the golden vectors here are
// transcribed from IRS published worked examples, so any consumer in any
// language validates its own arithmetic against the statute, not against a
// value it alone produced.
//
// # Precision
//
// All money is exact integer minor units (cents / pence); there is no floating
// point anywhere in the computation. Proportional apportionment across
// multiple replacement lots uses cumulative rounding (round half away from
// zero at each cumulative boundary), which guarantees the per-lot basis
// increases sum EXACTLY to the total disallowed loss with no residual cent.

// Cents is an exact monetary amount in the smallest currency unit (US cents,
// UK pence). Using integer minor units makes all wash-sale arithmetic exact —
// tax figures must reconcile to the cent, which binary floating point cannot
// guarantee.
type Cents int64

// LossSale describes a single sale of shares from one cost-basis lot,
// disposed of at a loss.
type LossSale struct {
	// Shares is the number of shares sold. Must be positive.
	Shares int64
	// CostBasis is the total cost basis of the shares sold. Must be >= 0.
	CostBasis Cents
	// Proceeds is the total sale proceeds. Must be >= 0. The sale is a loss
	// iff Proceeds < CostBasis.
	Proceeds Cents
	// AcquisitionDate is the trade date on which the sold shares were
	// acquired — used to compute the holding period that tacks onto the
	// replacement shares.
	AcquisitionDate Date
	// SaleDate is the trade date of this loss sale.
	SaleDate Date
}

// ReplacementLot describes a purchase of substantially-identical shares made
// inside the wash-sale window. The caller is responsible for having already
// established that each lot is inside the ±30-day window and substantially
// identical; this kernel performs the apportionment and basis math on the lots
// it is given.
type ReplacementLot struct {
	// Shares is the number of replacement shares in this lot. Must be positive.
	Shares int64
	// CostBasis is the total amount actually paid for this lot. Must be >= 0.
	CostBasis Cents
	// AcquisitionDate is the trade date of this replacement purchase; matched
	// shares are assigned to lots in ascending order of this date.
	AcquisitionDate Date
}

// LotAdjustment is the wash-sale basis and holding-period adjustment applied
// to one replacement lot.
type LotAdjustment struct {
	// LotIndex is the index of this lot in the caller's original replacements
	// slice (not the acquisition-sorted order).
	LotIndex int
	// MatchedShares is how many of this lot's shares absorbed disallowed loss.
	MatchedShares int64
	// BasisIncrease is the disallowed loss apportioned onto this lot and added
	// to its cost basis (IRC §1091(d)).
	BasisIncrease Cents
	// AdjustedCostBasis is CostBasis + BasisIncrease.
	AdjustedCostBasis Cents
	// TackedAcquisitionDate is the replacement lot's acquisition date rolled
	// back by the holding period of the sold shares, so future short/long-term
	// classification of the replacement shares includes the sold shares'
	// holding period (IRC §1223(3)).
	TackedAcquisitionDate Date
}

// WashSaleResult is the outcome of applying IRC §1091 to a loss sale and its
// replacement lots.
type WashSaleResult struct {
	// SharesSold echoes the number of shares in the loss sale.
	SharesSold int64
	// MatchedShares is min(SharesSold, total replacement shares) — the number
	// of sold shares whose loss is disallowed.
	MatchedShares int64
	// TotalLoss is the full realised loss on the sale (CostBasis - Proceeds).
	TotalLoss Cents
	// DisallowedLoss is the portion of TotalLoss disallowed by the wash-sale
	// rule and deferred into the replacement lots' basis.
	DisallowedLoss Cents
	// DeductibleLoss is the portion of TotalLoss that remains deductible this
	// year (TotalLoss - DisallowedLoss) — nonzero only on a partial wash sale.
	DeductibleLoss Cents
	// Adjustments holds the per-lot basis / holding-period adjustments, one
	// entry per replacement lot that absorbed matched shares, in acquisition
	// order.
	Adjustments []LotAdjustment
}

// ApplyWashSale computes the IRC §1091 wash-sale treatment of a loss sale
// against the given replacement lots.
//
// The loss is disallowed only on the matched shares — min(shares sold,
// replacement shares) — with the disallowed amount apportioned across the
// replacement lots in acquisition order and added to their basis, and the
// sold shares' holding period tacked onto each. The remaining loss on any
// unreplaced shares stays deductible.
//
// Errors:
//   - ErrNoLoss if Proceeds >= CostBasis (a gain is never a wash sale).
//   - ErrNonPositiveShares if the sale or any replacement lot has non-positive
//     shares.
//   - ErrNegativeMoney if any cost basis or proceeds is negative.
//
// Golden vector — IRS Publication 550, "Wash Sales" (more/less stock bought
// than sold): 100 shares bought for $1,000, sold for $750 ($250 loss); 50
// shares of the same stock bought within 30 days for $800. Only the loss on
// the 50 matched shares — $125 — is disallowed; the $125 loss on the other 50
// shares is deductible. The 50 replacement shares' $800 basis is increased by
// the $125 disallowed loss to $925.
func ApplyWashSale(sale LossSale, replacements []ReplacementLot) (WashSaleResult, error) {
	if sale.Shares <= 0 {
		return WashSaleResult{}, ErrNonPositiveShares
	}
	if sale.CostBasis < 0 || sale.Proceeds < 0 {
		return WashSaleResult{}, ErrNegativeMoney
	}
	if sale.Proceeds >= sale.CostBasis {
		return WashSaleResult{}, ErrNoLoss
	}
	for _, r := range replacements {
		if r.Shares <= 0 {
			return WashSaleResult{}, ErrNonPositiveShares
		}
		if r.CostBasis < 0 {
			return WashSaleResult{}, ErrNegativeMoney
		}
	}

	totalLoss := sale.CostBasis - sale.Proceeds // > 0

	// Order replacement lots by acquisition date (Reg. §1.1091-1(c)),
	// preserving the caller's original index and breaking date ties by that
	// index so the assignment is fully deterministic.
	order := make([]int, len(replacements))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool {
		return replacements[order[a]].AcquisitionDate.Before(replacements[order[b]].AcquisitionDate)
	})

	var totalReplacement int64
	for _, r := range replacements {
		totalReplacement += r.Shares
	}

	matched := sale.Shares
	if totalReplacement < matched {
		matched = totalReplacement
	}

	result := WashSaleResult{
		SharesSold:    sale.Shares,
		MatchedShares: matched,
		TotalLoss:     totalLoss,
	}

	if matched == 0 {
		// No replacement shares: not a wash sale, entire loss deductible.
		result.DeductibleLoss = totalLoss
		return result, nil
	}

	// Disallowed loss = totalLoss × matched / sharesSold, rounded to the cent.
	disallowed := Cents(roundDiv(int64(totalLoss)*matched, sale.Shares))
	result.DisallowedLoss = disallowed
	result.DeductibleLoss = totalLoss - disallowed

	// Holding period of the sold shares, tacked onto every replacement lot.
	soldHoldingDays := sale.AcquisitionDate.DaysUntil(sale.SaleDate)

	// Assign matched shares to lots in acquisition order, and apportion the
	// disallowed loss across them by cumulative rounding so the per-lot pieces
	// sum EXACTLY to `disallowed` with no leftover cent.
	remaining := matched
	var cumMatched int64
	var allocated int64
	for _, idx := range order {
		if remaining == 0 {
			break
		}
		lot := replacements[idx]
		take := lot.Shares
		if take > remaining {
			take = remaining
		}
		remaining -= take
		cumMatched += take

		cumTarget := roundDiv(int64(disallowed)*cumMatched, matched)
		basisInc := Cents(cumTarget - allocated)
		allocated = cumTarget

		result.Adjustments = append(result.Adjustments, LotAdjustment{
			LotIndex:              idx,
			MatchedShares:         take,
			BasisIncrease:         basisInc,
			AdjustedCostBasis:     lot.CostBasis + basisInc,
			TackedAcquisitionDate: lot.AcquisitionDate.AddDays(-soldHoldingDays),
		})
	}

	return result, nil
}

// roundDiv returns num/den rounded half away from zero. num must be
// non-negative and den positive — the only regime in which this package calls
// it (losses, share counts, and cumulative weights are all non-negative).
func roundDiv(num, den int64) int64 {
	return (2*num + den) / (2 * den)
}
