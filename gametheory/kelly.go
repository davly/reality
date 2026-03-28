package gametheory

import "math"

// ---------------------------------------------------------------------------
// Kelly Criterion
//
// Optimal bet sizing for long-term growth rate maximization. Used by
// RubberDuck for position sizing in evolutionary trading strategies and
// by any decision system that must allocate resources under uncertainty.
// ---------------------------------------------------------------------------

// KellyFraction computes the optimal bet fraction using the Kelly criterion
// for a single binary outcome bet.
//
// The Kelly formula maximizes the expected logarithmic growth rate:
//
//	f* = (p * b - q) / b
//
// where:
//   - p = probability of winning
//   - q = 1 - p = probability of losing
//   - b = net odds received on the bet (win amount per unit wagered)
//   - f* = fraction of bankroll to bet
//
// Returns:
//   - The optimal fraction to bet. Positive means bet on the outcome,
//     negative means bet against (if possible), zero means don't bet.
//   - Returns 0 if odds <= 0 (no bet possible)
//   - Returns 0 if prob <= 0 or prob >= 1 (degenerate)
//   - Clamps to [-1, 1] range (can't bet more than bankroll)
//
// Precision: exact (single arithmetic expression)
// Reference: Kelly, J.L. (1956) "A New Interpretation of Information Rate",
// Bell System Technical Journal 35(4):917-926.
func KellyFraction(prob, odds float64) float64 {
	if odds <= 0 || prob <= 0 || prob >= 1 {
		return 0
	}

	q := 1 - prob
	f := (prob*odds - q) / odds

	// Clamp to [-1, 1].
	if f > 1 {
		f = 1
	}
	if f < -1 {
		f = -1
	}

	return f
}

// KellyFractionMultiple computes the optimal bet fractions for multiple
// independent simultaneous bets, each with its own probability and odds.
// This is the multi-outcome extension of the Kelly criterion.
//
// For independent bets, the optimal fraction for each bet is computed
// independently using the single-bet Kelly formula. This is an approximation
// that is exact when the bets are truly independent and the total allocation
// is small relative to the bankroll.
//
// Parameters:
//   - probs: probability of winning for each bet
//   - odds: net odds for each bet (win amount per unit wagered)
//
// Returns a slice of optimal fractions, one per bet.
// Returns nil if inputs have different lengths or are empty.
//
// Note: For correlated bets or large total allocations, a portfolio
// optimization approach (quadratic programming) would be more appropriate.
//
// Precision: exact per-bet (single arithmetic expression each)
// Reference: Kelly (1956); Thorp, E.O. (2006) "The Kelly Criterion in
// Blackjack, Sports Betting, and the Stock Market", in Handbook of
// Asset and Liability Management, Vol. 1.
func KellyFractionMultiple(probs, odds []float64) []float64 {
	if len(probs) != len(odds) || len(probs) == 0 {
		return nil
	}

	fractions := make([]float64, len(probs))
	for i := range probs {
		fractions[i] = KellyFraction(probs[i], odds[i])
	}

	// Scale down if total allocation exceeds 1 (100% of bankroll).
	// This prevents overbetting when many simultaneous opportunities exist.
	totalPositive := 0.0
	for _, f := range fractions {
		if f > 0 {
			totalPositive += f
		}
	}

	if totalPositive > 1 {
		scale := 1.0 / totalPositive
		for i := range fractions {
			if fractions[i] > 0 {
				fractions[i] *= scale
			}
		}
	}

	return fractions
}

// KellyGrowthRate returns the expected logarithmic growth rate for a given
// bet fraction f on a binary outcome with probability prob and odds.
//
// Growth rate: G(f) = p * ln(1 + f*b) + q * ln(1 - f)
//
// This is the function that KellyFraction maximizes. Useful for comparing
// different bet sizes or visualizing the Kelly curve.
//
// Returns NaN if the bet would result in a negative bankroll
// (i.e., 1 + f*odds <= 0 or 1 - f <= 0).
//
// Precision: ~15 significant digits (float64)
// Reference: Kelly (1956)
func KellyGrowthRate(prob, odds, fraction float64) float64 {
	if prob <= 0 || prob >= 1 || odds <= 0 {
		return math.NaN()
	}

	winTerm := 1 + fraction*odds
	loseTerm := 1 - fraction

	if winTerm <= 0 || loseTerm <= 0 {
		return math.NaN()
	}

	q := 1 - prob
	return prob*math.Log(winTerm) + q*math.Log(loseTerm)
}
