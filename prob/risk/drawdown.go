package risk

import "math"

// MaxDrawdownFromPrices returns the maximum drawdown of a price (or equity /
// net-asset-value) series: the largest peak-to-trough decline, as a POSITIVE
// fraction of the running peak,
//
//	MDD = max_t ( (peak_{<=t} - price_t) / peak_{<=t} )
//
// where peak_{<=t} is the highest price seen up to and including t. A result
// of 0.30 means the series fell 30% from a prior high at its worst point.
//
// Valid range: at least 1 price; all prices must be > 0 (a drawdown is a
// fraction of a positive peak). Returns NaN for an empty slice or if any
// non-positive price is encountered while a positive peak is in force.
// Returns 0 for a monotonically non-decreasing series.
// Precision: one pass; exact up to float64 division.
func MaxDrawdownFromPrices(prices []float64) float64 {
	if len(prices) == 0 {
		return math.NaN()
	}
	peak := math.Inf(-1)
	var maxDD float64
	for _, p := range prices {
		if p > peak {
			peak = p
		}
		if peak <= 0 || math.IsNaN(p) {
			return math.NaN()
		}
		if dd := (peak - p) / peak; dd > maxDD {
			maxDD = dd
		}
	}
	return maxDD
}

// MaxDrawdownFromReturns returns the maximum drawdown implied by a series of
// per-period simple returns, by first compounding them into an equity curve
// that starts at 1.0 and then applying the price-series definition:
//
//	equity_0 = 1;  equity_t = prod_{i<=t} (1 + returns[i])
//	MDD = MaxDrawdownFromPrices( [1, equity_0, equity_1, ...] )
//
// The synthetic starting value 1.0 is included as the initial peak, so a
// series that only ever loses money reports a drawdown measured from par. The
// result is identical to MaxDrawdownFromPrices applied to any price series
// whose successive ratios equal (1 + returns[i]).
//
// Valid range: at least 1 return; every (1 + returns[i]) must be > 0 (a
// period return of -100% or worse drives equity to zero and the fraction is
// undefined thereafter). Returns NaN for an empty slice or on a wipe-out.
// Returns 0 when the equity curve never declines.
// Precision: one pass; ~15 significant digits (compounding accumulates
// float64 rounding across periods).
func MaxDrawdownFromReturns(returns []float64) float64 {
	if len(returns) == 0 {
		return math.NaN()
	}
	equity := 1.0
	peak := 1.0 // the par starting value is the first peak
	var maxDD float64
	for _, r := range returns {
		equity *= (1.0 + r)
		if equity <= 0 || math.IsNaN(equity) {
			return math.NaN()
		}
		if equity > peak {
			peak = equity
		}
		if dd := (peak - equity) / peak; dd > maxDD {
			maxDD = dd
		}
	}
	return maxDD
}

// CalmarRatio returns the Calmar ratio: an annualized return divided by the
// magnitude of the maximum drawdown,
//
//	Calmar = annualizedReturn / maxDrawdown
//
// It rewards return per unit of worst-case peak-to-trough pain. Both inputs
// are supplied by the caller (rather than recomputed from a series) so the
// annualization convention and the drawdown horizon are the caller's explicit
// choice: pair AnnualizeReturn with MaxDrawdownFromReturns/Prices over the
// same window. maxDrawdown must be the POSITIVE fraction returned by the
// MaxDrawdown functions.
//
// Valid range: maxDrawdown > 0. Returns NaN if maxDrawdown < 0; returns +Inf
// (or -Inf) when maxDrawdown == 0 and the annualized return is positive (or
// negative) — a strategy that never drew down has unbounded Calmar.
// Precision: one division; exact up to float64.
func CalmarRatio(annualizedReturn, maxDrawdown float64) float64 {
	if maxDrawdown < 0 || math.IsNaN(maxDrawdown) {
		return math.NaN()
	}
	return annualizedReturn / maxDrawdown
}
