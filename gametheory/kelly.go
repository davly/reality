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

// ---------------------------------------------------------------------------
// Continuous / Fractional Kelly + Ergodicity (time-average growth)
//
// The binary form above (f* = (pb-q)/b) is the discrete-outcome Kelly. For
// position sizing over a *return distribution* (the way RubberDuck's live
// ergodicity sizers actually allocate capital), the relevant object is the
// continuous Kelly fraction f* = mu / sigma^2 — the maximiser of the
// time-average log-growth rate g(f) = f*mu - f^2*sigma^2/2.
//
// These functions are the substrate-neutral reference that RubberDuck's two
// C# KellyOptimalSizer implementations (Core/Analysis/Ergodicity and
// Brain.Risk/Services) are pinned against by cross-language parity tests, so
// any future drift between them (or a refactor of the gap-shrinkage term)
// breaks a golden test instead of silently mis-sizing capital.
//
// References:
//   - Merton, R.C. (1969) "Lifetime Portfolio Selection under Uncertainty:
//     the Continuous-Time Case", Rev. Econ. Stat. 51(3):247-257 (f*=mu/sigma^2).
//   - Thorp, E.O. (2006) "The Kelly Criterion in Blackjack, Sports Betting,
//     and the Stock Market", Handbook of Asset & Liability Management, Vol.1.
//   - MacLean, Thorp, Ziemba (2011) "The Kelly Capital Growth Investment
//     Criterion: Theory and Practice", World Scientific.
//   - Peters, O. (2019) "The ergodicity problem in economics", Nature Physics
//     15:1216-1221 (ensemble-average vs time-average growth; the gap).
// ---------------------------------------------------------------------------

// KellyContinuous computes the single-asset continuous Kelly fraction that
// maximises the long-run log-growth rate for a return with mean mu and
// standard deviation sigma:
//
//	f* = mu / sigma^2
//
// This is the maximiser of g(f) = f*mu - f^2*sigma^2/2 (see ContinuousGrowthRate).
// The result is the *raw* Kelly fraction — it is deliberately NOT clamped, as
// full Kelly is famously aggressive; callers apply FractionalKelly and any hard
// cap themselves. A negative mu yields a negative fraction (short / abstain).
//
// Returns 0 when sigma <= 0 (no volatility to size against) or when any input
// is non-finite.
//
// Precision: exact (single arithmetic expression).
// Reference: Merton (1969); Thorp (2006).
func KellyContinuous(mu, sigma float64) float64 {
	if sigma <= 0 || math.IsNaN(mu) || math.IsInf(mu, 0) || math.IsInf(sigma, 0) {
		return 0
	}
	return mu / (sigma * sigma)
}

// KellyContinuousMulti computes the multi-asset continuous Kelly allocation
// vector for correlated assets:
//
//	f* = Sigma^{-1} mu
//
// where mu is the vector of expected excess returns and Sigma (cov) is the
// return covariance matrix (n x n, symmetric positive-definite in the
// well-posed case). The system Sigma * f = mu is solved by Gaussian
// elimination with partial pivoting (no external dependency).
//
// Returns nil if the dimensions are inconsistent (len(mu) != n, cov not n x n),
// n == 0, or the covariance matrix is singular / near-singular.
//
// Precision: ~15 significant digits (float64) for well-conditioned Sigma.
// Reference: Thorp (2006); MacLean, Thorp, Ziemba (2011).
func KellyContinuousMulti(mu []float64, cov [][]float64) []float64 {
	n := len(mu)
	if n == 0 || len(cov) != n {
		return nil
	}
	// Copy into a working matrix; validate squareness.
	a := make([][]float64, n)
	for i := range cov {
		if len(cov[i]) != n {
			return nil
		}
		a[i] = make([]float64, n)
		copy(a[i], cov[i])
	}
	b := make([]float64, n)
	copy(b, mu)

	// Forward elimination with partial pivoting.
	for col := 0; col < n; col++ {
		// Find the pivot row (largest absolute value in this column).
		pivot := col
		best := math.Abs(a[col][col])
		for r := col + 1; r < n; r++ {
			if v := math.Abs(a[r][col]); v > best {
				best = v
				pivot = r
			}
		}
		if best < 1e-300 {
			return nil // singular
		}
		if pivot != col {
			a[col], a[pivot] = a[pivot], a[col]
			b[col], b[pivot] = b[pivot], b[col]
		}
		// Eliminate below.
		for r := col + 1; r < n; r++ {
			factor := a[r][col] / a[col][col]
			if factor == 0 {
				continue
			}
			for c := col; c < n; c++ {
				a[r][c] -= factor * a[col][c]
			}
			b[r] -= factor * b[col]
		}
	}

	// Back substitution.
	x := make([]float64, n)
	for i := n - 1; i >= 0; i-- {
		sum := b[i]
		for c := i + 1; c < n; c++ {
			sum -= a[i][c] * x[c]
		}
		x[i] = sum / a[i][i]
	}
	return x
}

// FractionalKelly applies a fractional-Kelly multiplier lambda to a raw Kelly
// fraction f. Full Kelly (lambda = 1) maximises growth but has high variance;
// the "quarter Kelly" convention (lambda = 0.25) captures most of the growth at
// a fraction of the drawdown risk.
//
//	fractional = clamp(lambda, 0, 1) * f
//
// lambda is clamped to [0, 1]: a negative lambda yields 0 (no bet); a lambda
// above 1 is treated as full Kelly (never lever *up* past the raw fraction).
//
// Precision: exact (single arithmetic expression).
// Reference: Thorp (2006); MacLean, Thorp, Ziemba (2011).
func FractionalKelly(f, lambda float64) float64 {
	if lambda <= 0 {
		return 0
	}
	if lambda > 1 {
		lambda = 1
	}
	return lambda * f
}

// ContinuousGrowthRate returns the time-average logarithmic growth rate for a
// continuous position of fraction f on a return with mean mu and standard
// deviation sigma, under the standard second-order (log-normal) approximation:
//
//	g(f) = f*mu - f^2*sigma^2/2
//
// This is the objective that KellyContinuous maximises: setting g'(f) =
// mu - f*sigma^2 = 0 gives the maximiser f* = mu / sigma^2, at which
// g(f*) = mu^2 / (2*sigma^2).
//
// Precision: exact (single arithmetic expression).
// Reference: Peters (2019); Merton (1969).
func ContinuousGrowthRate(f, mu, sigma float64) float64 {
	return f*mu - f*f*sigma*sigma/2
}

// EnsembleMeanReturn is the arithmetic mean of a simple-return series — the
// "expected return" E[r] headline that an ensemble (many parallel trajectories)
// would realise. Returns NaN for an empty series.
//
// Precision: ~15 significant digits (float64).
// Reference: Peters (2019).
func EnsembleMeanReturn(returns []float64) float64 {
	if len(returns) == 0 {
		return math.NaN()
	}
	sum := 0.0
	for _, r := range returns {
		sum += r
	}
	return sum / float64(len(returns))
}

// TimeAverageGrowthRate is the mean log-return E[ln(1+r)] of a simple-return
// series — what a *single* multiplicative trajectory actually compounds at per
// step. Each return is floored at -0.9999 to avoid ln(0) = -Inf on a total-loss
// bar (mirroring the consumer's -99.99% cap). Returns NaN for an empty series.
//
// Precision: ~15 significant digits (float64).
// Reference: Peters (2019); Peters & Gell-Mann (2016).
func TimeAverageGrowthRate(returns []float64) float64 {
	if len(returns) == 0 {
		return math.NaN()
	}
	sum := 0.0
	for _, r := range returns {
		if r < -0.9999 {
			r = -0.9999
		}
		sum += math.Log(1 + r)
	}
	return sum / float64(len(returns))
}

// ErgodicityGap is the ensemble-vs-time divergence for a return series:
//
//	gap = E[r] - E[ln(1+r)]  =  EnsembleMeanReturn - TimeAverageGrowthRate
//
// By Jensen's inequality (ln is concave) the gap is always >= 0. A wide gap
// relative to the time-average growth signals that the apparent edge is
// dominated by estimation noise / volatility drag — the "ergodicity trap".
// Returns NaN for an empty series.
//
// Precision: ~15 significant digits (float64).
// Reference: Peters (2019).
func ErgodicityGap(returns []float64) float64 {
	return EnsembleMeanReturn(returns) - TimeAverageGrowthRate(returns)
}

// ergodicityGapRatio is the shared, arbitrated denominator for both shrinkage
// shapes: gap normalised by the magnitude of the time-average growth, floored
// at 0 (the gap is non-negative by Jensen; this guards float noise) and with a
// small epsilon guarding division when timeAvg ~ 0.
func ergodicityGapRatio(gap, timeAvg float64) float64 {
	const eps = 1e-12
	ratio := gap / math.Max(math.Abs(timeAvg), eps)
	if ratio < 0 {
		return 0
	}
	return ratio
}

// ErgodicityShrinkageExp is the gap-aware shrinkage multiplier in exponential
// form (RubberDuck.Core/Analysis/Ergodicity/KellyOptimalSizer's variant):
//
//	s = exp( -max(0, gap / max(|timeAvg|, eps)) )
//
// As gap -> 0, s -> 1 (no extra shrinkage); as the gap dominates timeAvg,
// s -> 0 (size goes to zero). Multiply a fractional-Kelly size by s to pull it
// down in high-estimation-noise regimes. Result is in (0, 1].
//
// Precision: ~15 significant digits (float64).
// Reference: Peters (2019); this is the estate's Core sizer term.
func ErgodicityShrinkageExp(gap, timeAvg float64) float64 {
	return math.Exp(-ergodicityGapRatio(gap, timeAvg))
}

// ErgodicityShrinkageReciprocal is the gap-aware shrinkage multiplier in
// reciprocal form (RubberDuck.Brain.Risk/Services/KellyOptimalSizer's variant):
//
//	s = 1 / (1 + max(0, gap / max(|timeAvg|, eps)))
//
// Same monotone behaviour as ErgodicityShrinkageExp (1 at gap 0, -> 0 as the
// gap dominates) but a heavier tail. Both variants exist here so the two C#
// twins — which had *silently diverged* onto different shapes — can each be
// pinned against a single Go reference. Result is in (0, 1].
//
// Precision: ~15 significant digits (float64).
// Reference: this is the estate's Brain.Risk sizer term.
func ErgodicityShrinkageReciprocal(gap, timeAvg float64) float64 {
	return 1 / (1 + ergodicityGapRatio(gap, timeAvg))
}

// PriorShrink blends a shrink-only ecosystem win-probability prior into a
// sizing fraction. The prior can ONLY make the size more conservative: a base
// rate at or above the neutral 0.5 leaves the fraction unchanged, below 0.5
// scales it down proportionally, and 0 abstains entirely:
//
//	priorShrink = clamp(winProb / 0.5, 0, 1)
//	out         = fraction * priorShrink
//
// A missing / out-of-range / non-finite winProb (NaN, <0, or >1) returns the
// fraction unchanged (the additive-safety property: no prior can inflate a bet).
//
// Precision: exact (single arithmetic expression).
// Reference: RubberDuck ecosystem-winrate prior (Brain.Risk sizer overload).
func PriorShrink(fraction, winProb float64) float64 {
	if math.IsNaN(winProb) || winProb < 0 || winProb > 1 {
		return fraction
	}
	const neutral = 0.5
	priorShrink := winProb / neutral
	if priorShrink > 1 {
		priorShrink = 1
	}
	return fraction * priorShrink
}
