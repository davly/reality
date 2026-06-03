package combinatorics

import "math"

// ---------------------------------------------------------------------------
// Lattice Path Enumeration & Binomial-Tree Option Pricing
// ---------------------------------------------------------------------------
//
// A binomial-tree option-pricing model is a discrete random walk: at each of
// n time steps the underlying moves up by factor u or down by factor d. After
// n steps the asset has visited one of n+1 terminal nodes, reachable by
// C(n, k) distinct lattice paths (k up-moves out of n). Pricing a contingent
// claim is therefore a weighted enumeration over lattice paths.
//
// Two combinatorial structures appear naturally:
//
//   - Catalan numbers C_n count Dyck paths — lattice walks of length 2n from
//     (0,0) to (2n,0) that stay weakly above zero. They count the number of
//     paths in a binomial tree that never touch a barrier from above.
//
//   - The reflection principle (André, 1887) counts paths that DO touch a
//     barrier as the count of paths to the reflected endpoint. This gives a
//     closed-form expression for barrier-option prices in O(n).
//
// References:
//   - Cox, Ross, Rubinstein (1979) "Option Pricing: A Simplified Approach"
//   - Hull, "Options, Futures, and Other Derivatives", chapters on binomial
//     trees and barrier options
//   - Stanley, "Enumerative Combinatorics" Vol. 1, on Dyck paths
//   - Feller, "An Introduction to Probability Theory" Vol. 1, on the
//     reflection principle

// PriceEuropeanBinomialTree prices a European call (or put if isPut) under a
// CRR binomial tree. The risk-neutral probability of an up-move is
//
//	p = (exp(r * dt) - d) / (u - d),  where dt = T/n
//
// and the price is the discounted expectation
//
//	V_0 = exp(-r*T) * sum_{k=0}^{n} C(n,k) * p^k * (1-p)^{n-k} * payoff(S_n)
//
// Each lattice path with k up-moves contributes weight p^k * (1-p)^{n-k};
// C(n,k) collects the paths with the same terminal node. Time complexity O(n).
//
// Parameters:
//   - s0:    spot price
//   - k:     strike price
//   - r:     risk-free rate (continuously compounded)
//   - sigma: volatility (annualised)
//   - t:     time to expiry (years)
//   - n:     number of binomial steps (n >= 1)
//   - isPut: false for call, true for put
//
// Returns the option price, or NaN if inputs are pathological (n < 1,
// sigma <= 0, t <= 0).
func PriceEuropeanBinomialTree(s0, k, r, sigma, t float64, n int, isPut bool) float64 {
	if n < 1 || sigma <= 0 || t <= 0 || s0 <= 0 || k <= 0 {
		return math.NaN()
	}
	dt := t / float64(n)
	u := math.Exp(sigma * math.Sqrt(dt))
	d := 1.0 / u
	disc := math.Exp(-r * dt)
	p := (math.Exp(r*dt) - d) / (u - d)
	if p <= 0 || p >= 1 {
		return math.NaN() // arbitrage condition violated
	}

	// Closed-form O(n) sum over terminal nodes weighted by binomial probability.
	price := 0.0
	for j := 0; j <= n; j++ {
		st := s0 * math.Pow(u, float64(j)) * math.Pow(d, float64(n-j))
		var payoff float64
		if isPut {
			payoff = math.Max(k-st, 0)
		} else {
			payoff = math.Max(st-k, 0)
		}
		price += BinomialCoeff(n, j) * math.Pow(p, float64(j)) *
			math.Pow(1-p, float64(n-j)) * payoff
	}
	return price * math.Pow(disc, float64(n))
}

// PriceAmericanBinomialTree prices an American option under a CRR binomial
// tree by backward induction. At each node the holder may exercise early; the
// continuation value is the discounted risk-neutral expectation of the next
// step.
//
// Algorithm: build the n+1 terminal payoffs, then walk back row by row taking
// max(intrinsic, discounted expectation). This visits O(n^2) nodes and uses
// O(n) memory by overwriting in place.
//
// The American premium (over the European price) reflects the value of the
// early-exercise right: paths that pass through deep-in-the-money nodes can
// be "stopped early", which the European formula cannot exploit.
//
// Parameters mirror PriceEuropeanBinomialTree.
func PriceAmericanBinomialTree(s0, k, r, sigma, t float64, n int, isPut bool) float64 {
	if n < 1 || sigma <= 0 || t <= 0 || s0 <= 0 || k <= 0 {
		return math.NaN()
	}
	dt := t / float64(n)
	u := math.Exp(sigma * math.Sqrt(dt))
	d := 1.0 / u
	disc := math.Exp(-r * dt)
	p := (math.Exp(r*dt) - d) / (u - d)
	if p <= 0 || p >= 1 {
		return math.NaN()
	}

	// Terminal payoffs at step n.
	values := make([]float64, n+1)
	for j := 0; j <= n; j++ {
		st := s0 * math.Pow(u, float64(j)) * math.Pow(d, float64(n-j))
		if isPut {
			values[j] = math.Max(k-st, 0)
		} else {
			values[j] = math.Max(st-k, 0)
		}
	}

	// Backward induction: at each node compare intrinsic vs continuation.
	for step := n - 1; step >= 0; step-- {
		for j := 0; j <= step; j++ {
			st := s0 * math.Pow(u, float64(j)) * math.Pow(d, float64(step-j))
			cont := disc * (p*values[j+1] + (1-p)*values[j])
			var intrinsic float64
			if isPut {
				intrinsic = math.Max(k-st, 0)
			} else {
				intrinsic = math.Max(st-k, 0)
			}
			values[j] = math.Max(intrinsic, cont)
		}
	}
	return values[0]
}

// BarrierOptionReflection prices an up-and-out European call by counting
// lattice paths that never cross the barrier, using André's reflection
// principle.
//
// In a symmetric random walk on the integer lattice, the number of paths from
// (0, 0) to (n, k) that touch or exceed level h equals the number of unrestricted
// paths from (0, 2h - 0) to (n, k), i.e. paths starting at the reflected source.
// Subtracting reflected-path counts from total counts gives the number of
// paths that stay strictly below the barrier.
//
// Concretely, if the barrier is breached when the cumulative number of
// up-moves exceeds threshold m, the number of allowed paths terminating with
// j up-moves is
//
//	N_allowed(j) = C(n, j) - C(n, 2*m - j + 1)        if 2m - j + 1 in [0, n]
//	             = C(n, j)                            otherwise
//
// Each allowed path contributes payoff * p^j * (1-p)^(n-j) to the price.
// Time complexity O(n) — the entire price is a single sum over terminal
// nodes, no node-by-node tree walk.
//
// Parameters:
//   - s0, k, r, sigma, t, n: as in PriceEuropeanBinomialTree
//   - barrier: the up-and-out level (option voids if S_t >= barrier at any step)
//
// Returns NaN on bad inputs. If barrier <= s0 the option is dead at issue
// (returns 0). For barriers far above any reachable node the result equals
// the plain European call.
func BarrierOptionReflection(s0, k, r, sigma, t, barrier float64, n int) float64 {
	if n < 1 || sigma <= 0 || t <= 0 || s0 <= 0 || k <= 0 || barrier <= 0 {
		return math.NaN()
	}
	if barrier <= s0 {
		return 0 // already knocked out
	}
	dt := t / float64(n)
	u := math.Exp(sigma * math.Sqrt(dt))
	d := 1.0 / u
	disc := math.Exp(-r * dt)
	p := (math.Exp(r*dt) - d) / (u - d)
	if p <= 0 || p >= 1 {
		return math.NaN()
	}

	// Find the smallest m such that s0 * u^m * d^(n-m) >= barrier; any path
	// reaching level m at any time breaches the barrier. Using log space:
	// m_breach = ceil(log(barrier/s0) / log(u/d) * something)
	// We instead translate: barrier breached iff cumulative excess of up over
	// down moves crosses threshold h where
	//   s0 * u^h >= barrier   =>   h = ceil(log(barrier/s0) / log(u))
	logRatio := math.Log(barrier / s0)
	logU := math.Log(u)
	hFloat := logRatio / logU
	h := int(math.Ceil(hFloat))
	if h <= 0 {
		return 0
	}
	// If even the all-up path doesn't breach, the barrier is irrelevant.
	if h > n {
		return PriceEuropeanBinomialTree(s0, k, r, sigma, t, n, false)
	}

	// Sum payoffs over terminal nodes, subtracting reflected-path counts for
	// nodes that could have been reached only via a barrier-breaching path.
	price := 0.0
	for j := 0; j <= n; j++ {
		st := s0 * math.Pow(u, float64(j)) * math.Pow(d, float64(n-j))
		payoff := math.Max(st-k, 0)
		if payoff == 0 {
			continue
		}
		// A terminal node whose OWN price already breaches the barrier is
		// fully knocked out: every path that reaches it has touched h. The
		// reflection below assumes the terminal sits strictly below the
		// barrier — without this guard, in-the-money nodes ABOVE the barrier
		// (reachable under high volatility) would be spuriously counted.
		// Terminal net displacement is 2j-n; barrier is at displacement h.
		if 2*j-n >= h {
			continue
		}
		// Total paths to this node minus the paths that ever touched the
		// barrier. André's reflection principle in NET-DISPLACEMENT
		// coordinates: the walk position is the net (up minus down)
		// displacement, the terminal sits at displacement 2j-n, and the
		// barrier is at displacement h. The number of n-step paths from 0 to
		// terminal up-count j that ever reach displacement h equals the number
		// of paths to the reflected endpoint, whose up-count is (n+h-j) — NOT
		// (2h-j), which incorrectly mixed the up-count and displacement
		// coordinate systems and overpriced up-and-out calls by up to ~8x.
		total := BinomialCoeff(n, j)
		reflected := 0.0
		jRefl := n + h - j
		if jRefl >= 0 && jRefl <= n {
			reflected = BinomialCoeff(n, jRefl)
		}
		allowed := total - reflected
		if allowed < 0 {
			allowed = 0 // numerical guard
		}
		price += allowed * math.Pow(p, float64(j)) * math.Pow(1-p, float64(n-j)) * payoff
	}
	return price * math.Pow(disc, float64(n))
}

// CountDyckPaths returns the number of Dyck paths of semilength n — the
// number of lattice paths from (0,0) to (2n, 0) using steps (1, +1) and
// (1, -1) that never go below the x-axis. This is the nth Catalan number.
//
// Equivalently: the number of binomial-tree paths of length 2n with equal
// up/down counts that stay weakly above their starting level. In barrier-
// option language this counts paths that stay below a one-sided barrier.
//
// Reference: Stanley, "Catalan Numbers" (2015), Exercise 1.
func CountDyckPaths(n int) float64 {
	return CatalanNumber(n)
}
