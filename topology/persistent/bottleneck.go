package persistent

import (
	"math"
	"sort"
)

// BottleneckDistance returns the bottleneck distance between two
// persistence diagrams (slices of bars).  The bottleneck distance is
//
//	d_B(D, D') = inf_{matching M}  max_{(p, q) in M}  ||p - q||_inf
//
// where the matching M is a bijection between the points of D and the
// points of D', with the diagonal {(t, t)} treated as an unbounded
// reservoir of "free" points so that unmatched points can be sent to
// their nearest diagonal projection at cost (Death - Birth) / 2.
//
// Per Cohen-Steiner-Edelsbrunner-Harer 2007, the bottleneck distance
// is a stable metric on persistence diagrams: small perturbations of
// the input filtration produce small bottleneck distance, with
// stability constant 1.  This is the Lipschitz property that makes PH
// regulator-defensible and the load-bearing reason Witness wants
// d_B(today, yesterday) on a daily window.
//
// Algorithm: binary search on the threshold delta.  For each candidate
// delta, build the bipartite graph in which p_i in D is connected to
// q_j in D' iff ||p_i - q_j||_inf <= delta, and to its diagonal
// projection iff (Death_i - Birth_i)/2 <= delta.  The diagram is at
// bottleneck distance <= delta iff this graph has a perfect matching
// (Hopcroft-Karp).  Lower bound: 0 if both diagrams empty, else the
// minimum max-distance from the trivial diagonal matching.  Upper
// bound: the larger of the two diagrams' max-persistence/2.
//
// Complexity: O(n^{2.5} log n) using Hopcroft-Karp inside binary
// search; cheap enough for Phase-A consumer scale (a 30-asset
// correlation barcode has at most ~30 H_0 bars + a few H_1 bars).
//
// Bars in the slice are filtered by Dim == dimension.  Essential
// (death = +Inf) bars are excluded from the matching: the bottleneck
// distance between diagrams that disagree on the count of essential
// bars is +Inf, which we surface to the caller as +Inf (callers
// typically compare finite-persistence subdiagrams).
//
// If both filtered diagrams are empty, returns 0.  If exactly one is
// empty, returns the largest half-persistence in the non-empty one
// (i.e. the cost of matching every point to its diagonal).
//
// Inputs are not mutated.  If d1 and d2 disagree on essential-bar
// count in the dimension of interest, returns +Inf.
func BottleneckDistance(d1, d2 []Bar, dimension int) float64 {
	a := filterFinite(d1, dimension)
	b := filterFinite(d2, dimension)

	// Essential bar count must match for finite bottleneck distance.
	ea := countEssential(d1, dimension)
	eb := countEssential(d2, dimension)
	if ea != eb {
		return math.Inf(1)
	}

	if len(a) == 0 && len(b) == 0 {
		return 0
	}
	if len(a) == 0 {
		return maxHalfPersistence(b)
	}
	if len(b) == 0 {
		return maxHalfPersistence(a)
	}

	// Collect candidate threshold values: every pairwise L^inf
	// distance, every diagonal projection cost, and 0.  The optimal
	// delta is one of these (proof: at optimum, increasing delta
	// further does not change the matching graph until the next
	// candidate).
	cands := candidateThresholds(a, b)

	// Bisect on the sorted candidates.
	lo, hi := 0, len(cands)-1
	for lo < hi {
		mid := (lo + hi) / 2
		if hasPerfectMatching(a, b, cands[mid]) {
			hi = mid
		} else {
			lo = mid + 1
		}
	}
	return cands[lo]
}

// filterFinite returns the finite-persistence bars in the given
// dimension (i.e. excludes essential = +Inf-death bars).
func filterFinite(bars []Bar, dim int) []Bar {
	out := make([]Bar, 0, len(bars))
	for _, b := range bars {
		if b.Dim != dim || b.IsEssential() {
			continue
		}
		out = append(out, b)
	}
	return out
}

// countEssential returns the number of essential (Death=+Inf) bars in
// the given dimension.  Bottleneck distance can only be finite when
// the two diagrams agree on this count.
func countEssential(bars []Bar, dim int) int {
	c := 0
	for _, b := range bars {
		if b.Dim == dim && b.IsEssential() {
			c++
		}
	}
	return c
}

// maxHalfPersistence returns the largest (Death - Birth) / 2 in a
// non-empty bar slice.  This is the cost of matching every bar to its
// nearest diagonal projection (the diagonal-only matching).
func maxHalfPersistence(bars []Bar) float64 {
	m := 0.0
	for _, b := range bars {
		h := (b.Death - b.Birth) / 2.0
		if h > m {
			m = h
		}
	}
	return m
}

// linfDistance returns the L^inf (Chebyshev) distance between two
// bars in (birth, death) coordinates.
func linfDistance(p, q Bar) float64 {
	dx := math.Abs(p.Birth - q.Birth)
	dy := math.Abs(p.Death - q.Death)
	if dx > dy {
		return dx
	}
	return dy
}

// candidateThresholds collects, sorts and deduplicates the set of
// scalar values that could be the optimal bottleneck delta: every
// pairwise L^inf distance between a in d1 and b in d2, plus every
// diagonal cost (b.Death - b.Birth)/2, plus 0.  The bottleneck
// optimum is provably one of these candidates because the matching
// graph is piecewise-constant in delta.
func candidateThresholds(a, b []Bar) []float64 {
	cands := make([]float64, 0, len(a)*len(b)+len(a)+len(b)+1)
	cands = append(cands, 0)
	for _, p := range a {
		cands = append(cands, (p.Death-p.Birth)/2.0)
	}
	for _, q := range b {
		cands = append(cands, (q.Death-q.Birth)/2.0)
	}
	for _, p := range a {
		for _, q := range b {
			cands = append(cands, linfDistance(p, q))
		}
	}
	sort.Float64s(cands)
	// Deduplicate (preserving order).
	out := cands[:0]
	prev := math.NaN()
	for _, c := range cands {
		if c != prev {
			out = append(out, c)
			prev = c
		}
	}
	return out
}

// hasPerfectMatching reports whether, at threshold delta, every bar
// in a can be matched (to a bar in b within delta L^inf, OR to its
// diagonal projection if (death-birth)/2 <= delta) AND every bar in b
// can be matched symmetrically.
//
// Encoding: each side has len + len-of-other "diagonal stand-ins" so
// that matching to the diagonal is always available subject to the
// half-persistence threshold.  We then run Hopcroft-Karp bipartite
// matching.
func hasPerfectMatching(a, b []Bar, delta float64) bool {
	la, lb := len(a), len(b)
	// Left vertices: la real-a + lb diagonal-stand-ins-for-b.
	// Right vertices: lb real-b + la diagonal-stand-ins-for-a.
	left := la + lb
	right := lb + la

	adj := make([][]int, left)
	for i := range adj {
		adj[i] = nil
	}

	// Real a -> real b (within delta) and a -> diagonal-of-a (right
	// index lb + i) if that bar's half-persistence <= delta.
	for i := 0; i < la; i++ {
		for j := 0; j < lb; j++ {
			if linfDistance(a[i], b[j]) <= delta+epsBottleneck {
				adj[i] = append(adj[i], j)
			}
		}
		if (a[i].Death-a[i].Birth)/2.0 <= delta+epsBottleneck {
			adj[i] = append(adj[i], lb+i)
		}
	}
	// Diagonal-stand-in for b[j] (left index la + j) -> real b[j]
	// (always available, since the diagonal can absorb any b) AND
	// any a's diagonal-stand-in (right index lb + i) so the
	// diagonal-to-diagonal pairing is free.
	for j := 0; j < lb; j++ {
		// Diagonal-stand-in-for-b[j] connects to real b[j] iff
		// b[j]'s half-persistence <= delta (otherwise b[j] cannot
		// be matched to the diagonal).
		if (b[j].Death-b[j].Birth)/2.0 <= delta+epsBottleneck {
			adj[la+j] = append(adj[la+j], j)
		}
		// Diagonal-to-diagonal pairing: free (cost 0 <= delta).
		for i := 0; i < la; i++ {
			adj[la+j] = append(adj[la+j], lb+i)
		}
	}

	matchL := make([]int, left)
	matchR := make([]int, right)
	for i := range matchL {
		matchL[i] = -1
	}
	for i := range matchR {
		matchR[i] = -1
	}

	matched := 0
	for u := 0; u < left; u++ {
		visited := make([]bool, right)
		if hkAugment(u, adj, matchL, matchR, visited) {
			matched++
		}
	}
	// We need a perfect matching on the larger side; both sides have
	// la + lb vertices by construction, so a perfect matching has
	// la + lb edges.
	return matched == la+lb
}

// hkAugment is the standard augmenting-path search used in
// Hopcroft-Karp / Hungarian-Kuhn bipartite matching.  We use the
// simpler Kuhn variant here because the matching size is small in
// Phase-A consumer scale (n <= 50 bars per dimension).
func hkAugment(u int, adj [][]int, matchL, matchR []int, visited []bool) bool {
	for _, v := range adj[u] {
		if visited[v] {
			continue
		}
		visited[v] = true
		if matchR[v] == -1 || hkAugment(matchR[v], adj, matchL, matchR, visited) {
			matchL[u] = v
			matchR[v] = u
			return true
		}
	}
	return false
}

// epsBottleneck is the slop added when comparing pairwise distances
// against the candidate delta to avoid spurious "just over" failures
// caused by float64 rounding when delta itself was computed as a
// pairwise distance.  Two bars at L^inf distance exactly equal to
// delta are matched.
const epsBottleneck = 1e-12
