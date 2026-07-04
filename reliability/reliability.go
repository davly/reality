// Package reliability provides reliability-block-diagram (RBD) availability
// math on directed dependency graphs. All functions are deterministic, use
// only the Go standard library plus reality's own graph package, and make
// zero external dependencies.
//
// Where reality/graph answers boolean reachability ("which services become
// unreachable if X dies"), this package answers the probabilistic question a
// reachability closure cannot: "given each dependency's measured availability,
// what availability can this service actually achieve, and which single
// dependency caps it?" Boolean reachability and availability propagation are
// different theories — the former is set membership, the latter is a product
// of probabilities over a coherent structure function.
//
// Component availability a_i is the steady-state probability that component i
// is up (a_i in [0,1]). Availabilities of distinct components are assumed
// independent, the standard RBD assumption.
//
// Sources:
//   - Rausand, M. & Hoyland, A. (2004). System Reliability Theory: Models,
//     Statistical Methods, and Applications, 2nd ed., Wiley. Ch. 4
//     (series eq. 4.6, parallel eq. 4.11, Birnbaum's measure sec. 4.5).
//   - Trivedi, K.S. (2001). Probability and Statistics with Reliability,
//     Queuing, and Computer Science Applications, 2nd ed., Wiley
//     (k-out-of-n sec. 3.5.3, availability A = MTBF/(MTBF+MTTR) sec. 4.2).
//   - Birnbaum, Z.W. (1969). On the importance of different components in a
//     multicomponent system. In Multivariate Analysis-II, ed. P.R. Krishnaiah,
//     Academic Press, pp. 581-592.
//
// Precision: all results are exact products/sums of the inputs to within IEEE
// 754 double rounding; golden tolerances are 1e-12 (accumulating products) or
// tighter for exact identities. Inputs are assumed in [0,1]; behaviour outside
// that range is undefined (garbage in, garbage out — no silent clamping).
package reliability

import (
	"math"

	"github.com/davly/reality/graph"
)

// SeriesAvailability returns the availability of a series system: the system
// is up iff every component is up, so A = product(a_i).
//
// The empty product is 1.0 (the series identity: a system with no components
// imposes no constraint). A single perfect component (1.0) is neutral; a single
// dead component (0.0) drives the whole series to 0.
//
// Source: Rausand & Hoyland (2004), eq. 4.6. Time complexity: O(n).
func SeriesAvailability(a []float64) float64 {
	p := 1.0
	for _, x := range a {
		p *= x
	}
	return p
}

// ParallelAvailability returns the availability of an active-parallel
// (redundant) system: the system is up iff at least one component is up, so
// A = 1 - product(1 - a_i).
//
// The empty parallel system is 0.0 (no branch can carry the load). A single
// perfect branch (1.0) forces the result to 1.0; a single dead branch (0.0)
// is neutral.
//
// Source: Rausand & Hoyland (2004), eq. 4.11. Time complexity: O(n).
func ParallelAvailability(a []float64) float64 {
	if len(a) == 0 {
		return 0.0
	}
	q := 1.0
	for _, x := range a {
		q *= (1 - x)
	}
	return 1 - q
}

// KofN returns the availability of a k-out-of-n system of identical,
// independent components each with availability a: the system is up iff at
// least k of the n components are up.
//
//	A = sum_{j=k}^{n} C(n,j) a^j (1-a)^(n-j)
//
// Special cases: k <= 0 returns 1.0 (zero components required is always
// satisfied); k > n returns 0.0 (impossible). k == n reduces to a series
// system (a^n); k == 1 reduces to a parallel system (1 - (1-a)^n).
//
// Source: Trivedi (2001), sec. 3.5.3 (this is the reliability of a k-out-of-n
// structure; the 2-out-of-3 case is classic triple-modular-redundancy).
// Time complexity: O(n).
func KofN(k, n int, a float64) float64 {
	if k <= 0 {
		return 1.0
	}
	if k > n {
		return 0.0
	}
	sum := 0.0
	for j := k; j <= n; j++ {
		sum += binom(n, j) * math.Pow(a, float64(j)) * math.Pow(1-a, float64(n-j))
	}
	return sum
}

// binom computes the binomial coefficient C(n,k) using the numerically stable
// multiplicative recurrence, accumulating in float64. Redundancy counts are
// small so this never overflows; the multiplicative form avoids computing
// large factorials. C(n,k) = C(n,k-1) * (n-k+1)/k.
func binom(n, k int) float64 {
	if k < 0 || k > n {
		return 0.0
	}
	if k > n-k {
		k = n - k // symmetry: fewer multiplications
	}
	c := 1.0
	for i := 0; i < k; i++ {
		c = c * float64(n-i) / float64(i+1)
	}
	return c
}

// AvailabilityFromMTBF converts mean-time-between-failures and mean-time-to-
// repair into steady-state availability: A = MTBF / (MTBF + MTTR).
//
// Returns NaN if MTBF + MTTR == 0 (undefined). MTTR == 0 gives 1.0.
//
// Source: Trivedi (2001), sec. 4.2. Time complexity: O(1).
func AvailabilityFromMTBF(mtbf, mttr float64) float64 {
	d := mtbf + mttr
	if d == 0 {
		return math.NaN()
	}
	return mtbf / d
}

// SystemAvailability computes the achievable availability of target in a
// dependency DAG. An edge (X, Y) means "X requires Y" (X depends on Y), the
// same orientation reality/graph uses. The target is available iff it and
// every node in its transitive required-dependency closure are up, so
//
//	A_sys(target) = product over {target} union deps*(target) of a_v
//
// where deps*(target) is the set of nodes reachable from target by following
// edges forward. Because the closure is a SET, a shared ("diamond") dependency
// is counted exactly ONCE — the naive recursion A(u) = a_u * product A(child)
// double-counts a shared descendant and is wrong; this function does not.
//
// A node present in the graph but absent from avail is treated as availability
// 1.0 (the series identity). This is optimistic — an unknown component does not
// lower the bound — so supply every component's availability for a meaningful
// result. Availabilities are read from avail by node name.
//
// Cycles (which a well-formed dependency graph must not contain) are traversed
// safely via a visited set; the result is the product over the reachable set.
//
// Time complexity: O(|V| + |E|). Source: series composition over a coherent
// structure, Rausand & Hoyland (2004) Ch. 4.
func SystemAvailability(edges []graph.Edge, avail map[string]float64, target string) float64 {
	adj := graph.AdjacencyList(edges)
	closure := dependencyClosure(adj, target)
	return productOver(closure, avail, nil)
}

// BirnbaumImportance returns the Birnbaum importance of component in the
// availability of target: the marginal change in system availability per unit
// change in the component's availability,
//
//	I_B(component) = dA_sys/da_component = h(a | a_component=1) - h(a | a_component=0)
//
// computed by pivotal decomposition (force the component up, force it down,
// take the difference). This is exact because the availability structure
// function is multilinear in each a_i. For a series system this equals the
// product of every OTHER component's availability in target's closure, i.e.
// A_sys / a_component: the component with the LOWEST availability has the
// HIGHEST importance, which is precisely the dependency to harden first.
//
// A component not in target's dependency closure has importance 0 (changing it
// cannot move the system). The target itself has importance equal to the
// product of its dependencies' availabilities.
//
// Time complexity: O(|V| + |E|). Source: Birnbaum (1969); Rausand & Hoyland
// (2004), sec. 4.5.
func BirnbaumImportance(edges []graph.Edge, avail map[string]float64, target, component string) float64 {
	adj := graph.AdjacencyList(edges)
	closure := dependencyClosure(adj, target)
	if _, in := closure[component]; !in {
		return 0.0
	}
	up := productOver(closure, avail, map[string]float64{component: 1.0})
	down := productOver(closure, avail, map[string]float64{component: 0.0})
	return up - down
}

// BirnbaumImportances returns the Birnbaum importance of every node in target's
// dependency closure (including target itself), keyed by node name. Nodes
// outside the closure are omitted (their importance is 0). This is the ranked
// hardening input: the node with the largest value is the highest-leverage
// component to improve.
//
// Time complexity: O(|closure| * (|V| + |E|)).
func BirnbaumImportances(edges []graph.Edge, avail map[string]float64, target string) map[string]float64 {
	adj := graph.AdjacencyList(edges)
	closure := dependencyClosure(adj, target)
	out := make(map[string]float64, len(closure))
	for node := range closure {
		up := productOver(closure, avail, map[string]float64{node: 1.0})
		down := productOver(closure, avail, map[string]float64{node: 0.0})
		out[node] = up - down
	}
	return out
}

// LimitingDependency returns the dependency (a node in target's closure OTHER
// than target itself) with the lowest availability — the bottleneck that caps
// target's achievable availability in a series system — together with that
// availability. Ties are broken by lexicographically smallest node name for
// determinism.
//
// If target has no dependencies (its closure is just itself), returns
// ("", NaN): the service's own availability is the only factor and there is no
// limiting dependency. A dependency absent from avail is treated as 1.0 and so
// never limits.
//
// Time complexity: O(|V| + |E|). This is the direct answer to "which single
// dependency should the operator harden?" — hardening the min-availability node
// in a series chain yields the largest achievable-availability gain (it also
// carries the highest Birnbaum importance).
func LimitingDependency(edges []graph.Edge, avail map[string]float64, target string) (string, float64) {
	adj := graph.AdjacencyList(edges)
	closure := dependencyClosure(adj, target)
	best := ""
	bestA := math.Inf(1)
	for node := range closure {
		if node == target {
			continue
		}
		a := 1.0
		if v, ok := avail[node]; ok {
			a = v
		}
		if a < bestA || (a == bestA && node < best) {
			bestA = a
			best = node
		}
	}
	if best == "" {
		return "", math.NaN()
	}
	return best, bestA
}

// dependencyClosure returns the set of nodes reachable from target by following
// dependency edges forward (target requires each), including target itself.
// A visited set makes it safe against cycles.
func dependencyClosure(adj map[string][]string, target string) map[string]struct{} {
	seen := make(map[string]struct{})
	var dfs func(n string)
	dfs = func(n string) {
		if _, ok := seen[n]; ok {
			return
		}
		seen[n] = struct{}{}
		for _, m := range adj[n] {
			dfs(m)
		}
	}
	dfs(target)
	return seen
}

// productOver multiplies the availability of every node in the set, reading
// from override first (if present), then avail, defaulting a missing node to
// 1.0 (the series identity).
func productOver(set map[string]struct{}, avail, override map[string]float64) float64 {
	p := 1.0
	for node := range set {
		if override != nil {
			if v, ok := override[node]; ok {
				p *= v
				continue
			}
		}
		if v, ok := avail[node]; ok {
			p *= v
		}
	}
	return p
}
