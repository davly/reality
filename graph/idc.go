package graph

import "strconv"

// Conditional causal-effect identification: the IDC algorithm of Shpitser &
// Pearl (2006), the conditional companion to the ID algorithm in idalgorithm.go.
//
// IDC decides whether P(Y | do(X), Z) — the effect of do(X) within the
// subpopulation defined by observing Z — is identifiable, and returns the
// functional. It works by the reduction: a conditioning variable Z' can be moved
// into the intervention set whenever Y is independent of Z' given the rest, in
// the graph with arrows into X and out of Z' removed; once no more can be moved,
// the remaining conditional is P(y|do x, z) = ID(y,z ; do x) / Σ_y ID(y,z ; do x).
//
// The d-separation test is run as m-separation on the ADMG: bidirected (latent-
// confounder) edges are expanded to explicit latent parents (the canonical DAG)
// and the existing tested DSeparated is reused.

// IdentifyConditionalEffect runs IDC for P(outcome | do(treatment), conditioning).
// Returns the identifying expression (in terms of the observational joint), an
// identifiability verdict, and an error for malformed input (unknown node, or
// non-disjoint treatment/outcome/conditioning sets). When z is empty this is
// exactly IdentifyEffect (unconditional ID).
func (g ADMG) IdentifyConditionalEffect(treatment, outcome, conditioning []string) (expr string, identifiable bool, err error) {
	known := setOf(g.nodes)
	for _, n := range concat(treatment, outcome, conditioning) {
		if _, ok := known[n]; !ok {
			return "", false, &idError{"unknown node: " + n}
		}
	}
	x, y, z := setOf(treatment), setOf(outcome), setOf(conditioning)
	if n, ok := firstOverlap(x, y); ok {
		return "", false, &idError{"treatment/outcome overlap at: " + n}
	}
	if n, ok := firstOverlap(x, z); ok {
		return "", false, &idError{"treatment/conditioning overlap at: " + n}
	}
	if n, ok := firstOverlap(y, z); ok {
		return "", false, &idError{"outcome/conditioning overlap at: " + n}
	}

	// IDC reduction: move each Z' that is m-separated from Y (given X and the
	// other Z) in G_{\bar X, \underline{Z'}} from conditioning into intervention.
	for {
		moved := false
		for _, zp := range idSortedKeys(z) {
			zpSet := map[string]struct{}{zp: {}}
			gm := g.removeIncomingTo(x).removeOutgoingFrom(zpSet)
			given := union(x, diff(z, zpSet))
			if gm.mSeparated(y, zpSet, given) {
				x = union(x, zpSet)
				z = diff(z, zpSet)
				moved = true
				break
			}
		}
		if !moved {
			break
		}
	}

	// P(y | do x, z) = ID(y ∪ z ; do x) / Σ_y ID(y ∪ z ; do x).
	e, ok := g.id(union(y, z), x, &exprP{}, &hedgeBox{})
	if !ok {
		return "", false, nil
	}
	if len(z) == 0 {
		return e.String(), true, nil // pure interventional distribution; denominator is 1
	}
	den := (&exprMarginal{over: idSortedKeys(y), inner: e}).String()
	return "[" + e.String() + "] / " + den, true, nil
}

// removeOutgoingFrom returns G with directed edges whose tail is in z removed
// (the G_{\underline{Z}} mutilation). Bidirected edges are unaffected.
func (g ADMG) removeOutgoingFrom(z map[string]struct{}) ADMG {
	var dir []Edge
	for _, e := range g.directed {
		if _, out := z[e[0]]; out {
			continue
		}
		dir = append(dir, e)
	}
	return ADMG{nodes: g.nodes, directed: dir, bidirected: g.bidirected}
}

// mSeparated tests whether sets a and b are m-separated given the conditioning
// set, accounting for latent confounding. Bidirected edges are expanded to
// explicit latent parents (the canonical DAG) and the existing directed-graph
// DSeparated is reused. Latent nodes are never in the conditioning set.
func (g ADMG) mSeparated(a, b, given map[string]struct{}) bool {
	return DSeparated(g.canonicalDAG(), idSortedKeys(a), idSortedKeys(b), idSortedKeys(given))
}

// canonicalDAG expands each bidirected edge U<->V into a fresh latent parent
// L -> U, L -> V, yielding a pure directed edge list on which d-separation equals
// m-separation on the ADMG. Latent names are unique and namespaced so they cannot
// collide with observed nodes.
func (g ADMG) canonicalDAG() []Edge {
	out := make([]Edge, len(g.directed), len(g.directed)+2*len(g.bidirected))
	copy(out, g.directed)
	for i, e := range g.bidirected {
		latent := "__U" + strconv.Itoa(i) + "::" + e[0] + "::" + e[1]
		out = append(out, Edge{latent, e[0]}, Edge{latent, e[1]})
	}
	return out
}

func concat(ss ...[]string) []string {
	var out []string
	for _, s := range ss {
		out = append(out, s...)
	}
	return out
}

func firstOverlap(a, b map[string]struct{}) (string, bool) {
	for _, k := range idSortedKeys(a) {
		if _, ok := b[k]; ok {
			return k, true
		}
	}
	return "", false
}
