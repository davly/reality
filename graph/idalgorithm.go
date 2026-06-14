package graph

import (
	"sort"
	"strings"
)

// Causal-effect identification: the complete ID algorithm of Shpitser & Pearl
// (2006), "Identification of Joint Interventional Distributions in Recursive
// Semi-Markovian Causal Models" (AAAI; also Tian & Pearl 2002).
//
// Given an ADMG (a DAG plus bidirected edges for latent confounders) and a
// treatment set X and outcome set Y, ID decides whether the interventional
// distribution P(Y | do(X)) is IDENTIFIABLE from the observational joint P(V),
// and if so returns an expression for it in terms of P(V). The procedure is
// COMPLETE: it returns identifiable iff the effect truly is, and otherwise
// reports a non-identifiability certificate (a hedge). Back-door adjustment,
// the front-door criterion, and (the failure of) instrumental variables are all
// special cases of this one decision procedure.
//
// The identifiability VERDICT depends only on graph structure (ancestor sets and
// confounded components); the returned expression is the standard c-component
// factorization. This file is additive to the existing directed-only causal
// primitives (DSeparated, BackDoorAdjustmentSet) — it adds the latent-confounder
// (bidirected) machinery they lack.

// ADMG is an acyclic directed mixed graph: directed edges U->V (U is a direct
// cause of V) plus bidirected edges U<->V (an unobserved common cause of U and
// V). Bidirected edges are unordered. Construct with NewADMG.
type ADMG struct {
	nodes      []string // sorted, deduplicated vertex set V
	directed   []Edge   // U -> V
	bidirected []Edge   // U <-> V (latent confounding), stored canonically (sorted endpoints)
}

// NewADMG builds an ADMG from a node list, directed edges (U->V) and bidirected
// edges (U<->V). Endpoints not in nodes are added. Bidirected endpoints are
// canonicalised (sorted) so U<->V and V<->U coincide. Self-loops are dropped.
func NewADMG(nodes []string, directed, bidirected []Edge) ADMG {
	set := map[string]struct{}{}
	for _, n := range nodes {
		if n != "" {
			set[n] = struct{}{}
		}
	}
	add := func(e Edge) {
		if e[0] != "" {
			set[e[0]] = struct{}{}
		}
		if e[1] != "" {
			set[e[1]] = struct{}{}
		}
	}
	var dir []Edge
	for _, e := range directed {
		if e[0] == "" || e[1] == "" || e[0] == e[1] {
			continue
		}
		add(e)
		dir = append(dir, e)
	}
	var bi []Edge
	for _, e := range bidirected {
		if e[0] == "" || e[1] == "" || e[0] == e[1] {
			continue
		}
		add(e)
		u, v := e[0], e[1]
		if u > v {
			u, v = v, u
		}
		bi = append(bi, Edge{u, v})
	}
	ns := make([]string, 0, len(set))
	for n := range set {
		ns = append(ns, n)
	}
	sort.Strings(ns)
	return ADMG{nodes: ns, directed: dir, bidirected: bi}
}

// IdentifyEffect runs the ID algorithm for P(Y | do(X)). It returns the
// identifying expression (as a string in terms of the observational joint P),
// whether the effect is identifiable, and an error for malformed input (unknown
// node, or overlapping X and Y). When identifiable is false the expression is
// empty and the effect is provably non-identifiable (a hedge exists).
func (g ADMG) IdentifyEffect(treatment, outcome []string) (expr string, identifiable bool, err error) {
	known := setOf(g.nodes)
	for _, n := range append(append([]string{}, treatment...), outcome...) {
		if _, ok := known[n]; !ok {
			return "", false, &idError{"unknown node: " + n}
		}
	}
	x, y := setOf(treatment), setOf(outcome)
	for n := range x {
		if _, ok := y[n]; ok {
			return "", false, &idError{"treatment and outcome overlap at: " + n}
		}
	}
	e, ok := g.id(y, x, &exprP{}, &hedgeBox{}) // P starts as the joint over all of V
	if !ok {
		return "", false, nil
	}
	return e.String(), true, nil
}

// IdentifyEffectWithWitness is IdentifyEffect plus, when the effect is NOT
// identifiable, the hedge certificate explaining why (nil when identifiable).
func (g ADMG) IdentifyEffectWithWitness(treatment, outcome []string) (expr string, identifiable bool, hedge *Hedge, err error) {
	known := setOf(g.nodes)
	for _, n := range append(append([]string{}, treatment...), outcome...) {
		if _, ok := known[n]; !ok {
			return "", false, nil, &idError{"unknown node: " + n}
		}
	}
	x, y := setOf(treatment), setOf(outcome)
	for n := range x {
		if _, ok := y[n]; ok {
			return "", false, nil, &idError{"treatment and outcome overlap at: " + n}
		}
	}
	hb := &hedgeBox{}
	e, ok := g.id(y, x, &exprP{}, hb)
	if !ok {
		return "", false, hb.h, nil
	}
	return e.String(), true, nil, nil
}

func join(xs []string) string {
	out := "{"
	for i, s := range xs {
		if i > 0 {
			out += ","
		}
		out += s
	}
	return out + "}"
}

type idError struct{ msg string }

func (e *idError) Error() string { return "graph: ID: " + e.msg }

// Hedge is the non-identifiability certificate (Shpitser & Pearl 2006). When an
// effect is not identifiable, the algorithm exhibits a hedge: a pair of nested
// C-forests ⟨Forest, Subforest⟩ over which the latent (bidirected) confounding
// entangles the treatment with the outcome's ancestors. Subforest ⊆ Forest,
// Subforest is disjoint from the treatment, and Forest meets the treatment — so
// no observational functional can separate the intervened from the confounded
// part. It tells you WHERE to intervene to restore identifiability (break a
// latent edge inside Forest, or randomise a treatment in it).
type Hedge struct {
	Forest    []string // F: the confounded C-forest at the failure point (meets X)
	Subforest []string // F': the inner C-forest rooted in An(Y), disjoint from X
}

func (h *Hedge) String() string {
	return "hedge⟨F=" + join(h.Forest) + ", F'=" + join(h.Subforest) + "⟩"
}

// hedgeBox carries the hedge captured at the deepest FAIL point up the recursion.
type hedgeBox struct{ h *Hedge }

// id is the recursive Shpitser-Pearl procedure on the current subgraph g.
// y, x are vertex sets; p is the current probabilistic expression; hb captures
// the hedge witness on failure. Returns the identifying expression and whether
// identification succeeded.
func (g ADMG) id(y, x map[string]struct{}, p expr, hb *hedgeBox) (expr, bool) {
	V := setOf(g.nodes)

	// Line 1: no intervention -> marginalise.
	if len(x) == 0 {
		return marginal(diff(V, y), p), true
	}

	// Line 2: restrict to ancestors of Y.
	anY := g.ancestors(y)
	if len(anY) < len(V) { // An(Y) is always within V; V != An(Y) iff it is strictly smaller
		gi := g.induced(anY)
		return gi.id(y, inter(x, anY), marginal(diff(V, anY), p), hb)
	}

	// Line 3: W = (V \ X) \ An(Y) in G_{\bar X}.
	gxbar := g.removeIncomingTo(x)
	anYxbar := gxbar.ancestors(y)
	w := diff(diff(V, x), anYxbar)
	if len(w) > 0 {
		return g.id(y, union(x, w), p, hb)
	}

	// Line 4: c-components of G[V \ X].
	gx := g.induced(diff(V, x))
	cc := gx.cComponents()
	if len(cc) > 1 {
		factors := make([]expr, 0, len(cc))
		for _, s := range cc {
			sub, ok := g.id(setOf(s), diff(V, setOf(s)), p, hb)
			if !ok {
				return nil, false
			}
			factors = append(factors, sub)
		}
		return marginal(diff(V, union(y, x)), product(factors)), true
	}

	// Single c-component S of G[V\X].
	s := setOf(cc[0])
	ccG := g.cComponents()

	// Line 5/6: if the whole graph is one c-component -> hedge -> NON-identifiable.
	if len(ccG) == 1 {
		hb.h = &Hedge{Forest: append([]string{}, g.nodes...), Subforest: idSortedKeys(s)}
		return nil, false
	}

	order := g.topoOrder()
	// Line 7: S is itself a c-component of G.
	if g.containsComponent(ccG, s) {
		return marginal(diff(s, y), g.qFactor(s, order)), true
	}

	// Line 8: S is a strict subset of some c-component S' of G.
	var sprime map[string]struct{}
	for _, c := range ccG {
		cs := setOf(c)
		if subsetEq(cs, s) { // s subset of cs (subsetEq(super, sub))
			sprime = cs
			break
		}
	}
	gsp := g.induced(sprime)
	return gsp.id(inter(y, sprime), inter(x, sprime), gsp.qFactor(sprime, order), hb)
}

// qFactor returns the c-component factorisation Q[S] = prod_{Vi in S} P(Vi | V^{(i-1)})
// where V^{(i-1)} are the predecessors of Vi in the topological order.
func (g ADMG) qFactor(s map[string]struct{}, order []string) expr {
	pos := map[string]int{}
	for i, n := range order {
		pos[n] = i
	}
	members := idSortedKeys(s)
	sort.Slice(members, func(i, j int) bool { return pos[members[i]] < pos[members[j]] })
	factors := make([]expr, 0, len(members))
	for _, v := range members {
		var given []string
		for _, u := range order {
			if pos[u] < pos[v] {
				given = append(given, u)
			}
		}
		factors = append(factors, &exprFactor{vars: []string{v}, given: given})
	}
	if len(factors) == 1 {
		return factors[0]
	}
	return product(factors)
}

// ---- graph operations -------------------------------------------------------

func (g ADMG) ancestors(seeds map[string]struct{}) map[string]struct{} {
	parents := parentMap(g.directed)
	return ancestorsOf(parents, seeds)
}

func (g ADMG) induced(keep map[string]struct{}) ADMG {
	var ns []string
	for _, n := range g.nodes {
		if _, ok := keep[n]; ok {
			ns = append(ns, n)
		}
	}
	in := func(e Edge) bool { _, a := keep[e[0]]; _, b := keep[e[1]]; return a && b }
	var dir, bi []Edge
	for _, e := range g.directed {
		if in(e) {
			dir = append(dir, e)
		}
	}
	for _, e := range g.bidirected {
		if in(e) {
			bi = append(bi, e)
		}
	}
	return ADMG{nodes: ns, directed: dir, bidirected: bi}
}

// removeIncomingTo returns G with directed edges pointing INTO any node of x
// removed (the G_{\bar X} mutilation). Bidirected edges are unaffected.
func (g ADMG) removeIncomingTo(x map[string]struct{}) ADMG {
	var dir []Edge
	for _, e := range g.directed {
		if _, into := x[e[1]]; into {
			continue
		}
		dir = append(dir, e)
	}
	return ADMG{nodes: g.nodes, directed: dir, bidirected: g.bidirected}
}

// cComponents partitions the vertices into maximal sets connected by bidirected
// edges (confounded components). Isolated vertices are singletons. Returns each
// component as a sorted slice; the list is ordered by first member.
func (g ADMG) cComponents() [][]string {
	parent := map[string]string{}
	var find func(string) string
	find = func(a string) string {
		if parent[a] == "" || parent[a] == a {
			parent[a] = a
			return a
		}
		r := find(parent[a])
		parent[a] = r
		return r
	}
	uni := func(a, b string) { parent[find(a)] = find(b) }
	for _, n := range g.nodes {
		parent[n] = n
	}
	for _, e := range g.bidirected {
		uni(e[0], e[1])
	}
	groups := map[string][]string{}
	for _, n := range g.nodes {
		r := find(n)
		groups[r] = append(groups[r], n)
	}
	out := make([][]string, 0, len(groups))
	for _, members := range groups {
		sort.Strings(members)
		out = append(out, members)
	}
	sort.Slice(out, func(i, j int) bool { return out[i][0] < out[j][0] })
	return out
}

func (g ADMG) containsComponent(cc [][]string, s map[string]struct{}) bool {
	for _, c := range cc {
		if len(c) == len(s) && subsetEq(s, setOf(c)) && subsetEq(setOf(c), s) {
			return true
		}
	}
	return false
}

// topoOrder returns a topological order of the directed part (Kahn's algorithm);
// ties broken lexicographically for determinism.
func (g ADMG) topoOrder() []string {
	children := childMap(g.directed)
	indeg := map[string]int{}
	for _, n := range g.nodes {
		indeg[n] = 0
	}
	for _, e := range g.directed {
		indeg[e[1]]++
	}
	var ready []string
	for _, n := range g.nodes {
		if indeg[n] == 0 {
			ready = append(ready, n)
		}
	}
	sort.Strings(ready)
	var order []string
	for len(ready) > 0 {
		n := ready[0]
		ready = ready[1:]
		order = append(order, n)
		kids := idSortedKeys(children[n])
		for _, c := range kids {
			indeg[c]--
			if indeg[c] == 0 {
				ready = append(ready, c)
			}
		}
		sort.Strings(ready)
	}
	return order
}

// ---- set helpers ------------------------------------------------------------

func setOf(xs []string) map[string]struct{} {
	m := make(map[string]struct{}, len(xs))
	for _, x := range xs {
		m[x] = struct{}{}
	}
	return m
}
func idSortedKeys(m map[string]struct{}) []string {
	ks := make([]string, 0, len(m))
	for k := range m {
		ks = append(ks, k)
	}
	sort.Strings(ks)
	return ks
}
func diff(a, b map[string]struct{}) map[string]struct{} {
	out := map[string]struct{}{}
	for k := range a {
		if _, ok := b[k]; !ok {
			out[k] = struct{}{}
		}
	}
	return out
}
func union(a, b map[string]struct{}) map[string]struct{} {
	out := map[string]struct{}{}
	for k := range a {
		out[k] = struct{}{}
	}
	for k := range b {
		out[k] = struct{}{}
	}
	return out
}
func inter(a, b map[string]struct{}) map[string]struct{} {
	out := map[string]struct{}{}
	for k := range a {
		if _, ok := b[k]; ok {
			out[k] = struct{}{}
		}
	}
	return out
}

// subsetEq reports whether sub is a subset of super (sub ⊆ super).
func subsetEq(super, sub map[string]struct{}) bool {
	for k := range sub {
		if _, ok := super[k]; !ok {
			return false
		}
	}
	return true
}

// ---- expression tree --------------------------------------------------------

type expr interface{ String() string }

// exprP is the observational joint P(V) placeholder (used only as the initial p;
// the returned expressions are always concrete factors/products/marginals).
type exprP struct{}

func (exprP) String() string { return "P(V)" }

type exprFactor struct {
	vars  []string
	given []string
}

func (f *exprFactor) String() string {
	v := strings.Join(f.vars, ",")
	if len(f.given) == 0 {
		return "P(" + v + ")"
	}
	g := append([]string{}, f.given...)
	sort.Strings(g)
	return "P(" + v + "|" + strings.Join(g, ",") + ")"
}

type exprProduct struct{ factors []expr }

func (p *exprProduct) String() string {
	parts := make([]string, len(p.factors))
	for i, f := range p.factors {
		parts[i] = f.String()
	}
	sort.Strings(parts)
	return "[" + strings.Join(parts, "*") + "]"
}

type exprMarginal struct {
	over  []string
	inner expr
}

func (m *exprMarginal) String() string {
	if len(m.over) == 0 {
		return m.inner.String()
	}
	o := append([]string{}, m.over...)
	sort.Strings(o)
	return "Σ_{" + strings.Join(o, ",") + "}(" + m.inner.String() + ")"
}

func marginal(over map[string]struct{}, inner expr) expr {
	if len(over) == 0 {
		return inner
	}
	return &exprMarginal{over: idSortedKeys(over), inner: inner}
}
func product(factors []expr) expr {
	if len(factors) == 1 {
		return factors[0]
	}
	return &exprProduct{factors: factors}
}
