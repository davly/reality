package graph

import "sort"

// This file adds causal-graph primitives on directed acyclic graphs (DAGs):
// d-separation testing and back-door adjustment-set discovery. These turn the
// graph package from a structural toolbox into a (small) causal-inference
// toolbox: d-separation answers "are X and Y conditionally independent given Z
// according to this causal structure?", and the back-door criterion answers
// "which variables must I adjust for to get an unbiased causal effect of a
// treatment on an outcome?".
//
// The DAG is represented with the existing Edge = [2]string type (an edge
// {U, V} means U -> V, read as "U is a direct cause of V"). All functions here
// are additive: they introduce no changes to existing graph algorithms.
//
// References:
//   - Pearl, J. (2009). Causality, 2nd ed. (d-separation; back-door criterion).
//   - Shachter, R. (1998). Bayes-Ball: The Rational Pastime. (linear-time
//     reachability formulation of d-separation, used by DSeparated below).

// parentMap builds child -> set-of-parents from an edge list.
// Self-loops and empty endpoints are ignored (a DAG has neither).
func parentMap(edges []Edge) map[string]map[string]struct{} {
	parents := make(map[string]map[string]struct{})
	for _, e := range edges {
		u, v := e[0], e[1]
		if u == "" || v == "" || u == v {
			continue
		}
		if parents[v] == nil {
			parents[v] = make(map[string]struct{})
		}
		parents[v][u] = struct{}{}
	}
	return parents
}

// childMap builds parent -> set-of-children from an edge list, deduplicated.
// Self-loops and empty endpoints are ignored.
func childMap(edges []Edge) map[string]map[string]struct{} {
	children := make(map[string]map[string]struct{})
	for _, e := range edges {
		u, v := e[0], e[1]
		if u == "" || v == "" || u == v {
			continue
		}
		if children[u] == nil {
			children[u] = make(map[string]struct{})
		}
		children[u][v] = struct{}{}
	}
	return children
}

// ancestorsOf returns the set of all ancestors of the given seed nodes,
// inclusive of the seeds themselves. An ancestor of a node is any node from
// which the node is reachable by following directed edges.
//
// This is used by the d-separation collider rule: a collider on a path is
// "open" (lets influence through) when the collider itself OR any of its
// descendants is in the conditioning set — equivalently, when the collider is
// an ancestor of (or equal to) some conditioned node.
//
// Time complexity: O(|V| + |E|).
func ancestorsOf(parents map[string]map[string]struct{}, seeds map[string]struct{}) map[string]struct{} {
	anc := make(map[string]struct{})
	stack := make([]string, 0, len(seeds))
	for s := range seeds {
		stack = append(stack, s)
	}
	for len(stack) > 0 {
		n := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if _, seen := anc[n]; seen {
			continue
		}
		anc[n] = struct{}{}
		for p := range parents[n] {
			if _, seen := anc[p]; !seen {
				stack = append(stack, p)
			}
		}
	}
	return anc
}

// reachState is a node together with the direction we are *leaving* it during
// the Bayes-Ball walk. Tracking direction is what lets a single graph walk
// apply the chain/fork/collider rules correctly.
type reachState struct {
	node string
	// up == true:   we arrived at node from one of its children (the previous
	//               edge was traversed against its arrow, child <- node), so we
	//               are currently moving "up" toward parents.
	// up == false:  we arrived at node from one of its parents (the previous
	//               edge was traversed with its arrow, parent -> node), so we
	//               are currently moving "down" toward children.
	up bool
}

// reachableFrom returns every node reachable from any source node along an
// *active* (d-connecting) path given the conditioning set z, using the
// Bayes-Ball reachability algorithm. The returned set includes a node iff
// there exists an active path from some source to that node.
//
// The three local rules at the middle node m of a consecutive triple are:
//
//   - chain  a -> m -> c  and fork a <- m -> c : the path is active through m
//     iff m is NOT in z.
//   - collider a -> m <- c : the path is active through m iff m IS in z, or
//     some descendant of m is in z (captured by m being in zAncestors).
//
// In reachability terms, from a state (m, up=true) we may continue:
//   - to each parent of m as (parent, up=true)   — fork/chain pass-through,
//     allowed iff m ∉ z;
//   - to each child of m  as (child, up=false)    — fork pass-through,
//     allowed iff m ∉ z.
//
// From a state (m, up=false) — we arrived along parent -> m — we may continue:
//   - to each child of m  as (child, up=false)    — chain pass-through,
//     allowed iff m ∉ z;
//   - to each parent of m as (parent, up=true)    — collider m, allowed iff
//     m is in zAncestors (m or a descendant is conditioned on).
//
// Time complexity: O(|V| + |E|) — each (node, direction) state is expanded once.
func reachableFrom(
	parents, children map[string]map[string]struct{},
	sources, z, zAncestors map[string]struct{},
) map[string]struct{} {
	visitedStates := make(map[reachState]struct{})
	reached := make(map[string]struct{})

	queue := make([]reachState, 0, len(sources))
	for s := range sources {
		// A source is treated as if reached by an incoming arrow, i.e. moving
		// up toward its parents AND down toward its children. We seed both the
		// "up" and "down" states so the walk can leave the source in either
		// direction (the source itself is never a collider on a path it starts).
		queue = append(queue, reachState{s, true})
		reached[s] = struct{}{}
	}

	for len(queue) > 0 {
		st := queue[len(queue)-1]
		queue = queue[:len(queue)-1]
		if _, seen := visitedStates[st]; seen {
			continue
		}
		visitedStates[st] = struct{}{}
		reached[st.node] = struct{}{}

		_, inZ := z[st.node]

		if st.up {
			// Arrived from a child (moving up). Fork/chain pass-through is open
			// only when this middle node is NOT conditioned on.
			if !inZ {
				for p := range parents[st.node] {
					queue = append(queue, reachState{p, true})
				}
				for c := range children[st.node] {
					queue = append(queue, reachState{c, false})
				}
			}
		} else {
			// Arrived from a parent (moving down).
			if !inZ {
				// Chain pass-through: continue down to children.
				for c := range children[st.node] {
					queue = append(queue, reachState{c, false})
				}
			}
			// Collider at this node: continue up to parents only if the
			// collider is "open", i.e. it (or a descendant) is conditioned on.
			if _, openCollider := zAncestors[st.node]; openCollider {
				for p := range parents[st.node] {
					queue = append(queue, reachState{p, true})
				}
			}
		}
	}

	return reached
}

// DSeparated reports whether sets X and Y are d-separated by the conditioning
// set Z in the given DAG. d-separation is the graphical criterion for
// conditional independence: if X and Y are d-separated given Z, then in every
// distribution that factorizes according to the DAG, X is independent of Y
// given Z.
//
// edges:  directed edges of the DAG; {U, V} means U -> V.
// x, y:   the two node sets being tested (order does not matter).
// z:      the conditioning set (may be nil/empty for marginal d-separation).
//
// Returns true when X and Y are d-separated (NO active path connects any node
// in X to any node in Y given Z), and false when they are d-connected.
//
// Convention / edge cases:
//   - Nodes in X, Y, or Z that do not appear in edges are treated as isolated
//     vertices (they connect to nothing), which is the standard reading.
//   - If X and Y overlap on a node that is NOT in Z, that shared node is
//     trivially d-connected to itself, so the result is false. (A node is
//     never d-separated from itself unless it is conditioned away into Z.)
//   - Empty X or empty Y vacuously yields true (nothing to connect).
//
// Precondition: edges must describe a DAG (no directed cycles). The algorithm
// terminates regardless, but the d-separation guarantee only holds for DAGs.
//
// Time complexity: O(|V| + |E|).
func DSeparated(edges []Edge, x, y, z []string) bool {
	xSet := toSet(x)
	ySet := toSet(y)
	zSet := toSet(z)

	if len(xSet) == 0 || len(ySet) == 0 {
		return true
	}

	parents := parentMap(edges)
	children := childMap(edges)

	// A collider is "open" when it, or any of its descendants, is in Z. That is
	// exactly the set of nodes that are ancestors-of-or-equal-to some Z node.
	zAncestors := ancestorsOf(parents, zSet)

	// Walk out from X along active paths; if any Y node is reached, X and Y are
	// d-connected (NOT d-separated).
	reached := reachableFrom(parents, children, xSet, zSet, zAncestors)
	for yn := range ySet {
		if _, hit := reached[yn]; hit {
			return false
		}
	}
	return true
}

// BackdoorAdjustmentSet returns a valid back-door adjustment set for estimating
// the causal effect of treatment on outcome in the given DAG, or (nil, false)
// if the back-door criterion cannot be satisfied by the candidate set this
// function considers.
//
// The back-door criterion (Pearl) says a set Z is admissible for adjusting the
// effect of T on Y when:
//  1. no node in Z is a descendant of T, and
//  2. Z blocks every "back-door" path from T to Y — every path that starts
//     with an arrow INTO T (T <- ...).
//
// Strategy: this function tests the canonical, widely-applicable candidate
//
//	Z = (ancestors of T ∪ ancestors of Y) \ {T, Y} \ descendants(T)
//
// which is the standard sufficient adjustment set whenever any admissible set
// exists among the non-descendants of T. It then *verifies* admissibility
// directly:
//   - condition (1) holds by construction (descendants of T are removed); and
//   - condition (2) is verified by deleting T's outgoing edges (so only
//     back-door paths remain) and checking that T and Y are d-separated by Z in
//     that mutilated graph.
//
// If the verified candidate is admissible it is returned (sorted, for
// determinism). If it is not — e.g. an unblockable back-door path exists, or no
// valid set exists because the only confounding is through an unobservable that
// is itself a descendant of T — the function returns (nil, false).
//
// Notes / edge cases:
//   - The empty set is a valid back-door set exactly when there is no
//     confounding (no open back-door path). In that case this returns
//     ([]string{}, true) — a non-nil, length-0 slice with ok==true.
//   - treatment == outcome, or either not present in the graph, returns
//     (nil, false): there is no well-posed effect to adjust for.
//   - If outcome is a descendant of treatment only through the front door (no
//     back-door confounding), the empty set is returned as admissible.
//
// Time complexity: O(|V| + |E|).
func BackdoorAdjustmentSet(edges []Edge, treatment, outcome string) ([]string, bool) {
	if treatment == "" || outcome == "" || treatment == outcome {
		return nil, false
	}

	allNodes := Nodes(edges)
	if _, ok := allNodes[treatment]; !ok {
		return nil, false
	}
	if _, ok := allNodes[outcome]; !ok {
		return nil, false
	}

	parents := parentMap(edges)
	children := childMap(edges)

	// Descendants of the treatment (inclusive) — forbidden from any back-door
	// adjustment set by criterion (1).
	descT := descendantsOf(children, map[string]struct{}{treatment: {}})

	// Candidate adjustment set: ancestors of treatment and outcome, minus the
	// treatment and outcome themselves, minus any descendant of the treatment.
	ancSeeds := map[string]struct{}{treatment: {}, outcome: {}}
	cand := ancestorsOf(parents, ancSeeds)

	zSet := make(map[string]struct{})
	for n := range cand {
		if n == treatment || n == outcome {
			continue
		}
		if _, isDesc := descT[n]; isDesc {
			continue
		}
		zSet[n] = struct{}{}
	}

	// Verify the back-door criterion (2): on the graph with treatment's
	// OUTGOING edges removed, only back-door paths from treatment remain. The
	// candidate Z is admissible iff treatment and outcome are d-separated there.
	mutilated := removeOutgoing(edges, treatment)
	if !DSeparated(mutilated, []string{treatment}, []string{outcome}, setToSlice(zSet)) {
		return nil, false
	}

	return setToSlice(zSet), true
}

// descendantsOf returns the set of all descendants of the seed nodes, inclusive
// of the seeds. A descendant is any node reachable by following directed edges.
// Time complexity: O(|V| + |E|).
func descendantsOf(children map[string]map[string]struct{}, seeds map[string]struct{}) map[string]struct{} {
	desc := make(map[string]struct{})
	stack := make([]string, 0, len(seeds))
	for s := range seeds {
		stack = append(stack, s)
	}
	for len(stack) > 0 {
		n := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if _, seen := desc[n]; seen {
			continue
		}
		desc[n] = struct{}{}
		for c := range children[n] {
			if _, seen := desc[c]; !seen {
				stack = append(stack, c)
			}
		}
	}
	return desc
}

// removeOutgoing returns a copy of edges with every edge whose source is node
// removed. Used to build the "mutilated" graph (only back-door paths from the
// treatment survive) for verifying the back-door criterion.
func removeOutgoing(edges []Edge, node string) []Edge {
	out := make([]Edge, 0, len(edges))
	for _, e := range edges {
		if e[0] == node {
			continue
		}
		out = append(out, e)
	}
	return out
}

// toSet builds a string set from a slice, skipping empty strings.
func toSet(xs []string) map[string]struct{} {
	s := make(map[string]struct{}, len(xs))
	for _, x := range xs {
		if x != "" {
			s[x] = struct{}{}
		}
	}
	return s
}

// setToSlice returns the set's members as a sorted slice (deterministic order).
// A non-nil, length-0 slice is returned for an empty set.
func setToSlice(s map[string]struct{}) []string {
	out := make([]string, 0, len(s))
	for k := range s {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
