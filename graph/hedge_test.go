package graph

import "testing"

func inSlice(xs []string, v string) bool {
	for _, s := range xs {
		if s == v {
			return true
		}
	}
	return false
}

// TestHedge_StructuralInvariants verifies that every non-identifiable effect
// yields a hedge satisfying its DEFINITION (Shpitser-Pearl): F' ⊆ F, F' disjoint
// from the treatment X, and F meets X. These are checkable properties of the
// certificate — an independent leg beyond the yes/no verdict.
func TestHedge_StructuralInvariants(t *testing.T) {
	cases := []struct {
		name       string
		nodes      []string
		directed   []Edge
		bidirected []Edge
		x, y       []string
	}{
		{"bow arc", []string{"X", "Y"}, []Edge{{"X", "Y"}}, []Edge{{"X", "Y"}}, []string{"X"}, []string{"Y"}},
		{"IV", []string{"Z", "X", "Y"}, []Edge{{"Z", "X"}, {"X", "Y"}}, []Edge{{"X", "Y"}}, []string{"X"}, []string{"Y"}},
		{"extended bow", []string{"X", "M", "Y"}, []Edge{{"X", "M"}, {"M", "Y"}, {"X", "Y"}}, []Edge{{"X", "Y"}}, []string{"X"}, []string{"Y"}},
		{"confounded treatment-mediator", []string{"X", "M", "Y"}, []Edge{{"X", "M"}, {"M", "Y"}}, []Edge{{"X", "M"}}, []string{"X"}, []string{"Y"}},
	}
	for _, c := range cases {
		g := NewADMG(c.nodes, c.directed, c.bidirected)
		_, id, h, err := g.IdentifyEffectWithWitness(c.x, c.y)
		if err != nil {
			t.Errorf("%s: err %v", c.name, err)
			continue
		}
		if id || h == nil {
			t.Errorf("%s: expected non-ID with a hedge, got id=%v hedge=%v", c.name, id, h)
			continue
		}
		if len(h.Forest) == 0 || len(h.Subforest) == 0 {
			t.Errorf("%s: empty hedge %v", c.name, h)
			continue
		}
		// F' ⊆ F
		for _, v := range h.Subforest {
			if !inSlice(h.Forest, v) {
				t.Errorf("%s: Subforest %v not within Forest %v", c.name, h.Subforest, h.Forest)
				break
			}
		}
		// F' ∩ X = ∅ and F ∩ X ≠ ∅
		for _, xv := range c.x {
			if inSlice(h.Subforest, xv) {
				t.Errorf("%s: treatment %s must NOT be in Subforest %v", c.name, xv, h.Subforest)
			}
		}
		metX := false
		for _, xv := range c.x {
			if inSlice(h.Forest, xv) {
				metX = true
			}
		}
		if !metX {
			t.Errorf("%s: Forest %v must meet the treatment %v", c.name, h.Forest, c.x)
		}
	}
}

// TestHedge_NilWhenIdentifiable: identifiable effects carry no hedge.
func TestHedge_NilWhenIdentifiable(t *testing.T) {
	cases := []struct {
		name       string
		nodes      []string
		directed   []Edge
		bidirected []Edge
	}{
		{"chain", []string{"X", "Y"}, []Edge{{"X", "Y"}}, nil},
		{"front-door", []string{"X", "M", "Y"}, []Edge{{"X", "M"}, {"M", "Y"}}, []Edge{{"X", "Y"}}},
		{"napkin", []string{"W", "Z", "X", "Y"}, []Edge{{"W", "Z"}, {"Z", "X"}, {"X", "Y"}}, []Edge{{"W", "X"}, {"W", "Y"}}},
	}
	for _, c := range cases {
		g := NewADMG(c.nodes, c.directed, c.bidirected)
		_, id, h, _ := g.IdentifyEffectWithWitness([]string{"X"}, []string{"Y"})
		if !id || h != nil {
			t.Errorf("%s: expected identifiable with nil hedge, got id=%v hedge=%v", c.name, id, h)
		}
	}
}

// TestHedge_BowArcWitness pins the exact hedge for the canonical bow arc.
func TestHedge_BowArcWitness(t *testing.T) {
	g := NewADMG([]string{"X", "Y"}, []Edge{{"X", "Y"}}, []Edge{{"X", "Y"}})
	_, id, h, _ := g.IdentifyEffectWithWitness([]string{"X"}, []string{"Y"})
	if id || h == nil {
		t.Fatalf("bow arc: expected non-ID hedge")
	}
	// F = {X,Y}, F' = {Y}: the confounded component is {X,Y}; the inner forest
	// rooted in An(Y) excluding X is {Y}.
	if !(len(h.Forest) == 2 && inSlice(h.Forest, "X") && inSlice(h.Forest, "Y")) {
		t.Errorf("bow arc Forest = %v, want {X,Y}", h.Forest)
	}
	if !(len(h.Subforest) == 1 && inSlice(h.Subforest, "Y")) {
		t.Errorf("bow arc Subforest = %v, want {Y}", h.Subforest)
	}
}
