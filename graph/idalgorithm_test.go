package graph

import "testing"

// The identifiability verdicts below are established results from the causal-
// inference literature (Pearl 2009; Tian & Pearl 2002; Shpitser & Pearl 2006).
// Matching all of them is an INDEPENDENT check on the implementation — these are
// theorems, not self-consistency.
func TestIdentifyEffect_LiteratureTruthTable(t *testing.T) {
	cases := []struct {
		name       string
		nodes      []string
		directed   []Edge
		bidirected []Edge
		x, y       []string
		wantID     bool
		note       string
	}{
		{
			name:     "chain X->Y (no confounding)",
			nodes:    []string{"X", "Y"},
			directed: []Edge{{"X", "Y"}},
			x:        []string{"X"}, y: []string{"Y"},
			wantID: true, note: "P(y|do x)=P(y|x)",
		},
		{
			name:     "bow arc X->Y, X<->Y",
			nodes:    []string{"X", "Y"},
			directed: []Edge{{"X", "Y"}}, bidirected: []Edge{{"X", "Y"}},
			x: []string{"X"}, y: []string{"Y"},
			wantID: false, note: "classic non-identifiable (Pearl)",
		},
		{
			name:     "front-door X->M->Y, X<->Y",
			nodes:    []string{"X", "M", "Y"},
			directed: []Edge{{"X", "M"}, {"M", "Y"}}, bidirected: []Edge{{"X", "Y"}},
			x: []string{"X"}, y: []string{"Y"},
			wantID: true, note: "front-door criterion (Pearl)",
		},
		{
			name:     "back-door observed confounder Z->X, Z->Y, X->Y",
			nodes:    []string{"Z", "X", "Y"},
			directed: []Edge{{"Z", "X"}, {"Z", "Y"}, {"X", "Y"}},
			x: []string{"X"}, y: []string{"Y"},
			wantID: true, note: "adjust for Z",
		},
		{
			name:     "instrumental variable Z->X->Y, X<->Y",
			nodes:    []string{"Z", "X", "Y"},
			directed: []Edge{{"Z", "X"}, {"X", "Y"}}, bidirected: []Edge{{"X", "Y"}},
			x: []string{"X"}, y: []string{"Y"},
			wantID: false, note: "IV is NOT nonparametrically identifiable",
		},
		{
			name:     "napkin graph",
			nodes:    []string{"W", "Z", "X", "Y"},
			directed: []Edge{{"W", "Z"}, {"Z", "X"}, {"X", "Y"}},
			bidirected: []Edge{{"W", "X"}, {"W", "Y"}},
			x: []string{"X"}, y: []string{"Y"},
			wantID: true, note: "famous identifiable case (Pearl)",
		},
		{
			name:     "extended bow X->Y, X<->Y with mediator X->M->Y, M<->Y",
			nodes:    []string{"X", "M", "Y"},
			directed: []Edge{{"X", "M"}, {"M", "Y"}, {"X", "Y"}}, bidirected: []Edge{{"X", "Y"}},
			x: []string{"X"}, y: []string{"Y"},
			wantID: false, note: "direct X->Y under X<->Y confounding is non-ID",
		},
		{
			name:     "two-step front-door X->M1->M2->Y, X<->Y",
			nodes:    []string{"X", "M1", "M2", "Y"},
			directed: []Edge{{"X", "M1"}, {"M1", "M2"}, {"M2", "Y"}}, bidirected: []Edge{{"X", "Y"}},
			x: []string{"X"}, y: []string{"Y"},
			wantID: true, note: "front-door generalises along the unconfounded mediator chain",
		},
		{
			name:     "confounded treatment-mediator X->M->Y, X<->M",
			nodes:    []string{"X", "M", "Y"},
			directed: []Edge{{"X", "M"}, {"M", "Y"}}, bidirected: []Edge{{"X", "M"}},
			x: []string{"X"}, y: []string{"Y"},
			wantID: false, note: "P(m|do x) is a bow arc (X->M, X<->M) -> non-ID -> P(y|do x) non-ID",
		},
	}
	for _, c := range cases {
		g := NewADMG(c.nodes, c.directed, c.bidirected)
		expr, gotID, err := g.IdentifyEffect(c.x, c.y)
		if err != nil {
			t.Errorf("%s: unexpected error %v", c.name, err)
			continue
		}
		if gotID != c.wantID {
			t.Errorf("%s: identifiable=%v want %v (%s)", c.name, gotID, c.wantID, c.note)
		}
		if gotID && expr == "" {
			t.Errorf("%s: identifiable but empty expression", c.name)
		}
		if !gotID && expr != "" {
			t.Errorf("%s: non-identifiable but returned expression %q", c.name, expr)
		}
	}
}

// TestIdentifyEffect_TextbookExpressions checks the returned functional on the
// two cases whose closed form everyone knows.
func TestIdentifyEffect_TextbookExpressions(t *testing.T) {
	// Chain X->Y: P(y|do x) = P(y|x).
	g := NewADMG([]string{"X", "Y"}, []Edge{{"X", "Y"}}, nil)
	expr, id, _ := g.IdentifyEffect([]string{"X"}, []string{"Y"})
	if !id || expr != "P(Y|X)" {
		t.Errorf("chain: got id=%v expr=%q, want true / P(Y|X)", id, expr)
	}

	// Back-door Z->X, Z->Y, X->Y: P(y|do x) = Σ_z P(y|x,z) P(z).
	g = NewADMG([]string{"Z", "X", "Y"}, []Edge{{"Z", "X"}, {"Z", "Y"}, {"X", "Y"}}, nil)
	expr, id, _ = g.IdentifyEffect([]string{"X"}, []string{"Y"})
	if !id {
		t.Fatalf("back-door: not identifiable")
	}
	// The adjustment formula Σ_z P(y|x,z)P(z): a marginal over Z whose product
	// contains the conditional P(Y|X,Z). The P(z) term may render either as
	// "P(Z)" or, equivalently, as the joint-marginal "Σ_{X,Y}(P(V))" — the
	// implementation returns a correct-but-unsimplified functional.
	if !contains(expr, "Σ_{Z}") || !contains(expr, "P(Y|X,Z)") {
		t.Errorf("back-door expr %q lacks the adjustment shape Σ_z P(y|x,z)·P(z)", expr)
	}
}

// TestIdentifyEffect_Validation covers malformed input.
func TestIdentifyEffect_Validation(t *testing.T) {
	g := NewADMG([]string{"X", "Y"}, []Edge{{"X", "Y"}}, nil)
	if _, _, err := g.IdentifyEffect([]string{"Q"}, []string{"Y"}); err == nil {
		t.Error("unknown treatment node: expected error")
	}
	if _, _, err := g.IdentifyEffect([]string{"X"}, []string{"X"}); err == nil {
		t.Error("overlapping X and Y: expected error")
	}
}

// TestCComponents checks the confounded-component partition directly.
func TestCComponents(t *testing.T) {
	// X<->Y and Z alone: two components {X,Y} and {Z}.
	g := NewADMG([]string{"X", "Y", "Z"}, nil, []Edge{{"X", "Y"}})
	cc := g.cComponents()
	if len(cc) != 2 {
		t.Fatalf("want 2 c-components, got %d: %v", len(cc), cc)
	}
	// Transitive bidirected chain A<->B<->C: one component.
	g = NewADMG([]string{"A", "B", "C"}, nil, []Edge{{"A", "B"}, {"B", "C"}})
	if cc := g.cComponents(); len(cc) != 1 || len(cc[0]) != 3 {
		t.Fatalf("want one 3-node c-component, got %v", cc)
	}
}

func contains(s, sub string) bool {
	return len(s) >= len(sub) && (s == sub || indexOf(s, sub) >= 0)
}
func indexOf(s, sub string) int {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return i
		}
	}
	return -1
}
