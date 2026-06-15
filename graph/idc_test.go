package graph

import "testing"

// TestIDC_ReducesToID is the strongest independent check: with no conditioning,
// IDC must return exactly the (theorem-validated) ID verdict on every case of the
// literature truth table. This pins IDC's spine to established results.
func TestIDC_ReducesToID(t *testing.T) {
	cases := []struct {
		name       string
		nodes      []string
		directed   []Edge
		bidirected []Edge
		x, y       []string
		wantID     bool
	}{
		{"chain", []string{"X", "Y"}, []Edge{{"X", "Y"}}, nil, []string{"X"}, []string{"Y"}, true},
		{"bow arc", []string{"X", "Y"}, []Edge{{"X", "Y"}}, []Edge{{"X", "Y"}}, []string{"X"}, []string{"Y"}, false},
		{"front-door", []string{"X", "M", "Y"}, []Edge{{"X", "M"}, {"M", "Y"}}, []Edge{{"X", "Y"}}, []string{"X"}, []string{"Y"}, true},
		{"IV", []string{"Z", "X", "Y"}, []Edge{{"Z", "X"}, {"X", "Y"}}, []Edge{{"X", "Y"}}, []string{"X"}, []string{"Y"}, false},
		{"napkin", []string{"W", "Z", "X", "Y"}, []Edge{{"W", "Z"}, {"Z", "X"}, {"X", "Y"}}, []Edge{{"W", "X"}, {"W", "Y"}}, []string{"X"}, []string{"Y"}, true},
	}
	for _, c := range cases {
		g := NewADMG(c.nodes, c.directed, c.bidirected)
		idcExpr, idcID, err := g.IdentifyConditionalEffect(c.x, c.y, nil)
		if err != nil {
			t.Errorf("%s: IDC err %v", c.name, err)
			continue
		}
		_, idID, _ := g.IdentifyEffect(c.x, c.y)
		if idcID != idID || idcID != c.wantID {
			t.Errorf("%s: IDC id=%v, ID id=%v, want %v", c.name, idcID, idID, c.wantID)
		}
		if idcID && idcExpr == "" {
			t.Errorf("%s: identifiable but empty expr", c.name)
		}
	}
}

// TestIDC_ConditionOnConfounder: P(y | do(x), z) in Z->X->Y, Z->Y. Conditioning
// on the observed confounder Z blocks the back-door, so the effect is
// identifiable. The IDC reduction moves Z into the intervention (Z is isolated in
// G_{\bar X, \underline Z}, hence m-separated from Y given X).
func TestIDC_ConditionOnConfounder(t *testing.T) {
	g := NewADMG([]string{"Z", "X", "Y"}, []Edge{{"Z", "X"}, {"Z", "Y"}, {"X", "Y"}}, nil)
	expr, id, err := g.IdentifyConditionalEffect([]string{"X"}, []string{"Y"}, []string{"Z"})
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if !id {
		t.Fatalf("P(y|do x, z) should be identifiable when conditioning on the confounder")
	}
	if expr == "" {
		t.Fatalf("identifiable but empty expression")
	}
}

// TestIDC_NonIdentifiableConditional: conditioning does not rescue a bow arc.
func TestIDC_NonIdentifiableConditional(t *testing.T) {
	// X->Y, X<->Y, with a pre-treatment covariate W->X. P(y|do x, w) still
	// inherits the unidentifiable X<->Y confounding of the effect on Y.
	g := NewADMG([]string{"W", "X", "Y"}, []Edge{{"W", "X"}, {"X", "Y"}}, []Edge{{"X", "Y"}})
	_, id, err := g.IdentifyConditionalEffect([]string{"X"}, []string{"Y"}, []string{"W"})
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if id {
		t.Fatalf("bow-arc effect must stay non-identifiable even conditioning on W")
	}
}

// TestMSeparation_CanonicalDAG checks the bidirected->latent expansion: X<->Y are
// NOT m-separated (a latent common cause connects them), while two unconnected
// nodes are. This discriminates canonicalDAG — dropping the latent expansion
// makes X<->Y trivially separated and this fails.
func TestMSeparation_CanonicalDAG(t *testing.T) {
	g := NewADMG([]string{"X", "Y"}, nil, []Edge{{"X", "Y"}})
	if g.mSeparated(setOf([]string{"X"}), setOf([]string{"Y"}), map[string]struct{}{}) {
		t.Error("X<->Y must NOT be m-separated (latent common cause)")
	}
	g2 := NewADMG([]string{"A", "B"}, nil, nil)
	if !g2.mSeparated(setOf([]string{"A"}), setOf([]string{"B"}), map[string]struct{}{}) {
		t.Error("disconnected A,B must be m-separated")
	}
}

func TestIDC_Validation(t *testing.T) {
	g := NewADMG([]string{"X", "Y", "Z"}, []Edge{{"X", "Y"}}, nil)
	if _, _, err := g.IdentifyConditionalEffect([]string{"Q"}, []string{"Y"}, nil); err == nil {
		t.Error("unknown node: expected error")
	}
	if _, _, err := g.IdentifyConditionalEffect([]string{"X"}, []string{"Y"}, []string{"X"}); err == nil {
		t.Error("treatment/conditioning overlap: expected error")
	}
	if _, _, err := g.IdentifyConditionalEffect([]string{"X"}, []string{"Y"}, []string{"Y"}); err == nil {
		t.Error("outcome/conditioning overlap: expected error")
	}
}
