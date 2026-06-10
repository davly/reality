package graph

import (
	"reflect"
	"sort"
	"testing"
)

// ═══════════════════════════════════════════════════════════════════════════
// DSeparated — classic structural cases
// ═══════════════════════════════════════════════════════════════════════════

// assertDSep is a small helper: asserts DSeparated(edges, X, Y, Z) == want.
func assertDSep(t *testing.T, label string, edges []Edge, x, y, z []string, want bool) {
	t.Helper()
	got := DSeparated(edges, x, y, z)
	if got != want {
		t.Errorf("%s: DSeparated(X=%v, Y=%v, Z=%v) = %v, want %v", label, x, y, z, got, want)
	}
}

func TestDSeparated_Chain(t *testing.T) {
	// Chain: X -> M -> Y.
	// Marginally, X and Y are d-CONNECTED (information flows X..Y).
	// Conditioning on the middle M BLOCKS the chain -> d-separated.
	edges := []Edge{{"X", "M"}, {"M", "Y"}}
	assertDSep(t, "chain unconditioned", edges, []string{"X"}, []string{"Y"}, nil, false)
	assertDSep(t, "chain | M", edges, []string{"X"}, []string{"Y"}, []string{"M"}, true)
}

func TestDSeparated_ForkConfounder(t *testing.T) {
	// Fork / common cause (confounder): X <- Z -> Y.
	// Marginally d-CONNECTED through the common cause Z.
	// Conditioning on Z BLOCKS the fork -> d-separated.
	edges := []Edge{{"Z", "X"}, {"Z", "Y"}}
	assertDSep(t, "fork unconditioned", edges, []string{"X"}, []string{"Y"}, nil, false)
	assertDSep(t, "fork | Z", edges, []string{"X"}, []string{"Y"}, []string{"Z"}, true)
}

func TestDSeparated_Collider(t *testing.T) {
	// Collider / common effect: X -> C <- Y.
	// Marginally d-SEPARATED (a collider blocks the path by default).
	// Conditioning ON the collider C OPENS the path -> d-connected.
	edges := []Edge{{"X", "C"}, {"Y", "C"}}
	assertDSep(t, "collider unconditioned", edges, []string{"X"}, []string{"Y"}, nil, true)
	assertDSep(t, "collider | C (opens)", edges, []string{"X"}, []string{"Y"}, []string{"C"}, false)
}

func TestDSeparated_ColliderDescendant(t *testing.T) {
	// Collider with a descendant: X -> C <- Y, and C -> D.
	// Conditioning on the DESCENDANT D of the collider also OPENS the path.
	edges := []Edge{{"X", "C"}, {"Y", "C"}, {"C", "D"}}
	assertDSep(t, "collider-desc unconditioned", edges, []string{"X"}, []string{"Y"}, nil, true)
	assertDSep(t, "collider-desc | D (opens via descendant)", edges, []string{"X"}, []string{"Y"}, []string{"D"}, false)
	// Conditioning on C directly still opens it.
	assertDSep(t, "collider-desc | C (opens)", edges, []string{"X"}, []string{"Y"}, []string{"C"}, false)
}

func TestDSeparated_ChainConditioningIrrelevantNode(t *testing.T) {
	// Chain X -> M -> Y with an extra unrelated node W (X -> W).
	// Conditioning on W does NOT block the X..Y chain.
	edges := []Edge{{"X", "M"}, {"M", "Y"}, {"X", "W"}}
	assertDSep(t, "chain | W (irrelevant)", edges, []string{"X"}, []string{"Y"}, []string{"W"}, false)
	assertDSep(t, "chain | M still blocks", edges, []string{"X"}, []string{"Y"}, []string{"M"}, true)
}

func TestDSeparated_LongChainPartialBlock(t *testing.T) {
	// X -> A -> B -> C -> Y.
	// Conditioning on ANY interior node blocks the single path.
	edges := []Edge{{"X", "A"}, {"A", "B"}, {"B", "C"}, {"C", "Y"}}
	assertDSep(t, "long chain open", edges, []string{"X"}, []string{"Y"}, nil, false)
	assertDSep(t, "long chain | B", edges, []string{"X"}, []string{"Y"}, []string{"B"}, true)
	assertDSep(t, "long chain | A", edges, []string{"X"}, []string{"Y"}, []string{"A"}, true)
}

func TestDSeparated_TwoPathsOneBlocked(t *testing.T) {
	// Two paths X..Y:
	//   path 1 (fork):     X <- Z -> Y
	//   path 2 (collider): X -> C <- Y
	// Conditioning on Z blocks path 1 but does NOT open path 2 (collider stays
	// closed because neither C nor a descendant is conditioned) -> d-separated.
	edges := []Edge{{"Z", "X"}, {"Z", "Y"}, {"X", "C"}, {"Y", "C"}}
	assertDSep(t, "two-paths unconditioned (fork open)", edges, []string{"X"}, []string{"Y"}, nil, false)
	assertDSep(t, "two-paths | Z (fork blocked, collider closed)", edges, []string{"X"}, []string{"Y"}, []string{"Z"}, true)
	// Conditioning on Z AND C: fork blocked, but collider now open -> connected.
	assertDSep(t, "two-paths | Z,C (collider opens)", edges, []string{"X"}, []string{"Y"}, []string{"Z", "C"}, false)
}

func TestDSeparated_SetVersusSet(t *testing.T) {
	// Multi-node X and Y sets.
	// Graph: A -> C, B -> C (collider at C), and D -> A.
	// X = {A, B}, Y = {D}. D -> A is a directed edge so D..A is connected.
	edges := []Edge{{"A", "C"}, {"B", "C"}, {"D", "A"}}
	assertDSep(t, "set X{A,B} vs Y{D} unconditioned", edges, []string{"A", "B"}, []string{"D"}, nil, false)
	// Conditioning on A blocks D -> A -> (A is in X though). Use Y={D}, X={B}:
	// B only connects to D via the collider C (B -> C <- A <- D); collider
	// closed by default -> d-separated.
	assertDSep(t, "B vs D unconditioned (collider blocks)", edges, []string{"B"}, []string{"D"}, nil, true)
	// Conditioning on C opens the collider: B -> C <- A <- D is now active.
	assertDSep(t, "B vs D | C (collider opens whole path)", edges, []string{"B"}, []string{"D"}, []string{"C"}, false)
}

func TestDSeparated_EmptyAndDisconnected(t *testing.T) {
	edges := []Edge{{"X", "M"}, {"M", "Y"}}
	// Empty X or empty Y -> vacuously d-separated.
	assertDSep(t, "empty X", edges, nil, []string{"Y"}, nil, true)
	assertDSep(t, "empty Y", edges, []string{"X"}, nil, nil, true)

	// Fully disconnected nodes.
	dis := []Edge{{"A", "B"}, {"C", "D"}}
	assertDSep(t, "disconnected components", dis, []string{"A"}, []string{"D"}, nil, true)
	assertDSep(t, "same component connected", dis, []string{"A"}, []string{"B"}, nil, false)
}

func TestDSeparated_NodeWithItself(t *testing.T) {
	// A node is d-connected to itself unless conditioned into Z.
	edges := []Edge{{"X", "M"}, {"M", "Y"}}
	assertDSep(t, "X vs X (self, open)", edges, []string{"X"}, []string{"X"}, nil, false)
}

// ═══════════════════════════════════════════════════════════════════════════
// DSeparated — table-driven golden suite of canonical triples
// ═══════════════════════════════════════════════════════════════════════════

func TestDSeparated_GoldenTable(t *testing.T) {
	type tc struct {
		name    string
		edges   []Edge
		x, y, z []string
		want    bool // true == d-separated
	}
	cases := []tc{
		{"chain open", []Edge{{"X", "M"}, {"M", "Y"}}, S("X"), S("Y"), nil, false},
		{"chain blocked", []Edge{{"X", "M"}, {"M", "Y"}}, S("X"), S("Y"), S("M"), true},
		{"fork open", []Edge{{"Z", "X"}, {"Z", "Y"}}, S("X"), S("Y"), nil, false},
		{"fork blocked", []Edge{{"Z", "X"}, {"Z", "Y"}}, S("X"), S("Y"), S("Z"), true},
		{"collider closed", []Edge{{"X", "C"}, {"Y", "C"}}, S("X"), S("Y"), nil, true},
		{"collider opened", []Edge{{"X", "C"}, {"Y", "C"}}, S("X"), S("Y"), S("C"), false},
		{"collider opened by descendant", []Edge{{"X", "C"}, {"Y", "C"}, {"C", "D"}}, S("X"), S("Y"), S("D"), false},
		{"reverse chain open", []Edge{{"Y", "M"}, {"M", "X"}}, S("X"), S("Y"), nil, false},
		{"reverse chain blocked", []Edge{{"Y", "M"}, {"M", "X"}}, S("X"), S("Y"), S("M"), true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := DSeparated(c.edges, c.x, c.y, c.z)
			if got != c.want {
				t.Errorf("DSeparated(X=%v,Y=%v,Z=%v) = %v, want %v", c.x, c.y, c.z, got, c.want)
			}
		})
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// BackdoorAdjustmentSet
// ═══════════════════════════════════════════════════════════════════════════

// assertBackdoor checks both the ok flag and the returned set (sorted compare).
func assertBackdoor(t *testing.T, label string, edges []Edge, treat, outcome string, wantOK bool, wantSet []string) {
	t.Helper()
	got, ok := BackdoorAdjustmentSet(edges, treat, outcome)
	if ok != wantOK {
		t.Fatalf("%s: ok = %v, want %v (set=%v)", label, ok, wantOK, got)
	}
	if !wantOK {
		if got != nil {
			t.Errorf("%s: expected nil set when not ok, got %v", label, got)
		}
		return
	}
	gotSorted := append([]string(nil), got...)
	sort.Strings(gotSorted)
	wantSorted := append([]string(nil), wantSet...)
	sort.Strings(wantSorted)
	if !reflect.DeepEqual(gotSorted, wantSorted) {
		t.Errorf("%s: set = %v, want %v", label, gotSorted, wantSorted)
	}
}

func TestBackdoor_ClassicConfounder(t *testing.T) {
	// Confounding (back-door): Z -> X (treatment), Z -> Y (outcome), X -> Y.
	// The open back-door path X <- Z -> Y must be blocked by adjusting for {Z}.
	edges := []Edge{{"Z", "X"}, {"Z", "Y"}, {"X", "Y"}}
	assertBackdoor(t, "classic confounder", edges, "X", "Y", true, []string{"Z"})
}

func TestBackdoor_NoConfounding(t *testing.T) {
	// Pure causal chain with no back-door path: X -> M -> Y.
	// No adjustment needed; the empty set is admissible.
	edges := []Edge{{"X", "M"}, {"M", "Y"}}
	assertBackdoor(t, "no confounding (empty set valid)", edges, "X", "Y", true, []string{})
}

func TestBackdoor_DirectEdgeNoConfounder(t *testing.T) {
	// Direct effect only: X -> Y, no common cause. Empty set is admissible.
	edges := []Edge{{"X", "Y"}}
	assertBackdoor(t, "direct edge, empty set", edges, "X", "Y", true, []string{})
}

func TestBackdoor_DoesNotAdjustForMediator(t *testing.T) {
	// Confounder Z plus a mediator M on the causal path:
	//   Z -> X, Z -> Y, X -> M -> Y.
	// M is a DESCENDANT of the treatment X and must NOT be in the adjustment
	// set (adjusting for a mediator blocks part of the causal effect).
	// Correct back-door set is {Z} only.
	edges := []Edge{{"Z", "X"}, {"Z", "Y"}, {"X", "M"}, {"M", "Y"}}
	got, ok := BackdoorAdjustmentSet(edges, "X", "Y")
	if !ok {
		t.Fatalf("expected an admissible set, got ok=false")
	}
	// Must include Z, must exclude the mediator M (descendant of X).
	gotSet := toSet(got)
	if _, hasZ := gotSet["Z"]; !hasZ {
		t.Errorf("adjustment set must contain confounder Z, got %v", got)
	}
	if _, hasM := gotSet["M"]; hasM {
		t.Errorf("adjustment set must NOT contain mediator M (descendant of X), got %v", got)
	}
}

func TestBackdoor_TwoConfounders(t *testing.T) {
	// Two separate confounders:
	//   A -> X, A -> Y  and  B -> X, B -> Y,  plus X -> Y.
	// Both back-door paths must be blocked -> adjustment set {A, B}.
	edges := []Edge{{"A", "X"}, {"A", "Y"}, {"B", "X"}, {"B", "Y"}, {"X", "Y"}}
	assertBackdoor(t, "two confounders", edges, "X", "Y", true, []string{"A", "B"})
}

func TestBackdoor_ConfounderViaChain(t *testing.T) {
	// Confounder reaches the treatment/outcome through chains:
	//   U -> A -> X,  U -> B -> Y,  X -> Y.
	// The back-door path X <- A <- U -> B -> Y is open. Adjusting for the
	// ancestors {U, A, B} blocks it. (Adjusting for U alone would also work,
	// but this function returns the canonical ancestor set.)
	edges := []Edge{{"U", "A"}, {"A", "X"}, {"U", "B"}, {"B", "Y"}, {"X", "Y"}}
	got, ok := BackdoorAdjustmentSet(edges, "X", "Y")
	if !ok {
		t.Fatalf("expected admissible set, got ok=false")
	}
	// Verify the returned set actually blocks the back-door path: in the graph
	// with X's outgoing edges removed, X and Y must be d-separated by it.
	mut := removeOutgoing(edges, "X")
	if !DSeparated(mut, []string{"X"}, []string{"Y"}, got) {
		t.Errorf("returned set %v does NOT block the back-door path", got)
	}
	// And it must not contain any descendant of X (here X's only descendant is Y).
	if _, bad := toSet(got)["Y"]; bad {
		t.Errorf("adjustment set must not contain outcome/descendant Y, got %v", got)
	}
}

func TestBackdoor_MdiatorOnlyBackdoorUnblockable(t *testing.T) {
	// M-bias-like / unblockable case: the only thing that could block the
	// back-door path is itself a collider whose conditioning OPENS another path.
	//   Treatment X, outcome Y.
	//   X <- A,  A -> C <- B,  B -> Y.  (A and B are roots; C is a collider.)
	//   Back-door path X <- A -> C <- B -> Y is CLOSED by default (collider C),
	//   so the empty set is already admissible — no confounding flows.
	edges := []Edge{{"A", "X"}, {"A", "C"}, {"B", "C"}, {"B", "Y"}}
	// X -> Y has no direct edge here; we still ask for a back-door set for the
	// (X, Y) pair. The back-door path is blocked by the collider, empty set OK.
	got, ok := BackdoorAdjustmentSet(edges, "X", "Y")
	if !ok {
		t.Fatalf("expected admissible (empty) set, got ok=false; set=%v", got)
	}
	// The canonical candidate (ancestors of X and Y) would include A and B.
	// Including A and B must NOT open the collider path — verify admissibility
	// directly via the mutilated graph.
	mut := removeOutgoing(edges, "X")
	if !DSeparated(mut, []string{"X"}, []string{"Y"}, got) {
		t.Errorf("returned set %v fails the back-door criterion", got)
	}
}

func TestBackdoor_InvalidInputs(t *testing.T) {
	edges := []Edge{{"Z", "X"}, {"Z", "Y"}, {"X", "Y"}}
	// treatment == outcome.
	if _, ok := BackdoorAdjustmentSet(edges, "X", "X"); ok {
		t.Error("treatment==outcome should be invalid")
	}
	// missing nodes.
	if _, ok := BackdoorAdjustmentSet(edges, "X", "NOPE"); ok {
		t.Error("missing outcome should be invalid")
	}
	if _, ok := BackdoorAdjustmentSet(edges, "NOPE", "Y"); ok {
		t.Error("missing treatment should be invalid")
	}
	// empty strings.
	if _, ok := BackdoorAdjustmentSet(edges, "", "Y"); ok {
		t.Error("empty treatment should be invalid")
	}
}

func TestBackdoor_ReturnedSetIsSortedAndNonNil(t *testing.T) {
	// Determinism: empty admissible set is a non-nil length-0 slice.
	edges := []Edge{{"X", "Y"}}
	got, ok := BackdoorAdjustmentSet(edges, "X", "Y")
	if !ok {
		t.Fatal("expected ok")
	}
	if got == nil {
		t.Error("expected non-nil empty slice for empty admissible set")
	}
	if len(got) != 0 {
		t.Errorf("expected empty set, got %v", got)
	}

	// Sorted order on a multi-element set.
	edges2 := []Edge{{"B", "X"}, {"B", "Y"}, {"A", "X"}, {"A", "Y"}, {"X", "Y"}}
	got2, _ := BackdoorAdjustmentSet(edges2, "X", "Y")
	if !sort.StringsAreSorted(got2) {
		t.Errorf("adjustment set not sorted: %v", got2)
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Internal helper sanity checks (ancestors/descendants)
// ═══════════════════════════════════════════════════════════════════════════

func TestAncestorsDescendants_Helpers(t *testing.T) {
	// A -> B -> C, A -> D.
	edges := []Edge{{"A", "B"}, {"B", "C"}, {"A", "D"}}
	parents := parentMap(edges)
	children := childMap(edges)

	anc := ancestorsOf(parents, map[string]struct{}{"C": {}})
	wantAnc := map[string]struct{}{"A": {}, "B": {}, "C": {}}
	if !reflect.DeepEqual(anc, wantAnc) {
		t.Errorf("ancestorsOf(C) = %v, want %v", keysOf(anc), keysOf(wantAnc))
	}

	desc := descendantsOf(children, map[string]struct{}{"A": {}})
	wantDesc := map[string]struct{}{"A": {}, "B": {}, "C": {}, "D": {}}
	if !reflect.DeepEqual(desc, wantDesc) {
		t.Errorf("descendantsOf(A) = %v, want %v", keysOf(desc), keysOf(wantDesc))
	}
}

// ── tiny local test helpers ────────────────────────────────────────────────

// S wraps a single label into a one-element slice (readability in tables).
func S(x string) []string { return []string{x} }

func keysOf(m map[string]struct{}) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
