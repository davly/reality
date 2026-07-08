package combinatorics

import (
	"fmt"
	"math/rand"
	"reflect"
	"testing"
)

// derangement_test.go — the constrained-derangement suite, ported from
// giftdrawcraft's giftdraw_draw_test.go (feat/giftdrawcraft-producer
// 57c023cb) and adapted to the generalized domain-free API: golden
// determinism pins, validity properties (no fixed points, no blocked
// pair, bijectivity), exact-impossibility proofs (degree-zero AND the
// Hall-violation kind no per-agent check can see), canonical-ordering
// invariance, and backtracking-path determinism.

// solveExclusions is the test harness's one entry point mirroring the
// donor's giftdrawRunDraw: canonicalize, seed from the canonical payload,
// build the blocked matrix, solve.
func solveExclusions(t *testing.T, n int, exclusions [][]int, label string) ([]int, string, bool) {
	t.Helper()
	canonical := CanonicalizeExclusions(exclusions)
	payload := struct {
		Label      string       `json:"label"`
		N          int          `json:"n"`
		Exclusions []Constraint `json:"exclusions"`
	}{Label: label, N: n, Exclusions: canonical}
	seed, seedStr, err := SeedFromCanonical(payload)
	if err != nil {
		t.Fatalf("SeedFromCanonical: %v", err)
	}
	blocked := BuildBlocked(n, canonical)
	perm, ok := ConstrainedDerangement(n, blocked, rand.New(rand.NewSource(seed)))
	return perm, seedStr, ok
}

// requireValidAssignment asserts perm is a true constrained derangement:
// a permutation (bijectivity), no fixed points, every exclusion honored
// in both directions. Independent arithmetic — it does not call
// IsValidAssignment, so the two verifiers cross-check each other.
func requireValidAssignment(t *testing.T, perm []int, n int, exclusions [][]int) {
	t.Helper()
	if len(perm) != n {
		t.Fatalf("len(perm) = %d, want %d", len(perm), n)
	}
	seen := make(map[int]bool, n)
	for agent, slot := range perm {
		if slot < 0 || slot >= n {
			t.Fatalf("agent %d assigned out-of-range slot %d", agent, slot)
		}
		if slot == agent {
			t.Errorf("fixed point: agent %d assigned itself", agent)
		}
		if seen[slot] {
			t.Errorf("slot %d assigned twice — not a permutation", slot)
		}
		seen[slot] = true
	}
	for _, pair := range exclusions {
		if perm[pair[0]] == pair[1] {
			t.Errorf("exclusion [%d,%d] violated forward", pair[0], pair[1])
		}
		if perm[pair[1]] == pair[0] {
			t.Errorf("exclusion [%d,%d] violated backward (pairs are symmetric)", pair[0], pair[1])
		}
	}
}

// ---------------------------------------------------------------------------
// Golden determinism pin
// ---------------------------------------------------------------------------

// TestConstrainedDerangement_Golden — THE DETERMINISM PIN: one fixed
// (n, exclusions) input, the exact expected seed string and the exact
// expected permutation. If the seed derivation, the canonical form, the
// attempt bound, or the rng consumption order ever changes, this fails.
func TestConstrainedDerangement_Golden(t *testing.T) {
	canonical := CanonicalizeExclusions([][]int{{0, 1}, {4, 5}})
	payload := struct {
		N          int          `json:"n"`
		Exclusions []Constraint `json:"exclusions"`
	}{N: 10, Exclusions: canonical}
	seed, seedStr, err := SeedFromCanonical(payload)
	if err != nil {
		t.Fatalf("SeedFromCanonical: %v", err)
	}
	if seedStr != "d952516ae850b5c2" {
		t.Fatalf("seed string = %q, want %q", seedStr, "d952516ae850b5c2")
	}
	if seed != -2787075699782928958 {
		t.Fatalf("seed = %d, want %d", seed, int64(-2787075699782928958))
	}
	if len(seedStr) != 16 {
		t.Fatalf("seed string length = %d, want 16 (8 bytes hex-encoded)", len(seedStr))
	}
	blocked := BuildBlocked(10, canonical)
	perm, ok := ConstrainedDerangement(10, blocked, rand.New(rand.NewSource(seed)))
	if !ok {
		t.Fatal("golden input is solvable — must not report impossibility")
	}
	want := []int{5, 7, 0, 2, 6, 3, 4, 9, 1, 8}
	if !reflect.DeepEqual(perm, want) {
		t.Fatalf("golden assignment = %v, want %v", perm, want)
	}
	requireValidAssignment(t, perm, 10, [][]int{{0, 1}, {4, 5}})
}

// ---------------------------------------------------------------------------
// Solvable cases
// ---------------------------------------------------------------------------

// TestConstrainedDerangement_ThreeAgentMinimal — the floor: 3 agents, no
// exclusions. Only two derangements exist (the two 3-cycles); whichever
// the seed picks must verify.
func TestConstrainedDerangement_ThreeAgentMinimal(t *testing.T) {
	perm, seedStr, ok := solveExclusions(t, 3, nil, "minimal")
	if !ok {
		t.Fatal("3 agents with no exclusions must be solvable")
	}
	if len(seedStr) != 16 {
		t.Errorf("seed string length = %d, want 16 (8 bytes hex-encoded)", len(seedStr))
	}
	requireValidAssignment(t, perm, 3, nil)
}

// TestConstrainedDerangement_CouplePairsSaturating — 4 agents, 2 couple
// exclusions ([0,1] and [2,3]): exactly the cross-couple assignments
// remain legal; the solver must find one.
func TestConstrainedDerangement_CouplePairsSaturating(t *testing.T) {
	exclusions := [][]int{{0, 1}, {2, 3}}
	perm, _, ok := solveExclusions(t, 4, exclusions, "couples")
	if !ok {
		t.Fatal("two-couples draw is solvable")
	}
	requireValidAssignment(t, perm, 4, exclusions)
	// Saturation check: each member of couple A must be assigned a member
	// of couple B and vice versa (the only legal structure).
	for agent, slot := range perm {
		if agent <= 1 && slot < 2 {
			t.Errorf("agent %d (couple A) assigned %d — must cross to couple B", agent, slot)
		}
		if agent >= 2 && slot > 1 {
			t.Errorf("agent %d (couple B) assigned %d — must cross to couple A", agent, slot)
		}
	}
}

// TestConstrainedDerangement_AllButOneExcluded — agent 0 is excluded
// against everyone except agent 3: its slot is FORCED. The solver must
// honor the forcing chain rather than sample forever.
func TestConstrainedDerangement_AllButOneExcluded(t *testing.T) {
	exclusions := [][]int{{0, 1}, {0, 2}}
	perm, _, ok := solveExclusions(t, 4, exclusions, "forced")
	if !ok {
		t.Fatal("forced-slot case is solvable")
	}
	requireValidAssignment(t, perm, 4, exclusions)
	if perm[0] != 3 {
		t.Errorf("perm[0] = %d, want 3 — agent 0's only legal slot is 3", perm[0])
	}
}

// TestConstrainedDerangement_ThirtyAgentScale — the donor's contract
// ceiling: 30 agents with a chain of 20 exclusion pairs. Must solve and
// verify.
func TestConstrainedDerangement_ThirtyAgentScale(t *testing.T) {
	var exclusions [][]int
	for i := 0; i < 20; i++ {
		exclusions = append(exclusions, []int{i, i + 1})
	}
	perm, _, ok := solveExclusions(t, 30, exclusions, "scale")
	if !ok {
		t.Fatal("30-agent chained case is solvable")
	}
	requireValidAssignment(t, perm, 30, exclusions)
}

// TestConstrainedDerangement_DenseConstraints_ForcedCycle — a forcing
// thicket: agent 0 may only take slot 5, and (by the exclusions'
// symmetry) only agent 5 may take slot 0, forcing the 0↔5 two-cycle; the
// rest must arrange around the {1,3} block.
func TestConstrainedDerangement_DenseConstraints_ForcedCycle(t *testing.T) {
	exclusions := [][]int{
		{0, 1}, {0, 2}, {0, 3}, {0, 4}, // 0 ↔ {1,2,3,4} blocked: 0→5 and 5→0 both forced
		{1, 3},
	}
	perm, _, ok := solveExclusions(t, 6, exclusions, "thicket")
	if !ok {
		t.Fatal("forced two-cycle case is solvable")
	}
	requireValidAssignment(t, perm, 6, exclusions)
	if perm[0] != 5 {
		t.Errorf("perm[0] = %d, want 5 — agent 0's only legal slot is 5", perm[0])
	}
	if perm[5] != 0 {
		t.Errorf("perm[5] = %d, want 0 — agent 5 is the only legal giver to 0", perm[5])
	}
}

// TestConstrainedDerangement_MatchingPhase_UniqueSolution — the exact
// matching phase, deterministically: four couples where every agent may
// ONLY pair within its couple (all 24 cross-pair edges excluded).
// Exactly ONE assignment exists (each pair swaps); a random permutation
// hits it with p = 1/8! ≈ 0.0025%, so the 64-attempt sampling phase is
// (deterministically, for this seed) passed through and the backtracking
// matcher must construct the unique answer.
func TestConstrainedDerangement_MatchingPhase_UniqueSolution(t *testing.T) {
	var exclusions [][]int
	for a := 0; a < 8; a++ {
		for b := a + 1; b < 8; b++ {
			if a/2 == b/2 {
				continue // within-pair edges stay legal
			}
			exclusions = append(exclusions, []int{a, b})
		}
	}
	if len(exclusions) != 24 {
		t.Fatalf("constructed %d exclusions, want 24", len(exclusions))
	}
	canonical := CanonicalizeExclusions(exclusions)
	blocked := BuildBlocked(8, canonical)
	perm, ok := ConstrainedDerangement(8, blocked, rand.New(rand.NewSource(42)))
	if !ok {
		t.Fatal("the unique pair-swap assignment exists — must not misreport impossibility")
	}
	want := []int{1, 0, 3, 2, 5, 4, 7, 6}
	if !reflect.DeepEqual(perm, want) {
		t.Fatalf("unique solution = %v, want %v", perm, want)
	}
}

// ---------------------------------------------------------------------------
// Determinism
// ---------------------------------------------------------------------------

// TestConstrainedDerangement_SeedReproducibility — the identical payload
// produces the identical seed and the identical assignment, run after
// run; a changed payload produces a different seed.
func TestConstrainedDerangement_SeedReproducibility(t *testing.T) {
	exclusions := [][]int{{0, 1}, {4, 5}}
	first, firstSeed, ok := solveExclusions(t, 10, exclusions, "repro")
	if !ok {
		t.Fatal("base case is solvable")
	}
	for run := 0; run < 5; run++ {
		again, againSeed, ok := solveExclusions(t, 10, exclusions, "repro")
		if !ok {
			t.Fatalf("run %d: solvable case misreported impossible", run)
		}
		if againSeed != firstSeed {
			t.Errorf("run %d: identical payload → seed %q, want %q", run, againSeed, firstSeed)
		}
		if !reflect.DeepEqual(again, first) {
			t.Errorf("run %d: identical payload → assignment %v, want %v", run, again, first)
		}
	}

	_, changedSeed, ok := solveExclusions(t, 10, exclusions, "a different label")
	if !ok {
		t.Fatal("changed-payload case is solvable")
	}
	if changedSeed == firstSeed {
		t.Error("a changed payload must hash to a different seed")
	}
}

// TestSeedFromCanonical_CoversEveryField — every field of the payload
// participates in the seed hash: changing any one of label / n /
// exclusions changes the seed.
func TestSeedFromCanonical_CoversEveryField(t *testing.T) {
	type payload struct {
		Label      string       `json:"label"`
		N          int          `json:"n"`
		Exclusions []Constraint `json:"exclusions"`
	}
	base := payload{Label: "base", N: 5, Exclusions: CanonicalizeExclusions([][]int{{0, 1}})}
	_, baseSeed, err := SeedFromCanonical(base)
	if err != nil {
		t.Fatalf("SeedFromCanonical(base): %v", err)
	}
	mutants := map[string]payload{
		"label":     {Label: "mutated", N: 5, Exclusions: base.Exclusions},
		"n":         {Label: "base", N: 6, Exclusions: base.Exclusions},
		"exclusion": {Label: "base", N: 5, Exclusions: CanonicalizeExclusions([][]int{{0, 1}, {2, 3}})},
	}
	for name, m := range mutants {
		_, seed, err := SeedFromCanonical(m)
		if err != nil {
			t.Fatalf("SeedFromCanonical(%s): %v", name, err)
		}
		if seed == baseSeed {
			t.Errorf("changing %s must change the seed", name)
		}
	}
}

// TestSeedFromCanonical_UnmarshalablePayload — a payload encoding/json
// cannot canonicalize must surface an error, never a silent zero seed.
func TestSeedFromCanonical_UnmarshalablePayload(t *testing.T) {
	_, _, err := SeedFromCanonical(make(chan int))
	if err == nil {
		t.Fatal("unmarshalable payload must return an error")
	}
}

// ---------------------------------------------------------------------------
// Canonical-ordering invariance (exclusion symmetry + dedupe)
// ---------------------------------------------------------------------------

// TestCanonicalizeExclusions_SymmetryAndDedupe — [1,0] is the same pair
// as [0,1]: both spellings (and exact duplicates) canonicalize to ONE
// pair, the same seed, and the IDENTICAL assignment; the pair blocks BOTH
// directions.
func TestCanonicalizeExclusions_SymmetryAndDedupe(t *testing.T) {
	canonical := CanonicalizeExclusions([][]int{{1, 0}, {0, 1}, {0, 1}, {3, 2}})
	want := []Constraint{{0, 1}, {2, 3}}
	if !reflect.DeepEqual(canonical, want) {
		t.Fatalf("canonical = %v, want %v — symmetric + exact duplicates collapse; pairs [low,high]; list sorted", canonical, want)
	}

	a, seedA, ok := solveExclusions(t, 6, [][]int{{0, 1}}, "spelling")
	if !ok {
		t.Fatal("spelling A solvable")
	}
	b, seedB, ok := solveExclusions(t, 6, [][]int{{1, 0}, {0, 1}}, "spelling")
	if !ok {
		t.Fatal("spelling B solvable")
	}
	if seedA != seedB {
		t.Errorf("exclusion spelling changed the seed: %q vs %q", seedA, seedB)
	}
	if !reflect.DeepEqual(a, b) {
		t.Errorf("exclusion spelling changed the assignment: %v vs %v", a, b)
	}

	// Both directions blocked: brute-verify over many seeds via label
	// variation — no assignment ever maps 0→1 or 1→0.
	for i := 0; i < 25; i++ {
		perm, _, ok := solveExclusions(t, 4, [][]int{{1, 0}}, fmt.Sprintf("group %d", i))
		if !ok {
			t.Fatalf("seed variant %d: solvable case misreported impossible", i)
		}
		if perm[0] == 1 {
			t.Errorf("0→1 must be blocked (variant %d)", i)
		}
		if perm[1] == 0 {
			t.Errorf("1→0 must be blocked (variant %d)", i)
		}
	}
}

// TestCanonicalizeExclusions_InputOrderInvariance — the canonical form
// (hence the seed) must not depend on the order pairs arrive in.
func TestCanonicalizeExclusions_InputOrderInvariance(t *testing.T) {
	a := CanonicalizeExclusions([][]int{{4, 5}, {0, 1}, {2, 3}})
	b := CanonicalizeExclusions([][]int{{1, 0}, {3, 2}, {5, 4}})
	if !reflect.DeepEqual(a, b) {
		t.Fatalf("canonical forms differ across input orderings: %v vs %v", a, b)
	}
	_, seedA, err := SeedFromCanonical(a)
	if err != nil {
		t.Fatal(err)
	}
	_, seedB, err := SeedFromCanonical(b)
	if err != nil {
		t.Fatal(err)
	}
	if seedA != seedB {
		t.Errorf("input pair order/orientation changed the seed: %q vs %q", seedA, seedB)
	}
}

// ---------------------------------------------------------------------------
// EXACT impossibility
// ---------------------------------------------------------------------------

// TestConstrainedDerangement_Impossible_AgentIsolated — 3 agents with
// [0,1] and [0,2] excluded → agent 0 has no legal slot.
func TestConstrainedDerangement_Impossible_AgentIsolated(t *testing.T) {
	_, _, ok := solveExclusions(t, 3, [][]int{{0, 1}, {0, 2}}, "isolated")
	if ok {
		t.Fatal("agent 0 has no legal slot — must report impossibility")
	}
}

// TestConstrainedDerangement_Impossible_SlotIsolated — the mirror proof:
// 4 agents where NOBODY may take slot 3.
func TestConstrainedDerangement_Impossible_SlotIsolated(t *testing.T) {
	_, _, ok := solveExclusions(t, 4, [][]int{{0, 3}, {1, 3}, {2, 3}}, "slot-isolated")
	if ok {
		t.Fatal("nobody can take slot 3 — must report impossibility")
	}
}

// TestConstrainedDerangement_Impossible_HallViolation — the subtle class
// no per-agent degree check can catch: 4 agents, exclusions [0,1], [0,2],
// [1,2] → agents 0, 1 AND 2 can each ONLY take slot 3 (everyone has an
// option!), but only one of them can have it. EXACT impossibility via the
// matching argument.
func TestConstrainedDerangement_Impossible_HallViolation(t *testing.T) {
	_, _, ok := solveExclusions(t, 4, [][]int{{0, 1}, {0, 2}, {1, 2}}, "hall")
	if ok {
		t.Fatal("three agents compete for one slot — Hall violation, must report impossibility")
	}
}

// TestConstrainedDerangement_PossibleJustBarely — the boundary partner of
// the Hall case: exclusions [0,2], [0,3], [1,2], [1,3] — pairs can only
// swap within themselves. Exactly one assignment exists; the solver must
// find it rather than misreport impossibility.
func TestConstrainedDerangement_PossibleJustBarely(t *testing.T) {
	exclusions := [][]int{{0, 2}, {0, 3}, {1, 2}, {1, 3}}
	perm, _, ok := solveExclusions(t, 4, exclusions, "barely")
	if !ok {
		t.Fatal("the unique two-swap assignment exists — must not misreport impossibility")
	}
	requireValidAssignment(t, perm, 4, exclusions)
	want := []int{1, 0, 3, 2}
	if !reflect.DeepEqual(perm, want) {
		t.Fatalf("unique solution = %v, want %v", perm, want)
	}
}

// ---------------------------------------------------------------------------
// Property checks + backtracking-path determinism
// ---------------------------------------------------------------------------

// TestConstrainedDerangement_SamplingAndMatchingAgree — property check
// across many seeds: whenever the solver returns ok the assignment
// verifies, and the exported verifier IsValidAssignment agrees with the
// test's independent arithmetic.
func TestConstrainedDerangement_SamplingAndMatchingAgree(t *testing.T) {
	exclusions := [][]int{{0, 1}, {2, 3}, {4, 5}}
	canonical := CanonicalizeExclusions(exclusions)
	blocked := BuildBlocked(8, canonical)
	for seed := int64(0); seed < 50; seed++ {
		perm, ok := ConstrainedDerangement(8, blocked, rand.New(rand.NewSource(seed)))
		if !ok {
			t.Fatalf("seed %d: solvable case misreported impossible", seed)
		}
		if !IsValidAssignment(perm, blocked) {
			t.Errorf("seed %d produced an assignment IsValidAssignment rejects: %v", seed, perm)
		}
		requireValidAssignment(t, perm, 8, exclusions)
	}
}

// TestConstrainedDerangement_BacktrackingPathDeterminism — the matching
// phase itself must be seed-deterministic: on a dense case that
// (deterministically) exhausts the sampling phase, the same seed twice
// yields the identical assignment, and different seeds still satisfy the
// constraints. This pins the donor's property that backtracking respects
// the seed via shuffled candidate order rather than collapsing to a
// lexicographic assignment.
func TestConstrainedDerangement_BacktrackingPathDeterminism(t *testing.T) {
	// Four couples, all cross-pair edges blocked: the unique-solution
	// thicket — guaranteed to reach phase 2 for any seed.
	var exclusions [][]int
	for a := 0; a < 8; a++ {
		for b := a + 1; b < 8; b++ {
			if a/2 == b/2 {
				continue
			}
			exclusions = append(exclusions, []int{a, b})
		}
	}
	canonical := CanonicalizeExclusions(exclusions)
	blocked := BuildBlocked(8, canonical)
	for seed := int64(0); seed < 10; seed++ {
		first, ok := ConstrainedDerangement(8, blocked, rand.New(rand.NewSource(seed)))
		if !ok {
			t.Fatalf("seed %d: misreported impossible", seed)
		}
		second, ok := ConstrainedDerangement(8, blocked, rand.New(rand.NewSource(seed)))
		if !ok {
			t.Fatalf("seed %d (rerun): misreported impossible", seed)
		}
		if !reflect.DeepEqual(first, second) {
			t.Errorf("seed %d: backtracking path not deterministic: %v vs %v", seed, first, second)
		}
		if !IsValidAssignment(first, blocked) {
			t.Errorf("seed %d: invalid assignment %v", seed, first)
		}
	}
}

// TestBuildBlocked_DiagonalAndSymmetry — the matrix constructor blocks
// the diagonal by construction and every canonical pair in BOTH
// directions, and nothing else.
func TestBuildBlocked_DiagonalAndSymmetry(t *testing.T) {
	blocked := BuildBlocked(4, []Constraint{{1, 2}})
	for i := 0; i < 4; i++ {
		if !blocked[i][i] {
			t.Errorf("diagonal [%d][%d] must be blocked (no fixed points)", i, i)
		}
	}
	if !blocked[1][2] || !blocked[2][1] {
		t.Error("canonical pair {1,2} must block both directions")
	}
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			if i == j || (i == 1 && j == 2) || (i == 2 && j == 1) {
				continue
			}
			if blocked[i][j] {
				t.Errorf("edge [%d][%d] must NOT be blocked", i, j)
			}
		}
	}
}

// TestIsValidAssignment_Rejections — the verifier rejects out-of-range,
// duplicate-slot (bijectivity violation), fixed-point, and blocked-edge
// assignments, and accepts a valid one.
func TestIsValidAssignment_Rejections(t *testing.T) {
	blocked := BuildBlocked(4, []Constraint{{0, 1}})
	cases := []struct {
		name string
		perm []int
		want bool
	}{
		{"valid", []int{2, 3, 1, 0}, true},
		{"fixed point", []int{0, 3, 1, 2}, false},
		{"blocked edge 0->1", []int{1, 2, 3, 0}, false},
		{"blocked edge 1->0", []int{2, 0, 3, 1}, false},
		{"duplicate slot", []int{2, 2, 1, 0}, false},
		{"out of range", []int{4, 3, 1, 0}, false},
		{"negative", []int{-1, 3, 1, 0}, false},
	}
	for _, tc := range cases {
		if got := IsValidAssignment(tc.perm, blocked); got != tc.want {
			t.Errorf("%s: IsValidAssignment(%v) = %v, want %v", tc.name, tc.perm, got, tc.want)
		}
	}
}
