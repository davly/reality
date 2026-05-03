package canonical

import (
	"strings"
	"testing"
)

// =========================================================================
// Expansion coverage for pkg/canonical. Pre-this: 3 tests (R85 marker /
// non-empty primitives list / zero divergences).
//
// Adds: primitive name format invariants, no duplicates, expected
// primitive set membership, IsCanonicalSource const-pin.
// =========================================================================

func TestIsCanonicalSource_True(t *testing.T) {
	// Locking the const value as part of R85 contract.
	if IsCanonicalSource != true {
		t.Errorf("IsCanonicalSource = %v, want true", IsCanonicalSource)
	}
}

func TestCanonicalPrimitives_AllUseDottedFormat(t *testing.T) {
	// Convention: each primitive is "<package>.<symbol>". Catches typos
	// like missing dots or all-lowercase legacy names.
	for i, p := range CanonicalPrimitives() {
		if !strings.Contains(p, ".") {
			t.Errorf("primitive[%d] %q missing dot separator", i, p)
		}
	}
}

func TestCanonicalPrimitives_NoDuplicates(t *testing.T) {
	prims := CanonicalPrimitives()
	seen := make(map[string]bool)
	for _, p := range prims {
		if seen[p] {
			t.Errorf("duplicate primitive: %q", p)
		}
		seen[p] = true
	}
}

func TestCanonicalPrimitives_AllLowercase(t *testing.T) {
	// Snake-case dotted convention — all primitives should be lowercase.
	for i, p := range CanonicalPrimitives() {
		if p != strings.ToLower(p) {
			t.Errorf("primitive[%d] %q has uppercase chars (snake_case expected)", i, p)
		}
	}
}

func TestCanonicalPrimitives_ContainsExpectedFoundationals(t *testing.T) {
	// Specific primitives that the R85 contract pins reality as the source for.
	prims := CanonicalPrimitives()
	expected := []string{"math.pi", "physics.c", "color.srgb_to_linear"}
	for _, want := range expected {
		found := false
		for _, p := range prims {
			if p == want {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("expected primitive %q missing from CanonicalPrimitives()", want)
		}
	}
}

func TestCanonicalPrimitives_NoLeadingTrailingWhitespace(t *testing.T) {
	for i, p := range CanonicalPrimitives() {
		if p != strings.TrimSpace(p) {
			t.Errorf("primitive[%d] %q has leading/trailing whitespace", i, p)
		}
	}
}

func TestCanonicalPrimitives_DeterministicAcrossInvocations(t *testing.T) {
	a := CanonicalPrimitives()
	b := CanonicalPrimitives()
	if len(a) != len(b) {
		t.Fatalf("len differs across calls: %d vs %d", len(a), len(b))
	}
	for i := range a {
		if a[i] != b[i] {
			t.Errorf("position %d differs across calls: %q vs %q", i, a[i], b[i])
		}
	}
}

func TestCanonicalPrimitives_AtLeastThreeEntries(t *testing.T) {
	// Floor — protects against accidental shrinkage when adding new primitives.
	if got := len(CanonicalPrimitives()); got < 3 {
		t.Errorf("len(CanonicalPrimitives()) = %d, want >= 3", got)
	}
}
