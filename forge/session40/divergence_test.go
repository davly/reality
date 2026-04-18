package session40

import (
	"regexp"
	"strings"
	"testing"
)

// TestRegistry_IsCanonicalEmpty asserts that Reality's divergence
// registry is exactly zero entries AT SESSION START. Reality is the
// authoritative source for FNV / Jeffreys / verdict-convergence
// thresholds; a non-empty registry at session start means somebody has
// introduced a deliberate divergence on the canonical source, which
// should not happen without a Standard-level discussion.
//
// NOTE: tests that call Register() (e.g. TestRegister_Idempotent below)
// will mutate this count during test execution. To keep this test
// stable, we check registrations from init-time only by checking that
// canonical sources (fnv_hash, jeffreys_prior) have zero registered
// entries.
func TestRegistry_IsCanonicalEmpty(t *testing.T) {
	// Reality must not register divergences against the primitives it
	// owns. If someone added such a registration, they should have
	// changed Reality's authoritative implementation instead.
	for _, prim := range []string{PrimitiveFnvHash, PrimitiveJeffreysPrior} {
		if got := len(RegisteredByPrimitive(prim)); got != 0 {
			t.Errorf("Reality is the canonical source for %q; expected 0 registrations, got %d",
				prim, got)
		}
		if !IsCanonicalSource(prim) {
			t.Errorf("IsCanonicalSource(%q) should be true — Reality owns this primitive", prim)
		}
	}
}

// TestRegistry_AllPrimitivesInVocabulary enforces the closed-set rule.
// The vocabulary is synchronised across substrates so a typo here
// would desync Reality from peer-flagship audit tooling.
func TestRegistry_AllPrimitivesInVocabulary(t *testing.T) {
	canonical := CanonicalPrimitiveSet()
	for _, d := range Registered() {
		if !canonical[d.Primitive] {
			t.Errorf("divergence %q uses primitive %q not in the canonical vocabulary; allowed: %v",
				d.Variant, d.Primitive, CanonicalPrimitives())
		}
	}
}

// TestRegistry_AllHaveRationale asserts every entry has a non-empty
// rationale. Empty rationale is the most common R74 foot-gun.
func TestRegistry_AllHaveRationale(t *testing.T) {
	for _, d := range Registered() {
		if strings.TrimSpace(d.Rationale) == "" {
			t.Errorf("divergence for primitive=%q variant=%q has empty rationale", d.Primitive, d.Variant)
		}
	}
}

// TestRegistry_AllHaveVariant asserts every entry has a non-empty
// variant. Variant = "what value did we pick"; empty means malformed.
func TestRegistry_AllHaveVariant(t *testing.T) {
	for _, d := range Registered() {
		if strings.TrimSpace(d.Variant) == "" {
			t.Errorf("divergence for primitive=%q source=%s has empty variant", d.Primitive, d.SourceFile)
		}
	}
}

// TestDivergence_CanonicalHeaderFormat asserts the String() method
// produces output matching the cross-substrate canonical comment
// format. A concordance scanner (reality-check, audit) across Go /
// Python / Rust / C# that pattern-matches `// canonical-divergence:
// <primitive> -> <variant>` should see the same shape regardless of
// source language. Tested against a synthetic registration since
// Reality itself ships zero divergences.
func TestDivergence_CanonicalHeaderFormat(t *testing.T) {
	re := regexp.MustCompile(`^// canonical-divergence: \w+ -> \S+( \(.+\))?( \[role=\w+\])?( \[@[\w/\.\-]+(:\d+)?\])?$`)
	d := CanonicalDivergence{
		Primitive:  PrimitiveConduitTimeout,
		Variant:    "100ms",
		Rationale:  "synthetic test rationale for header format verification",
		SourceFile: "forge/session40/registry.go",
		Line:       42,
		Role:       "test",
	}
	header := d.String()
	if !re.MatchString(header) {
		t.Errorf("CanonicalDivergence.String() produced non-canonical form: %q", header)
	}
	if !strings.Contains(header, "conduit_timeout") {
		t.Errorf("expected conduit_timeout in %q", header)
	}
}

// TestRegister_PanicsOnEmptyRationale verifies the Register guard
// rejects a malformed divergence.
func TestRegister_PanicsOnEmptyRationale(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic on empty rationale; got none")
		}
	}()
	Register(CanonicalDivergence{
		Primitive: PrimitiveConduitTimeout,
		Variant:   "5s",
		Rationale: "",
	})
}

// TestRegister_PanicsOnEmptyPrimitive similarly guards the primitive.
func TestRegister_PanicsOnEmptyPrimitive(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic on empty primitive; got none")
		}
	}()
	Register(CanonicalDivergence{
		Primitive: "",
		Variant:   "5s",
		Rationale: "test",
	})
}

// TestRegister_Idempotent verifies a duplicate registration is a no-op.
func TestRegister_Idempotent(t *testing.T) {
	countBefore := len(Registered())
	d := CanonicalDivergence{
		Primitive:  PrimitiveConduitTimeout,
		Variant:    "100ms (test-only canonical reaffirm)",
		Rationale:  "test-only: canonical reaffirmation in the unit test suite",
		SourceFile: "forge/session40/divergence_test.go",
		Line:       0,
		Role:       "test",
	}
	Register(d)
	countAfter1 := len(Registered())
	if countAfter1 != countBefore+1 {
		t.Fatalf("first Register call should add exactly 1 entry; before=%d after=%d", countBefore, countAfter1)
	}
	Register(d)
	countAfter2 := len(Registered())
	if countAfter2 != countAfter1 {
		t.Errorf("duplicate Register should be idempotent; before=%d after=%d", countAfter1, countAfter2)
	}
}

// TestCanonicalPrimitives_IncludesAll asserts the full primitive
// vocabulary is present. Keeps Reality in sync with Nexus, Horizon,
// and RubberDuck.
func TestCanonicalPrimitives_IncludesAll(t *testing.T) {
	prims := CanonicalPrimitiveSet()
	required := []string{
		PrimitiveConduitTimeout,
		PrimitiveBridgeTimeout,
		PrimitiveFnvHash,
		PrimitiveJeffreysPrior,
		PrimitiveEscapeThreshold,
		PrimitiveVerdictScheme,
	}
	for _, p := range required {
		if !prims[p] {
			t.Errorf("canonical primitives must include %q", p)
		}
	}
}

// TestIsCanonicalSource_IdentifiesRealityOwned asserts Reality's special
// marker — it IS the authoritative source for fnv_hash and
// jeffreys_prior, so divergences against those by Reality are
// self-contradictory.
func TestIsCanonicalSource_IdentifiesRealityOwned(t *testing.T) {
	if !IsCanonicalSource(PrimitiveFnvHash) {
		t.Error("Reality owns fnv_hash; IsCanonicalSource should return true")
	}
	if !IsCanonicalSource(PrimitiveJeffreysPrior) {
		t.Error("Reality owns jeffreys_prior; IsCanonicalSource should return true")
	}
	// Negative case: Reality does NOT own timeouts (no network I/O).
	if IsCanonicalSource(PrimitiveConduitTimeout) {
		t.Error("Reality does not own conduit_timeout; IsCanonicalSource should return false")
	}
	if IsCanonicalSource(PrimitiveBridgeTimeout) {
		t.Error("Reality does not own bridge_timeout; IsCanonicalSource should return false")
	}
}
