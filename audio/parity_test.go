package audio

// R80b cross-substrate parity test for the Welford convergence
// primitive. Reads the canonical golden vector at
// testdata/welford-parity-golden.json and asserts that this Go
// implementation produces exactly the same (N, Mean, M2) at each step.
//
// Sister substrate parity tests:
//   - flagships/folio/backend/internal/forge/welford_parity_test.go (Go)
//   - flagships/folio/web/src/lib/welford-parity.test.ts (TypeScript via Node)
//   - flagships/rubberduck/tests/RubberDuck.Property.Tests/WelfordParityTests.cs (C#)
//   - flagships/{pigeonhole,howler,dipstick}/mobile/shared/.../WelfordParityTest.kt (Kotlin)
//
// Each substrate copies a vendored copy of the golden JSON into its
// own test resources (because cross-language imports across repos are
// brittle). The golden JSON is regenerated only when intentionally
// changing the input stream — see testdata/README.md.

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

type goldenStep struct {
	N    int     `json:"n"`
	Mean float64 `json:"mean"`
	M2   float64 `json:"m2"`
}

type goldenVector struct {
	Description string       `json:"description"`
	Inputs      []float64    `json:"inputs"`
	AfterStep   []goldenStep `json:"afterStep"`
}

// loadGolden returns the canonical Welford-1D golden vector.
// Used by parity tests in this package and (verbatim copies) by sister
// substrates.
func loadGolden(t *testing.T) goldenVector {
	t.Helper()
	b, err := os.ReadFile("testdata/welford-parity-golden.json")
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g goldenVector
	if err := json.Unmarshal(b, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	if len(g.Inputs) != len(g.AfterStep) {
		t.Fatalf("golden malformed: %d inputs but %d after-step rows",
			len(g.Inputs), len(g.AfterStep))
	}
	if len(g.Inputs) == 0 {
		t.Fatal("golden empty")
	}
	return g
}

func TestWelford_ParityWithGoldenVector(t *testing.T) {
	g := loadGolden(t)

	fp := NewFingerprint(1)
	for i, x := range g.Inputs {
		UpdateFingerprint(&fp, []float64{x})
		expected := g.AfterStep[i]
		if fp.N != expected.N {
			t.Errorf("step %d: N = %d, golden = %d", i, fp.N, expected.N)
		}
		if math.Abs(fp.Mean[0]-expected.Mean) > 1e-12 {
			t.Errorf("step %d: Mean = %.17g, golden = %.17g (diff %g)",
				i, fp.Mean[0], expected.Mean, math.Abs(fp.Mean[0]-expected.Mean))
		}
		if math.Abs(fp.M2[0]-expected.M2) > 1e-9 {
			t.Errorf("step %d: M2 = %.17g, golden = %.17g (diff %g)",
				i, fp.M2[0], expected.M2, math.Abs(fp.M2[0]-expected.M2))
		}
	}
}

// TestWelford_GoldenIsSelfConsistent re-runs the canonical Welford
// against the golden inputs and ensures the outputs match the
// recorded after-step values. This is a tautology against the canonical
// generator BUT serves to catch the case where the golden file gets
// corrupted (e.g. mis-pasted) — if the golden becomes inconsistent
// with the canonical, the parity tests across all substrates would
// pass-or-fail in lockstep, so we'd lose the cross-substrate signal.
//
// This test detects exactly that: it ensures the canonical Go
// implementation IS what generated the golden JSON. Sister substrates
// trust this assumption when comparing against the same JSON.
func TestWelford_GoldenIsSelfConsistent(t *testing.T) {
	g := loadGolden(t)
	fp := NewFingerprint(1)
	for i, x := range g.Inputs {
		UpdateFingerprint(&fp, []float64{x})
		exp := g.AfterStep[i]
		if fp.N != exp.N || fp.Mean[0] != exp.Mean || fp.M2[0] != exp.M2 {
			t.Errorf("golden self-inconsistent at step %d: got (n=%d, mean=%.17g, m2=%.17g), "+
				"golden (n=%d, mean=%.17g, m2=%.17g)",
				i, fp.N, fp.Mean[0], fp.M2[0], exp.N, exp.Mean, exp.M2)
		}
	}
}
