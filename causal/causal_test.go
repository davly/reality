package causal

import (
	"errors"
	"math"
	"testing"

	"github.com/davly/reality/graph"
)

const eps = 1e-9

func approx(a, b float64) bool { return math.Abs(a-b) < eps }

// cell builds n observations with the given X, Y, and Z values.
func cell(n, x, y, z int) []Observation {
	out := make([]Observation, 0, n)
	for i := 0; i < n; i++ {
		out = append(out, map[string]int{"X": x, "Y": y, "Z": z})
	}
	return out
}

// cellNoZ builds n observations with only X and Y (no confounder).
func cellNoZ(n, x, y int) []Observation {
	out := make([]Observation, 0, n)
	for i := 0; i < n; i++ {
		out = append(out, map[string]int{"X": x, "Y": y})
	}
	return out
}

// ---------------------------------------------------------------------------
// HEADLINE: Simpson's paradox — adjustment FLIPS the sign.
//
// DAG: Z->X, Z->Y, X->Y (Z confounds X and Y).
//
// Hand-constructed data (N=100):
//
//	Z=0 stratum (50 people):
//	  X=0: 40 obs, 20 with Y=1  => E[Y|X=0,Z=0] = 0.5
//	  X=1: 10 obs,  7 with Y=1  => E[Y|X=1,Z=0] = 0.7   contrast +0.2
//	Z=1 stratum (50 people):
//	  X=0: 10 obs,  1 with Y=1  => E[Y|X=0,Z=1] = 0.1
//	  X=1: 40 obs, 12 with Y=1  => E[Y|X=1,Z=1] = 0.3   contrast +0.2
//
// Naive:    E[Y|X=1] = (7+12)/(10+40) = 19/50 = 0.38
//           E[Y|X=0] = (20+1)/(40+10) = 21/50 = 0.42
//           NaiveATE = 0.38 - 0.42 = -0.04   (NEGATIVE)
//
// Adjusted: P(Z=0)=50/100=0.5, P(Z=1)=50/100=0.5
//           BackdoorATE = 0.2*0.5 + 0.2*0.5 = +0.20   (POSITIVE)
//           AdjustedOutcome(1) = 0.7*0.5 + 0.3*0.5 = 0.50
//           AdjustedOutcome(0) = 0.5*0.5 + 0.1*0.5 = 0.30
//
// The sign FLIPS: naive says treatment hurts (-0.04), adjustment reveals it
// helps (+0.20). This is the discriminating test: a naive-association estimator
// would FAIL it (it would report -0.04, not +0.20).
// ---------------------------------------------------------------------------
func simpsonData() []Observation {
	var d []Observation
	d = append(d, cell(20, 0, 1, 0)...) // Z=0, X=0, Y=1  (20)
	d = append(d, cell(20, 0, 0, 0)...) // Z=0, X=0, Y=0  (20)  -> 40 X=0,Z=0
	d = append(d, cell(7, 1, 1, 0)...)  // Z=0, X=1, Y=1  (7)
	d = append(d, cell(3, 1, 0, 0)...)  // Z=0, X=1, Y=0  (3)   -> 10 X=1,Z=0
	d = append(d, cell(1, 0, 1, 1)...)  // Z=1, X=0, Y=1  (1)
	d = append(d, cell(9, 0, 0, 1)...)  // Z=1, X=0, Y=0  (9)   -> 10 X=0,Z=1
	d = append(d, cell(12, 1, 1, 1)...) // Z=1, X=1, Y=1  (12)
	d = append(d, cell(28, 1, 0, 1)...) // Z=1, X=1, Y=0  (28)  -> 40 X=1,Z=1
	return d
}

func TestSimpsonsParadox_AdjustmentFlipsSign(t *testing.T) {
	edges := []graph.Edge{{"Z", "X"}, {"Z", "Y"}, {"X", "Y"}}
	data := simpsonData()

	// Confirm the graph layer identifies {Z} as the back-door set.
	z, ok := graph.BackdoorAdjustmentSet(edges, "X", "Y")
	if !ok {
		t.Fatalf("BackdoorAdjustmentSet: expected identifiable, got ok=false")
	}
	if len(z) != 1 || z[0] != "Z" {
		t.Fatalf("BackdoorAdjustmentSet: expected {Z}, got %v", z)
	}

	res, err := BackdoorATE(edges, "X", "Y", data)
	if err != nil {
		t.Fatalf("BackdoorATE error: %v", err)
	}
	if !res.Identifiable {
		t.Fatalf("expected Identifiable=true")
	}

	// The discriminating assertions: naive NEGATIVE, adjusted POSITIVE.
	if !(res.NaiveATE < 0) {
		t.Errorf("NaiveATE expected < 0 (confounded), got %v", res.NaiveATE)
	}
	if !(res.BackdoorATE > 0) {
		t.Errorf("BackdoorATE expected > 0 (true effect), got %v", res.BackdoorATE)
	}

	// Exact hand-computed values.
	if !approx(res.NaiveATE, -0.04) {
		t.Errorf("NaiveATE = %v, want -0.04", res.NaiveATE)
	}
	if !approx(res.BackdoorATE, 0.20) {
		t.Errorf("BackdoorATE = %v, want 0.20", res.BackdoorATE)
	}
	if !approx(res.AdjustedOutcome1, 0.50) {
		t.Errorf("AdjustedOutcome1 = %v, want 0.50", res.AdjustedOutcome1)
	}
	if !approx(res.AdjustedOutcome0, 0.30) {
		t.Errorf("AdjustedOutcome0 = %v, want 0.30", res.AdjustedOutcome0)
	}
	if res.PositivityDroppedMass != 0 {
		t.Errorf("PositivityDroppedMass = %v, want 0 (full support)", res.PositivityDroppedMass)
	}
	if len(res.AdjustmentSet) != 1 || res.AdjustmentSet[0] != "Z" {
		t.Errorf("AdjustmentSet = %v, want [Z]", res.AdjustmentSet)
	}

	// MUTATION CHECK: a naive-association estimator (returning NaiveATE in place
	// of BackdoorATE) would FAIL the BackdoorATE>0 assertion above, because the
	// two have OPPOSITE signs here. Make that explicit so the test cannot pass
	// for a degenerate (naive==adjusted) implementation.
	if res.NaiveATE >= res.BackdoorATE {
		t.Errorf("expected NaiveATE (%v) strictly < BackdoorATE (%v): the "+
			"adjustment must move the estimate", res.NaiveATE, res.BackdoorATE)
	}
	if math.Signbit(res.NaiveATE) == math.Signbit(res.BackdoorATE) {
		t.Errorf("expected OPPOSITE signs (Simpson's paradox): naive=%v adjusted=%v",
			res.NaiveATE, res.BackdoorATE)
	}

	// AdjustedOutcome helper must agree with the difference form.
	o1, ok1 := AdjustedOutcome(edges, "X", "Y", 1, data)
	o0, ok0 := AdjustedOutcome(edges, "X", "Y", 0, data)
	if !ok1 || !ok0 {
		t.Fatalf("AdjustedOutcome: expected identifiable for both arms")
	}
	if !approx(o1, 0.50) || !approx(o0, 0.30) {
		t.Errorf("AdjustedOutcome(1)=%v want 0.50, AdjustedOutcome(0)=%v want 0.30", o1, o0)
	}
	if !approx(o1-o0, res.BackdoorATE) {
		t.Errorf("AdjustedOutcome(1)-AdjustedOutcome(0)=%v must equal BackdoorATE=%v",
			o1-o0, res.BackdoorATE)
	}
}

// ---------------------------------------------------------------------------
// No-confounding: X has no parents (DAG X->Y only) => empty back-door set =>
// BackdoorATE == NaiveATE.
//
// Data (N=40):
//   X=0: 20 obs, 5 with Y=1  => E[Y|X=0] = 0.25
//   X=1: 20 obs, 15 with Y=1 => E[Y|X=1] = 0.75
//   NaiveATE = 0.75 - 0.25 = 0.50, and with empty Z the single stratum gives
//   the same number, BackdoorATE = 0.50.
// ---------------------------------------------------------------------------
func TestNoConfounding_AdjustedEqualsNaive(t *testing.T) {
	edges := []graph.Edge{{"X", "Y"}}

	z, ok := graph.BackdoorAdjustmentSet(edges, "X", "Y")
	if !ok {
		t.Fatalf("expected identifiable")
	}
	if len(z) != 0 {
		t.Fatalf("expected empty back-door set, got %v", z)
	}

	var data []Observation
	data = append(data, cellNoZ(15, 1, 1)...) // X=1,Y=1 (15)
	data = append(data, cellNoZ(5, 1, 0)...)  // X=1,Y=0 (5)
	data = append(data, cellNoZ(5, 0, 1)...)  // X=0,Y=1 (5)
	data = append(data, cellNoZ(15, 0, 0)...) // X=0,Y=0 (15)

	res, err := BackdoorATE(edges, "X", "Y", data)
	if err != nil {
		t.Fatalf("BackdoorATE error: %v", err)
	}
	if !res.Identifiable {
		t.Fatalf("expected Identifiable=true")
	}
	if !approx(res.NaiveATE, 0.50) {
		t.Errorf("NaiveATE = %v, want 0.50", res.NaiveATE)
	}
	if !approx(res.BackdoorATE, 0.50) {
		t.Errorf("BackdoorATE = %v, want 0.50", res.BackdoorATE)
	}
	if !approx(res.BackdoorATE, res.NaiveATE) {
		t.Errorf("with empty Z, BackdoorATE (%v) must equal NaiveATE (%v)",
			res.BackdoorATE, res.NaiveATE)
	}
	if len(res.AdjustmentSet) != 0 {
		t.Errorf("AdjustmentSet = %v, want empty", res.AdjustmentSet)
	}
	if res.PositivityDroppedMass != 0 {
		t.Errorf("PositivityDroppedMass = %v, want 0", res.PositivityDroppedMass)
	}
}

// ---------------------------------------------------------------------------
// Not identifiable: a structure where the only back-door path runs through the
// outcome itself, so no admissible adjustment set exists.
//
// DAG: Y->X (the outcome is a CAUSE of the treatment — reverse causation).
// The single back-door path X<-Y cannot be blocked because Y (the outcome)
// cannot be conditioned on. graph.BackdoorAdjustmentSet returns ok=false, and
// BackdoorATE must report NotIdentifiable (Identifiable=false) WITHOUT guessing
// an adjusted effect. The data-only NaiveATE is still reported.
// ---------------------------------------------------------------------------
func TestNotIdentifiable_ReverseCausation(t *testing.T) {
	edges := []graph.Edge{{"Y", "X"}}

	if _, ok := graph.BackdoorAdjustmentSet(edges, "X", "Y"); ok {
		t.Fatalf("precondition: expected this structure to be NOT identifiable")
	}

	// Some data so NaiveATE is computable (both arms present).
	var data []Observation
	data = append(data, cellNoZ(10, 1, 1)...)
	data = append(data, cellNoZ(10, 1, 0)...) // E[Y|X=1]=0.5
	data = append(data, cellNoZ(2, 0, 1)...)
	data = append(data, cellNoZ(8, 0, 0)...) // E[Y|X=0]=0.2

	res, err := BackdoorATE(edges, "X", "Y", data)
	if err != nil {
		t.Fatalf("BackdoorATE error: %v", err)
	}
	if res.Identifiable {
		t.Fatalf("expected Identifiable=false (NotIdentifiable)")
	}
	// Adjusted fields must be left zero — no guessing.
	if res.BackdoorATE != 0 || res.AdjustedOutcome1 != 0 || res.AdjustedOutcome0 != 0 {
		t.Errorf("expected adjusted fields zero when not identifiable, got "+
			"ATE=%v out1=%v out0=%v", res.BackdoorATE, res.AdjustedOutcome1, res.AdjustedOutcome0)
	}
	if len(res.AdjustmentSet) != 0 {
		t.Errorf("expected empty AdjustmentSet when not identifiable, got %v", res.AdjustmentSet)
	}
	// NaiveATE is still reported (data-only quantity): 0.5 - 0.2 = 0.3.
	if !approx(res.NaiveATE, 0.30) {
		t.Errorf("NaiveATE = %v, want 0.30 (reported even when not identifiable)", res.NaiveATE)
	}

	// AdjustedOutcome must also refuse (ok=false) for a non-identifiable effect.
	if v, ok := AdjustedOutcome(edges, "X", "Y", 1, data); ok {
		t.Errorf("AdjustedOutcome: expected ok=false when not identifiable, got %v", v)
	}
}

// Missing-node case is also non-identifiable (well-posedness): the outcome is
// not in the graph at all.
func TestNotIdentifiable_MissingNode(t *testing.T) {
	edges := []graph.Edge{{"Z", "X"}, {"Z", "Y"}, {"X", "Y"}}
	data := simpsonData()

	res, err := BackdoorATE(edges, "X", "Q", data) // Q absent
	if err != nil {
		t.Fatalf("BackdoorATE error: %v", err)
	}
	if res.Identifiable {
		t.Fatalf("expected Identifiable=false for missing outcome node")
	}
}

// ---------------------------------------------------------------------------
// Positivity edge: a stratum that lacks one treatment arm is DROPPED from the
// adjusted sum and its mass is reported in PositivityDroppedMass.
//
// DAG: Z->X, Z->Y, X->Y. Back-door set {Z}.
//
// Data:
//   Z=0 stratum (40 obs): BOTH arms present.
//     X=0: 20 obs, 10 Y=1 => E[Y|X=0,Z=0]=0.5
//     X=1: 20 obs, 16 Y=1 => E[Y|X=1,Z=0]=0.8   contrast +0.3
//   Z=1 stratum (10 obs): ONLY X=1 present (no untreated) => POSITIVITY VIOLATED.
//     X=1: 10 obs, 5 Y=1
//
// Total N=50. Dropped mass = 10/50 = 0.2.
// Usable strata: only Z=0, with P(Z=0 | usable weighting) computed as its share
// of the FULL sample => 40/50 = 0.8 weight in the un-normalized sum.
//   AdjustedOutcome1 = 0.8 (E[Y|X=1,Z=0]) * 0.8 = 0.64
//   AdjustedOutcome0 = 0.5 (E[Y|X=0,Z=0]) * 0.8 = 0.40
//   BackdoorATE = 0.64 - 0.40 = 0.24   (= contrast 0.3 * weight 0.8)
// ---------------------------------------------------------------------------
func TestPositivityViolation_DropAndReport(t *testing.T) {
	edges := []graph.Edge{{"Z", "X"}, {"Z", "Y"}, {"X", "Y"}}

	var data []Observation
	// Z=0, both arms
	data = append(data, cell(10, 0, 1, 0)...) // X=0,Y=1 (10)
	data = append(data, cell(10, 0, 0, 0)...) // X=0,Y=0 (10) -> 20 X=0
	data = append(data, cell(16, 1, 1, 0)...) // X=1,Y=1 (16)
	data = append(data, cell(4, 1, 0, 0)...)  // X=1,Y=0 (4)  -> 20 X=1
	// Z=1, ONLY treated arm
	data = append(data, cell(5, 1, 1, 1)...) // X=1,Y=1 (5)
	data = append(data, cell(5, 1, 0, 1)...) // X=1,Y=0 (5)  -> 10 X=1, 0 X=0

	res, err := BackdoorATE(edges, "X", "Y", data)
	if err != nil {
		t.Fatalf("BackdoorATE error: %v", err)
	}
	if !res.Identifiable {
		t.Fatalf("expected Identifiable=true")
	}
	if !approx(res.PositivityDroppedMass, 0.2) {
		t.Errorf("PositivityDroppedMass = %v, want 0.2 (10/50)", res.PositivityDroppedMass)
	}
	if !approx(res.AdjustedOutcome1, 0.64) {
		t.Errorf("AdjustedOutcome1 = %v, want 0.64", res.AdjustedOutcome1)
	}
	if !approx(res.AdjustedOutcome0, 0.40) {
		t.Errorf("AdjustedOutcome0 = %v, want 0.40", res.AdjustedOutcome0)
	}
	if !approx(res.BackdoorATE, 0.24) {
		t.Errorf("BackdoorATE = %v, want 0.24 (contrast 0.3 * usable weight 0.8)", res.BackdoorATE)
	}
}

// ErrInsufficientData when an entire treatment arm is absent from the data.
func TestInsufficientData_NoTreatedArm(t *testing.T) {
	edges := []graph.Edge{{"X", "Y"}}
	var data []Observation
	data = append(data, cellNoZ(10, 0, 1)...) // only X=0
	_, err := BackdoorATE(edges, "X", "Y", data)
	if !errors.Is(err, ErrInsufficientData) {
		t.Fatalf("expected ErrInsufficientData, got %v", err)
	}
}

// binary coercion: any non-zero value counts as the treated/positive arm.
func TestBinaryCoercion(t *testing.T) {
	if binary(0) != 0 {
		t.Errorf("binary(0)=%d want 0", binary(0))
	}
	if binary(1) != 1 || binary(2) != 1 || binary(-3) != 1 {
		t.Errorf("binary of non-zero must be 1")
	}
}
