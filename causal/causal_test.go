package causal

import (
	"errors"
	"math"
	"math/rand"
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
//
//	E[Y|X=0] = (20+1)/(40+10) = 21/50 = 0.42
//	NaiveATE = 0.38 - 0.42 = -0.04   (NEGATIVE)
//
// Adjusted: P(Z=0)=50/100=0.5, P(Z=1)=50/100=0.5
//
//	BackdoorATE = 0.2*0.5 + 0.2*0.5 = +0.20   (POSITIVE)
//	AdjustedOutcome(1) = 0.7*0.5 + 0.3*0.5 = 0.50
//	AdjustedOutcome(0) = 0.5*0.5 + 0.1*0.5 = 0.30
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
//
//	X=0: 20 obs, 5 with Y=1  => E[Y|X=0] = 0.25
//	X=1: 20 obs, 15 with Y=1 => E[Y|X=1] = 0.75
//	NaiveATE = 0.75 - 0.25 = 0.50, and with empty Z the single stratum gives
//	the same number, BackdoorATE = 0.50.
//
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
//
//	Z=0 stratum (40 obs): BOTH arms present.
//	  X=0: 20 obs, 10 Y=1 => E[Y|X=0,Z=0]=0.5
//	  X=1: 20 obs, 16 Y=1 => E[Y|X=1,Z=0]=0.8   contrast +0.3
//	Z=1 stratum (10 obs): ONLY X=1 present (no untreated) => POSITIVITY VIOLATED.
//	  X=1: 10 obs, 5 Y=1
//
// Total N=50. Dropped mass = 10/50 = 0.2.
// Usable strata: only Z=0, with P(Z=0 | usable weighting) computed as its share
// of the FULL sample => 40/50 = 0.8 weight in the un-normalized sum.
//
//	AdjustedOutcome1 = 0.8 (E[Y|X=1,Z=0]) * 0.8 = 0.64
//	AdjustedOutcome0 = 0.5 (E[Y|X=0,Z=0]) * 0.8 = 0.40
//	BackdoorATE = 0.64 - 0.40 = 0.24   (= contrast 0.3 * weight 0.8)
//
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

// ---------------------------------------------------------------------------
// REFUTATION layer (BackdoorATEWithRefutation): DoWhy-style falsification
// checks on the IDENTIFIED back-door estimate.
//
// Salvage subset: placebo-treatment + bootstrap-subset only. The
// random-common-cause refuter is deliberately NOT implemented because the
// adjustment set is graph-derived (graph.BackdoorAdjustmentSet), never from
// data columns — an injected synthetic covariate can never enter Z, so it is a
// provable no-op. See the doc comment on BackdoorATEWithRefutation / Refutation.
// ---------------------------------------------------------------------------

func TestRefutation_PlaceboCollapsesToZero(t *testing.T) {
	edges := []graph.Edge{{"Z", "X"}, {"Z", "Y"}, {"X", "Y"}}
	data := simpsonData()

	res, err := BackdoorATEWithRefutation(edges, "X", "Y", data, RefuteOptions{Seed: 1})
	if err != nil {
		t.Fatalf("BackdoorATEWithRefutation error: %v", err)
	}
	// No regression: point estimate identical to BackdoorATE's 0.20.
	if !res.Identifiable {
		t.Fatalf("expected Identifiable=true")
	}
	if !approx(res.BackdoorATE, 0.20) {
		t.Errorf("BackdoorATE = %v, want 0.20 (refutation must not move the point estimate)", res.BackdoorATE)
	}
	if res.Refutation == nil {
		t.Fatalf("expected Refutation != nil for an identifiable effect")
	}
	if res.Refutation.PlaceboTrials != 100 {
		t.Errorf("PlaceboTrials = %d, want default 100", res.Refutation.PlaceboTrials)
	}
	// The MEAN placebo ATE over many permutations must collapse toward 0 (the
	// permuted treatment is independent of Y given Z). This is far smaller than
	// the true effect (0.20), so it discriminates a real effect from noise.
	if math.Abs(res.Refutation.PlaceboATE) >= 0.05 {
		t.Errorf("PlaceboATE = %v, want |PlaceboATE| < 0.05 (mean placebo must collapse to ~0)", res.Refutation.PlaceboATE)
	}
	if !res.Refutation.PlaceboPassed {
		t.Errorf("expected PlaceboPassed=true, got false (PlaceboATE=%v, tol=%v)",
			res.Refutation.PlaceboATE, res.Refutation.PlaceboTolerance)
	}
	if !approx(res.Refutation.PlaceboTolerance, 0.05) {
		t.Errorf("PlaceboTolerance = %v, want default 0.05", res.Refutation.PlaceboTolerance)
	}
}

func TestRefutation_BootstrapStable(t *testing.T) {
	edges := []graph.Edge{{"Z", "X"}, {"Z", "Y"}, {"X", "Y"}}
	data := simpsonData()

	res, err := BackdoorATEWithRefutation(edges, "X", "Y", data, RefuteOptions{Seed: 1})
	if err != nil {
		t.Fatalf("BackdoorATEWithRefutation error: %v", err)
	}
	if res.Refutation == nil {
		t.Fatalf("expected Refutation != nil")
	}
	if res.Refutation.Resamples != 200 {
		t.Errorf("Resamples = %d, want default 200", res.Refutation.Resamples)
	}
	// Bootstrap mean should sit near the point estimate (0.20).
	if math.Abs(res.Refutation.BootstrapMean-0.20) > 0.05 {
		t.Errorf("BootstrapMean = %v, want within 0.05 of 0.20", res.Refutation.BootstrapMean)
	}
	// Spread positive but well below the signal (0.20) => stable relative to the
	// effect. On this small N=100 fixture the genuine bootstrap std is ~0.1, so
	// the band is < 0.15 (about half the point estimate), not arbitrarily tiny.
	if !(res.Refutation.BootstrapStd > 0 && res.Refutation.BootstrapStd < 0.15) {
		t.Errorf("BootstrapStd = %v, want 0 < std < 0.15 (stable relative to 0.20 effect)", res.Refutation.BootstrapStd)
	}
}

func TestRefutation_Deterministic(t *testing.T) {
	edges := []graph.Edge{{"Z", "X"}, {"Z", "Y"}, {"X", "Y"}}
	data := simpsonData()

	r1, err := BackdoorATEWithRefutation(edges, "X", "Y", data, RefuteOptions{Seed: 1})
	if err != nil {
		t.Fatalf("run 1 error: %v", err)
	}
	r2, err := BackdoorATEWithRefutation(edges, "X", "Y", data, RefuteOptions{Seed: 1})
	if err != nil {
		t.Fatalf("run 2 error: %v", err)
	}
	if r1.Refutation == nil || r2.Refutation == nil {
		t.Fatalf("expected non-nil Refutation on both runs")
	}
	if r1.Refutation.PlaceboATE != r2.Refutation.PlaceboATE {
		t.Errorf("PlaceboATE not reproducible: %v vs %v", r1.Refutation.PlaceboATE, r2.Refutation.PlaceboATE)
	}
	if r1.Refutation.BootstrapMean != r2.Refutation.BootstrapMean {
		t.Errorf("BootstrapMean not reproducible: %v vs %v", r1.Refutation.BootstrapMean, r2.Refutation.BootstrapMean)
	}
	if r1.Refutation.BootstrapStd != r2.Refutation.BootstrapStd {
		t.Errorf("BootstrapStd not reproducible: %v vs %v", r1.Refutation.BootstrapStd, r2.Refutation.BootstrapStd)
	}
}

func TestRefutation_NotIdentifiableSkipsRefuter(t *testing.T) {
	edges := []graph.Edge{{"Y", "X"}} // reverse causation => not identifiable

	var data []Observation
	data = append(data, cellNoZ(10, 1, 1)...)
	data = append(data, cellNoZ(10, 1, 0)...)
	data = append(data, cellNoZ(2, 0, 1)...)
	data = append(data, cellNoZ(8, 0, 0)...)

	res, err := BackdoorATEWithRefutation(edges, "X", "Y", data, RefuteOptions{Seed: 1})
	if err != nil {
		t.Fatalf("BackdoorATEWithRefutation error: %v", err)
	}
	if res.Identifiable {
		t.Fatalf("expected Identifiable=false")
	}
	if res.Refutation != nil {
		t.Errorf("expected Refutation == nil when not identifiable, got %+v", res.Refutation)
	}
}

// TestRefutation_NoRandomCommonCauseField is a compile-time guard recording the
// deliberate omission of the random-common-cause refuter (a provable no-op given
// the graph-derived adjustment set). It references exactly the real Refutation
// fields; if anyone adds a random-common-cause field, this enumeration becomes a
// reminder that the omission was intentional and source-grounded.
func TestRefutation_NoRandomCommonCauseField(t *testing.T) {
	// UNKEYED literal: lists every field in order. Adding any field (e.g. a
	// random-common-cause one) will break this compile, forcing a reviewer to
	// re-confirm the deliberate, source-grounded omission.
	r := Refutation{
		0,     // PlaceboATE
		0,     // PlaceboTrials
		false, // PlaceboPassed
		0,     // PlaceboTolerance
		0,     // Resamples
		0,     // BootstrapMean
		0,     // BootstrapStd
	}
	_ = r
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

// ---------------------------------------------------------------------------
// Gold-standard refinement: targeted coverage of the refutation layer's
// risk-surface branches that the headline tests do not exercise — custom
// RefuteOptions are honoured (not silently overridden by defaults), the
// PlaceboPassed=false branch fires when the tolerance is tighter than the
// (small but non-zero) placebo mean, and the unexported refuter helpers handle
// degenerate inputs (empty data / zero trials / zero resamples) without
// dividing by zero. These are additive: they touch no production code.
// ---------------------------------------------------------------------------

// Custom RefuteOptions must be threaded through verbatim (not clobbered by the
// zero-value defaulting), and must remain deterministic for a fixed seed.
func TestRefutation_CustomOptionsHonored(t *testing.T) {
	edges := []graph.Edge{{"Z", "X"}, {"Z", "Y"}, {"X", "Y"}}
	data := simpsonData()

	opts := RefuteOptions{Seed: 42, Resamples: 50, PlaceboTrials: 30, PlaceboTolerance: 0.1}
	res, err := BackdoorATEWithRefutation(edges, "X", "Y", data, opts)
	if err != nil {
		t.Fatalf("BackdoorATEWithRefutation error: %v", err)
	}
	if res.Refutation == nil {
		t.Fatalf("expected Refutation != nil")
	}
	// The non-default knobs must be reflected exactly — proving the defaulting
	// block only fires for <=0 values and does not override caller intent.
	if res.Refutation.PlaceboTrials != 30 {
		t.Errorf("PlaceboTrials = %d, want 30 (custom)", res.Refutation.PlaceboTrials)
	}
	if res.Refutation.Resamples != 50 {
		t.Errorf("Resamples = %d, want 50 (custom)", res.Refutation.Resamples)
	}
	if !approx(res.Refutation.PlaceboTolerance, 0.1) {
		t.Errorf("PlaceboTolerance = %v, want 0.1 (custom)", res.Refutation.PlaceboTolerance)
	}
	// Point estimate is still produced by the BackdoorATE path: unchanged at 0.20.
	if !approx(res.BackdoorATE, 0.20) {
		t.Errorf("BackdoorATE = %v, want 0.20 (refutation must not move it)", res.BackdoorATE)
	}
	// Deterministic for this seed: a second identical call reproduces it.
	res2, _ := BackdoorATEWithRefutation(edges, "X", "Y", data, opts)
	if res.Refutation.PlaceboATE != res2.Refutation.PlaceboATE ||
		res.Refutation.BootstrapMean != res2.Refutation.BootstrapMean ||
		res.Refutation.BootstrapStd != res2.Refutation.BootstrapStd {
		t.Errorf("custom-seed run not reproducible: %+v vs %+v", res.Refutation, res2.Refutation)
	}
}

// PlaceboPassed must report FALSE when the configured tolerance is tighter than
// the (small, non-zero) placebo mean. The mean placebo collapses to ~ -0.0091
// on this seeded fixture, so a tolerance of 0.005 sits just below it: the gate
// must flip to false. This exercises the |PlaceboATE| <= tol comparison from the
// failing side (the headline test only covers the passing side).
func TestRefutation_PlaceboFailsUnderTightTolerance(t *testing.T) {
	edges := []graph.Edge{{"Z", "X"}, {"Z", "Y"}, {"X", "Y"}}
	data := simpsonData()

	res, err := BackdoorATEWithRefutation(edges, "X", "Y", data, RefuteOptions{Seed: 1, PlaceboTolerance: 0.005})
	if err != nil {
		t.Fatalf("BackdoorATEWithRefutation error: %v", err)
	}
	if res.Refutation == nil {
		t.Fatalf("expected Refutation != nil")
	}
	// Precondition: the placebo mean is non-zero but well inside 0.005..0.05.
	abs := math.Abs(res.Refutation.PlaceboATE)
	if !(abs > 0.005 && abs < 0.05) {
		t.Fatalf("fixture precondition broken: |PlaceboATE|=%v not in (0.005,0.05)", abs)
	}
	if res.Refutation.PlaceboPassed {
		t.Errorf("PlaceboPassed = true, want false (|PlaceboATE|=%v > tol=0.005)", abs)
	}
}

// The unexported refuter helpers must be degenerate-safe: empty data or a
// zero trial/resample count returns the zero value rather than dividing by
// zero (NaN) or panicking. These guards are not reachable through the public
// API (BackdoorATEWithRefutation errors out on insufficient data and defaults
// non-positive counts), so they are verified directly in-package.
func TestRefuterHelpers_DegenerateInputs(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	z := []string{"Z"}
	data := simpsonData()

	if got := placeboATE(rng, z, "X", "Y", nil, 10); got != 0 {
		t.Errorf("placeboATE(empty data) = %v, want 0", got)
	}
	if got := placeboATE(rng, z, "X", "Y", data, 0); got != 0 {
		t.Errorf("placeboATE(trials=0) = %v, want 0", got)
	}
	if m, s := bootstrapATE(rng, z, "X", "Y", nil, 10); m != 0 || s != 0 {
		t.Errorf("bootstrapATE(empty data) = (%v,%v), want (0,0)", m, s)
	}
	if m, s := bootstrapATE(rng, z, "X", "Y", data, 0); m != 0 || s != 0 {
		t.Errorf("bootstrapATE(resamples=0) = (%v,%v), want (0,0)", m, s)
	}
}

// adjustedOutcomes must return all-zero (and not NaN) on empty data — the guard
// that protects every refuter draw when a bootstrap/placebo sample collapses.
func TestAdjustedOutcomes_EmptyData(t *testing.T) {
	out1, out0, dropped := adjustedOutcomes([]string{"Z"}, "X", "Y", nil)
	if out1 != 0 || out0 != 0 || dropped != 0 {
		t.Errorf("adjustedOutcomes(nil) = (%v,%v,%v), want (0,0,0)", out1, out0, dropped)
	}
	if math.IsNaN(out1) || math.IsNaN(out0) || math.IsNaN(dropped) {
		t.Errorf("adjustedOutcomes(nil) produced NaN")
	}
}
