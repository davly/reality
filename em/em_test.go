package em

import (
	"math"
	"testing"

	"github.com/davly/reality/testutil"
)

// ═══════════════════════════════════════════════════════════════════════════
// Golden-file tests — shared test vectors across Go, Python, C++, C#
// ═══════════════════════════════════════════════════════════════════════════

func TestGolden_CoulombForce(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/em/coulomb_force.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			q1 := testutil.InputFloat64(t, tc, "q1")
			q2 := testutil.InputFloat64(t, tc, "q2")
			r := testutil.InputFloat64(t, tc, "r")
			got := CoulombForce(q1, q2, r)
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

func TestGolden_ElectricField(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/em/electric_field.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			q := testutil.InputFloat64(t, tc, "q")
			r := testutil.InputFloat64(t, tc, "r")
			got := ElectricField(q, r)
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

func TestGolden_OhmsLaw(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/em/ohms_law.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			V := testutil.InputFloat64(t, tc, "V")
			R := testutil.InputFloat64(t, tc, "R")
			got := OhmsLaw(V, R)
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

func TestGolden_PowerElectric(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/em/power_electric.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			V := testutil.InputFloat64(t, tc, "V")
			I := testutil.InputFloat64(t, tc, "I")
			got := PowerElectric(V, I)
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

func TestGolden_ResistorsInSeries(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/em/resistors.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			r := testutil.InputFloat64Slice(t, tc, "resistances")
			got := ResistorsInSeries(r)
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

func TestGolden_ResistorsInParallel(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/em/resistors_parallel.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			r := testutil.InputFloat64Slice(t, tc, "resistances")
			got := ResistorsInParallel(r)
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

func TestGolden_CapacitorEnergy(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/em/energy_storage.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			C := testutil.InputFloat64(t, tc, "C")
			V := testutil.InputFloat64(t, tc, "V")
			got := CapacitorEnergy(C, V)
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

func TestGolden_InductorEnergy(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/em/inductor_energy.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			L := testutil.InputFloat64(t, tc, "L")
			I := testutil.InputFloat64(t, tc, "I")
			got := InductorEnergy(L, I)
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

func TestGolden_RCTimeConstant(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/em/rc_time_constant.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			R := testutil.InputFloat64(t, tc, "R")
			C := testutil.InputFloat64(t, tc, "C")
			got := RCTimeConstant(R, C)
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

func TestGolden_ResonantFrequencyLC(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/em/resonant_frequency.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			L := testutil.InputFloat64(t, tc, "L")
			C := testutil.InputFloat64(t, tc, "C")
			got := ResonantFrequencyLC(L, C)
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — CoulombForce
// ═══════════════════════════════════════════════════════════════════════════

func TestCoulombForce_Repulsive(t *testing.T) {
	// Same-sign charges repel.
	F := CoulombForce(1e-6, 1e-6, 0.1)
	if F <= 0 {
		t.Errorf("same-sign charges should repel, got F=%v", F)
	}
}

func TestCoulombForce_Attractive(t *testing.T) {
	// Opposite-sign charges attract.
	F := CoulombForce(1e-6, -1e-6, 0.1)
	if F >= 0 {
		t.Errorf("opposite-sign charges should attract, got F=%v", F)
	}
}

func TestCoulombForce_InverseSquare(t *testing.T) {
	// Doubling distance should quarter the force.
	F1 := CoulombForce(1, 1, 1)
	F2 := CoulombForce(1, 1, 2)
	assertClose(t, "inv-sq", F1/F2, 4.0, 1e-10)
}

func TestCoulombForce_Symmetry(t *testing.T) {
	// F(q1,q2,r) == F(q2,q1,r)
	F1 := CoulombForce(2e-6, 3e-6, 0.5)
	F2 := CoulombForce(3e-6, 2e-6, 0.5)
	assertClose(t, "symmetry", F1, F2, 1e-20)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — ElectricField
// ═══════════════════════════════════════════════════════════════════════════

func TestElectricField_InverseSquare(t *testing.T) {
	E1 := ElectricField(1, 1)
	E2 := ElectricField(1, 3)
	assertClose(t, "E-inv-sq", E1/E2, 9.0, 1e-10)
}

func TestElectricField_Proportional(t *testing.T) {
	// E is proportional to q.
	E1 := ElectricField(1, 1)
	E2 := ElectricField(5, 1)
	assertClose(t, "E-prop", E2/E1, 5.0, 1e-10)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — OhmsLaw
// ═══════════════════════════════════════════════════════════════════════════

func TestOhmsLaw_Basic(t *testing.T) {
	assertClose(t, "ohm-12v-4r", OhmsLaw(12, 4), 3.0, 1e-15)
}

func TestOhmsLaw_ZeroResistance(t *testing.T) {
	got := OhmsLaw(10, 0)
	if !math.IsInf(got, 1) {
		t.Errorf("expected +Inf for zero resistance, got %v", got)
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — PowerElectric
// ═══════════════════════════════════════════════════════════════════════════

func TestPowerElectric_Basic(t *testing.T) {
	assertClose(t, "pwr-120v-10a", PowerElectric(120, 10), 1200.0, 1e-15)
}

func TestPowerElectric_Zero(t *testing.T) {
	assertClose(t, "pwr-zero", PowerElectric(0, 10), 0.0, 0)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — Resistors
// ═══════════════════════════════════════════════════════════════════════════

func TestResistorsInSeries_Empty(t *testing.T) {
	assertClose(t, "series-empty", ResistorsInSeries(nil), 0.0, 0)
}

func TestResistorsInSeries_Single(t *testing.T) {
	assertClose(t, "series-single", ResistorsInSeries([]float64{42}), 42.0, 1e-15)
}

func TestResistorsInParallel_TwoEqual(t *testing.T) {
	assertClose(t, "par-two-100", ResistorsInParallel([]float64{100, 100}), 50.0, 1e-10)
}

func TestResistorsInParallel_ShortCircuit(t *testing.T) {
	assertClose(t, "par-short", ResistorsInParallel([]float64{100, 0}), 0.0, 0)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — Energy Storage
// ═══════════════════════════════════════════════════════════════════════════

func TestCapacitorEnergy_Basic(t *testing.T) {
	assertClose(t, "cap-1f-1v", CapacitorEnergy(1.0, 1.0), 0.5, 1e-15)
}

func TestCapacitorEnergy_VoltageSquared(t *testing.T) {
	// Energy quadruples when voltage doubles.
	E1 := CapacitorEnergy(1, 1)
	E2 := CapacitorEnergy(1, 2)
	assertClose(t, "cap-v-sq", E2/E1, 4.0, 1e-10)
}

func TestInductorEnergy_Basic(t *testing.T) {
	assertClose(t, "ind-1h-1a", InductorEnergy(1.0, 1.0), 0.5, 1e-15)
}

func TestInductorEnergy_CurrentSquared(t *testing.T) {
	E1 := InductorEnergy(1, 1)
	E2 := InductorEnergy(1, 3)
	assertClose(t, "ind-i-sq", E2/E1, 9.0, 1e-10)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — RC and LC
// ═══════════════════════════════════════════════════════════════════════════

func TestRCTimeConstant_Basic(t *testing.T) {
	assertClose(t, "rc-1k-1u", RCTimeConstant(1000, 1e-6), 1e-3, 1e-15)
}

func TestResonantFrequencyLC_Basic(t *testing.T) {
	f := ResonantFrequencyLC(1, 1)
	expected := 1.0 / (2 * math.Pi)
	assertClose(t, "lc-1h-1f", f, expected, 1e-12)
}

func TestResonantFrequencyLC_InverseRelation(t *testing.T) {
	// Doubling L should reduce f by 1/sqrt(2).
	f1 := ResonantFrequencyLC(1, 1)
	f2 := ResonantFrequencyLC(2, 1)
	assertClose(t, "lc-inv", f1/f2, math.Sqrt(2), 1e-10)
}

// ═══════════════════════════════════════════════════════════════════════════
// Test helpers
// ═══════════════════════════════════════════════════════════════════════════

func assertClose(t *testing.T, label string, got, want, tol float64) {
	t.Helper()
	if math.Abs(got-want) > tol {
		t.Errorf("%s: got %v, want %v (tol %v)", label, got, want, tol)
	}
}
