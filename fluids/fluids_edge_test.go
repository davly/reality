package fluids

import (
	"math"
	"testing"
)

// ═══════════════════════════════════════════════════════════════════════════
// ReynoldsNumber — boundary / edge
// ═══════════════════════════════════════════════════════════════════════════

func TestReynoldsNumber_ZeroLength(t *testing.T) {
	assertClose(t, "Re L=0", ReynoldsNumber(1000, 1, 0, 0.001), 0.0, 1e-15)
}

func TestReynoldsNumber_WaterInPipe(t *testing.T) {
	// Water: rho=998, v=1 m/s, L=0.05 m, mu=0.001 Pa.s
	Re := ReynoldsNumber(998, 1, 0.05, 0.001)
	assertClose(t, "water pipe Re", Re, 49900.0, 1.0)
}

func TestReynoldsNumber_AirFlow(t *testing.T) {
	// Air: rho=1.225, v=10 m/s, L=1 m, mu=1.81e-5 Pa.s
	Re := ReynoldsNumber(1.225, 10, 1, 1.81e-5)
	if Re < 600000 || Re > 700000 {
		t.Errorf("air Re out of range: got %v", Re)
	}
}

func TestReynoldsNumber_ZeroViscosityReturnsInf(t *testing.T) {
	Re := ReynoldsNumber(1000, 1, 1, 0)
	if !math.IsInf(Re, 1) {
		t.Errorf("expected +Inf for zero viscosity, got %v", Re)
	}
}

func TestReynoldsNumber_NegativeVelocity(t *testing.T) {
	Re := ReynoldsNumber(1000, -2, 0.1, 0.001)
	// Re is negative with negative velocity
	assertClose(t, "Re neg v", Re, -200000.0, 1e-8)
}

// ═══════════════════════════════════════════════════════════════════════════
// BernoulliPressure — boundary / edge
// ═══════════════════════════════════════════════════════════════════════════

func TestBernoulliPressure_HeightIncrease(t *testing.T) {
	// Higher downstream -> lower pressure (gravity)
	p2 := BernoulliPressure(1000, 1, 101325, 0, 1, 10, 9.81)
	if p2 >= 101325 {
		t.Errorf("higher elevation should reduce pressure, got p2=%v", p2)
	}
}

func TestBernoulliPressure_ZeroDensity(t *testing.T) {
	// rho=0: no kinetic or gravitational terms
	p2 := BernoulliPressure(0, 10, 101325, 0, 20, 100, 9.81)
	assertClose(t, "zero density", p2, 101325, 1e-6)
}

func TestBernoulliPressure_VelocityDecrease(t *testing.T) {
	// Slower downstream -> higher pressure (diffuser)
	p2 := BernoulliPressure(1000, 5, 101325, 0, 1, 0, 9.81)
	if p2 <= 101325 {
		t.Errorf("slower flow should increase pressure, got p2=%v", p2)
	}
}

func TestBernoulliPressure_SymmetryOfKineticTerm(t *testing.T) {
	// Swapping v1/v2 should give symmetric pressure change
	p2a := BernoulliPressure(1000, 1, 100000, 0, 5, 0, 9.81)
	p2b := BernoulliPressure(1000, 5, 100000, 0, 1, 0, 9.81)
	// The pressure changes should be equal and opposite around 100000
	diffA := 100000 - p2a
	diffB := p2b - 100000
	assertClose(t, "kinetic symmetry", diffA, diffB, 1e-6)
}

// ═══════════════════════════════════════════════════════════════════════════
// PipeFlowFriction — boundary / edge
// ═══════════════════════════════════════════════════════════════════════════

func TestPipeFlowFriction_LaminarBoundary2299(t *testing.T) {
	f := PipeFlowFriction(2299, 0.001, 0.1)
	assertClose(t, "laminar at 2299", f, 64.0/2299.0, 1e-12)
}

func TestPipeFlowFriction_TurbulentTransition(t *testing.T) {
	// Re=2300 (just turbulent)
	f := PipeFlowFriction(2300, 0.001, 0.1)
	if f <= 0 || math.IsNaN(f) {
		t.Errorf("turbulent friction at transition should be positive, got %v", f)
	}
}

func TestPipeFlowFriction_SmoothPipe(t *testing.T) {
	f := PipeFlowFriction(100000, 0, 0.1)
	if f <= 0 || f > 0.05 {
		t.Errorf("smooth pipe friction out of range: got %v", f)
	}
}

func TestPipeFlowFriction_HighRe(t *testing.T) {
	f := PipeFlowFriction(1e7, 0.001, 0.1)
	if f <= 0 || f > 0.1 {
		t.Errorf("high Re friction out of range: got %v", f)
	}
}

func TestPipeFlowFriction_VeryRoughPipe(t *testing.T) {
	f := PipeFlowFriction(50000, 0.01, 0.1) // relative roughness 0.1
	if f <= 0 || f > 0.15 {
		t.Errorf("very rough pipe friction out of range: got %v", f)
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// DarcyWeisbach — boundary / edge
// ═══════════════════════════════════════════════════════════════════════════

func TestDarcyWeisbach_ProportionalToLength(t *testing.T) {
	dp1 := DarcyWeisbach(0.02, 10, 0.1, 1000, 2)
	dp2 := DarcyWeisbach(0.02, 20, 0.1, 1000, 2)
	assertClose(t, "double length", dp2, 2*dp1, 1e-6)
}

func TestDarcyWeisbach_ProportionalToVSquared(t *testing.T) {
	dp1 := DarcyWeisbach(0.02, 10, 0.1, 1000, 1)
	dp2 := DarcyWeisbach(0.02, 10, 0.1, 1000, 2)
	assertClose(t, "v squared", dp2, 4*dp1, 1e-6)
}

func TestDarcyWeisbach_InverseToDiameter(t *testing.T) {
	dp1 := DarcyWeisbach(0.02, 10, 0.1, 1000, 2)
	dp2 := DarcyWeisbach(0.02, 10, 0.2, 1000, 2)
	assertClose(t, "inverse to diameter", dp2, dp1/2, 1e-6)
}

// ═══════════════════════════════════════════════════════════════════════════
// DragForce — boundary / edge
// ═══════════════════════════════════════════════════════════════════════════

func TestDragForce_ZeroCd(t *testing.T) {
	assertClose(t, "drag Cd=0", DragForce(0, 1.225, 10, 1), 0.0, 1e-15)
}

func TestDragForce_ProportionalToVSquared(t *testing.T) {
	d1 := DragForce(0.47, 1.225, 10, 1)
	d2 := DragForce(0.47, 1.225, 20, 1)
	assertClose(t, "drag v^2", d2, 4*d1, 1e-6)
}

func TestDragForce_NegativeVelocitySquaresPositive(t *testing.T) {
	dPos := DragForce(0.47, 1.225, 10, 1)
	dNeg := DragForce(0.47, 1.225, -10, 1)
	assertClose(t, "neg velocity same drag", dNeg, dPos, 1e-12)
}

func TestDragForce_ProportionalToArea(t *testing.T) {
	d1 := DragForce(0.47, 1.225, 10, 1)
	d2 := DragForce(0.47, 1.225, 10, 2)
	assertClose(t, "drag area", d2, 2*d1, 1e-12)
}

// ═══════════════════════════════════════════════════════════════════════════
// LiftForce — boundary / edge
// ═══════════════════════════════════════════════════════════════════════════

func TestLiftForce_ProportionalToVSquared(t *testing.T) {
	l1 := LiftForce(1.2, 1.225, 10, 10)
	l2 := LiftForce(1.2, 1.225, 20, 10)
	assertClose(t, "lift v^2", l2, 4*l1, 1e-6)
}

func TestLiftForce_ProportionalToCl(t *testing.T) {
	l1 := LiftForce(1.0, 1.225, 30, 10)
	l2 := LiftForce(2.0, 1.225, 30, 10)
	assertClose(t, "lift Cl", l2, 2*l1, 1e-6)
}

// ═══════════════════════════════════════════════════════════════════════════
// TerminalVelocity — boundary / edge
// ═══════════════════════════════════════════════════════════════════════════

func TestTerminalVelocity_ProportionalToSqrtMass(t *testing.T) {
	vt1 := TerminalVelocity(10, 9.81, 0.5, 1.225, 1)
	vt2 := TerminalVelocity(40, 9.81, 0.5, 1.225, 1)
	assertClose(t, "sqrt mass proportionality", vt2/vt1, 2.0, 1e-10)
}

func TestTerminalVelocity_ZeroCd(t *testing.T) {
	vt := TerminalVelocity(1, 9.81, 0, 1.225, 1) // Cd=0
	if !math.IsNaN(vt) {
		t.Errorf("expected NaN for zero Cd, got %v", vt)
	}
}

func TestTerminalVelocity_NegativeArea(t *testing.T) {
	vt := TerminalVelocity(1, 9.81, 0.5, 1.225, -1)
	if !math.IsNaN(vt) {
		t.Errorf("expected NaN for negative area, got %v", vt)
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// StokesLaw — boundary / edge
// ═══════════════════════════════════════════════════════════════════════════

func TestStokesLaw_ZeroRadius(t *testing.T) {
	assertClose(t, "stokes r=0", StokesLaw(0.001, 0, 1), 0.0, 1e-15)
}

func TestStokesLaw_ProportionalToVelocity(t *testing.T) {
	f1 := StokesLaw(0.001, 0.01, 1)
	f2 := StokesLaw(0.001, 0.01, 2)
	assertClose(t, "stokes linear v", f2, 2*f1, 1e-12)
}

func TestStokesLaw_ProportionalToRadius(t *testing.T) {
	f1 := StokesLaw(0.001, 0.01, 1)
	f2 := StokesLaw(0.001, 0.02, 1)
	assertClose(t, "stokes linear r", f2, 2*f1, 1e-12)
}

func TestStokesLaw_ProportionalToViscosity(t *testing.T) {
	f1 := StokesLaw(0.001, 0.01, 1)
	f2 := StokesLaw(0.002, 0.01, 1)
	assertClose(t, "stokes linear mu", f2, 2*f1, 1e-12)
}

// ═══════════════════════════════════════════════════════════════════════════
// FlowRates — boundary / edge
// ═══════════════════════════════════════════════════════════════════════════

func TestMassFlowRate_ConsistencyWithVolumetric(t *testing.T) {
	rho, v, A := 1000.0, 2.0, 0.01
	mdot := MassFlowRate(rho, v, A)
	Q := VolumetricFlowRate(v, A)
	assertClose(t, "mdot = rho * Q", mdot, rho*Q, 1e-12)
}

func TestMassFlowRate_ZeroArea(t *testing.T) {
	assertClose(t, "mdot A=0", MassFlowRate(1000, 2, 0), 0.0, 1e-15)
}

func TestVolumetricFlowRate_LargeValues(t *testing.T) {
	Q := VolumetricFlowRate(100, 10)
	assertClose(t, "Q large", Q, 1000.0, 1e-12)
}
