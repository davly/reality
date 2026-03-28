package fluids

import (
	"math"
	"testing"

	"github.com/davly/reality/testutil"
)

// ═══════════════════════════════════════════════════════════════════════════
// Golden-file tests — shared test vectors across Go, Python, C++, C#
// ═══════════════════════════════════════════════════════════════════════════

func TestGolden_ReynoldsNumber(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/fluids/reynolds.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			rho := testutil.InputFloat64(t, tc, "rho")
			v := testutil.InputFloat64(t, tc, "v")
			L := testutil.InputFloat64(t, tc, "L")
			mu := testutil.InputFloat64(t, tc, "mu")
			got := ReynoldsNumber(rho, v, L, mu)
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

func TestGolden_BernoulliPressure(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/fluids/bernoulli.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			rho := testutil.InputFloat64(t, tc, "rho")
			v1 := testutil.InputFloat64(t, tc, "v1")
			p1 := testutil.InputFloat64(t, tc, "p1")
			h1 := testutil.InputFloat64(t, tc, "h1")
			v2 := testutil.InputFloat64(t, tc, "v2")
			h2 := testutil.InputFloat64(t, tc, "h2")
			g := testutil.InputFloat64(t, tc, "g")
			got := BernoulliPressure(rho, v1, p1, h1, v2, h2, g)
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

func TestGolden_DarcyWeisbach(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/fluids/darcy_weisbach.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			f := testutil.InputFloat64(t, tc, "f")
			L := testutil.InputFloat64(t, tc, "L")
			D := testutil.InputFloat64(t, tc, "D")
			rho := testutil.InputFloat64(t, tc, "rho")
			v := testutil.InputFloat64(t, tc, "v")
			got := DarcyWeisbach(f, L, D, rho, v)
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

func TestGolden_DragForce(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/fluids/drag_force.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			Cd := testutil.InputFloat64(t, tc, "Cd")
			rho := testutil.InputFloat64(t, tc, "rho")
			v := testutil.InputFloat64(t, tc, "v")
			A := testutil.InputFloat64(t, tc, "A")
			got := DragForce(Cd, rho, v, A)
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

func TestGolden_TerminalVelocity(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/fluids/terminal_velocity.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			m := testutil.InputFloat64(t, tc, "m")
			g := testutil.InputFloat64(t, tc, "g")
			Cd := testutil.InputFloat64(t, tc, "Cd")
			rho := testutil.InputFloat64(t, tc, "rho")
			A := testutil.InputFloat64(t, tc, "A")
			got := TerminalVelocity(m, g, Cd, rho, A)
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — Reynolds Number
// ═══════════════════════════════════════════════════════════════════════════

func TestReynoldsNumber_Laminar(t *testing.T) {
	// Water (rho=1000, mu=0.001) at v=0.001 m/s, L=0.01 m -> Re=10
	assertClose(t, "Re-laminar", ReynoldsNumber(1000, 0.001, 0.01, 0.001), 10.0, 1e-12)
}

func TestReynoldsNumber_Turbulent(t *testing.T) {
	// Water in large pipe: rho=1000, v=2, L=0.1, mu=0.001 -> Re=200000
	assertClose(t, "Re-turbulent", ReynoldsNumber(1000, 2, 0.1, 0.001), 200000.0, 1e-8)
}

func TestReynoldsNumber_ZeroViscosity(t *testing.T) {
	got := ReynoldsNumber(1000, 1, 1, 0)
	if !math.IsInf(got, 1) {
		t.Errorf("expected +Inf for zero viscosity, got %v", got)
	}
}

func TestReynoldsNumber_ZeroVelocity(t *testing.T) {
	assertClose(t, "Re-zero-v", ReynoldsNumber(1000, 0, 0.1, 0.001), 0.0, 0)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — Bernoulli Equation
// ═══════════════════════════════════════════════════════════════════════════

func TestBernoulliPressure_SameConditions(t *testing.T) {
	// Identical upstream and downstream -> p2 = p1
	p2 := BernoulliPressure(1000, 5, 101325, 10, 5, 10, 9.80665)
	assertClose(t, "bernoulli-same", p2, 101325.0, 1e-10)
}

func TestBernoulliPressure_VelocityIncrease(t *testing.T) {
	// Horizontal pipe, velocity doubles: rho=1000, v1=1, p1=100000, v2=2
	// p2 = 100000 + 0.5*1000*(1-4) = 100000 - 1500 = 98500
	p2 := BernoulliPressure(1000, 1, 100000, 0, 2, 0, 9.80665)
	assertClose(t, "bernoulli-v-inc", p2, 98500.0, 1e-10)
}

func TestBernoulliPressure_ElevationDrop(t *testing.T) {
	// Same velocity, elevation drop of 10m in water
	// p2 = p1 + rho*g*(h1-h2) = 100000 + 1000*9.80665*10 = 198066.5
	p2 := BernoulliPressure(1000, 1, 100000, 10, 1, 0, 9.80665)
	assertClose(t, "bernoulli-elev", p2, 198066.5, 1e-6)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — Pipe Flow Friction (Colebrook-White)
// ═══════════════════════════════════════════════════════════════════════════

func TestPipeFlowFriction_Laminar(t *testing.T) {
	// Re=1000 (laminar) -> f = 64/1000 = 0.064
	assertClose(t, "friction-laminar", PipeFlowFriction(1000, 0.001, 0.1), 0.064, 1e-15)
}

func TestPipeFlowFriction_LaminarEdge(t *testing.T) {
	// Re=2299 (still laminar) -> f = 64/2299
	assertClose(t, "friction-laminar-edge", PipeFlowFriction(2299, 0.001, 0.1), 64.0/2299, 1e-12)
}

func TestPipeFlowFriction_TurbulentSmooth(t *testing.T) {
	// Smooth pipe (roughness=0) at Re=100000
	// Colebrook smooth: 1/sqrt(f) = -2*log10(2.51/(Re*sqrt(f)))
	// Known Moody chart value ~ 0.0180
	f := PipeFlowFriction(100000, 0, 0.1)
	if f < 0.017 || f > 0.019 {
		t.Errorf("smooth turbulent friction out of range: got %v", f)
	}
}

func TestPipeFlowFriction_TurbulentRough(t *testing.T) {
	// Rough pipe: Re=100000, roughness=0.001, D=0.1 (rel rough=0.01)
	// Expected from Moody chart ~ 0.038
	f := PipeFlowFriction(100000, 0.001, 0.1)
	if f < 0.036 || f > 0.042 {
		t.Errorf("rough turbulent friction out of range: got %v", f)
	}
}

func TestPipeFlowFriction_ZeroRe(t *testing.T) {
	got := PipeFlowFriction(0, 0.001, 0.1)
	if !math.IsNaN(got) {
		t.Errorf("expected NaN for Re=0, got %v", got)
	}
}

func TestPipeFlowFriction_NegativeRe(t *testing.T) {
	got := PipeFlowFriction(-100, 0.001, 0.1)
	if !math.IsNaN(got) {
		t.Errorf("expected NaN for negative Re, got %v", got)
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — Darcy-Weisbach
// ═══════════════════════════════════════════════════════════════════════════

func TestDarcyWeisbach_Known(t *testing.T) {
	// f=0.02, L=100, D=0.1, rho=1000, v=2
	// ΔP = 0.02*(100/0.1)*(1000*4/2) = 0.02*1000*2000 = 40000
	assertClose(t, "DW-known", DarcyWeisbach(0.02, 100, 0.1, 1000, 2), 40000.0, 1e-8)
}

func TestDarcyWeisbach_ZeroVelocity(t *testing.T) {
	assertClose(t, "DW-zero-v", DarcyWeisbach(0.03, 50, 0.05, 1000, 0), 0.0, 0)
}

func TestDarcyWeisbach_ZeroFriction(t *testing.T) {
	assertClose(t, "DW-zero-f", DarcyWeisbach(0, 100, 0.1, 1000, 5), 0.0, 0)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — Drag & Lift
// ═══════════════════════════════════════════════════════════════════════════

func TestDragForce_Car(t *testing.T) {
	// Cd=0.3, rho=1.225, v=30 m/s, A=2.2 m^2
	// F = 0.5*0.3*1.225*900*2.2 = 363.825
	assertClose(t, "drag-car", DragForce(0.3, 1.225, 30, 2.2), 363.825, 1e-8)
}

func TestDragForce_ZeroVelocity(t *testing.T) {
	assertClose(t, "drag-zero", DragForce(0.5, 1000, 0, 1), 0.0, 0)
}

func TestLiftForce_Wing(t *testing.T) {
	// Cl=1.5, rho=1.225, v=70 m/s (takeoff), A=120 m^2 (wing area)
	// F = 0.5*1.5*1.225*4900*120 = 540225
	assertClose(t, "lift-wing", LiftForce(1.5, 1.225, 70, 120), 540225.0, 1e-6)
}

func TestLiftForce_ZeroVelocity(t *testing.T) {
	assertClose(t, "lift-zero", LiftForce(1.0, 1.225, 0, 100), 0.0, 0)
}

func TestLiftForce_NegativeCl(t *testing.T) {
	// Negative Cl gives downforce (like race car spoiler)
	F := LiftForce(-0.5, 1.225, 50, 2)
	if F >= 0 {
		t.Errorf("expected negative force (downforce) for Cl<0, got %v", F)
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — Terminal Velocity
// ═══════════════════════════════════════════════════════════════════════════

func TestTerminalVelocity_Skydiver(t *testing.T) {
	// m=80 kg, Cd=1.0, rho=1.225, A=0.7 m^2
	// v_t = sqrt(2*80*9.80665/(1*1.225*0.7)) ~ 42.71 m/s
	vt := TerminalVelocity(80, 9.80665, 1.0, 1.225, 0.7)
	if vt < 42 || vt > 43 {
		t.Errorf("skydiver terminal velocity out of range: got %v", vt)
	}
}

func TestTerminalVelocity_ZeroDenom(t *testing.T) {
	got := TerminalVelocity(10, 9.80665, 0, 1.225, 1)
	if !math.IsNaN(got) {
		t.Errorf("expected NaN for Cd=0, got %v", got)
	}
}

func TestTerminalVelocity_ZeroMass(t *testing.T) {
	assertClose(t, "tv-zero-mass", TerminalVelocity(0, 9.80665, 1, 1.225, 1), 0.0, 1e-15)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — Stokes Law
// ═══════════════════════════════════════════════════════════════════════════

func TestStokesLaw_Known(t *testing.T) {
	// mu=0.001 Pa·s (water), r=0.001 m, v=0.01 m/s
	// F = 6*pi*0.001*0.001*0.01 = 6*pi*1e-8 ~ 1.885e-7
	expected := 6.0 * math.Pi * 0.001 * 0.001 * 0.01
	assertClose(t, "stokes-known", StokesLaw(0.001, 0.001, 0.01), expected, 1e-20)
}

func TestStokesLaw_ZeroVelocity(t *testing.T) {
	assertClose(t, "stokes-zero-v", StokesLaw(0.001, 0.001, 0), 0.0, 0)
}

func TestStokesLaw_UnitValues(t *testing.T) {
	// F = 6*pi*1*1*1 = 6*pi
	assertClose(t, "stokes-unit", StokesLaw(1, 1, 1), 6*math.Pi, 1e-14)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — Flow Rates
// ═══════════════════════════════════════════════════════════════════════════

func TestMassFlowRate_Known(t *testing.T) {
	// Water: rho=1000, v=2, A=0.01 -> mdot=20
	assertClose(t, "mdot-known", MassFlowRate(1000, 2, 0.01), 20.0, 1e-12)
}

func TestMassFlowRate_ZeroVelocity(t *testing.T) {
	assertClose(t, "mdot-zero", MassFlowRate(1000, 0, 0.01), 0.0, 0)
}

func TestVolumetricFlowRate_Known(t *testing.T) {
	// v=3, A=0.05 -> Q=0.15
	assertClose(t, "Q-known", VolumetricFlowRate(3, 0.05), 0.15, 1e-15)
}

func TestVolumetricFlowRate_ZeroVelocity(t *testing.T) {
	assertClose(t, "Q-zero", VolumetricFlowRate(0, 0.05), 0.0, 0)
}

func TestVolumetricFlowRate_ZeroArea(t *testing.T) {
	assertClose(t, "Q-zero-area", VolumetricFlowRate(10, 0), 0.0, 0)
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

func assertClose(t *testing.T, label string, got, want, tol float64) {
	t.Helper()
	if math.Abs(got-want) > tol {
		t.Errorf("%s: got %v, want %v (tol %v)", label, got, want, tol)
	}
}
