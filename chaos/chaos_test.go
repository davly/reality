package chaos

import (
	"math"
	"testing"

	"github.com/davly/reality/testutil"
)

// ═══════════════════════════════════════════════════════════════════════════
// Golden-file tests — shared test vectors across Go, Python, C++, C#
// ═══════════════════════════════════════════════════════════════════════════

func TestGolden_LorenzTrajectory(t *testing.T) {
	gf := testutil.LoadGolden(t, "../testdata/chaos/lorenz.json")
	lorenz := LorenzSystem(10, 28, 8.0/3.0)
	y0 := []float64{1.0, 1.0, 1.0}
	traj := SolveODE(lorenz, y0, 0, 0.09, 0.01)

	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			step := testutil.InputInt(t, tc, "step")
			if step >= len(traj) {
				t.Fatalf("step %d out of range (trajectory has %d points)", step, len(traj))
			}
			testutil.AssertFloat64Slice(t, tc, traj[step])
		})
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — RK4 ODE solver
// ═══════════════════════════════════════════════════════════════════════════

func TestRK4_ExponentialDecay(t *testing.T) {
	// y' = -y, y(0) = 1. Exact solution: y(t) = exp(-t).
	f := func(tt float64, y []float64, dydt []float64) {
		dydt[0] = -y[0]
	}
	y := []float64{1.0}
	out := make([]float64, 1)
	dt := 0.001
	tt := 0.0

	for i := 0; i < 1000; i++ {
		RK4Step(f, tt, y, dt, out)
		copy(y, out)
		tt += dt
	}

	// At t=1: y = exp(-1) ≈ 0.367879441...
	expected := math.Exp(-1.0)
	assertClose(t, "RK4 exponential decay at t=1", y[0], expected, 1e-12)
}

func TestRK4_ExponentialDecayLargerStep(t *testing.T) {
	// Same system, larger dt to verify RK4 still accurate.
	f := func(tt float64, y []float64, dydt []float64) {
		dydt[0] = -y[0]
	}
	y := []float64{1.0}
	out := make([]float64, 1)
	dt := 0.01
	tt := 0.0

	for i := 0; i < 100; i++ {
		RK4Step(f, tt, y, dt, out)
		copy(y, out)
		tt += dt
	}

	expected := math.Exp(-1.0)
	assertClose(t, "RK4 exponential decay dt=0.01", y[0], expected, 1e-10)
}

func TestRK4_HarmonicOscillatorEnergy(t *testing.T) {
	// y'' = -y, written as [y, y']. Energy E = 0.5*(y^2 + y'^2) = const.
	f := func(tt float64, y []float64, dydt []float64) {
		dydt[0] = y[1]
		dydt[1] = -y[0]
	}
	y := []float64{1.0, 0.0} // cos(t), -sin(t)
	out := make([]float64, 2)
	dt := 0.01

	E0 := 0.5 * (y[0]*y[0] + y[1]*y[1])

	for i := 0; i < 10000; i++ {
		RK4Step(f, float64(i)*dt, y, dt, out)
		copy(y, out)
	}

	E1 := 0.5 * (y[0]*y[0] + y[1]*y[1])
	assertClose(t, "harmonic oscillator energy conservation", E1, E0, 1e-8)
}

func TestRK4_HarmonicOscillatorSolution(t *testing.T) {
	// y'' = -y, y(0) = 1, y'(0) = 0 => y(t) = cos(t).
	f := func(tt float64, y []float64, dydt []float64) {
		dydt[0] = y[1]
		dydt[1] = -y[0]
	}
	y := []float64{1.0, 0.0}
	out := make([]float64, 2)
	dt := 0.001
	tFinal := math.Pi // Half period.

	steps := int(tFinal / dt)
	for i := 0; i < steps; i++ {
		RK4Step(f, float64(i)*dt, y, dt, out)
		copy(y, out)
	}

	// y(pi) = cos(pi) = -1.
	assertClose(t, "harmonic oscillator y(pi)=cos(pi)", y[0], -1.0, 1e-6)
	// y'(pi) = -sin(pi) = 0.
	assertClose(t, "harmonic oscillator y'(pi)=-sin(pi)", y[1], 0.0, 1e-3)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — Euler step
// ═══════════════════════════════════════════════════════════════════════════

func TestEuler_ExponentialDecay(t *testing.T) {
	f := func(tt float64, y []float64, dydt []float64) {
		dydt[0] = -y[0]
	}
	y := []float64{1.0}
	out := make([]float64, 1)
	dt := 0.001

	for i := 0; i < 1000; i++ {
		EulerStep(f, float64(i)*dt, y, dt, out)
		copy(y, out)
	}

	// Euler is first-order, so expect less accuracy than RK4.
	expected := math.Exp(-1.0)
	assertClose(t, "Euler exponential decay", y[0], expected, 0.001)
}

func TestEuler_SingleStep(t *testing.T) {
	// y' = 2, y(0) = 0. After one step with dt=0.1: y = 0.2.
	f := func(tt float64, y []float64, dydt []float64) {
		dydt[0] = 2.0
	}
	y := []float64{0.0}
	out := make([]float64, 1)
	EulerStep(f, 0, y, 0.1, out)
	assertClose(t, "Euler constant rate", out[0], 0.2, 1e-15)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — SolveODE
// ═══════════════════════════════════════════════════════════════════════════

func TestSolveODE_TrajectoryLength(t *testing.T) {
	f := func(tt float64, y []float64, dydt []float64) {
		dydt[0] = -y[0]
	}
	traj := SolveODE(f, []float64{1.0}, 0, 1.0, 0.1)

	// 10 steps from 0 to 1 with dt=0.1, plus initial state = 11 points.
	if len(traj) != 11 {
		t.Errorf("expected 11 points, got %d", len(traj))
	}
}

func TestSolveODE_InvalidInput(t *testing.T) {
	f := func(tt float64, y []float64, dydt []float64) {
		dydt[0] = -y[0]
	}
	// Negative dt.
	traj := SolveODE(f, []float64{1.0}, 0, 1.0, -0.1)
	if traj != nil {
		t.Error("expected nil for negative dt")
	}
	// tEnd < t0.
	traj = SolveODE(f, []float64{1.0}, 2.0, 1.0, 0.1)
	if traj != nil {
		t.Error("expected nil for tEnd < t0")
	}
}

func TestSolveODE_InitialStatePreserved(t *testing.T) {
	f := func(tt float64, y []float64, dydt []float64) {
		dydt[0] = -y[0]
	}
	y0 := []float64{5.0}
	traj := SolveODE(f, y0, 0, 0.5, 0.1)

	// First point should be the initial state.
	assertClose(t, "initial state", traj[0][0], 5.0, 1e-15)
	// y0 should not be mutated.
	assertClose(t, "y0 not mutated", y0[0], 5.0, 1e-15)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — Lorenz system
// ═══════════════════════════════════════════════════════════════════════════

func TestLorenz_SensitiveToInitialConditions(t *testing.T) {
	lorenz := LorenzSystem(10, 28, 8.0/3.0)
	y1 := []float64{1.0, 1.0, 1.0}
	y2 := []float64{1.0 + 1e-8, 1.0, 1.0} // Tiny perturbation.

	traj1 := SolveODE(lorenz, y1, 0, 50, 0.01)
	traj2 := SolveODE(lorenz, y2, 0, 50, 0.01)

	// After sufficient time, trajectories should diverge significantly.
	last := len(traj1) - 1
	dist := euclideanDist(traj1[last], traj2[last])
	if dist < 1.0 {
		t.Errorf("Lorenz trajectories should diverge, but final distance is only %v", dist)
	}
}

func TestLorenz_StaysOnAttractor(t *testing.T) {
	// The Lorenz attractor is bounded. After transient, all points should
	// stay within a known bounding box.
	lorenz := LorenzSystem(10, 28, 8.0/3.0)
	y0 := []float64{1.0, 1.0, 1.0}
	traj := SolveODE(lorenz, y0, 0, 50, 0.01)

	// Skip transient (first 1000 steps).
	for i := 1000; i < len(traj); i++ {
		x, y, z := traj[i][0], traj[i][1], traj[i][2]
		if math.Abs(x) > 50 || math.Abs(y) > 50 || z < -5 || z > 60 {
			t.Fatalf("trajectory escaped attractor at step %d: [%v, %v, %v]", i, x, y, z)
		}
	}
}

func TestLorenz_DerivativesCorrect(t *testing.T) {
	lorenz := LorenzSystem(10, 28, 8.0/3.0)
	y := []float64{1.0, 2.0, 3.0}
	dydt := make([]float64, 3)
	lorenz(0, y, dydt)

	// dx/dt = 10*(2-1) = 10
	assertClose(t, "Lorenz dx/dt", dydt[0], 10.0, 1e-15)
	// dy/dt = 1*(28-3) - 2 = 23
	assertClose(t, "Lorenz dy/dt", dydt[1], 23.0, 1e-15)
	// dz/dt = 1*2 - (8/3)*3 = 2 - 8 = -6
	assertClose(t, "Lorenz dz/dt", dydt[2], -6.0, 1e-15)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — Rossler system
// ═══════════════════════════════════════════════════════════════════════════

func TestRossler_DerivativesCorrect(t *testing.T) {
	rossler := RosslerSystem(0.2, 0.2, 5.7)
	y := []float64{1.0, 2.0, 3.0}
	dydt := make([]float64, 3)
	rossler(0, y, dydt)

	// dx/dt = -2 - 3 = -5
	assertClose(t, "Rossler dx/dt", dydt[0], -5.0, 1e-14)
	// dy/dt = 1 + 0.2*2 = 1.4
	assertClose(t, "Rossler dy/dt", dydt[1], 1.4, 1e-14)
	// dz/dt = 0.2 + 3*(1 - 5.7) = 0.2 - 14.1 = -13.9
	assertClose(t, "Rossler dz/dt", dydt[2], -13.9, 1e-14)
}

func TestRossler_BoundedAttractor(t *testing.T) {
	rossler := RosslerSystem(0.2, 0.2, 5.7)
	y0 := []float64{1.0, 1.0, 0.0}
	traj := SolveODE(rossler, y0, 0, 200, 0.01)

	// After transient, Rossler should stay bounded.
	for i := 5000; i < len(traj); i++ {
		for d := 0; d < 3; d++ {
			if math.Abs(traj[i][d]) > 50 {
				t.Fatalf("Rossler escaped at step %d dim %d: %v", i, d, traj[i][d])
			}
		}
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — Lotka-Volterra predator-prey
// ═══════════════════════════════════════════════════════════════════════════

func TestLotkaVolterra_PopulationsOscillate(t *testing.T) {
	lv := LotkaVolterra(1.1, 0.4, 0.1, 0.4)
	y0 := []float64{10.0, 10.0}
	traj := SolveODE(lv, y0, 0, 30, 0.001)

	// Find a local max, then min in prey population (oscillation).
	foundMax := false
	foundMinAfterMax := false
	prevPrey := traj[0][0]
	increasing := true

	for i := 1; i < len(traj); i++ {
		prey := traj[i][0]
		if increasing && prey < prevPrey {
			foundMax = true
			increasing = false
		}
		if !increasing && prey > prevPrey {
			if foundMax {
				foundMinAfterMax = true
				break
			}
		}
		prevPrey = prey
	}

	if !foundMinAfterMax {
		t.Error("Lotka-Volterra: prey population did not oscillate (no max then min found)")
	}
}

func TestLotkaVolterra_HamiltonianConserved(t *testing.T) {
	// Conserved quantity: H = delta*x - gamma*ln(x) + beta*y - alpha*ln(y).
	alpha, beta, delta, gamma := 1.1, 0.4, 0.1, 0.4
	lv := LotkaVolterra(alpha, beta, delta, gamma)
	y0 := []float64{10.0, 10.0}
	traj := SolveODE(lv, y0, 0, 50, 0.001)

	hamiltonian := func(prey, pred float64) float64 {
		return delta*prey - gamma*math.Log(prey) + beta*pred - alpha*math.Log(pred)
	}

	H0 := hamiltonian(traj[0][0], traj[0][1])
	for i := 1; i < len(traj); i += 1000 {
		Hi := hamiltonian(traj[i][0], traj[i][1])
		if math.Abs(Hi-H0) > 0.01 {
			t.Errorf("Hamiltonian not conserved at step %d: H0=%v, Hi=%v, diff=%v",
				i, H0, Hi, math.Abs(Hi-H0))
		}
	}
}

func TestLotkaVolterra_PositivePopulations(t *testing.T) {
	lv := LotkaVolterra(1.1, 0.4, 0.1, 0.4)
	y0 := []float64{10.0, 10.0}
	traj := SolveODE(lv, y0, 0, 50, 0.001)

	for i, pt := range traj {
		if pt[0] < 0 || pt[1] < 0 {
			t.Fatalf("negative population at step %d: prey=%v, pred=%v", i, pt[0], pt[1])
		}
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — SIR epidemic model
// ═══════════════════════════════════════════════════════════════════════════

func TestSIR_PopulationConserved(t *testing.T) {
	// S + I + R must remain constant.
	sir := SIRModel(0.3, 0.1)
	y0 := []float64{0.99, 0.01, 0.0}
	traj := SolveODE(sir, y0, 0, 100, 0.01)

	total0 := traj[0][0] + traj[0][1] + traj[0][2]
	for i := 0; i < len(traj); i += 100 {
		total := traj[i][0] + traj[i][1] + traj[i][2]
		assertClose(t, "SIR population conservation", total, total0, 1e-8)
	}
}

func TestSIR_R0_GreaterThan1_EpidemicGrows(t *testing.T) {
	// R_0 = beta/gamma = 0.3/0.1 = 3 > 1 → epidemic.
	sir := SIRModel(0.3, 0.1)
	y0 := []float64{0.99, 0.01, 0.0}
	traj := SolveODE(sir, y0, 0, 100, 0.01)

	// Infected should increase initially.
	peakI := 0.0
	for _, pt := range traj {
		if pt[1] > peakI {
			peakI = pt[1]
		}
	}
	if peakI < y0[1]*2 {
		t.Errorf("SIR with R0>1: expected epidemic peak > 2*I0, got %v", peakI)
	}
}

func TestSIR_R0_LessThan1_EpidemicDiesOut(t *testing.T) {
	// R_0 = beta/gamma = 0.05/0.1 = 0.5 < 1 → epidemic dies out.
	sir := SIRModel(0.05, 0.1)
	y0 := []float64{0.99, 0.01, 0.0}
	traj := SolveODE(sir, y0, 0, 200, 0.01)

	// Infected should monotonically decrease.
	lastI := traj[len(traj)-1][1]
	if lastI > y0[1] {
		t.Errorf("SIR with R0<1: final infected %v > initial %v", lastI, y0[1])
	}
	if lastI > 0.001 {
		t.Errorf("SIR with R0<1: epidemic should have died out, got I=%v", lastI)
	}
}

func TestSIR_NonNegative(t *testing.T) {
	sir := SIRModel(0.3, 0.1)
	y0 := []float64{0.99, 0.01, 0.0}
	traj := SolveODE(sir, y0, 0, 200, 0.01)

	for i, pt := range traj {
		for j := 0; j < 3; j++ {
			if pt[j] < -1e-10 {
				t.Fatalf("SIR compartment %d negative at step %d: %v", j, i, pt[j])
			}
		}
	}
}

func TestSIR_DerivativesCorrect(t *testing.T) {
	sir := SIRModel(0.3, 0.1)
	y := []float64{0.9, 0.1, 0.0}
	dydt := make([]float64, 3)
	sir(0, y, dydt)

	// dS/dt = -0.3 * 0.9 * 0.1 = -0.027
	assertClose(t, "SIR dS/dt", dydt[0], -0.027, 1e-15)
	// dI/dt = 0.3*0.9*0.1 - 0.1*0.1 = 0.027 - 0.01 = 0.017
	assertClose(t, "SIR dI/dt", dydt[1], 0.017, 1e-15)
	// dR/dt = 0.1 * 0.1 = 0.01
	assertClose(t, "SIR dR/dt", dydt[2], 0.01, 1e-15)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — Van der Pol oscillator
// ═══════════════════════════════════════════════════════════════════════════

func TestVanDerPol_LimitCycle(t *testing.T) {
	// With mu > 0, the system should converge to a limit cycle.
	vdp := VanDerPol(1.0)
	y0 := []float64{0.1, 0.0} // Start near origin.
	traj := SolveODE(vdp, y0, 0, 50, 0.01)

	// After transient, amplitude should stabilize near 2.
	last := len(traj) - 1
	amplitude := math.Sqrt(traj[last][0]*traj[last][0] + traj[last][1]*traj[last][1])
	if amplitude < 0.5 {
		t.Errorf("Van der Pol: amplitude too small after transient: %v", amplitude)
	}
}

func TestVanDerPol_Mu0_HarmonicOscillator(t *testing.T) {
	// mu=0 → simple harmonic oscillator, energy conserved.
	vdp := VanDerPol(0)
	y0 := []float64{1.0, 0.0}
	traj := SolveODE(vdp, y0, 0, 10, 0.001)

	E0 := 0.5 * (y0[0]*y0[0] + y0[1]*y0[1])
	last := traj[len(traj)-1]
	E1 := 0.5 * (last[0]*last[0] + last[1]*last[1])
	assertClose(t, "VdP mu=0 energy", E1, E0, 1e-6)
}

func TestVanDerPol_DerivativesCorrect(t *testing.T) {
	vdp := VanDerPol(2.0)
	y := []float64{1.0, 0.5}
	dydt := make([]float64, 2)
	vdp(0, y, dydt)

	// dx/dt = y[1] = 0.5
	assertClose(t, "VdP dx/dt", dydt[0], 0.5, 1e-15)
	// dy/dt = 2*(1-1)*0.5 - 1.0 = -1.0
	assertClose(t, "VdP dy/dt", dydt[1], -1.0, 1e-15)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — Logistic map
// ═══════════════════════════════════════════════════════════════════════════

func TestLogisticMap_R2_5_FixedPoint(t *testing.T) {
	// For r=2.5, the fixed point is (r-1)/r = 0.6.
	x := 0.1
	for i := 0; i < 1000; i++ {
		x = LogisticMap(2.5, x)
	}
	assertClose(t, "logistic r=2.5 fixed point", x, 0.6, 1e-10)
}

func TestLogisticMap_R3_2_Period2(t *testing.T) {
	// For r=3.2, the system enters a period-2 cycle.
	// After warmup, the orbit alternates between two values.
	x := 0.1
	for i := 0; i < 10000; i++ {
		x = LogisticMap(3.2, x)
	}
	// x is now on the period-2 cycle. Record two consecutive values.
	xa := x
	xb := LogisticMap(3.2, xa)
	xc := LogisticMap(3.2, xb) // Should return to xa.

	// After two iterations, we should return to the starting point.
	assertClose(t, "logistic r=3.2 period-2 return", xc, xa, 1e-10)
	// The two cycle points should differ.
	if math.Abs(xb-xa) < 0.01 {
		t.Error("logistic r=3.2: cycle points xa and xb should differ")
	}
}

func TestLogisticMap_R3_5_Period4(t *testing.T) {
	// For r=3.5, the system enters a period-4 cycle.
	x := 0.1
	for i := 0; i < 10000; i++ {
		x = LogisticMap(3.5, x)
	}
	// Record 4 consecutive values and verify the cycle returns.
	xa := x
	xb := LogisticMap(3.5, xa)
	xc := LogisticMap(3.5, xb)
	xd := LogisticMap(3.5, xc)
	xe := LogisticMap(3.5, xd) // Should return to xa.

	assertClose(t, "logistic r=3.5 period-4 return", xe, xa, 1e-10)
	// Verify it's not period-2 (xa != xc).
	if math.Abs(xa-xc) < 0.01 {
		t.Error("logistic r=3.5: should be period-4, not period-2")
	}
}

func TestLogisticMap_R4_Chaotic(t *testing.T) {
	// r=4: chaotic regime. Two nearby initial conditions should diverge.
	// Avoid x=0.5 which maps to 1.0 then 0.0 (degenerate).
	x1 := 0.3
	x2 := 0.3 + 1e-10

	for i := 0; i < 50; i++ {
		x1 = LogisticMap(4.0, x1)
		x2 = LogisticMap(4.0, x2)
	}

	if math.Abs(x1-x2) < 0.01 {
		t.Errorf("logistic r=4: trajectories should diverge, diff=%v", math.Abs(x1-x2))
	}
}

func TestLogisticMap_KnownValues(t *testing.T) {
	// x = 0.5, r = 2: x_next = 2 * 0.5 * 0.5 = 0.5 (fixed point).
	assertClose(t, "logistic r=2 x=0.5", LogisticMap(2.0, 0.5), 0.5, 1e-15)

	// x = 0.25, r = 4: x_next = 4 * 0.25 * 0.75 = 0.75.
	assertClose(t, "logistic r=4 x=0.25", LogisticMap(4.0, 0.25), 0.75, 1e-15)

	// x = 0, r = anything: x_next = 0.
	assertClose(t, "logistic x=0", LogisticMap(3.0, 0.0), 0.0, 1e-15)

	// x = 1, r = anything: x_next = 0.
	assertClose(t, "logistic x=1", LogisticMap(3.0, 1.0), 0.0, 1e-15)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — Game of Life
// ═══════════════════════════════════════════════════════════════════════════

func TestGameOfLife_BlinkerOscillates(t *testing.T) {
	// Blinker: period-2 oscillator.
	// Step 0:    Step 1:    Step 2 (= Step 0):
	// .X.        ...        .X.
	// .X.        XXX        .X.
	// .X.        ...        .X.
	rows, cols := 5, 5
	grid := makeGrid(rows, cols)
	grid[1][2] = true
	grid[2][2] = true
	grid[3][2] = true

	out := makeGrid(rows, cols)
	GameOfLife(grid, rows, cols, out)

	// After one step: horizontal blinker.
	if !out[2][1] || !out[2][2] || !out[2][3] {
		t.Error("blinker step 1: expected horizontal line at row 2")
	}
	if out[1][2] || out[3][2] {
		t.Error("blinker step 1: cells at (1,2) and (3,2) should be dead")
	}

	// Step 2: back to vertical.
	out2 := makeGrid(rows, cols)
	GameOfLife(out, rows, cols, out2)

	if !out2[1][2] || !out2[2][2] || !out2[3][2] {
		t.Error("blinker step 2: expected vertical line at col 2")
	}
	if out2[2][1] || out2[2][3] {
		t.Error("blinker step 2: cells at (2,1) and (2,3) should be dead")
	}
}

func TestGameOfLife_BlockStillLife(t *testing.T) {
	// Block: 2x2 still life.
	rows, cols := 4, 4
	grid := makeGrid(rows, cols)
	grid[1][1] = true
	grid[1][2] = true
	grid[2][1] = true
	grid[2][2] = true

	out := makeGrid(rows, cols)
	GameOfLife(grid, rows, cols, out)

	// Block should be unchanged.
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			if grid[r][c] != out[r][c] {
				t.Errorf("block changed at (%d,%d): was %v, now %v", r, c, grid[r][c], out[r][c])
			}
		}
	}
}

func TestGameOfLife_EmptyGridStaysEmpty(t *testing.T) {
	rows, cols := 3, 3
	grid := makeGrid(rows, cols)
	out := makeGrid(rows, cols)
	GameOfLife(grid, rows, cols, out)

	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			if out[r][c] {
				t.Errorf("empty grid produced life at (%d,%d)", r, c)
			}
		}
	}
}

func TestGameOfLife_GliderMoves(t *testing.T) {
	// Glider: moves diagonally every 4 steps.
	rows, cols := 6, 6
	grid := makeGrid(rows, cols)
	// Standard glider pointing down-right:
	//  .X.
	//  ..X
	//  XXX
	grid[0][1] = true
	grid[1][2] = true
	grid[2][0] = true
	grid[2][1] = true
	grid[2][2] = true

	// Run 4 steps.
	out := makeGrid(rows, cols)
	for i := 0; i < 4; i++ {
		GameOfLife(grid, rows, cols, out)
		grid, out = out, grid
		out = makeGrid(rows, cols)
	}

	// After 4 steps, glider should have moved 1 cell down and 1 cell right.
	if !grid[1][2] || !grid[2][3] || !grid[3][1] || !grid[3][2] || !grid[3][3] {
		t.Error("glider did not move correctly after 4 steps")
	}
}

func TestGameOfLife_TorusWrapping(t *testing.T) {
	// Test that cells at edges wrap around.
	rows, cols := 3, 3
	grid := makeGrid(rows, cols)
	// Place cells that form a blinker wrapping around the edge.
	grid[0][0] = true
	grid[1][0] = true
	grid[2][0] = true

	out := makeGrid(rows, cols)
	GameOfLife(grid, rows, cols, out)

	// On torus: (0,0) has neighbors wrapping, this is effectively a blinker.
	// Middle cell (1,0) should survive with 2 neighbors.
	// Expected: horizontal line at row 1 wrapping around.
	if !out[1][0] {
		t.Error("torus: center of vertical blinker should survive")
	}
	if !out[1][2] {
		t.Error("torus: wrapping neighbor should be alive")
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — Lyapunov exponent
// ═══════════════════════════════════════════════════════════════════════════

func TestLyapunov_LogisticR4_Ln2(t *testing.T) {
	// For the logistic map at r=4, the Lyapunov exponent = ln(2) ≈ 0.693.
	f := func(x float64) float64 {
		return 4.0 * x * (1 - x)
	}
	lambda := LyapunovExponent(f, 0.1, 100000)
	assertClose(t, "Lyapunov logistic r=4", lambda, math.Ln2, 0.01)
}

func TestLyapunov_LogisticR2_5_Negative(t *testing.T) {
	// For r=2.5, fixed point is stable → negative Lyapunov exponent.
	f := func(x float64) float64 {
		return 2.5 * x * (1 - x)
	}
	lambda := LyapunovExponent(f, 0.1, 10000)
	if lambda >= 0 {
		t.Errorf("Lyapunov r=2.5: expected negative, got %v", lambda)
	}
}

func TestLyapunov_Identity_Zero(t *testing.T) {
	// f(x) = x has |f'| = 1, so Lyapunov exponent = ln(1) = 0.
	f := func(x float64) float64 { return x }
	lambda := LyapunovExponent(f, 0.5, 10000)
	assertClose(t, "Lyapunov identity map", lambda, 0.0, 0.001)
}

func TestLyapunov_Contraction(t *testing.T) {
	// f(x) = 0.5*x contracts, Lyapunov = ln(0.5) ≈ -0.693.
	f := func(x float64) float64 { return 0.5 * x }
	lambda := LyapunovExponent(f, 1.0, 10000)
	assertClose(t, "Lyapunov contraction", lambda, math.Log(0.5), 0.001)
}

func TestLyapunov_ZeroIterations(t *testing.T) {
	f := func(x float64) float64 { return x }
	lambda := LyapunovExponent(f, 0.5, 0)
	assertClose(t, "Lyapunov zero iterations", lambda, 0.0, 1e-15)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — Bifurcation diagram
// ═══════════════════════════════════════════════════════════════════════════

func TestBifurcation_BasicStructure(t *testing.T) {
	bif := BifurcationDiagram(LogisticMap, 2.5, 4.0, 10, 100, 5)

	// Should have (10+1) * 5 = 55 points.
	if len(bif) != 55 {
		t.Errorf("expected 55 bifurcation points, got %d", len(bif))
	}

	// All X values should be in [0, 1] for logistic map.
	for _, pt := range bif {
		if pt.X < 0 || pt.X > 1 {
			t.Errorf("bifurcation X out of range: R=%v, X=%v", pt.R, pt.X)
		}
	}
}

func TestBifurcation_FixedPointRegion(t *testing.T) {
	// For r in [2.5, 2.8], logistic map has a single fixed point.
	bif := BifurcationDiagram(LogisticMap, 2.5, 2.8, 5, 500, 10)

	for _, pt := range bif {
		// Fixed point = (r-1)/r.
		expected := (pt.R - 1) / pt.R
		if math.Abs(pt.X-expected) > 0.001 {
			t.Errorf("R=%v: expected fixed point %v, got %v", pt.R, expected, pt.X)
		}
	}
}

func TestBifurcation_InvalidInput(t *testing.T) {
	bif := BifurcationDiagram(LogisticMap, 0, 4, 0, 100, 5)
	if bif != nil {
		t.Error("expected nil for rSteps=0")
	}

	bif = BifurcationDiagram(LogisticMap, 0, 4, 10, 100, 0)
	if bif != nil {
		t.Error("expected nil for samples=0")
	}
}

func TestBifurcation_RRangeCorrect(t *testing.T) {
	bif := BifurcationDiagram(LogisticMap, 2.0, 3.0, 2, 100, 1)

	// Should sample r = 2.0, 2.5, 3.0.
	rValues := make(map[float64]bool)
	for _, pt := range bif {
		rValues[pt.R] = true
	}
	if len(rValues) != 3 {
		t.Errorf("expected 3 distinct R values, got %d", len(rValues))
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — Recurrence plot
// ═══════════════════════════════════════════════════════════════════════════

func TestRecurrence_Diagonal(t *testing.T) {
	// Diagonal should always be true (point is recurrent with itself).
	traj := [][]float64{{0}, {1}, {2}, {3}, {4}}
	rp := RecurrencePlot(traj, 0.5)

	for i := 0; i < len(traj); i++ {
		if !rp[i][i] {
			t.Errorf("diagonal element (%d,%d) should be true", i, i)
		}
	}
}

func TestRecurrence_Symmetric(t *testing.T) {
	traj := [][]float64{{0, 0}, {1, 0}, {0, 1}, {1, 1}, {0.5, 0.5}}
	rp := RecurrencePlot(traj, 1.0)

	for i := 0; i < len(traj); i++ {
		for j := 0; j < len(traj); j++ {
			if rp[i][j] != rp[j][i] {
				t.Errorf("recurrence matrix not symmetric at (%d,%d)", i, j)
			}
		}
	}
}

func TestRecurrence_AllClose(t *testing.T) {
	// All points identical → all entries true.
	traj := [][]float64{{1, 2}, {1, 2}, {1, 2}}
	rp := RecurrencePlot(traj, 0.01)

	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			if !rp[i][j] {
				t.Errorf("identical points should all be recurrent: (%d,%d)", i, j)
			}
		}
	}
}

func TestRecurrence_AllFar(t *testing.T) {
	// Points far apart with small threshold → only diagonal.
	traj := [][]float64{{0}, {100}, {200}}
	rp := RecurrencePlot(traj, 0.1)

	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			expected := i == j
			if rp[i][j] != expected {
				t.Errorf("recurrence at (%d,%d): got %v, expected %v", i, j, rp[i][j], expected)
			}
		}
	}
}

func TestRecurrence_EmptyTrajectory(t *testing.T) {
	rp := RecurrencePlot(nil, 1.0)
	if rp != nil {
		t.Error("expected nil for empty trajectory")
	}
}

func TestRecurrence_PeriodicOrbit(t *testing.T) {
	// Periodic orbit: [0] -> [1] -> [0] -> [1] -> ...
	// Recurrence at threshold 0.5: (0,2), (0,4), (1,3), (2,4) should be true.
	traj := [][]float64{{0}, {1}, {0}, {1}, {0}}
	rp := RecurrencePlot(traj, 0.5)

	// Same-value pairs should be recurrent.
	if !rp[0][2] || !rp[0][4] || !rp[2][4] {
		t.Error("periodic orbit: even-index points should be recurrent")
	}
	if !rp[1][3] {
		t.Error("periodic orbit: odd-index points should be recurrent")
	}
	// Different-value pairs should not be recurrent.
	if rp[0][1] || rp[1][2] {
		t.Error("periodic orbit: different-value pairs should not be recurrent")
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

func assertClose(t *testing.T, name string, got, expected, tol float64) {
	t.Helper()
	if math.IsNaN(got) && math.IsNaN(expected) {
		return
	}
	if math.Abs(got-expected) > tol {
		t.Errorf("%s: got %v, expected %v (diff %v > tol %v)",
			name, got, expected, math.Abs(got-expected), tol)
	}
}

func makeGrid(rows, cols int) [][]bool {
	grid := make([][]bool, rows)
	for i := range grid {
		grid[i] = make([]bool, cols)
	}
	return grid
}
