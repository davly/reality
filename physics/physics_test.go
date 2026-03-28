package physics

import (
	"math"
	"testing"

	"github.com/davly/reality/constants"
	"github.com/davly/reality/testutil"
)

// ═══════════════════════════════════════════════════════════════════════════
// Golden-file tests — shared test vectors across Go, Python, C++, C#
// ═══════════════════════════════════════════════════════════════════════════

func TestGolden_ProjectilePosition(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/physics/projectile.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			v0 := testutil.InputFloat64(t, tc, "v0")
			theta := testutil.InputFloat64(t, tc, "theta")
			tt := testutil.InputFloat64(t, tc, "t")
			g := testutil.InputFloat64(t, tc, "g")
			x, y := ProjectilePosition(v0, theta, tt, g)
			testutil.AssertFloat64Slice(t, tc, []float64{x, y})
		})
	}
}

func TestGolden_HookesLaw(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/physics/hookes_law.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			E := testutil.InputFloat64(t, tc, "E")
			epsilon := testutil.InputFloat64(t, tc, "epsilon")
			got := HookesLaw(E, epsilon)
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

func TestGolden_VonMisesStress(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/physics/von_mises_stress.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			s1 := testutil.InputFloat64(t, tc, "s1")
			s2 := testutil.InputFloat64(t, tc, "s2")
			s3 := testutil.InputFloat64(t, tc, "s3")
			got := VonMisesStress(s1, s2, s3)
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — Classical Mechanics
// ═══════════════════════════════════════════════════════════════════════════

func TestNewtonSecondLaw_Basic(t *testing.T) {
	// F = 10 N, m = 2 kg -> a = 5 m/s^2
	assertClose(t, "F=10,m=2", NewtonSecondLaw(10, 2), 5.0, 1e-15)
}

func TestNewtonSecondLaw_NegativeForce(t *testing.T) {
	assertClose(t, "F=-20,m=4", NewtonSecondLaw(-20, 4), -5.0, 1e-15)
}

func TestNewtonSecondLaw_ZeroForce(t *testing.T) {
	assertClose(t, "F=0,m=5", NewtonSecondLaw(0, 5), 0.0, 0)
}

func TestNewtonSecondLaw_ZeroMass(t *testing.T) {
	got := NewtonSecondLaw(10, 0)
	if !math.IsInf(got, 1) {
		t.Errorf("expected +Inf, got %v", got)
	}
}

func TestProjectilePosition_Origin(t *testing.T) {
	x, y := ProjectilePosition(10, math.Pi/4, 0, constants.StandardGravity)
	assertClose(t, "proj-origin-x", x, 0.0, 1e-15)
	assertClose(t, "proj-origin-y", y, 0.0, 1e-15)
}

func TestProjectilePosition_45Degrees_MaxRange(t *testing.T) {
	// 45° maximizes range for given v0 and g on flat ground.
	// Time of flight = 2*v0*sin(45)/g
	v0 := 100.0
	g := constants.StandardGravity
	tFlight := 2 * v0 * math.Sin(math.Pi/4) / g
	x, y := ProjectilePosition(v0, math.Pi/4, tFlight, g)
	// y should be ~0 at landing
	assertClose(t, "proj-land-y", y, 0.0, 1e-8)
	// Range at 45° = v0^2 * sin(2*45) / g = v0^2/g
	expectedRange := v0 * v0 / g
	assertClose(t, "proj-range-45", x, expectedRange, 1e-8)
}

func TestGravitationalForce_EarthSun(t *testing.T) {
	// Earth mass ~ 5.972e24 kg, Sun mass ~ 1.989e30 kg, distance ~ 1.496e11 m
	mEarth := 5.972e24
	mSun := 1.989e30
	r := 1.496e11
	F := GravitationalForce(mEarth, mSun, r)
	// Expected ~ 3.54e22 N
	if F < 3.5e22 || F > 3.6e22 {
		t.Errorf("earth-sun force out of range: got %e", F)
	}
}

func TestGravitationalForce_TwoUnitMasses(t *testing.T) {
	// m1=m2=1 kg, r=1 m -> F = G
	F := GravitationalForce(1, 1, 1)
	assertClose(t, "F=G", F, constants.GravitationalConst, 1e-25)
}

func TestOrbitalVelocity_EarthSurface(t *testing.T) {
	// Orbital velocity at Earth surface: sqrt(G*M_earth/R_earth)
	// M_earth ~ 5.972e24 kg, R_earth ~ 6.371e6 m
	// Expected ~ 7910 m/s
	v := OrbitalVelocity(5.972e24, 6.371e6)
	if v < 7900 || v > 7920 {
		t.Errorf("orbital velocity out of range: got %v", v)
	}
}

func TestOrbitalVelocity_UnitValues(t *testing.T) {
	// v = sqrt(G*1/1) = sqrt(G)
	v := OrbitalVelocity(1, 1)
	assertClose(t, "orbit-unit", v, math.Sqrt(constants.GravitationalConst), 1e-15)
}

func TestSpringForce_Undamped(t *testing.T) {
	// k=10, x=2, c=0, v=0 -> F = -20
	assertClose(t, "spring-undamped", SpringForce(10, 2, 0, 0), -20.0, 1e-15)
}

func TestSpringForce_Damped(t *testing.T) {
	// k=10, x=2, c=5, v=3 -> F = -10*2 - 5*3 = -35
	assertClose(t, "spring-damped", SpringForce(10, 2, 5, 3), -35.0, 1e-15)
}

func TestSpringForce_Equilibrium(t *testing.T) {
	assertClose(t, "spring-eq", SpringForce(100, 0, 50, 0), 0.0, 0)
}

func TestElasticCollision_EqualMasses(t *testing.T) {
	// Equal masses: velocities swap.
	v1f, v2f := ElasticCollision(1, 5, 1, 0)
	assertClose(t, "elastic-eq-v1f", v1f, 0.0, 1e-15)
	assertClose(t, "elastic-eq-v2f", v2f, 5.0, 1e-15)
}

func TestElasticCollision_WallCollision(t *testing.T) {
	// m2 >> m1: ball bounces back. m1=1, m2=1e10
	v1f, _ := ElasticCollision(1, 10, 1e10, 0)
	assertClose(t, "elastic-wall", v1f, -10.0, 1e-4)
}

func TestElasticCollision_MomentumConservation(t *testing.T) {
	m1, v1, m2, v2 := 3.0, 4.0, 5.0, -2.0
	v1f, v2f := ElasticCollision(m1, v1, m2, v2)
	pBefore := m1*v1 + m2*v2
	pAfter := m1*v1f + m2*v2f
	assertClose(t, "elastic-momentum", pAfter, pBefore, 1e-12)
}

func TestElasticCollision_EnergyConservation(t *testing.T) {
	m1, v1, m2, v2 := 3.0, 4.0, 5.0, -2.0
	v1f, v2f := ElasticCollision(m1, v1, m2, v2)
	keBefore := 0.5*m1*v1*v1 + 0.5*m2*v2*v2
	keAfter := 0.5*m1*v1f*v1f + 0.5*m2*v2f*v2f
	assertClose(t, "elastic-energy", keAfter, keBefore, 1e-12)
}

func TestPendulum_SmallAngle(t *testing.T) {
	// For small theta, sin(theta) ~ theta.
	// alpha ~ -(g/L)*theta for small theta.
	theta := 0.01 // small
	L := 1.0
	g := constants.StandardGravity
	alpha := Pendulum(theta, L, g, 0)
	expected := -(g / L) * math.Sin(theta)
	assertClose(t, "pendulum-small", alpha, expected, 1e-12)
}

func TestPendulum_VerticalDown(t *testing.T) {
	// theta=0 -> no acceleration (equilibrium)
	alpha := Pendulum(0, 1.0, constants.StandardGravity, 0)
	assertClose(t, "pendulum-eq", alpha, 0.0, 1e-15)
}

func TestPendulum_WithDamping(t *testing.T) {
	theta := math.Pi / 6.0
	L := 2.0
	g := constants.StandardGravity
	damping := 0.5
	alpha := Pendulum(theta, L, g, damping)
	expected := -(g/L)*math.Sin(theta) - damping*math.Sin(theta)
	assertClose(t, "pendulum-damped", alpha, expected, 1e-12)
}

func TestKineticEnergy_Known(t *testing.T) {
	assertClose(t, "KE(2,3)", KineticEnergy(2, 3), 9.0, 1e-15)
}

func TestKineticEnergy_Zero(t *testing.T) {
	assertClose(t, "KE(5,0)", KineticEnergy(5, 0), 0.0, 0)
}

func TestKineticEnergy_NegativeVelocity(t *testing.T) {
	// KE is always positive regardless of velocity sign.
	assertClose(t, "KE(2,-3)", KineticEnergy(2, -3), 9.0, 1e-15)
}

func TestPotentialEnergy_Known(t *testing.T) {
	assertClose(t, "PE(10,g,5)", PotentialEnergy(10, constants.StandardGravity, 5), 490.3325, 1e-10)
}

func TestPotentialEnergy_Zero(t *testing.T) {
	assertClose(t, "PE(10,g,0)", PotentialEnergy(10, constants.StandardGravity, 0), 0.0, 0)
}

func TestPotentialEnergy_NegativeHeight(t *testing.T) {
	pe := PotentialEnergy(10, constants.StandardGravity, -5)
	if pe >= 0 {
		t.Errorf("expected negative PE for negative height, got %v", pe)
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — Thermodynamics
// ═══════════════════════════════════════════════════════════════════════════

func TestIdealGas_STP(t *testing.T) {
	// 1 mol of ideal gas at STP: T=273.15 K, V=22.414e-3 m^3
	// P should be ~101325 Pa (1 atm)
	P := IdealGas(1.0, 273.15, 22.414e-3)
	assertClose(t, "ideal-gas-STP", P, constants.AtmPressure, 50) // within 50 Pa
}

func TestIdealGas_DoubleTemperature(t *testing.T) {
	// Doubling T at constant n,V doubles P.
	P1 := IdealGas(1, 300, 0.01)
	P2 := IdealGas(1, 600, 0.01)
	assertClose(t, "ideal-gas-2T", P2, 2*P1, 1e-6)
}

func TestIdealGas_HalfVolume(t *testing.T) {
	// Halving V at constant n,T doubles P.
	P1 := IdealGas(1, 300, 0.02)
	P2 := IdealGas(1, 300, 0.01)
	assertClose(t, "ideal-gas-halfV", P2, 2*P1, 1e-6)
}

func TestStefanBoltzmann_SunSurface(t *testing.T) {
	// Sun surface: T~5778K, 1 m^2, emissivity~1
	// Expected ~ 6.32e7 W/m^2 (luminous flux)
	P := StefanBoltzmann(5778, 1.0, 1.0)
	if P < 6.3e7 || P > 6.4e7 {
		t.Errorf("sun surface power out of range: got %e", P)
	}
}

func TestStefanBoltzmann_ZeroTemp(t *testing.T) {
	assertClose(t, "SB-zero", StefanBoltzmann(0, 1, 1), 0.0, 0)
}

func TestStefanBoltzmann_HalfEmissivity(t *testing.T) {
	P1 := StefanBoltzmann(1000, 1, 1.0)
	P2 := StefanBoltzmann(1000, 1, 0.5)
	assertClose(t, "SB-half-emissivity", P2, P1/2, 1e-6)
}

func TestCarnotEfficiency_MaxEfficiency(t *testing.T) {
	// Tc = 0 -> eta = 1 (theoretical maximum)
	assertClose(t, "carnot-max", CarnotEfficiency(500, 0), 1.0, 1e-15)
}

func TestCarnotEfficiency_SameTemperature(t *testing.T) {
	// Tc = Th -> eta = 0 (no work possible)
	assertClose(t, "carnot-zero", CarnotEfficiency(300, 300), 0.0, 1e-15)
}

func TestCarnotEfficiency_ThermalPlant(t *testing.T) {
	// Th = 600K, Tc = 300K -> eta = 0.5
	assertClose(t, "carnot-plant", CarnotEfficiency(600, 300), 0.5, 1e-15)
}

func TestCarnotEfficiency_ZeroHotReservoir(t *testing.T) {
	assertClose(t, "carnot-Th=0", CarnotEfficiency(0, 300), 0.0, 0)
}

func TestHeatEquation1DStep_Convergence(t *testing.T) {
	// Start with a spike in the middle, run several steps.
	// Temperature should diffuse toward uniform.
	n := 11
	u := make([]float64, n)
	u[5] = 100.0 // spike at center

	out := make([]float64, n)
	alpha := 1.0
	dx := 1.0
	dt := 0.1 // alpha*dt/dx^2 = 0.1 (stable)

	// Run 1000 steps.
	for step := 0; step < 1000; step++ {
		HeatEquation1DStep(u, dt, dx, alpha, out)
		copy(u, out)
	}

	// After many steps, interior should approach 0 (Dirichlet BC at 0).
	for i := 1; i < n-1; i++ {
		if math.Abs(u[i]) > 0.01 {
			t.Errorf("heat eq not converging: u[%d] = %v after 1000 steps", i, u[i])
		}
	}
}

func TestHeatEquation1DStep_BoundaryPreserved(t *testing.T) {
	u := []float64{100.0, 50.0, 0.0}
	out := make([]float64, 3)
	HeatEquation1DStep(u, 0.1, 1.0, 1.0, out)
	assertClose(t, "heat-bc-left", out[0], 100.0, 0)
	assertClose(t, "heat-bc-right", out[2], 0.0, 0)
}

func TestHeatEquation1DStep_UniformStays(t *testing.T) {
	// Uniform temperature should not change.
	u := []float64{50.0, 50.0, 50.0, 50.0, 50.0}
	out := make([]float64, 5)
	HeatEquation1DStep(u, 0.1, 1.0, 1.0, out)
	for i := range out {
		assertClose(t, "heat-uniform", out[i], 50.0, 1e-15)
	}
}

func TestFourierHeatConduction_Known(t *testing.T) {
	// k=50 W/(m*K), A=2 m^2, dTdx=10 K/m -> Q = -50*2*10 = -1000 W
	assertClose(t, "fourier", FourierHeatConduction(50, 2, 10), -1000.0, 1e-15)
}

func TestFourierHeatConduction_NoGradient(t *testing.T) {
	assertClose(t, "fourier-no-grad", FourierHeatConduction(50, 2, 0), 0.0, 0)
}

func TestNewtonCooling_Known(t *testing.T) {
	// h=25, A=0.5, Ts=100, Tinf=25 -> Q = 25*0.5*75 = 937.5
	assertClose(t, "newton-cooling", NewtonCooling(25, 0.5, 100, 25), 937.5, 1e-15)
}

func TestNewtonCooling_SameTemp(t *testing.T) {
	assertClose(t, "cooling-same", NewtonCooling(25, 0.5, 50, 50), 0.0, 0)
}

func TestThermalExpansion_Steel(t *testing.T) {
	// Steel: alpha ~ 12e-6 /K, L0=1m, deltaT=100K
	// deltaL = 1 * 12e-6 * 100 = 1.2e-3 m
	assertClose(t, "expansion-steel", ThermalExpansion(1.0, 12e-6, 100), 1.2e-3, 1e-15)
}

func TestThermalExpansion_NoChange(t *testing.T) {
	assertClose(t, "expansion-zero", ThermalExpansion(1.0, 12e-6, 0), 0.0, 0)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — Materials
// ═══════════════════════════════════════════════════════════════════════════

func TestHookesLaw_Linearity(t *testing.T) {
	E := 200e9
	s1 := HookesLaw(E, 0.001)
	s2 := HookesLaw(E, 0.002)
	assertClose(t, "hooke-linear", s2, 2*s1, 1e-6)
}

func TestVonMisesStress_Uniaxial(t *testing.T) {
	// Uniaxial: sigma_vm = |sigma|
	vm := VonMisesStress(250, 0, 0)
	assertClose(t, "vm-uniaxial", vm, 250.0, 1e-12)
}

func TestVonMisesStress_Hydrostatic(t *testing.T) {
	// Hydrostatic: all equal -> von Mises = 0
	assertClose(t, "vm-hydro", VonMisesStress(500, 500, 500), 0.0, 1e-12)
}

func TestTrescaStress_Uniaxial(t *testing.T) {
	// tau_max = (100 - 0) / 2 = 50
	assertClose(t, "tresca-uni", TrescaStress(100, 0, 0), 50.0, 1e-15)
}

func TestTrescaStress_PureShear(t *testing.T) {
	// (100, -100, 0) -> tau = (100 - (-100))/2 = 100
	assertClose(t, "tresca-shear", TrescaStress(100, -100, 0), 100.0, 1e-15)
}

func TestTrescaStress_Hydrostatic(t *testing.T) {
	assertClose(t, "tresca-hydro", TrescaStress(100, 100, 100), 0.0, 1e-15)
}

func TestStressIntensityFactor_InfinitePlate(t *testing.T) {
	// Y=1.0, sigma=100 MPa, a=0.01 m -> K_I = 100e6 * sqrt(pi*0.01)
	sigma := 100e6
	a := 0.01
	K := StressIntensityFactor(sigma, a, 1.0)
	expected := sigma * math.Sqrt(math.Pi*a)
	assertClose(t, "SIF-inf-plate", K, expected, 1e-6)
}

func TestGriffithCriterion_Known(t *testing.T) {
	// E=200e9, gamma=1.0 J/m^2, a=0.001 m
	sc := GriffithCriterion(200e9, 1.0, 0.001)
	expected := math.Sqrt(2 * 200e9 * 1.0 / (math.Pi * 0.001))
	assertClose(t, "griffith", sc, expected, 1e-2)
}

func TestParisLaw_Known(t *testing.T) {
	// C=1e-11, m=3, deltaK=20e6 -> da/dN = 1e-11 * (20e6)^3
	rate := ParisLaw(1e-11, 3, 20e6)
	expected := 1e-11 * math.Pow(20e6, 3)
	assertClose(t, "paris", rate, expected, 1e-6)
}

func TestCoffinManson_Known(t *testing.T) {
	// epsilonF=0.5, c=-0.6, Nf=1000
	strain := CoffinManson(0.5, -0.6, 1000)
	expected := 0.5 * math.Pow(2000, -0.6)
	assertClose(t, "coffin-manson", strain, expected, 1e-12)
}

func TestCreepArrhenius_Known(t *testing.T) {
	// A=1e-10, Q=200e3, R=GasConstant, T=800, sigma=100e6, n=4
	rate := CreepArrhenius(1e-10, 200e3, constants.GasConstant, 800, 100e6, 4)
	expected := 1e-10 * math.Pow(100e6, 4) * math.Exp(-200e3/(constants.GasConstant*800))
	assertClose(t, "creep", rate, expected, 1e-20)
}

func TestCompositeMixture_AllFiber(t *testing.T) {
	// Vf=1.0 -> E_c = Ef
	assertClose(t, "composite-all-fiber", CompositeMixture(1.0, 230e9, 3e9), 230e9, 1e-6)
}

func TestCompositeMixture_AllMatrix(t *testing.T) {
	// Vf=0.0 -> E_c = Em
	assertClose(t, "composite-all-matrix", CompositeMixture(0.0, 230e9, 3e9), 3e9, 1e-6)
}

func TestCompositeMixture_Half(t *testing.T) {
	// Vf=0.5 -> E_c = 0.5*(230e9 + 3e9) = 116.5e9
	assertClose(t, "composite-half", CompositeMixture(0.5, 230e9, 3e9), 116.5e9, 1e-6)
}

func TestEulerBuckling_PinnedPinned(t *testing.T) {
	// E=200e9, I=1e-6, L=1, K=1 -> P_cr = pi^2 * 200e9 * 1e-6 / 1
	Pcr := EulerBuckling(200e9, 1e-6, 1, 1)
	expected := math.Pi * math.Pi * 200e9 * 1e-6
	assertClose(t, "euler-pinned", Pcr, expected, 1e-2)
}

func TestEulerBuckling_FixedFixed(t *testing.T) {
	// K=0.5 -> P_cr is 4x pinned-pinned
	Pcr_pp := EulerBuckling(200e9, 1e-6, 1, 1)
	Pcr_ff := EulerBuckling(200e9, 1e-6, 1, 0.5)
	assertClose(t, "euler-fixed", Pcr_ff, 4*Pcr_pp, 1e-2)
}

func TestBeamDeflection_Known(t *testing.T) {
	// P=1000 N, L=2 m, E=200e9, I=1e-6
	d := BeamDeflection(1000, 2, 200e9, 1e-6)
	expected := 1000.0 * 8.0 / (48 * 200e9 * 1e-6)
	assertClose(t, "beam-defl", d, expected, 1e-10)
}

func TestBeamDeflection_ZeroLoad(t *testing.T) {
	assertClose(t, "beam-zero", BeamDeflection(0, 2, 200e9, 1e-6), 0.0, 0)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — Optics
// ═══════════════════════════════════════════════════════════════════════════

func TestSnellRefraction_NormalIncidence(t *testing.T) {
	// theta_I = 0 -> theta_R = 0 (no refraction at normal incidence)
	assertClose(t, "snell-normal", SnellRefraction(1.0, 1.5, 0), 0.0, 1e-15)
}

func TestSnellRefraction_AirToGlass(t *testing.T) {
	// n1=1 (air), n2=1.5 (glass), theta_I=30° (pi/6)
	thetaR := SnellRefraction(1.0, 1.5, math.Pi/6)
	// sin(theta_R) = (1/1.5)*sin(pi/6) = (1/1.5)*0.5 = 1/3
	expected := math.Asin(1.0 / 3.0)
	assertClose(t, "snell-air-glass-30", thetaR, expected, 1e-15)
}

func TestSnellRefraction_TotalInternalReflection(t *testing.T) {
	// Glass to air at angle beyond critical angle.
	// Critical angle for n1=1.5 -> n2=1: theta_c = asin(1/1.5) ~ 41.81°
	// Use theta_I = 45° (> critical angle)
	thetaR := SnellRefraction(1.5, 1.0, math.Pi/4)
	if !math.IsNaN(thetaR) {
		t.Errorf("expected NaN for TIR, got %v", thetaR)
	}
}

func TestSnellRefraction_CriticalAngle(t *testing.T) {
	// At exactly the critical angle, theta_R = pi/2 (90°).
	critAngle := math.Asin(1.0 / 1.5)
	thetaR := SnellRefraction(1.5, 1.0, critAngle)
	assertClose(t, "snell-critical", thetaR, math.Pi/2, 1e-10)
}

func TestSnellRefraction_SameMedium(t *testing.T) {
	// Same refractive index: theta_R = theta_I
	theta := 0.5
	assertClose(t, "snell-same", SnellRefraction(1.5, 1.5, theta), theta, 1e-15)
}

func TestFresnelReflectance_NormalIncidence(t *testing.T) {
	// R = ((n1-n2)/(n1+n2))^2 for normal incidence
	R := FresnelReflectance(1.0, 1.5, 0)
	expected := 0.04 // ((1-1.5)/(1+1.5))^2 = 0.04
	assertClose(t, "fresnel-normal", R, expected, 1e-12)
}

func TestFresnelReflectance_SameMedium(t *testing.T) {
	// Same n: no reflection
	R := FresnelReflectance(1.5, 1.5, 0)
	assertClose(t, "fresnel-same", R, 0.0, 1e-15)
}

func TestFresnelReflectance_TotalInternalReflection(t *testing.T) {
	// Beyond critical angle: R = 1.0
	R := FresnelReflectance(1.5, 1.0, math.Pi/4)
	assertClose(t, "fresnel-TIR", R, 1.0, 1e-15)
}

func TestFresnelReflectance_GlazingAngle(t *testing.T) {
	// Near pi/2: reflectance approaches 1.0
	R := FresnelReflectance(1.0, 1.5, 1.5) // ~86°
	if R < 0.5 {
		t.Errorf("expected high reflectance at grazing angle, got %v", R)
	}
}

func TestBeerLambertLaw_NoAbsorption(t *testing.T) {
	assertClose(t, "beer-none", BeerLambertLaw(100, 0, 10), 100.0, 1e-15)
}

func TestBeerLambertLaw_Known(t *testing.T) {
	// I0=100, mu=0.5, x=2 -> I = 100*exp(-1) ~ 36.788
	I := BeerLambertLaw(100, 0.5, 2)
	expected := 100.0 * math.Exp(-1.0)
	assertClose(t, "beer-known", I, expected, 1e-12)
}

func TestBeerLambertLaw_ZeroThickness(t *testing.T) {
	assertClose(t, "beer-zero-x", BeerLambertLaw(500, 2.0, 0), 500.0, 1e-15)
}

func TestBeerLambertLaw_HighAbsorption(t *testing.T) {
	// Very thick medium: intensity drops to near zero.
	I := BeerLambertLaw(1000, 1.0, 100)
	if I > 1e-30 {
		t.Errorf("expected near-zero intensity, got %v", I)
	}
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
