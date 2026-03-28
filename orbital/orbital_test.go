package orbital

import (
	"math"
	"testing"

	"github.com/davly/reality/constants"
	"github.com/davly/reality/testutil"
)

// ═══════════════════════════════════════════════════════════════════════════
// Golden-file tests — shared test vectors across Go, Python, C++, C#
// ═══════════════════════════════════════════════════════════════════════════

func TestGolden_KeplerOrbit(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/orbital/kepler_orbit.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			a := testutil.InputFloat64(t, tc, "a")
			e := testutil.InputFloat64(t, tc, "e")
			i := testutil.InputFloat64(t, tc, "i")
			omega := testutil.InputFloat64(t, tc, "omega")
			capOmega := testutil.InputFloat64(t, tc, "capOmega")
			nu := testutil.InputFloat64(t, tc, "nu")
			x, y, z := KeplerOrbit(a, e, i, omega, capOmega, nu)
			testutil.AssertFloat64Slice(t, tc, []float64{x, y, z})
		})
	}
}

func TestGolden_OrbitalPeriod(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/orbital/orbital_period.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			a := testutil.InputFloat64(t, tc, "a")
			mu := testutil.InputFloat64(t, tc, "mu")
			got := OrbitalPeriod(a, mu)
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

func TestGolden_OrbitalVelocity(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/orbital/orbital_velocity.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			mu := testutil.InputFloat64(t, tc, "mu")
			r := testutil.InputFloat64(t, tc, "r")
			a := testutil.InputFloat64(t, tc, "a")
			got := OrbitalVelocity(mu, r, a)
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

func TestGolden_HohmannTransfer(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/orbital/hohmann_transfer.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			r1 := testutil.InputFloat64(t, tc, "r1")
			r2 := testutil.InputFloat64(t, tc, "r2")
			mu := testutil.InputFloat64(t, tc, "mu")
			dv1, dv2 := HohmannTransfer(r1, r2, mu)
			testutil.AssertFloat64Slice(t, tc, []float64{dv1, dv2})
		})
	}
}

func TestGolden_EscapeVelocity(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/orbital/escape_velocity.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			M := testutil.InputFloat64(t, tc, "M")
			r := testutil.InputFloat64(t, tc, "r")
			got := EscapeVelocity(M, r)
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

func TestGolden_HillSphere(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/orbital/hill_sphere.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			a := testutil.InputFloat64(t, tc, "a")
			m := testutil.InputFloat64(t, tc, "m")
			M := testutil.InputFloat64(t, tc, "M")
			got := HillSphere(a, m, M)
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

func TestGolden_SynodicPeriod(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/orbital/synodic_period.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			T1 := testutil.InputFloat64(t, tc, "T1")
			T2 := testutil.InputFloat64(t, tc, "T2")
			got := SynodicPeriod(T1, T2)
			// Handle special +Inf case.
			if str, ok := tc.Expected.(string); ok && str == "Infinity" {
				if !math.IsInf(got, 1) {
					t.Errorf("[%s] got %v, expected +Inf", tc.Description, got)
				}
				return
			}
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

func TestGolden_TrueAnomalyFromMean(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/orbital/true_anomaly.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			M := testutil.InputFloat64(t, tc, "M")
			e := testutil.InputFloat64(t, tc, "e")
			maxIter := testutil.InputInt(t, tc, "maxIter")
			got := TrueAnomalyFromMean(M, e, maxIter)
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — KeplerOrbit
// ═══════════════════════════════════════════════════════════════════════════

func TestKeplerOrbit_CircularEquatorial_RadiusCorrect(t *testing.T) {
	// Any true anomaly on circular equatorial orbit should have r = a.
	for _, nu := range []float64{0, math.Pi / 4, math.Pi / 2, math.Pi, 3 * math.Pi / 2} {
		x, y, z := KeplerOrbit(7e6, 0, 0, 0, 0, nu)
		r := math.Sqrt(x*x + y*y + z*z)
		assertClose(t, "radius", r, 7e6, 1e-6)
	}
}

func TestKeplerOrbit_Periapsis_CorrectRadius(t *testing.T) {
	// At nu=0, r = a(1-e)
	a := 1e7
	e := 0.3
	x, y, z := KeplerOrbit(a, e, 0, 0, 0, 0)
	r := math.Sqrt(x*x + y*y + z*z)
	assertClose(t, "periapsis", r, a*(1-e), 1e-6)
}

func TestKeplerOrbit_Apoapsis_CorrectRadius(t *testing.T) {
	// At nu=pi, r = a(1+e)
	a := 1e7
	e := 0.3
	x, y, z := KeplerOrbit(a, e, 0, 0, 0, math.Pi)
	r := math.Sqrt(x*x + y*y + z*z)
	assertClose(t, "apoapsis", r, a*(1+e), 1e-6)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — OrbitalPeriod
// ═══════════════════════════════════════════════════════════════════════════

func TestOrbitalPeriod_EarthSun(t *testing.T) {
	// Earth around Sun: ~365.25 days = ~3.156e7 seconds.
	muSun := 1.32712440018e20
	aEarth := 1.496e11
	T := OrbitalPeriod(aEarth, muSun)
	// Should be close to one sidereal year.
	assertClose(t, "earth-year", T, 3.1558e7, 1e4)
}

func TestOrbitalPeriod_Proportionality(t *testing.T) {
	// T^2 ∝ a^3 (Kepler's third law).
	mu := 3.986004418e14
	a1 := 7e6
	a2 := 14e6
	T1 := OrbitalPeriod(a1, mu)
	T2 := OrbitalPeriod(a2, mu)
	ratio := (T2 * T2) / (T1 * T1)
	expected := (a2 * a2 * a2) / (a1 * a1 * a1)
	assertClose(t, "kepler-3rd", ratio, expected, 1e-6)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — OrbitalVelocity
// ═══════════════════════════════════════════════════════════════════════════

func TestOrbitalVelocity_Circular(t *testing.T) {
	// For circular orbit r=a, vis-viva gives v = sqrt(mu/a).
	mu := 3.986004418e14
	a := 7e6
	v := OrbitalVelocity(mu, a, a)
	assertClose(t, "circular", v, math.Sqrt(mu/a), 1e-6)
}

func TestOrbitalVelocity_PeriapsisGTApoapsis(t *testing.T) {
	// Velocity at periapsis should be greater than at apoapsis.
	mu := 3.986004418e14
	a := 1e7
	e := 0.5
	rPeri := a * (1 - e)
	rApo := a * (1 + e)
	vPeri := OrbitalVelocity(mu, rPeri, a)
	vApo := OrbitalVelocity(mu, rApo, a)
	if vPeri <= vApo {
		t.Errorf("periapsis velocity %v should exceed apoapsis velocity %v", vPeri, vApo)
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — HohmannTransfer
// ═══════════════════════════════════════════════════════════════════════════

func TestHohmannTransfer_PositiveDeltaV(t *testing.T) {
	dv1, dv2 := HohmannTransfer(7e6, 42e6, 3.986004418e14)
	if dv1 <= 0 {
		t.Errorf("dv1 should be positive, got %v", dv1)
	}
	if dv2 <= 0 {
		t.Errorf("dv2 should be positive, got %v", dv2)
	}
}

func TestHohmannTransfer_SameOrbit_ZeroDeltaV(t *testing.T) {
	dv1, dv2 := HohmannTransfer(7e6, 7e6, 3.986004418e14)
	assertClose(t, "same-dv1", dv1, 0, 1e-10)
	assertClose(t, "same-dv2", dv2, 0, 1e-10)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — EscapeVelocity
// ═══════════════════════════════════════════════════════════════════════════

func TestEscapeVelocity_EarthSurface(t *testing.T) {
	v := EscapeVelocity(5.972e24, 6.371e6)
	// Should be ~11.2 km/s.
	assertClose(t, "earth-escape", v, 11186.0, 10.0)
}

func TestEscapeVelocity_Relation(t *testing.T) {
	// Escape velocity = sqrt(2) * circular orbital velocity.
	M := 5.972e24
	r := 6.371e6
	vEsc := EscapeVelocity(M, r)
	vCirc := math.Sqrt(constants.GravitationalConst * M / r)
	assertClose(t, "sqrt2-relation", vEsc, math.Sqrt2*vCirc, 1e-6)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — HillSphere
// ═══════════════════════════════════════════════════════════════════════════

func TestHillSphere_EarthSun(t *testing.T) {
	rH := HillSphere(1.496e11, 5.972e24, 1.989e30)
	// ~1.5 million km.
	if rH < 1e9 || rH > 2e9 {
		t.Errorf("Earth Hill sphere should be ~1.5e9 m, got %v", rH)
	}
}

func TestHillSphere_Proportionality(t *testing.T) {
	// rH ∝ a, so doubling a doubles rH.
	rH1 := HillSphere(1e11, 1e24, 1e30)
	rH2 := HillSphere(2e11, 1e24, 1e30)
	assertClose(t, "hill-prop", rH2/rH1, 2.0, 1e-10)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — SynodicPeriod
// ═══════════════════════════════════════════════════════════════════════════

func TestSynodicPeriod_Symmetry(t *testing.T) {
	s1 := SynodicPeriod(365.256, 686.980)
	s2 := SynodicPeriod(686.980, 365.256)
	assertClose(t, "symmetry", s1, s2, 1e-10)
}

func TestSynodicPeriod_CoOrbital(t *testing.T) {
	s := SynodicPeriod(100, 100)
	if !math.IsInf(s, 1) {
		t.Errorf("co-orbital should be +Inf, got %v", s)
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — TrueAnomalyFromMean
// ═══════════════════════════════════════════════════════════════════════════

func TestTrueAnomalyFromMean_CircularIdentity(t *testing.T) {
	// For e=0, true anomaly = mean anomaly.
	for _, M := range []float64{0, 0.5, 1.0, math.Pi, 2 * math.Pi - 0.1} {
		nu := TrueAnomalyFromMean(M, 0, 30)
		assertClose(t, "circular-id", nu, M, 1e-12)
	}
}

func TestTrueAnomalyFromMean_AtPiSymmetry(t *testing.T) {
	// At M=pi, E=pi for any e, so nu=pi.
	for _, e := range []float64{0, 0.1, 0.3, 0.5, 0.8} {
		nu := TrueAnomalyFromMean(math.Pi, e, 50)
		assertClose(t, "pi-symmetry", nu, math.Pi, 1e-8)
	}
}

func TestTrueAnomalyFromMean_NegativeMNormalized(t *testing.T) {
	// Negative M should be normalised to [0, 2pi).
	nu := TrueAnomalyFromMean(-math.Pi, 0.1, 30)
	if nu < 0 || nu >= 2*math.Pi {
		t.Errorf("expected nu in [0, 2pi), got %v", nu)
	}
}

func TestTrueAnomalyFromMean_ConvergesForHighE(t *testing.T) {
	// Even high eccentricity (0.95) should converge with enough iterations.
	nu := TrueAnomalyFromMean(1.0, 0.95, 100)
	if math.IsNaN(nu) || nu < 0 || nu >= 2*math.Pi {
		t.Errorf("high-e convergence failed: nu = %v", nu)
	}
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
