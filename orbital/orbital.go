// Package orbital provides astrodynamics functions for computing orbits,
// transfers, and anomalies. All functions are pure, deterministic, and
// zero-dependency. Physical constants are imported from the constants package.
//
// Functions follow the Reality convention: numbers in, numbers out.
// Every function documents its formula, valid range, precision, and reference.
//
// The gravitational parameter μ = G*M is used throughout rather than separate
// G and M, as μ is typically known to higher precision for celestial bodies.
package orbital

import (
	"math"

	"github.com/davly/reality/constants"
)

// ---------------------------------------------------------------------------
// Position from Keplerian Elements
// ---------------------------------------------------------------------------

// KeplerOrbit computes the 3D Cartesian position (x, y, z) from classical
// Keplerian orbital elements.
//
// Parameters:
//   - a: semi-major axis (m or any consistent unit)
//   - e: eccentricity (dimensionless, 0 <= e < 1 for ellipse)
//   - i: inclination (radians)
//   - omega: argument of periapsis (radians)
//   - capOmega: right ascension of ascending node / RAAN (radians)
//   - nu: true anomaly (radians)
//
// Formula:
//
//	r = a(1 - e²) / (1 + e·cos(ν))
//	x_orb = r·cos(ν)
//	y_orb = r·sin(ν)
//	Rotation via 3-1-3 Euler angles (Ω, i, ω) to inertial frame.
//
// Valid range: e in [0, 1), a > 0, i in [0, pi]
// Precision: limited by float64 trig (~15 significant digits)
// Reference: Bate, Mueller, White "Fundamentals of Astrodynamics" (1971),
// ch. 2.5; Vallado "Fundamentals of Astrodynamics and Applications" 4th ed.
func KeplerOrbit(a, e, i, omega, capOmega, nu float64) (x, y, z float64) {
	// Radius in the orbital plane.
	r := a * (1 - e*e) / (1 + e*math.Cos(nu))

	// Position in the orbital plane.
	xOrb := r * math.Cos(nu)
	yOrb := r * math.Sin(nu)

	// Precompute trig values.
	cosOmega := math.Cos(omega)
	sinOmega := math.Sin(omega)
	cosCapOmega := math.Cos(capOmega)
	sinCapOmega := math.Sin(capOmega)
	cosI := math.Cos(i)
	sinI := math.Sin(i)

	// Rotation matrix elements (3-1-3 Euler: Ω, i, ω).
	x = xOrb*(cosOmega*cosCapOmega-sinOmega*cosI*sinCapOmega) -
		yOrb*(sinOmega*cosCapOmega+cosOmega*cosI*sinCapOmega)
	y = xOrb*(cosOmega*sinCapOmega+sinOmega*cosI*cosCapOmega) -
		yOrb*(sinOmega*sinCapOmega-cosOmega*cosI*cosCapOmega)
	z = xOrb*(sinOmega*sinI) + yOrb*(cosOmega*sinI)

	return x, y, z
}

// ---------------------------------------------------------------------------
// Orbital Period
// ---------------------------------------------------------------------------

// OrbitalPeriod computes the orbital period of a body in an elliptical orbit.
//
// Formula: T = 2π√(a³/μ)
// Parameters:
//   - a: semi-major axis (m)
//   - mu: gravitational parameter μ = G*M (m³/s²)
//
// Valid range: a > 0, mu > 0
// Precision: limited by float64 sqrt and pi (~15 significant digits)
// Reference: Kepler's Third Law; Bate, Mueller, White (1971) eq. 1.6-4
func OrbitalPeriod(a, mu float64) float64 {
	return 2.0 * math.Pi * math.Sqrt(a*a*a/mu)
}

// ---------------------------------------------------------------------------
// Vis-Viva Orbital Velocity
// ---------------------------------------------------------------------------

// OrbitalVelocity computes the orbital velocity at distance r from the central
// body using the vis-viva equation. This generalises circular orbital velocity
// to elliptical orbits.
//
// Formula: v = √(μ(2/r - 1/a))
// Parameters:
//   - mu: gravitational parameter μ = G*M (m³/s²)
//   - r: current distance from central body (m)
//   - a: semi-major axis of the orbit (m)
//
// Valid range: mu > 0, r > 0, a > 0
// Precision: limited by float64 sqrt (~15 significant digits)
// Reference: vis-viva equation; Bate, Mueller, White (1971) eq. 1.6-2
func OrbitalVelocity(mu, r, a float64) float64 {
	return math.Sqrt(mu * (2.0/r - 1.0/a))
}

// ---------------------------------------------------------------------------
// Hohmann Transfer
// ---------------------------------------------------------------------------

// HohmannTransfer computes the two delta-v burns for a Hohmann transfer
// orbit between two circular orbits of radii r1 and r2 (r1 < r2).
//
// Formula:
//
//	a_t = (r1 + r2) / 2
//	dv1 = √(μ/r1) * (√(2r2/(r1+r2)) - 1)
//	dv2 = √(μ/r2) * (1 - √(2r1/(r1+r2)))
//
// Parameters:
//   - r1: radius of inner orbit (m)
//   - r2: radius of outer orbit (m)
//   - mu: gravitational parameter μ = G*M (m³/s²)
//
// Returns:
//   - dv1: delta-v for the first burn (m/s), always >= 0
//   - dv2: delta-v for the second burn (m/s), always >= 0
//
// Valid range: r1 > 0, r2 > r1, mu > 0
// Precision: limited by float64 sqrt (~15 significant digits)
// Reference: Hohmann, W. "Die Erreichbarkeit der Himmelskörper" (1925);
// Bate, Mueller, White (1971) ch. 6.2
func HohmannTransfer(r1, r2, mu float64) (dv1, dv2 float64) {
	v1 := math.Sqrt(mu / r1)
	v2 := math.Sqrt(mu / r2)
	sum := r1 + r2
	dv1 = v1 * (math.Sqrt(2.0*r2/sum) - 1.0)
	dv2 = v2 * (1.0 - math.Sqrt(2.0*r1/sum))
	return dv1, dv2
}

// ---------------------------------------------------------------------------
// Escape Velocity
// ---------------------------------------------------------------------------

// EscapeVelocity computes the minimum velocity to escape a body's gravitational
// field from distance r.
//
// Formula: v = √(2GM/r)
// Parameters:
//   - M: mass of the central body (kg)
//   - r: distance from the body's center (m)
//
// Valid range: M > 0, r > 0
// Precision: limited by uncertainty in G (~2.2e-5 relative) and float64 sqrt
// Reference: conservation of energy; Bate, Mueller, White (1971) eq. 1.6-12
func EscapeVelocity(M, r float64) float64 {
	return math.Sqrt(2.0 * constants.GravitationalConst * M / r)
}

// ---------------------------------------------------------------------------
// Hill Sphere
// ---------------------------------------------------------------------------

// HillSphere computes the approximate radius of the Hill sphere — the region
// within which a smaller body dominates gravitational attraction over the
// larger body it orbits.
//
// Formula: r_H ≈ a * (m / (3M))^(1/3)
// Parameters:
//   - a: semi-major axis of the smaller body's orbit around the larger (m)
//   - m: mass of the smaller body (kg)
//   - M: mass of the larger body (kg)
//
// Valid range: a > 0, m > 0, M > 0, m << M
// Precision: limited by float64 cbrt (~15 significant digits)
// Reference: Hill, G.W. "Researches in the Lunar Theory" (1878);
// Murray & Dermott "Solar System Dynamics" (1999) eq. 3.147
func HillSphere(a, m, M float64) float64 {
	return a * math.Cbrt(m/(3.0*M))
}

// ---------------------------------------------------------------------------
// Synodic Period
// ---------------------------------------------------------------------------

// SynodicPeriod computes the synodic period — the time between successive
// conjunctions of two bodies as seen from the central body.
//
// Formula: 1/T_syn = |1/T1 - 1/T2|
// Parameters:
//   - T1: orbital period of the first body (any consistent time unit)
//   - T2: orbital period of the second body (same unit as T1)
//
// Valid range: T1 > 0, T2 > 0, T1 != T2
// If T1 == T2, returns +Inf (infinite synodic period for co-orbital bodies).
// Precision: exact (arithmetic only)
// Reference: standard celestial mechanics; Meeus "Astronomical Algorithms" (1998) ch. 36
func SynodicPeriod(T1, T2 float64) float64 {
	diff := math.Abs(1.0/T1 - 1.0/T2)
	if diff == 0 {
		return math.Inf(1)
	}
	return 1.0 / diff
}

// ---------------------------------------------------------------------------
// Kepler's Equation Solver
// ---------------------------------------------------------------------------

// TrueAnomalyFromMean converts mean anomaly M to true anomaly ν by solving
// Kepler's equation M = E - e*sin(E) for eccentric anomaly E via Newton's
// method, then converting to true anomaly.
//
// Parameters:
//   - M: mean anomaly (radians)
//   - e: eccentricity (dimensionless, 0 <= e < 1)
//   - maxIter: maximum Newton-Raphson iterations (typically 20-50 suffices)
//
// Formula:
//
//	Newton iteration: E_{n+1} = E_n - (E_n - e·sin(E_n) - M) / (1 - e·cos(E_n))
//	True anomaly: ν = 2·atan2(√(1+e)·sin(E/2), √(1-e)·cos(E/2))
//
// Valid range: 0 <= e < 1, maxIter >= 1
// Precision: typically converges to machine epsilon within 5-10 iterations
// for e < 0.9. High eccentricity (e > 0.99) may need more iterations.
// Reference: Kepler, J. "Astronomia Nova" (1609); Danby "Fundamentals of
// Celestial Mechanics" (1992) ch. 6.6 for Newton's method convergence.
func TrueAnomalyFromMean(M, e float64, maxIter int) float64 {
	// Normalise M to [0, 2π).
	M = math.Mod(M, 2.0*math.Pi)
	if M < 0 {
		M += 2.0 * math.Pi
	}

	// Initial guess: E = M (good for low eccentricity).
	E := M

	// Newton-Raphson iteration to solve M = E - e*sin(E).
	for iter := 0; iter < maxIter; iter++ {
		sinE := math.Sin(E)
		cosE := math.Cos(E)
		dE := (E - e*sinE - M) / (1.0 - e*cosE)
		E -= dE
		if math.Abs(dE) < 1e-15 {
			break
		}
	}

	// Convert eccentric anomaly E to true anomaly ν.
	halfE := E / 2.0
	nu := 2.0 * math.Atan2(
		math.Sqrt(1.0+e)*math.Sin(halfE),
		math.Sqrt(1.0-e)*math.Cos(halfE),
	)

	// Ensure ν is in [0, 2π).
	if nu < 0 {
		nu += 2.0 * math.Pi
	}

	return nu
}
