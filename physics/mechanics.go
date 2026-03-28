// Package physics provides classical mechanics, thermodynamics, material
// physics, and optics functions. All functions are pure, deterministic, and
// zero-dependency. Physical constants are imported from the constants package.
//
// Functions follow the Reality convention: numbers in, numbers out.
// Output buffers are pre-allocated where applicable. Every function documents
// its formula, valid range, precision, and reference.
package physics

import (
	"math"

	"github.com/davly/reality/constants"
)

// ---------------------------------------------------------------------------
// Classical Mechanics
// ---------------------------------------------------------------------------

// NewtonSecondLaw computes acceleration from force and mass.
//
// Formula: a = F / m
// Valid range: m != 0. If m == 0, returns +Inf or -Inf depending on sign of F
// (0/0 returns NaN per IEEE 754).
// Precision: exact (single division)
// Reference: Newton, I. (1687) "Principia Mathematica", Second Law of Motion
func NewtonSecondLaw(F, m float64) float64 {
	return F / m
}

// ProjectilePosition computes the (x, y) position of a projectile at time t.
//
// Formula:
//
//	x = v0 * cos(theta) * t
//	y = v0 * sin(theta) * t - 0.5 * g * t^2
//
// Parameters:
//   - v0: initial velocity (m/s)
//   - theta: launch angle (radians)
//   - t: elapsed time (s)
//   - g: gravitational acceleration (m/s^2), e.g. constants.StandardGravity
//
// Valid range: v0 >= 0, theta in [0, pi], t >= 0, g > 0
// Precision: limited by float64 trig (~15 significant digits)
// Reference: standard kinematic equations for projectile motion
func ProjectilePosition(v0, theta, t, g float64) (x, y float64) {
	cosTheta := math.Cos(theta)
	sinTheta := math.Sin(theta)
	x = v0 * cosTheta * t
	y = v0*sinTheta*t - 0.5*g*t*t
	return x, y
}

// GravitationalForce computes the gravitational force between two masses.
//
// Formula: F = G * m1 * m2 / r^2
// Valid range: r != 0, m1 > 0, m2 > 0. Returns +Inf if r == 0.
// Precision: limited by uncertainty in G (~2.2e-5 relative)
// Reference: Newton's Law of Universal Gravitation; NIST CODATA 2018 for G
func GravitationalForce(m1, m2, r float64) float64 {
	return constants.GravitationalConst * m1 * m2 / (r * r)
}

// OrbitalVelocity computes the circular orbital velocity around a central mass.
//
// Formula: v = sqrt(G * M / r)
// Valid range: M > 0, r > 0
// Precision: limited by uncertainty in G (~2.2e-5 relative) and float64 sqrt
// Reference: derived from equating gravitational and centripetal force;
// Kepler's third law for circular orbits
func OrbitalVelocity(M, r float64) float64 {
	return math.Sqrt(constants.GravitationalConst * M / r)
}

// SpringForce computes the force on a damped harmonic oscillator.
//
// Formula: F = -k*x - c*v
// Parameters:
//   - k: spring constant (N/m)
//   - x: displacement from equilibrium (m)
//   - c: damping coefficient (N*s/m)
//   - v: velocity (m/s)
//
// Valid range: k >= 0, c >= 0. Negative k/c are allowed but physically unusual.
// Precision: exact (two multiplications and a subtraction)
// Reference: Hooke's Law with viscous damping; see any classical mechanics text
func SpringForce(k, x, c, v float64) float64 {
	return -k*x - c*v
}

// ElasticCollision computes the final velocities of two objects after a
// perfectly elastic 1D collision.
//
// Formula:
//
//	v1f = ((m1 - m2) * v1 + 2 * m2 * v2) / (m1 + m2)
//	v2f = ((m2 - m1) * v2 + 2 * m1 * v1) / (m1 + m2)
//
// Valid range: m1 + m2 != 0. Returns NaN if total mass is zero.
// Precision: exact (arithmetic operations only)
// Reference: conservation of momentum and kinetic energy in elastic collisions
func ElasticCollision(m1, v1, m2, v2 float64) (v1f, v2f float64) {
	totalMass := m1 + m2
	v1f = ((m1-m2)*v1 + 2*m2*v2) / totalMass
	v2f = ((m2-m1)*v2 + 2*m1*v1) / totalMass
	return v1f, v2f
}

// Pendulum computes the angular acceleration of a simple pendulum with
// a velocity-proportional damping approximation.
//
// Formula: α = -(g/L) * sin(theta) - damping * sin(theta)
//
// The damping term uses sin(theta) as a proxy for angular velocity direction,
// providing a position-dependent restoring-plus-drag model. For a full ODE
// simulation with explicit angular velocity, integrate externally and pass
// damping = 0 to get the pure gravitational torque.
//
// Parameters:
//   - theta: angular displacement (radians)
//   - L: pendulum length (m), must be > 0
//   - g: gravitational acceleration (m/s^2)
//   - damping: damping coefficient (1/s^2)
//
// Valid range: L > 0. Returns NaN if L == 0.
// Precision: limited by float64 sin (~15 significant digits)
// Reference: nonlinear pendulum equation; see Thornton & Marion
// "Classical Dynamics of Particles and Systems" ch. 3
func Pendulum(theta, L, g, damping float64) float64 {
	return -(g/L)*math.Sin(theta) - damping*math.Sin(theta)
}

// KineticEnergy computes the translational kinetic energy of an object.
//
// Formula: KE = 0.5 * m * v^2
// Valid range: m >= 0, v any real number
// Precision: exact (multiplication only)
// Reference: classical mechanics, Leibniz "vis viva"
func KineticEnergy(m, v float64) float64 {
	return 0.5 * m * v * v
}

// PotentialEnergy computes the gravitational potential energy relative to
// a reference height of zero.
//
// Formula: PE = m * g * h
// Valid range: m >= 0, g > 0, h any real number (negative = below reference)
// Precision: exact (multiplication only)
// Reference: standard gravitational potential energy in uniform field
func PotentialEnergy(m, g, h float64) float64 {
	return m * g * h
}
