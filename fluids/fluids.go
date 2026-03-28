// Package fluids provides classical fluid mechanics functions. All functions
// are pure, deterministic, and zero-dependency. Physical constants are imported
// from the constants package where applicable.
//
// Functions follow the Reality convention: numbers in, numbers out.
// Every function documents its formula, valid range, precision, and reference.
package fluids

import "math"

// ---------------------------------------------------------------------------
// Dimensionless Numbers
// ---------------------------------------------------------------------------

// ReynoldsNumber computes the Reynolds number, which characterises the ratio
// of inertial forces to viscous forces in a flowing fluid.
//
// Formula: Re = ρ * v * L / μ
// Parameters:
//   - rho: fluid density (kg/m³)
//   - v:   flow velocity (m/s)
//   - L:   characteristic length (m)
//   - mu:  dynamic viscosity (Pa·s)
//
// Valid range: mu != 0. Returns +Inf/-Inf if mu == 0 (per IEEE 754).
// Precision: exact (multiplication and single division)
// Reference: Reynolds, O. (1883) "An Experimental Investigation of the
// Circumstances Which Determine Whether the Motion of Water Shall Be Direct
// or Sinuous, and of the Law of Resistance in Parallel Channels"
func ReynoldsNumber(rho, v, L, mu float64) float64 {
	return rho * v * L / mu
}

// ---------------------------------------------------------------------------
// Bernoulli & Pipe Flow
// ---------------------------------------------------------------------------

// BernoulliPressure computes the downstream pressure p2 from the steady-state
// incompressible Bernoulli equation along a streamline.
//
// Formula: p2 = p1 + 0.5*ρ*(v1² - v2²) + ρ*g*(h1 - h2)
// Parameters:
//   - rho: fluid density (kg/m³)
//   - v1:  upstream velocity (m/s)
//   - p1:  upstream static pressure (Pa)
//   - h1:  upstream elevation (m)
//   - v2:  downstream velocity (m/s)
//   - h2:  downstream elevation (m)
//   - g:   gravitational acceleration (m/s²)
//
// Valid range: all finite. Negative pressures are physically unusual but
// mathematically permitted.
// Precision: exact (arithmetic only)
// Reference: Bernoulli, D. (1738) "Hydrodynamica"; steady, inviscid,
// incompressible flow along a streamline.
func BernoulliPressure(rho, v1, p1, h1, v2, h2, g float64) float64 {
	return p1 + 0.5*rho*(v1*v1-v2*v2) + rho*g*(h1-h2)
}

// PipeFlowFriction computes the Darcy friction factor for flow in a pipe
// using the Colebrook–White equation (solved iteratively).
//
// For laminar flow (Re < 2300), the exact Hagen–Poiseuille result is used:
//   f = 64 / Re
//
// For turbulent flow (Re >= 2300), the Colebrook–White implicit equation is
// solved by fixed-point iteration (typically converges in < 20 iterations):
//   1/√f = -2 log₁₀(ε/(3.7D) + 2.51/(Re√f))
//
// Parameters:
//   - Re:        Reynolds number (dimensionless), must be > 0
//   - roughness: absolute pipe roughness ε (m)
//   - diameter:  pipe internal diameter D (m)
//
// Valid range: Re > 0, diameter > 0, roughness >= 0.
// Precision: iterative solve to ~1e-10 relative change; Swamee–Jain seed
// Reference: Colebrook, C.F. (1939) "Turbulent Flow in Pipes, with
// Particular Reference to the Transition Region Between the Smooth and
// Rough Pipe Laws"; Moody, L.F. (1944) Moody chart.
func PipeFlowFriction(Re, roughness, diameter float64) float64 {
	if Re <= 0 {
		return math.NaN()
	}
	// Laminar regime
	if Re < 2300 {
		return 64.0 / Re
	}

	// Turbulent: Swamee–Jain initial estimate
	relRough := roughness / diameter
	logArg := relRough/3.7 + 5.74/math.Pow(Re, 0.9)
	f := 0.25 / (math.Log10(logArg) * math.Log10(logArg))

	// Colebrook–White fixed-point iteration
	for i := 0; i < 100; i++ {
		sqrtF := math.Sqrt(f)
		rhs := -2.0 * math.Log10(relRough/3.7+2.51/(Re*sqrtF))
		fNew := 1.0 / (rhs * rhs)
		if math.Abs(fNew-f) < 1e-12 {
			return fNew
		}
		f = fNew
	}
	return f
}

// DarcyWeisbach computes the pressure drop due to friction in a pipe.
//
// Formula: ΔP = f * (L/D) * (ρ*v²/2)
// Parameters:
//   - f:   Darcy friction factor (dimensionless)
//   - L:   pipe length (m)
//   - D:   pipe internal diameter (m)
//   - rho: fluid density (kg/m³)
//   - v:   mean flow velocity (m/s)
//
// Valid range: D != 0. All parameters >= 0 for physical correctness.
// Precision: exact (arithmetic only)
// Reference: Weisbach, J. (1845); Darcy, H. (1857)
func DarcyWeisbach(f, L, D, rho, v float64) float64 {
	return f * (L / D) * (rho * v * v / 2.0)
}

// ---------------------------------------------------------------------------
// Aerodynamic Forces
// ---------------------------------------------------------------------------

// DragForce computes the aerodynamic drag on an object.
//
// Formula: F = 0.5 * Cd * ρ * v² * A
// Parameters:
//   - Cd:  drag coefficient (dimensionless)
//   - rho: fluid density (kg/m³)
//   - v:   velocity relative to fluid (m/s)
//   - A:   reference area (m²)
//
// Valid range: all >= 0 for physical correctness. Negative v squares positive.
// Precision: exact (multiplication only)
// Reference: Rayleigh, Lord (1876) drag equation; standard aerodynamics texts
func DragForce(Cd, rho, v, A float64) float64 {
	return 0.5 * Cd * rho * v * v * A
}

// LiftForce computes the aerodynamic lift on an object.
//
// Formula: F = 0.5 * Cl * ρ * v² * A
// Parameters:
//   - Cl:  lift coefficient (dimensionless)
//   - rho: fluid density (kg/m³)
//   - v:   velocity relative to fluid (m/s)
//   - A:   reference area (m², typically wing planform area)
//
// Valid range: all >= 0 for physical correctness. Negative Cl gives downforce.
// Precision: exact (multiplication only)
// Reference: Kutta–Joukowski theorem; standard aerodynamics texts
func LiftForce(Cl, rho, v, A float64) float64 {
	return 0.5 * Cl * rho * v * v * A
}

// TerminalVelocity computes the terminal (steady-state) falling speed where
// gravitational force equals aerodynamic drag.
//
// Formula: v_t = √(2mg / (Cd * ρ * A))
// Parameters:
//   - m:   mass (kg)
//   - g:   gravitational acceleration (m/s²)
//   - Cd:  drag coefficient (dimensionless)
//   - rho: fluid density (kg/m³)
//   - A:   reference area (m²)
//
// Valid range: Cd*ρ*A > 0, m >= 0, g >= 0. Returns NaN if denominator <= 0.
// Precision: exact (single sqrt)
// Reference: derived from drag equation = weight; see Batchelor "An
// Introduction to Fluid Dynamics" (1967)
func TerminalVelocity(m, g, Cd, rho, A float64) float64 {
	denom := Cd * rho * A
	if denom <= 0 {
		return math.NaN()
	}
	return math.Sqrt(2.0 * m * g / denom)
}

// ---------------------------------------------------------------------------
// Low Reynolds-Number Flow
// ---------------------------------------------------------------------------

// StokesLaw computes the viscous drag force on a small sphere moving
// through a fluid at low Reynolds number (creeping flow, Re << 1).
//
// Formula: F = 6π * μ * r * v
// Parameters:
//   - mu: dynamic viscosity (Pa·s)
//   - r:  sphere radius (m)
//   - v:  velocity (m/s)
//
// Valid range: all >= 0 for physical correctness. Strictly valid for Re << 1.
// Precision: limited by float64 representation of π (~15 significant digits)
// Reference: Stokes, G.G. (1851) "On the Effect of the Internal Friction of
// Fluids on the Motion of Pendulums"
func StokesLaw(mu, r, v float64) float64 {
	return 6.0 * math.Pi * mu * r * v
}

// ---------------------------------------------------------------------------
// Flow Rates
// ---------------------------------------------------------------------------

// MassFlowRate computes the mass flow rate through a cross-section.
//
// Formula: ṁ = ρ * v * A
// Parameters:
//   - rho: fluid density (kg/m³)
//   - v:   mean flow velocity (m/s)
//   - A:   cross-sectional area (m²)
//
// Valid range: all >= 0 for physical correctness.
// Precision: exact (multiplication only)
// Reference: continuity equation for incompressible flow
func MassFlowRate(rho, v, A float64) float64 {
	return rho * v * A
}

// VolumetricFlowRate computes the volume flow rate through a cross-section.
//
// Formula: Q = v * A
// Parameters:
//   - v: mean flow velocity (m/s)
//   - A: cross-sectional area (m²)
//
// Valid range: all >= 0 for physical correctness.
// Precision: exact (single multiplication)
// Reference: continuity equation for incompressible flow
func VolumetricFlowRate(v, A float64) float64 {
	return v * A
}
