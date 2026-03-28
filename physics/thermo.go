package physics

import (
	"github.com/davly/reality/constants"
)

// ---------------------------------------------------------------------------
// Thermodynamics
// ---------------------------------------------------------------------------

// IdealGas computes the pressure of an ideal gas from the ideal gas law.
//
// Formula: P = n * R * T / V
// Parameters:
//   - n: amount of substance (mol)
//   - T: absolute temperature (K)
//   - V: volume (m^3)
//
// Valid range: V != 0, n >= 0, T >= 0. Returns +Inf if V == 0.
// Precision: limited by float64 representation of R (~15 significant digits)
// Reference: Ideal Gas Law; Clapeyron (1834); uses NIST value for R = N_A * k_B
func IdealGas(n, T, V float64) float64 {
	return n * constants.GasConstant * T / V
}

// StefanBoltzmann computes the total radiant power emitted by a blackbody
// surface (or grey body with emissivity < 1).
//
// Formula: P = emissivity * sigma * A * T^4
// Parameters:
//   - T: absolute temperature (K)
//   - A: surface area (m^2)
//   - emissivity: emissivity coefficient in [0, 1]
//
// Valid range: T >= 0, A >= 0, emissivity in [0, 1]
// Precision: limited by float64 pow and sigma constant
// Reference: Stefan-Boltzmann Law; Stefan (1879), Boltzmann (1884)
func StefanBoltzmann(T, A, emissivity float64) float64 {
	T2 := T * T
	return emissivity * constants.StefanBoltzmann * A * T2 * T2
}

// CarnotEfficiency computes the maximum theoretical efficiency of a heat
// engine operating between two thermal reservoirs.
//
// Formula: eta = 1 - Tc / Th
// Parameters:
//   - Th: hot reservoir temperature (K), must be > 0
//   - Tc: cold reservoir temperature (K), must be >= 0
//
// Valid range: Th > 0, Tc >= 0, Tc <= Th. Returns 0 if Th <= 0.
// If Tc > Th, the result is negative (physically meaningless but mathematically correct).
// Precision: exact (single division and subtraction)
// Reference: Carnot, S. (1824) "Reflections on the Motive Power of Fire"
func CarnotEfficiency(Th, Tc float64) float64 {
	if Th <= 0 {
		return 0
	}
	return 1.0 - Tc/Th
}

// HeatEquation1DStep performs one explicit forward Euler time step of the
// 1D heat equation using a finite-difference method.
//
// Formula: u_new[i] = u[i] + alpha * dt / dx^2 * (u[i-1] - 2*u[i] + u[i+1])
//
// Boundary conditions: Dirichlet (fixed). out[0] = u[0] and out[n-1] = u[n-1]
// are preserved from the input.
//
// Parameters:
//   - u: current temperature distribution (length n >= 3)
//   - dt: time step (s)
//   - dx: spatial step (m)
//   - alpha: thermal diffusivity (m^2/s)
//   - out: pre-allocated output buffer (length >= len(u))
//
// Stability constraint: alpha * dt / dx^2 <= 0.5 (not enforced; caller's responsibility)
// Valid range: len(u) >= 3, dx != 0, dt >= 0, alpha >= 0
// Precision: O(dt) in time, O(dx^2) in space (first-order explicit scheme)
// Reference: Fourier, J. (1822) "Théorie analytique de la chaleur";
// finite difference method, see LeVeque "Finite Difference Methods for ODEs and PDEs"
func HeatEquation1DStep(u []float64, dt, dx, alpha float64, out []float64) {
	n := len(u)
	if n < 3 {
		return
	}

	r := alpha * dt / (dx * dx)

	// Dirichlet boundary conditions: endpoints stay fixed.
	out[0] = u[0]
	out[n-1] = u[n-1]

	// Interior points: explicit finite difference.
	for i := 1; i < n-1; i++ {
		out[i] = u[i] + r*(u[i-1]-2*u[i]+u[i+1])
	}
}

// FourierHeatConduction computes the rate of heat transfer through a material
// by conduction (Fourier's law).
//
// Formula: Q = -k * A * dTdx
// Parameters:
//   - k: thermal conductivity (W/(m*K))
//   - A: cross-sectional area perpendicular to heat flow (m^2)
//   - dTdx: temperature gradient (K/m), positive in direction of increasing T
//
// Valid range: k >= 0, A >= 0, dTdx any real number
// Precision: exact (multiplication only)
// Reference: Fourier, J. (1822) "Théorie analytique de la chaleur"
func FourierHeatConduction(k, A, dTdx float64) float64 {
	return -k * A * dTdx
}

// NewtonCooling computes the convective heat transfer rate using Newton's
// law of cooling.
//
// Formula: Q = h * A * (Ts - Tinf)
// Parameters:
//   - h: convective heat transfer coefficient (W/(m^2*K))
//   - A: surface area (m^2)
//   - Ts: surface temperature (K or C — units must be consistent)
//   - Tinf: ambient (fluid) temperature (K or C)
//
// Valid range: h >= 0, A >= 0, Ts and Tinf any real number
// Precision: exact (multiplication only)
// Reference: Newton, I. (1701) "Scala graduum Caloris"
func NewtonCooling(h, A, Ts, Tinf float64) float64 {
	return h * A * (Ts - Tinf)
}

// ThermalExpansion computes the change in length due to thermal expansion.
//
// Formula: deltaL = L0 * alpha * deltaT
// Parameters:
//   - L0: original length (m)
//   - alpha: coefficient of linear thermal expansion (1/K)
//   - deltaT: temperature change (K or C — delta is the same in both)
//
// Valid range: L0 >= 0, alpha > 0, deltaT any real number
// Precision: exact (multiplication only)
// Reference: standard linear thermal expansion; see Callister
// "Materials Science and Engineering"
func ThermalExpansion(L0, alpha, deltaT float64) float64 {
	return L0 * alpha * deltaT
}
