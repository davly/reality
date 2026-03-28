// Package em provides fundamental electromagnetic functions: Coulomb's law,
// electric fields, Ohm's law, circuit analysis, and energy storage.
// All functions are pure, deterministic, and zero-dependency.
//
// Functions follow the Reality convention: numbers in, numbers out.
// Every function documents its formula, valid range, precision, and reference.
//
// Coulomb's constant k is derived from the vacuum permittivity constant in
// the constants package: k = 1 / (4π ε₀).
package em

import (
	"math"

	"github.com/davly/reality/constants"
)

// coulombConst is Coulomb's constant k = 1 / (4π ε₀), in N·m²/C².
// Derived from the VacuumPermittivity constant.
// Value: ~8.9875517923e9 N·m²/C².
var coulombConst = 1.0 / (4.0 * math.Pi * constants.VacuumPermittivity)

// ---------------------------------------------------------------------------
// Electrostatics
// ---------------------------------------------------------------------------

// CoulombForce computes the electrostatic force between two point charges.
//
// Formula: F = k·q1·q2 / r²
// Parameters:
//   - q1: first charge (coulombs)
//   - q2: second charge (coulombs)
//   - r: distance between charges (meters)
//
// Returns the force in newtons. Positive means repulsive (same-sign charges),
// negative means attractive (opposite-sign charges).
//
// Valid range: r != 0. Returns ±Inf if r == 0.
// Precision: limited by uncertainty in ε₀ and float64 arithmetic
// Reference: Coulomb, C.A. (1785); Griffiths "Introduction to Electrodynamics"
// 4th ed. eq. 2.1
func CoulombForce(q1, q2, r float64) float64 {
	return coulombConst * q1 * q2 / (r * r)
}

// ElectricField computes the electric field magnitude from a point charge
// at distance r.
//
// Formula: E = k·q / r²
// Parameters:
//   - q: charge (coulombs)
//   - r: distance from charge (meters)
//
// Returns the field magnitude in V/m (or N/C). Sign indicates direction:
// positive away from a positive charge.
//
// Valid range: r != 0. Returns ±Inf if r == 0.
// Precision: limited by uncertainty in ε₀ and float64 arithmetic
// Reference: Griffiths "Introduction to Electrodynamics" 4th ed. eq. 2.3
func ElectricField(q, r float64) float64 {
	return coulombConst * q / (r * r)
}

// ---------------------------------------------------------------------------
// Ohm's Law and Power
// ---------------------------------------------------------------------------

// OhmsLaw computes the current flowing through a resistor.
//
// Formula: I = V / R
// Parameters:
//   - V: voltage across the resistor (volts)
//   - R: resistance (ohms)
//
// Valid range: R != 0. Returns ±Inf if R == 0.
// Precision: exact (single division)
// Reference: Ohm, G.S. (1827) "Die galvanische Kette, mathematisch bearbeitet"
func OhmsLaw(V, R float64) float64 {
	return V / R
}

// PowerElectric computes the electrical power dissipated or delivered.
//
// Formula: P = V · I
// Parameters:
//   - V: voltage (volts)
//   - I: current (amperes)
//
// Returns power in watts. Positive for power dissipated; sign follows
// the passive sign convention.
//
// Valid range: unrestricted
// Precision: exact (single multiplication)
// Reference: Joule's first law (P = IV); any introductory circuits text
func PowerElectric(V, I float64) float64 {
	return V * I
}

// ---------------------------------------------------------------------------
// Resistor Networks
// ---------------------------------------------------------------------------

// ResistorsInSeries computes the total resistance of resistors in series.
//
// Formula: R_total = R1 + R2 + ... + Rn
// Parameters:
//   - resistances: slice of resistance values (ohms), must have len >= 1
//
// Returns 0 if the slice is empty.
// Valid range: all R_i >= 0
// Precision: exact (summation)
// Reference: Kirchhoff's voltage law; any introductory circuits text
func ResistorsInSeries(resistances []float64) float64 {
	total := 0.0
	for _, r := range resistances {
		total += r
	}
	return total
}

// ResistorsInParallel computes the total resistance of resistors in parallel.
//
// Formula: 1/R_total = 1/R1 + 1/R2 + ... + 1/Rn
// Parameters:
//   - resistances: slice of resistance values (ohms), must have len >= 1
//
// Returns 0 if the slice is empty. Returns 0 if any resistance is 0 (short circuit).
// Valid range: all R_i > 0
// Precision: limited by float64 accumulation
// Reference: Kirchhoff's current law; any introductory circuits text
func ResistorsInParallel(resistances []float64) float64 {
	if len(resistances) == 0 {
		return 0
	}
	sumInv := 0.0
	for _, r := range resistances {
		if r == 0 {
			return 0
		}
		sumInv += 1.0 / r
	}
	return 1.0 / sumInv
}

// ---------------------------------------------------------------------------
// Energy Storage
// ---------------------------------------------------------------------------

// CapacitorEnergy computes the energy stored in a capacitor.
//
// Formula: E = 0.5 · C · V²
// Parameters:
//   - C: capacitance (farads)
//   - V: voltage across the capacitor (volts)
//
// Valid range: C >= 0, V unrestricted
// Precision: exact (multiplication)
// Reference: Griffiths "Introduction to Electrodynamics" 4th ed. eq. 2.55
func CapacitorEnergy(C, V float64) float64 {
	return 0.5 * C * V * V
}

// InductorEnergy computes the energy stored in an inductor's magnetic field.
//
// Formula: E = 0.5 · L · I²
// Parameters:
//   - L: inductance (henries)
//   - I: current through the inductor (amperes)
//
// Valid range: L >= 0, I unrestricted
// Precision: exact (multiplication)
// Reference: Griffiths "Introduction to Electrodynamics" 4th ed. eq. 7.29
func InductorEnergy(L, I float64) float64 {
	return 0.5 * L * I * I
}

// ---------------------------------------------------------------------------
// RC and LC Circuits
// ---------------------------------------------------------------------------

// RCTimeConstant computes the time constant of an RC circuit.
//
// Formula: τ = R · C
// Parameters:
//   - R: resistance (ohms)
//   - C: capacitance (farads)
//
// Returns τ in seconds. After one time constant, a charging capacitor reaches
// ~63.2% of its final voltage.
//
// Valid range: R >= 0, C >= 0
// Precision: exact (single multiplication)
// Reference: any introductory circuits text; time constant of first-order system
func RCTimeConstant(R, C float64) float64 {
	return R * C
}

// ResonantFrequencyLC computes the resonant frequency of an LC circuit.
//
// Formula: f = 1 / (2π√(LC))
// Parameters:
//   - L: inductance (henries)
//   - C: capacitance (farads)
//
// Returns frequency in hertz (Hz).
//
// Valid range: L > 0, C > 0
// Precision: limited by float64 sqrt and pi (~15 significant digits)
// Reference: Thomson, W. (Lord Kelvin) (1853); Griffiths "Introduction to
// Electrodynamics" 4th ed. eq. 9.178
func ResonantFrequencyLC(L, C float64) float64 {
	return 1.0 / (2.0 * math.Pi * math.Sqrt(L*C))
}
