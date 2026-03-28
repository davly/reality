package constants

// Physical constants — SI 2019 exact definitions where applicable,
// NIST CODATA 2018 recommended values otherwise.
//
// The 2019 SI redefinition fixed exact values for the Planck constant,
// elementary charge, Boltzmann constant, and Avogadro constant. Constants
// derived from these are therefore also exact.
//
// For non-exact constants (GravitationalConst, VacuumPermittivity, etc.),
// values are from NIST CODATA 2018 adjusted values.
// Reference: https://physics.nist.gov/cuu/Constants/

// SpeedOfLight is the speed of light in vacuum, in meters per second.
// Source: SI 2019 exact definition. c = 299,792,458 m/s (exact).
const SpeedOfLight = 299792458.0 // m/s

// Planck is the Planck constant, in joule-seconds.
// Source: SI 2019 exact definition. h = 6.62607015e-34 J*s (exact).
const Planck = 6.62607015e-34 // J*s

// PlanckReduced is the reduced Planck constant (h-bar), h / (2*pi),
// in joule-seconds.
// Source: derived from SI 2019 exact h and mathematical pi.
// Value: 1.054571817...e-34 J*s.
// Precision: limited by float64 representation of pi, not by h.
const PlanckReduced = Planck / (2 * Pi) // J*s

// Boltzmann is the Boltzmann constant, in joules per kelvin.
// Source: SI 2019 exact definition. k_B = 1.380649e-23 J/K (exact).
const Boltzmann = 1.380649e-23 // J/K

// Avogadro is the Avogadro constant, in reciprocal moles.
// Source: SI 2019 exact definition. N_A = 6.02214076e23 mol^-1 (exact).
const Avogadro = 6.02214076e23 // mol^-1

// ElementaryCharge is the elementary electric charge, in coulombs.
// Source: SI 2019 exact definition. e = 1.602176634e-19 C (exact).
const ElementaryCharge = 1.602176634e-19 // C

// GravitationalConst is Newton's gravitational constant, in
// m^3 kg^-1 s^-2.
// Source: NIST CODATA 2018 recommended value.
// Uncertainty: 2.2e-5 relative standard uncertainty.
const GravitationalConst = 6.67430e-11 // m^3 kg^-1 s^-2

// VacuumPermittivity is the permittivity of free space (electric constant),
// in farads per meter.
// Source: NIST CODATA 2018 recommended value, derived from
// epsilon_0 = 1 / (mu_0 * c^2). No longer exact after SI 2019.
// Value: 8.8541878128e-12 F/m.
const VacuumPermittivity = 8.8541878128e-12 // F/m

// VacuumPermeability is the permeability of free space (magnetic constant),
// in henries per meter.
// Source: NIST CODATA 2018 recommended value. No longer exact after SI 2019
// (was exactly 4*pi*1e-7 in the old SI).
// Value: 1.25663706212e-6 H/m.
const VacuumPermeability = 1.25663706212e-6 // H/m

// StefanBoltzmann is the Stefan-Boltzmann constant, in W m^-2 K^-4.
// Source: derived from SI 2019 exact constants.
// sigma = 2 * pi^5 * k_B^4 / (15 * h^3 * c^2)
// Value: 5.670374419e-8 W m^-2 K^-4 (exact given exact inputs).
const StefanBoltzmann = 5.670374419e-8 // W m^-2 K^-4

// GasConstant is the molar gas constant, in J mol^-1 K^-1.
// Source: SI 2019 derived exact value. R = N_A * k_B.
// Value: 8.314462618... J mol^-1 K^-1 (exact product of two exact constants,
// limited only by float64 representation).
const GasConstant = Avogadro * Boltzmann // J mol^-1 K^-1

// StandardGravity is the standard acceleration due to gravity at Earth's
// surface, in meters per second squared.
// Source: ISO 80000-3:2019 exact definition. g_n = 9.80665 m/s^2 (exact).
// Note: this is a defined constant, not a measured value.
const StandardGravity = 9.80665 // m/s^2

// AtmPressure is the standard atmosphere, in pascals.
// Source: ISO 80000-3:2019 exact definition. 1 atm = 101325 Pa (exact).
const AtmPressure = 101325.0 // Pa
