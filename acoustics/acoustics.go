// Package acoustics provides sound and wave propagation functions. All
// functions are pure, deterministic, and zero-dependency. Physical constants
// are imported from the constants package where applicable.
//
// Functions follow the Reality convention: numbers in, numbers out.
// Every function documents its formula, valid range, precision, and reference.
package acoustics

import "math"

// ---------------------------------------------------------------------------
// Sound Speed & Propagation
// ---------------------------------------------------------------------------

// SoundSpeed computes the speed of sound in an ideal gas.
//
// Formula: c = √(γRT/M)
// Parameters:
//   - gamma: heat capacity ratio Cp/Cv (dimensionless), e.g. 1.4 for air
//   - R:     universal gas constant (J/(mol·K)), typically 8.314462618
//   - T:     absolute temperature (K)
//   - M:     molar mass (kg/mol), e.g. 0.02896 for dry air
//
// Valid range: gamma > 0, R > 0, T >= 0, M > 0. Returns 0 for T=0, NaN for
// negative arguments inside sqrt.
// Precision: limited by float64 sqrt (~15 significant digits)
// Reference: Laplace's correction to Newton's formula for speed of sound;
// see Kinsler et al. "Fundamentals of Acoustics" (2000) ch. 5
func SoundSpeed(gamma, R, T, M float64) float64 {
	return math.Sqrt(gamma * R * T / M)
}

// SoundIntensity computes the sound intensity at distance r from a point
// source radiating uniformly in all directions (inverse-square law).
//
// Formula: I = P / (4πr²)
// Parameters:
//   - P: acoustic power (W)
//   - r: distance from source (m)
//
// Valid range: r > 0, P >= 0. Returns +Inf if r == 0.
// Precision: limited by float64 representation of π (~15 significant digits)
// Reference: inverse-square law for spherical wave propagation;
// Kinsler et al. "Fundamentals of Acoustics" (2000) ch. 6
func SoundIntensity(P, r float64) float64 {
	return P / (4.0 * math.Pi * r * r)
}

// ---------------------------------------------------------------------------
// Decibel Scales
// ---------------------------------------------------------------------------

// DecibelSPL computes the sound pressure level in decibels relative to a
// reference pressure.
//
// Formula: dB = 20 * log₁₀(p / pRef)
// Parameters:
//   - p:    sound pressure (Pa)
//   - pRef: reference pressure (Pa), typically 20e-6 Pa for air
//
// Valid range: p > 0, pRef > 0. Returns -Inf for p == 0, NaN for negative p.
// Precision: limited by float64 log10 (~15 significant digits)
// Reference: IEC 61672-1 standard for sound level meters;
// reference pressure 20 μPa is the threshold of human hearing at 1 kHz
func DecibelSPL(p, pRef float64) float64 {
	return 20.0 * math.Log10(p/pRef)
}

// DecibelFromIntensity computes the sound intensity level in decibels relative
// to a reference intensity.
//
// Formula: dB = 10 * log₁₀(I / IRef)
// Parameters:
//   - I:    sound intensity (W/m²)
//   - IRef: reference intensity (W/m²), typically 1e-12 W/m²
//
// Valid range: I > 0, IRef > 0. Returns -Inf for I == 0, NaN for negative I.
// Precision: limited by float64 log10 (~15 significant digits)
// Reference: IEC 61672-1; reference intensity 1e-12 W/m² corresponds to
// ~20 μPa in air at standard conditions
func DecibelFromIntensity(I, IRef float64) float64 {
	return 10.0 * math.Log10(I/IRef)
}

// ---------------------------------------------------------------------------
// Room Acoustics
// ---------------------------------------------------------------------------

// SabineRT60 computes the reverberation time (T60) using the Sabine equation.
// T60 is the time for sound to decay by 60 dB after the source is turned off.
//
// Formula: T60 = 0.161 * V / A
// Parameters:
//   - V: room volume (m³)
//   - A: total absorption (m², Sabine absorption units = Σ αᵢSᵢ)
//
// Valid range: V >= 0, A > 0. Returns +Inf if A == 0.
// Precision: exact (multiplication and division)
// Reference: Sabine, W.C. (1898) "Reverberation"; constant 0.161 assumes
// metric units (24 ln(10) / c ≈ 0.161 s/m at c = 343 m/s)
func SabineRT60(V, A float64) float64 {
	return 0.161 * V / A
}

// ---------------------------------------------------------------------------
// Doppler Effect
// ---------------------------------------------------------------------------

// DopplerShift computes the observed frequency when a source and receiver
// are in relative motion along the line connecting them.
//
// Formula: f = f₀ * (c + vr) / (c + vs)
// Parameters:
//   - f0: emitted frequency (Hz)
//   - vs: source velocity (m/s), positive = moving away from receiver
//   - vr: receiver velocity (m/s), positive = moving toward source
//   - c:  speed of sound in medium (m/s)
//
// Sign convention:
//   - vs > 0: source receding (frequency decreases)
//   - vr > 0: receiver approaching (frequency increases)
//
// Valid range: c + vs != 0. Returns +Inf/-Inf or NaN at sonic/supersonic
// source velocities where c + vs == 0.
// Precision: exact (arithmetic only)
// Reference: Doppler, C. (1842) "Über das farbige Licht der Doppelsterne";
// adapted for the acoustic case with medium-referenced velocities
func DopplerShift(f0, vs, vr, c float64) float64 {
	return f0 * (c + vr) / (c + vs)
}

// ---------------------------------------------------------------------------
// Resonance & Waves
// ---------------------------------------------------------------------------

// ResonantFrequency computes the nth harmonic resonant frequency of an
// open pipe (both ends open).
//
// Formula: f_n = n * c / (2L)
// Parameters:
//   - L: pipe length (m)
//   - n: harmonic number (1 = fundamental, 2 = first overtone, ...)
//   - c: speed of sound (m/s)
//
// Valid range: L > 0, n >= 1, c > 0. Returns +Inf if L == 0.
// Precision: exact (arithmetic only)
// Reference: standing wave condition for open-open pipe; see Kinsler et al.
// "Fundamentals of Acoustics" (2000) ch. 10
func ResonantFrequency(L float64, n int, c float64) float64 {
	return float64(n) * c / (2.0 * L)
}

// WaveLength computes the wavelength of a wave from its frequency and
// propagation speed.
//
// Formula: λ = c / f
// Parameters:
//   - f: frequency (Hz)
//   - c: propagation speed (m/s)
//
// Valid range: f > 0, c > 0. Returns +Inf if f == 0.
// Precision: exact (single division)
// Reference: fundamental wave relation v = fλ
func WaveLength(f, c float64) float64 {
	return c / f
}

// ---------------------------------------------------------------------------
// Weighting Curves
// ---------------------------------------------------------------------------

// AWeighting computes the approximate A-weighting adjustment in dB for a
// given frequency. A-weighting models the human ear's frequency-dependent
// sensitivity.
//
// The formula uses the IEC 61672-1 analytic approximation:
//
//	R_A(f) = 12194² * f⁴ / ((f² + 20.6²)(f² + 12194²) * √((f² + 107.7²)(f² + 737.9²)))
//	A(f) = 20*log₁₀(R_A) + 2.00
//
// Parameters:
//   - f: frequency (Hz)
//
// Valid range: f > 0. Returns -Inf for f == 0.
// Precision: limited by float64 log and sqrt (~12 significant digits at extreme frequencies)
// Reference: IEC 61672-1:2013 "Electroacoustics — Sound level meters",
// Annex E; frequency weighting network A
func AWeighting(f float64) float64 {
	f2 := f * f
	num := 12194.0 * 12194.0 * f2 * f2
	d1 := f2 + 20.6*20.6
	d2 := f2 + 12194.0*12194.0
	d3 := math.Sqrt((f2 + 107.7*107.7) * (f2 + 737.9*737.9))
	ra := num / (d1 * d2 * d3)
	return 20.0*math.Log10(ra) + 2.0
}
