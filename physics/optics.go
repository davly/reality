package physics

import "math"

// ---------------------------------------------------------------------------
// Optics
// ---------------------------------------------------------------------------

// SnellRefraction computes the angle of refraction using Snell's law.
//
// Formula: theta_R = arcsin((n1 / n2) * sin(theta_I))
// Parameters:
//   - n1: refractive index of the incident medium (dimensionless)
//   - n2: refractive index of the refracting medium (dimensionless)
//   - thetaI: angle of incidence (radians), measured from the surface normal
//
// Valid range: n1 > 0, n2 > 0, thetaI in [0, pi/2].
// Returns NaN if total internal reflection occurs (sin(theta_R) > 1).
// Precision: limited by float64 trig (~15 significant digits)
// Reference: Snell, W. (1621); Descartes (1637) "La Dioptrique"
func SnellRefraction(n1, n2, thetaI float64) float64 {
	sinThetaR := (n1 / n2) * math.Sin(thetaI)
	if sinThetaR > 1.0 || sinThetaR < -1.0 {
		return math.NaN() // total internal reflection
	}
	return math.Asin(sinThetaR)
}

// FresnelReflectance computes the average (unpolarized) reflectance at a
// dielectric interface using the Fresnel equations.
//
// Formula:
//
//	Rs = ((n1*cos(theta_I) - n2*cos(theta_T)) / (n1*cos(theta_I) + n2*cos(theta_T)))^2
//	Rp = ((n1*cos(theta_T) - n2*cos(theta_I)) / (n1*cos(theta_T) + n2*cos(theta_I)))^2
//	R  = (Rs + Rp) / 2
//
// where theta_T is found via Snell's law.
//
// Parameters:
//   - n1: refractive index of the incident medium (dimensionless)
//   - n2: refractive index of the refracting medium (dimensionless)
//   - thetaI: angle of incidence (radians)
//
// Valid range: n1 > 0, n2 > 0, thetaI in [0, pi/2].
// Returns 1.0 (total reflection) if total internal reflection occurs.
// Precision: limited by float64 trig (~15 significant digits)
// Reference: Fresnel, A. (1823); see Hecht "Optics" ch. 4
func FresnelReflectance(n1, n2, thetaI float64) float64 {
	sinThetaI := math.Sin(thetaI)
	sinThetaT := (n1 / n2) * sinThetaI

	// Total internal reflection.
	if sinThetaT > 1.0 || sinThetaT < -1.0 {
		return 1.0
	}

	cosThetaI := math.Cos(thetaI)
	cosThetaT := math.Sqrt(1.0 - sinThetaT*sinThetaT)

	// s-polarization (TE)
	numS := n1*cosThetaI - n2*cosThetaT
	denS := n1*cosThetaI + n2*cosThetaT
	Rs := (numS / denS) * (numS / denS)

	// p-polarization (TM)
	numP := n1*cosThetaT - n2*cosThetaI
	denP := n1*cosThetaT + n2*cosThetaI
	Rp := (numP / denP) * (numP / denP)

	return (Rs + Rp) / 2.0
}

// BeerLambertLaw computes the transmitted intensity of light passing through
// an absorbing medium.
//
// Formula: I = I0 * exp(-mu * x)
// Parameters:
//   - I0: incident intensity (W/m^2 or any consistent unit)
//   - mu: linear attenuation coefficient (1/m)
//   - x: path length through the medium (m)
//
// Valid range: I0 >= 0, mu >= 0, x >= 0
// Precision: limited by float64 exp (~15 significant digits)
// Reference: Bouguer (1729), Lambert (1760), Beer (1852);
// commonly called Beer-Lambert or Beer-Lambert-Bouguer law
func BeerLambertLaw(I0, mu, x float64) float64 {
	return I0 * math.Exp(-mu*x)
}
