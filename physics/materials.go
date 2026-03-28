package physics

import "math"

// ---------------------------------------------------------------------------
// Material Physics
// ---------------------------------------------------------------------------

// HookesLaw computes the stress in a linearly elastic material.
//
// Formula: sigma = E * epsilon
// Parameters:
//   - E: Young's modulus (Pa)
//   - epsilon: strain (dimensionless)
//
// Valid range: E > 0, epsilon any real number (linear elastic regime)
// Precision: exact (single multiplication)
// Reference: Hooke, R. (1676) "Ut tensio, sic vis";
// see Timoshenko "Theory of Elasticity"
func HookesLaw(E, epsilon float64) float64 {
	return E * epsilon
}

// VonMisesStress computes the von Mises equivalent stress from three
// principal stresses. This is the most widely used yield criterion for
// ductile materials.
//
// Formula: sigma_vm = sqrt(0.5 * ((s1-s2)^2 + (s2-s3)^2 + (s3-s1)^2))
// Parameters:
//   - s1, s2, s3: principal stresses (Pa)
//
// Valid range: any real numbers for s1, s2, s3
// Precision: limited by float64 sqrt (~15 significant digits)
// Reference: von Mises, R. (1913) "Mechanik der festen Körper im
// plastisch-deformablen Zustand"
func VonMisesStress(s1, s2, s3 float64) float64 {
	d12 := s1 - s2
	d23 := s2 - s3
	d31 := s3 - s1
	return math.Sqrt(0.5 * (d12*d12 + d23*d23 + d31*d31))
}

// TrescaStress computes the Tresca (maximum shear stress) criterion from
// three principal stresses. Yielding occurs when the maximum shear stress
// equals the material's shear yield strength.
//
// Formula: tau_max = (max(s1,s2,s3) - min(s1,s2,s3)) / 2
// Parameters:
//   - s1, s2, s3: principal stresses (Pa)
//
// Valid range: any real numbers
// Precision: exact (comparisons and arithmetic)
// Reference: Tresca, H. (1864) "Mémoire sur l'écoulement des corps solides"
func TrescaStress(s1, s2, s3 float64) float64 {
	maxS := s1
	if s2 > maxS {
		maxS = s2
	}
	if s3 > maxS {
		maxS = s3
	}
	minS := s1
	if s2 < minS {
		minS = s2
	}
	if s3 < minS {
		minS = s3
	}
	return (maxS - minS) / 2.0
}

// StressIntensityFactor computes the mode-I stress intensity factor at the
// tip of a crack in a stressed body.
//
// Formula: K_I = Y * sigma * sqrt(pi * a)
// Parameters:
//   - sigma: applied far-field stress (Pa)
//   - a: crack half-length (m)
//   - Y: geometry correction factor (dimensionless; 1.0 for infinite plate)
//
// Valid range: sigma >= 0, a >= 0, Y > 0
// Precision: limited by float64 sqrt (~15 significant digits)
// Reference: Irwin, G.R. (1957) "Analysis of Stresses and Strains Near the
// End of a Crack Traversing a Plate"
func StressIntensityFactor(sigma, a, Y float64) float64 {
	return Y * sigma * math.Sqrt(math.Pi*a)
}

// GriffithCriterion computes the critical stress for brittle fracture
// (Griffith's criterion for crack propagation).
//
// Formula: sigma_c = sqrt(2 * E * gamma / (pi * a))
// Parameters:
//   - E: Young's modulus (Pa)
//   - gamma: surface energy per unit area (J/m^2)
//   - a: crack half-length (m)
//
// Valid range: E > 0, gamma > 0, a > 0. Returns +Inf if a == 0.
// Precision: limited by float64 sqrt (~15 significant digits)
// Reference: Griffith, A.A. (1921) "The Phenomena of Rupture and Flow in Solids"
func GriffithCriterion(E, gamma, a float64) float64 {
	return math.Sqrt(2 * E * gamma / (math.Pi * a))
}

// ParisLaw computes the fatigue crack growth rate per cycle.
//
// Formula: da/dN = C * (deltaK)^m
// Parameters:
//   - C: material constant (units depend on m)
//   - m: Paris exponent (dimensionless, typically 2-4 for metals)
//   - deltaK: stress intensity factor range (Pa*sqrt(m))
//
// Valid range: C > 0, m > 0, deltaK >= 0
// Precision: limited by float64 pow
// Reference: Paris, P.C. and Erdogan, F. (1963) "A Critical Analysis of
// Crack Propagation Laws"
func ParisLaw(C, m, deltaK float64) float64 {
	return C * math.Pow(deltaK, m)
}

// CoffinManson computes the plastic strain amplitude for low-cycle fatigue.
//
// Formula: delta_epsilon / 2 = epsilon_f * (2 * Nf)^c
// Parameters:
//   - epsilonF: fatigue ductility coefficient (dimensionless)
//   - c: fatigue ductility exponent (negative, typically -0.5 to -0.7)
//   - Nf: number of cycles to failure
//
// Valid range: epsilonF > 0, c < 0 (typically), Nf > 0
// Precision: limited by float64 pow
// Reference: Coffin, L.F. (1954) and Manson, S.S. (1953); independently
// developed low-cycle fatigue relationship
func CoffinManson(epsilonF, c float64, Nf int) float64 {
	return epsilonF * math.Pow(2*float64(Nf), c)
}

// CreepArrhenius computes the steady-state creep strain rate using an
// Arrhenius-type power law model.
//
// Formula: epsilon_dot = A * sigma^n * exp(-Q / (R * T))
// Parameters:
//   - A: material constant (units depend on n)
//   - Q: activation energy (J/mol)
//   - R: gas constant (J/(mol*K)), use constants.GasConstant
//   - T: absolute temperature (K)
//   - sigma: applied stress (Pa)
//   - n: stress exponent (dimensionless, typically 3-8)
//
// Valid range: A > 0, T > 0, sigma >= 0, R > 0
// Precision: limited by float64 exp and pow
// Reference: Norton, F.H. (1929) "The Creep of Steel at High Temperature";
// Arrhenius activation energy model
func CreepArrhenius(A, Q, R, T, sigma, n float64) float64 {
	return A * math.Pow(sigma, n) * math.Exp(-Q/(R*T))
}

// CompositeMixture computes the effective elastic modulus of a fiber-reinforced
// composite using the rule of mixtures (Voigt upper bound).
//
// Formula: E_c = Vf * Ef + (1 - Vf) * Em
// Parameters:
//   - Vf: fiber volume fraction (dimensionless, 0 to 1)
//   - Ef: fiber elastic modulus (Pa)
//   - Em: matrix elastic modulus (Pa)
//
// Valid range: Vf in [0, 1], Ef > 0, Em > 0
// Precision: exact (arithmetic only)
// Reference: Voigt, W. (1889) rule of mixtures; see Jones
// "Mechanics of Composite Materials"
func CompositeMixture(Vf, Ef, Em float64) float64 {
	return Vf*Ef + (1-Vf)*Em
}

// EulerBuckling computes the critical buckling load for a slender column.
//
// Formula: P_cr = pi^2 * E * I / (K * L)^2
// Parameters:
//   - E: Young's modulus (Pa)
//   - I: second moment of area (m^4)
//   - L: unsupported length (m)
//   - K: effective length factor (dimensionless; 1.0 for pinned-pinned,
//     0.5 for fixed-fixed, 2.0 for fixed-free, 0.7 for fixed-pinned)
//
// Valid range: E > 0, I > 0, L > 0, K > 0. Returns +Inf if K*L == 0.
// Precision: limited by float64 representation of pi
// Reference: Euler, L. (1744) "Methodus inveniendi lineas curvas maximi
// minimive proprietate gaudentes"
func EulerBuckling(E, I, L, K float64) float64 {
	kl := K * L
	return math.Pi * math.Pi * E * I / (kl * kl)
}

// BeamDeflection computes the maximum deflection of a simply supported
// beam with a concentrated load at mid-span.
//
// Formula: delta = P * L^3 / (48 * E * I)
// Parameters:
//   - P: concentrated load at midspan (N)
//   - L: beam span (m)
//   - E: Young's modulus (Pa)
//   - I: second moment of area (m^4)
//
// Valid range: P >= 0, L > 0, E > 0, I > 0
// Precision: exact (arithmetic only)
// Reference: Euler-Bernoulli beam theory; see Gere & Timoshenko
// "Mechanics of Materials"
func BeamDeflection(P, L, E, I float64) float64 {
	return P * L * L * L / (48 * E * I)
}
