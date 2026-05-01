package copula

import (
	"math"
)

// StudentTCopulaCDF returns the n-variate Student-t copula CDF at u
// under correlation matrix sigma and degrees of freedom df:
//
//	C_t(u; sigma, df) = T_n(t_df^{-1}(u_1), ..., t_df^{-1}(u_n); sigma, df)
//
// where t_df^{-1} is the univariate Student-t inverse CDF and T_n is
// the n-variate Student-t CDF with correlation matrix sigma and df
// degrees of freedom.
//
// # Why t-copula matters for Solvency II
//
// The Gaussian copula has zero tail dependence — extreme co-movements
// are asymptotically independent.  Empirical evidence on financial /
// insurance loss data (Embrechts-Lindskog-McNeil 2003) shows positive
// tail dependence, especially under crisis / catastrophe regimes
// (fire-water, terror-cyber, pandemic-mortality).  The Student-t
// copula introduces tail dependence parameterised by df: lower df
// produces stronger tail dependence with the same Pearson correlation
// matrix.  EIOPA's *internal-model* approval pathway (alternative to
// the Standard Formula) explicitly recognises t-copula aggregation
// for cat-cluster sub-modules.
//
// # v1 dimensionality
//
// v1 supports n in {2, 3} only.  Bivariate uses Genz-Bretz 2009 §4.2
// 1D integral form
//
//	T_2(x1, x2; rho, df) = (1 / sqrt(2*pi)) *
//	  integral_{-inf}^{x1} t_df pdf(z) *
//	    [T_1((x2 - rho*z) / sqrt((1-rho^2)*(df+z^2)/(df+1));
//	         df+1)] dz
//
// evaluated by Gauss-Legendre quadrature after substituting u = T_df(z)
// so the integration domain is bounded.  Trivariate uses the analogous
// Plackett conditional reduction to a 1D integral over a bivariate
// t-CDF (Genz-Bretz 2009 §4.4), with conditional df shifted by one.
//
// Higher n requires Genz QMC (deferred to v2).
//
// # Inputs
//
//   - u: vector of marginal-CDF values, every entry strictly in (0, 1).
//   - sigma: n*n correlation matrix, must be PSD.
//   - df: degrees of freedom, must be >= 1.
//
// Reference: Genz, A. & Bretz, F. (2009).  Computation of Multivariate
// Normal and t Probabilities.  Lecture Notes in Statistics 195.
// Springer.  Embrechts, P., Lindskog, F. & McNeil, A. (2003).
// Modelling Dependence with Copulas and Applications to Risk
// Management.  In Handbook of Heavy Tailed Distributions in Finance.
// Demarta, S. & McNeil, A. (2005).  The t Copula and Related Copulas.
// International Statistical Review 73 (1): 111-129.
func StudentTCopulaCDF(u []float64, sigma [][]float64, df float64) (float64, error) {
	if df < 1 || math.IsNaN(df) {
		return 0, ErrDfTooSmall
	}
	if err := validateU(u); err != nil {
		return 0, err
	}
	flat, n, err := flattenAndValidateSigma(sigma, len(u))
	if err != nil {
		return 0, err
	}
	if !cholFactorisable(flat, n) {
		return 0, ErrSigmaNotPSD
	}

	// Map each margin through the univariate t-inverse CDF.
	z := make([]float64, n)
	for i, ui := range u {
		z[i] = StudentTQuantile(ui, df)
	}

	switch n {
	case 2:
		return BivariateTCDF(z[0], z[1], sigma[0][1], df), nil
	case 3:
		return TrivariateTCDF(z[0], z[1], z[2],
			sigma[0][1], sigma[0][2], sigma[1][2], df), nil
	default:
		return 0, ErrSigmaDimensionMismatch
	}
}

// BivariateTCDF returns P(X1 <= x1, X2 <= x2) for a bivariate Student-t
// distribution with correlation r and df degrees of freedom.
//
// Implementation: Genz-Bretz 2009 1D integral form, evaluated by
// 16-point Gauss-Legendre after the u = T_df(z) substitution that
// maps (-inf, x1] to (0, T_df(x1)].
//
// Boundary cases r = +/-1 are handled in closed form.
//
// Special case df -> inf: converges to BivariateNormalCDF.
func BivariateTCDF(x1, x2, r, df float64) float64 {
	if math.IsNaN(x1) || math.IsNaN(x2) || math.IsNaN(r) || math.IsNaN(df) {
		return math.NaN()
	}
	if r >= 1.0-1e-12 {
		return StudentTCDF(math.Min(x1, x2), df)
	}
	if r <= -1.0+1e-12 {
		return math.Max(0, StudentTCDF(x1, df)+StudentTCDF(x2, df)-1)
	}
	if math.Abs(r) < 1e-12 {
		return StudentTCDF(x1, df) * StudentTCDF(x2, df)
	}

	// Genz-Bretz 2009 §4.2: integral form
	//   T_2(x1, x2; rho, df) =
	//     integral_{-inf}^{x1} t_df pdf(z) * Phi_t(z) dz
	// where
	//   Phi_t(z) = T_1( (x2 - rho*z) / sqrt((1 - rho^2) * (df + z^2) /
	//                                       (df + 1)),
	//                   df + 1 )
	//
	// Substitute u = T_df(z); domain becomes [0, T_df(x1)] and the
	// pdf-cancellation removes the Jacobian (du = t_df pdf(z) dz):
	//
	//   T_2(x1, x2; rho, df) =
	//     integral_0^{T_df(x1)} Phi_t(t_df^{-1}(u)) du.
	//
	// To minimise the quadrature integration error and preserve the
	// Phi_2(x1, x2) == Phi_2(x2, x1) symmetry that the closed-form
	// distribution satisfies, integrate over whichever variable has
	// the smaller cumulative T_df(x_i) — that bounds the integration
	// domain and reduces tail-mass that 16-node Gauss-Legendre under-
	// resolves.
	if math.Abs(StudentTCDF(x2, df)) < math.Abs(StudentTCDF(x1, df)) {
		x1, x2 = x2, x1
	}
	return tBivariateOriented(x1, x2, r, df)
}

// tBivariateOriented is the Genz-Bretz §4.2 oriented integral with x1
// as the integration upper bound.  Caller picks the smaller-CDF
// orientation for stability.
func tBivariateOriented(x1, x2, r, df float64) float64 {
	upper := StudentTCDF(x1, df)
	if upper <= 0 {
		return 0
	}
	a, b := 0.0, upper
	hw := 0.5 * (b - a)
	mid := 0.5 * (a + b)

	var integral float64
	for k, node := range gaussLegendre16Nodes {
		u := mid + hw*node
		if u <= 0 || u >= 1 {
			continue
		}
		z := StudentTQuantile(u, df)
		num := x2 - r*z
		den := math.Sqrt((1 - r*r) * (df + z*z) / (df + 1))
		if den <= 0 {
			integral += gaussLegendre16Weights[k] * StudentTCDF(x2, df+1)
			continue
		}
		integral += gaussLegendre16Weights[k] * StudentTCDF(num/den, df+1)
	}
	integral *= hw

	if integral < 0 {
		integral = 0
	}
	if integral > 1 {
		integral = 1
	}
	return integral
}

// TrivariateTCDF returns P(X1 <= x1, X2 <= x2, X3 <= x3) for the
// trivariate Student-t with pairwise correlations r12, r13, r23 and df
// degrees of freedom.
//
// Plackett-style conditional reduction (Genz-Bretz 2009 §4.4):
// integrate the bivariate t-CDF on (X2, X3 | X1 = z) under the
// shifted-df conditional law over the marginal of X1.  The conditional
// (X2, X3 | X1 = z) is bivariate t with df+1 degrees of freedom,
// shifted means r12*z, r13*z, scaled variances (1 - r12^2)*(df+z^2)/(df+1)
// and (1 - r13^2)*(df+z^2)/(df+1), and conditional correlation
//
//	rho_23.1 = (r23 - r12*r13) / sqrt((1 - r12^2)*(1 - r13^2)).
//
// We integrate u = T_df(z) over [0, T_df(x1)] with 16-point
// Gauss-Legendre.  Accurate to ~1e-6 for moderate df.
func TrivariateTCDF(x1, x2, x3, r12, r13, r23, df float64) float64 {
	if math.IsNaN(x1) || math.IsNaN(x2) || math.IsNaN(x3) ||
		math.IsNaN(r12) || math.IsNaN(r13) || math.IsNaN(r23) ||
		math.IsNaN(df) {
		return math.NaN()
	}
	if math.Abs(r12) < 1e-12 && math.Abs(r13) < 1e-12 && math.Abs(r23) < 1e-12 {
		return StudentTCDF(x1, df) * StudentTCDF(x2, df) * StudentTCDF(x3, df)
	}

	upper := StudentTCDF(x1, df)
	if upper <= 0 {
		return 0
	}

	denom23 := math.Sqrt((1 - r12*r12) * (1 - r13*r13))
	if denom23 <= 0 {
		// Boundary degeneracy; fall back to bivariate t-CDF on the
		// remaining axes with the surviving correlation.
		if 1-r12*r12 <= 0 {
			eff1 := math.Min(x1, x2)
			if r12 < 0 {
				eff1 = math.Min(x1, -x2)
			}
			return BivariateTCDF(eff1, x3, r13, df)
		}
		eff1 := math.Min(x1, x3)
		if r13 < 0 {
			eff1 = math.Min(x1, -x3)
		}
		return BivariateTCDF(eff1, x2, r12, df)
	}
	rho231 := (r23 - r12*r13) / denom23

	a, b := 0.0, upper
	hw := 0.5 * (b - a)
	mid := 0.5 * (a + b)

	var integral float64
	for k, node := range gaussLegendre16Nodes {
		u := mid + hw*node
		if u <= 0 || u >= 1 {
			continue
		}
		z := StudentTQuantile(u, df)
		scale := math.Sqrt((df + z*z) / (df + 1))
		s2 := math.Sqrt(1-r12*r12) * scale
		s3 := math.Sqrt(1-r13*r13) * scale
		a2 := (x2 - r12*z) / s2
		a3 := (x3 - r13*z) / s3
		integral += gaussLegendre16Weights[k] * BivariateTCDF(a2, a3, rho231, df+1)
	}
	integral *= hw

	if integral < 0 {
		integral = 0
	}
	if integral > 1 {
		integral = 1
	}
	return integral
}
