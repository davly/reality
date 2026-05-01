package copula

import (
	"errors"
	"math"
)

// CopulaCDF is the type signature for an n-variate copula cumulative
// distribution function: it accepts a uniform-margins vector u in
// (0, 1)^n and returns the joint copula CDF value in [0, 1].
//
// The two canonical CopulaCDF closures shipped by this package are:
//
//	GaussianCopulaCDFFn(sigma)         — closure with sigma pinned
//	StudentTCopulaCDFFn(sigma, df)     — closure with sigma + df pinned
//
// Sklar's theorem then composes any such CopulaCDF with a vector of
// marginal CDFs to produce a joint distribution function on R^n.
type CopulaCDF func(u []float64) (float64, error)

// MarginalCDF is the type signature for a univariate marginal CDF —
// a strictly-monotone non-decreasing map from R to (0, 1).
type MarginalCDF func(x float64) float64

// JointCDF is the type signature for the n-variate joint distribution
// function produced by Sklar's reconstruction: it accepts an x vector
// in R^n and returns the joint CDF value in [0, 1].
type JointCDF func(x []float64) (float64, error)

// ErrSklarMarginalDimensionMismatch is returned when SklarJointFromMarginals
// receives a marginals vector whose length differs from the copula's
// expected dimensionality.
var ErrSklarMarginalDimensionMismatch = errors.New(
	"copula: marginals vector length must equal the copula dimensionality")

// SklarJointFromMarginals constructs a joint CDF on R^n from a vector
// of n marginal CDFs and an n-variate copula.  This is the *operational*
// form of Sklar's 1959 theorem:
//
//	F(x_1, ..., x_n) = C(F_1(x_1), ..., F_n(x_n))
//
// The marginals are arbitrary univariate CDFs (Normal, Lognormal,
// Tweedie, GPD, empirical step-function — any strictly-monotone map
// to (0, 1)).  The copula is the dependence structure.  Together they
// uniquely characterise the joint distribution; conversely, every
// joint distribution decomposes uniquely (for continuous marginals)
// into marginals + copula.  This is the one-line reason copulas exist:
// they isolate dependence from marginal shape.
//
// # Solvency II SCR aggregation in two lines
//
//	marginals := []MarginalCDF{F_market, F_life, F_nonlife, F_health,
//	                            F_counterparty}
//	scr := SklarJointFromMarginals(marginals,
//	                                GaussianCopulaCDFFn(eiopaAnnexIV))
//
// Now `scr(x_5d)` returns the joint CDF of the 5-module SCR aggregate
// at any aggregate-loss vector x_5d, with marginals shaped by the
// individual sub-module risk distributions (compound-Poisson, etc.)
// and dependence structure given by the EIOPA Annex IV regulator-
// prescribed correlation matrix.  This is the math relic-insurance's
// R98 envelope `Solvency_II_Art35` statutory_basis was always citing.
//
// # Inputs
//
//   - marginals: a slice of n univariate CDFs.  Each must map R to
//     (0, 1) (the boundary values cause copula CDFs to error via
//     ErrUOutOfRange).  Length determines the dimensionality.
//   - copula: an n-variate CopulaCDF closure (use
//     GaussianCopulaCDFFn or StudentTCopulaCDFFn).
//
// Returns a JointCDF that performs the marginal-then-copula composition
// at evaluation time.  No error is returned at construction; errors
// propagate from the copula CDF on evaluation.  The marginals slice
// header is defensively copied; the closures themselves are not.
//
// Reference: Sklar, A. (1959).  Fonctions de répartition à n
// dimensions et leurs marges.  Publications de l'Institut de
// Statistique de l'Université de Paris 8: 229-231.  Nelsen, R. B.
// (2006).  An Introduction to Copulas, 2nd ed.  Springer.
func SklarJointFromMarginals(marginals []MarginalCDF, copula CopulaCDF) JointCDF {
	// Defensive copy of the slice header.
	m := make([]MarginalCDF, len(marginals))
	copy(m, marginals)

	return func(x []float64) (float64, error) {
		if len(x) != len(m) {
			return 0, ErrSklarMarginalDimensionMismatch
		}
		u := make([]float64, len(x))
		for i, xi := range x {
			ui := m[i](xi)
			// Clamp tightly inside (0, 1) to avoid the boundary
			// singularities of the inverse-normal probit.
			if ui <= 0 {
				ui = 1e-12
			}
			if ui >= 1 {
				ui = 1 - 1e-12
			}
			u[i] = ui
		}
		return copula(u)
	}
}

// GaussianCopulaCDFFn returns a CopulaCDF closure that wraps
// GaussianCopulaCDF with the given correlation matrix sigma pinned in.
// Convenience for use with SklarJointFromMarginals.
func GaussianCopulaCDFFn(sigma [][]float64) CopulaCDF {
	return func(u []float64) (float64, error) {
		return GaussianCopulaCDF(u, sigma)
	}
}

// StudentTCopulaCDFFn returns a CopulaCDF closure that wraps
// StudentTCopulaCDF with the correlation matrix sigma and df pinned
// in.  Convenience for use with SklarJointFromMarginals.
func StudentTCopulaCDFFn(sigma [][]float64, df float64) CopulaCDF {
	return func(u []float64) (float64, error) {
		return StudentTCopulaCDF(u, sigma, df)
	}
}

// GaussianCopulaCorrelationFromTau is the Kruskal 1958 closed-form
// link between Kendall's tau and Gaussian copula correlation:
//
//	rho = sin(pi * tau / 2)
//
// This is the canonical one-shot estimator: given a paired sample
// (x, y), compute KendallTau(x, y) and pass the result here to obtain
// the Gaussian-copula correlation parameter.  Avoids the rank-Pearson
// detour the FW C# `GaussianCopulaCorrelation` uses (which empirically
// agrees but is more expensive and has the probit-clamping issue).
//
// Range: tau in [-1, 1] -> rho in [-1, 1] monotonically.
//
// Reference: Kruskal, W. H. (1958).  Ordinal Measures of Association.
// JASA 53: 814-861, eq. (3.6).
func GaussianCopulaCorrelationFromTau(tau float64) float64 {
	return math.Sin(math.Pi * tau / 2.0)
}
