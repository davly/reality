package copula

import "errors"

// ErrEmptyU is returned when a copula CDF is called with a zero-length
// uniform-margins vector.  A copula CDF requires at least 2 dimensions —
// scalar copulas degenerate to the identity and bring no information.
var ErrEmptyU = errors.New("copula: u must be non-empty (and len(u) >= 2 for non-degenerate copulas)")

// ErrUOutOfRange is returned when an entry of u is outside the open
// unit interval (0, 1).  Copula domains are PIT (probability-integral-
// transform) outputs of marginal CDFs, which by definition land in (0, 1)
// for continuous marginals.  Boundary points 0 and 1 cause the inverse
// normal CDF to diverge (probit(0) = -Inf, probit(1) = +Inf), so we
// reject them eagerly rather than emit Inf-tainted joint CDFs.
var ErrUOutOfRange = errors.New("copula: every u_i must lie in the open interval (0, 1)")

// ErrSigmaNotPSD is returned when the supplied correlation matrix sigma
// is not positive semi-definite (Cholesky factorisation fails).  The
// EIOPA Annex IV correlation matrices that Solvency II SCR aggregation
// uses are PSD by construction, but user-supplied matrices may not be
// — defensive validation is essential because a non-PSD sigma silently
// produces nonsense from any Gaussian-copula formula.
var ErrSigmaNotPSD = errors.New("copula: sigma must be positive semi-definite (Cholesky factorisation failed)")

// ErrSigmaDimensionMismatch is returned when the dimensionality of sigma
// does not match the dimensionality of u, or when sigma is non-square,
// or when the dimensionality is unsupported (n < 2 or n > 3 in v1).
//
// v1 supports n in {2, 3} only — bivariate and trivariate Gaussian /
// t-copula CDFs.  Higher-dimensional CDFs require Genz QMC or similar
// (deferred to v2).
var ErrSigmaDimensionMismatch = errors.New("copula: sigma must be square, of dimension matching u, with 2 <= n <= 3 in v1")

// ErrDfTooSmall is returned when the Student-t copula is called with a
// degrees-of-freedom parameter below 1.  The t-distribution has finite
// mean only for df > 1 and finite variance only for df > 2; df < 1 is
// not a copula in any practical regime.
var ErrDfTooSmall = errors.New("copula: degrees of freedom must be >= 1 for the Student-t copula")

// ErrLengthMismatch is returned by KendallTau and Sklar reconstruction
// helpers when paired-sample inputs do not agree in length.
var ErrLengthMismatch = errors.New("copula: paired inputs must have equal length")
