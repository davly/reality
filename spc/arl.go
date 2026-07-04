package spc

// Average-run-length (ARL) calibration for CUSUM and EWMA control charts.
//
// A control chart's ARL is the expected number of samples until it signals.
// The IN-CONTROL ARL0 is the mean number of samples between FALSE alarms (large
// is good: ARL0 = 500 means one false alarm every 500 samples on average); the
// OUT-OF-CONTROL ARL1(delta) is the mean number of samples to DETECT a sustained
// mean shift of delta (small is good). Calibrating a chart means choosing its
// design constants (CUSUM decision interval h and reference value k; EWMA weight
// lambda and width multiplier L) so that ARL0 hits a chosen false-alarm budget —
// e.g. "one false alarm per 500 points" — instead of folklore constants.
//
// All statistics here are STANDARDISED: the monitored quantity is assumed to be
// z = (x - mu0) / sigma, so a "shift" is measured in standard-deviation units and
// k, h, L are unitless. A caller who scales its own data by a robust sigma (as
// nexus's cusumChangepoints does) is already in these units.
//
// # CUSUM: Siegmund's approximation
//
// The two-sided tabular CUSUM accumulates C+_i = max(0, C+_{i-1} + z_i - k) and
// C-_i = max(0, C-_{i-1} - z_i - k) and signals when either exceeds h. Siegmund
// (1985) gives a closed-form ARL approximation for the one-sided CUSUM with drift
// parameter Delta:
//
//	ARL = ( exp(-2*Delta*b) + 2*Delta*b - 1 ) / ( 2*Delta^2 ),   Delta != 0
//	ARL = b^2,                                                    Delta = 0
//
// where b = h + 1.166 applies Siegmund's correction for the mean overshoot of a
// random walk past its boundary. For the UPPER CUSUM watching a true shift delta,
// Delta = delta - k; for the LOWER CUSUM, Delta = -delta - k. The two-sided ARL
// combines the two one-sided charts as competing risks (the chart signals at the
// first of the two):
//
//	1/ARL_two-sided = 1/ARL_upper + 1/ARL_lower.
//
// At delta = 0 both sides are equal, so ARL0_two-sided = ARL0_one-sided / 2.
// Siegmund's approximation is most accurate for small-to-moderate shifts
// (|delta| <= ~1.5 sigma) and for the in-control ARL0; it progressively
// UNDER-estimates ARL for large shifts (where a chart signals almost immediately
// anyway), so a large-shift ARL1 should be read as a lower bound.
//
// # EWMA: exact limits + Brook-Evans Markov-chain ARL
//
// The EWMA statistic is Z_i = lambda*z_i + (1-lambda)*Z_{i-1}, Z_0 = 0. Its exact
// variance at time t is
//
//	Var(Z_t) = sigma^2 * (lambda/(2-lambda)) * (1 - (1-lambda)^(2t)),
//
// which grows from lambda^2*sigma^2 at t=1 to the steady state sigma^2*lambda/(2-lambda).
// EWMALimits returns the exact time-varying control limits mu0 +/- L*sqrt(Var);
// the (lambda/(2-lambda))*(1-(1-lambda)^(2t)) term is the variance-inflation factor.
//
// The EWMA ARL has no elementary closed form; it is computed by the Brook-Evans
// (1972) Markov-chain method used by Lucas & Saccucci (1990) to tabulate EWMA
// ARLs. The in-control band [-H, H], H = L*sigma*sqrt(lambda/(2-lambda)), is split
// into m equal cells; the substochastic transition matrix P is built from the
// normal CDF of one EWMA update, and the ARL vector solves (I - P) * mu = 1 by
// Gaussian elimination, read at the centre (Z_0 = 0) cell. As lambda -> 1 the EWMA
// collapses to a Shewhart individuals chart and this reproduces the exact Shewhart
// ARL 1/(2*(1-Phi(L))) independent of m.
//
// References:
//   - Siegmund, D. (1985), "Sequential Analysis: Tests and Confidence Intervals",
//     Springer, ch. on the CUSUM ARL (the b = h + 1.166 overshoot correction).
//   - Montgomery, D.C., "Introduction to Statistical Quality Control", 7th ed.
//     (2013), ch. 9 (Table 9.3, two-sided CUSUM k=0.5, h=5 ARLs; the EWMA scheme).
//   - Lucas, J.M. & Saccucci, M.S. (1990), "Exponentially Weighted Moving Average
//     Control Schemes: Properties and Enhancements", Technometrics 32(1), 1-29.
//   - Brook, D. & Evans, D.A. (1972), "An Approach to the Probability Distribution
//     of CUSUM Run Length", Biometrika 59(3), 539-549 (the Markov-chain method).

import (
	"errors"
	"math"

	"github.com/davly/reality/optim"
)

// siegmundOvershoot is Siegmund's (1985) correction for the expected overshoot of
// a random walk past its boundary: the effective boundary is b = h + 1.166 rather
// than h. The 1.166 = 2 * rho, where rho ~ 0.583 is the limiting mean overshoot of
// a standardised random walk.
const siegmundOvershoot = 1.166

// defaultEWMAStates is the Markov-chain grid resolution EWMAARL uses. It is odd so
// that the centre cell straddles Z_0 = 0 exactly; 201 cells give ARL0 within ~1%
// of the Lucas & Saccucci (1990) tabulated values.
const defaultEWMAStates = 201

var (
	// ErrNonPositiveK is returned when the CUSUM reference value k is <= 0.
	ErrNonPositiveK = errors.New("spc: CUSUM reference value k must be positive")
	// ErrNonPositiveH is returned when the CUSUM decision interval h is <= 0.
	ErrNonPositiveH = errors.New("spc: CUSUM decision interval h must be positive")
	// ErrARLTarget is returned when a requested ARL0 is not attainable (must be
	// greater than the ARL0 the chart already has at h -> 0).
	ErrARLTarget = errors.New("spc: target ARL0 is below the attainable minimum")
	// ErrLambdaRange is returned when the EWMA weight lambda is not in (0, 1].
	ErrLambdaRange = errors.New("spc: EWMA lambda must be in (0, 1]")
	// ErrNonPositiveL is returned when the EWMA/limit width multiplier L is <= 0.
	ErrNonPositiveL = errors.New("spc: width multiplier L must be positive")
	// ErrNonPositiveSigmaARL is returned when a supplied sigma is <= 0.
	ErrNonPositiveSigmaARL = errors.New("spc: sigma must be positive")
	// ErrTimeIndex is returned when an EWMA time index t is < 1.
	ErrTimeIndex = errors.New("spc: time index t must be >= 1")
	// ErrStates is returned when the Markov-chain cell count is too small.
	ErrStates = errors.New("spc: Markov-chain state count m must be >= 11")
)

// cusumARLOneSidedDrift is Siegmund's one-sided CUSUM ARL as a function of the
// drift parameter Delta and corrected boundary b = h + siegmundOvershoot. It is
// the shared kernel of the exported one- and two-sided functions.
//
// Precision: exact float64 evaluation of Siegmund's closed form; the APPROXIMATION
// error versus the true ARL is the model's, not the arithmetic's.
func cusumARLOneSidedDrift(delta, b float64) float64 {
	if math.Abs(delta) < 1e-12 {
		// Limit as Delta -> 0: (exp(-2*Delta*b)+2*Delta*b-1)/(2*Delta^2) -> b^2.
		return b * b
	}
	return (math.Exp(-2*delta*b) + 2*delta*b - 1) / (2 * delta * delta)
}

// CUSUMARLOneSided returns Siegmund's ARL for a ONE-sided (upper) tabular CUSUM
// with reference value k and decision interval h, monitoring a standardised
// process whose true mean has shifted by shift sigma. The drift parameter is
// Delta = shift - k; a lower CUSUM's ARL is CUSUMARLOneSided(-shift, k, h).
//
// With shift = 0 this is the one-sided in-control ARL0. Requires k > 0 and h > 0.
//
// Precision: exact evaluation of Siegmund's closed form. Model accuracy is best
// for |shift| <= ~1.5 sigma; large shifts are progressively under-estimated.
func CUSUMARLOneSided(shift, k, h float64) (float64, error) {
	if k <= 0 {
		return 0, ErrNonPositiveK
	}
	if h <= 0 {
		return 0, ErrNonPositiveH
	}
	return cusumARLOneSidedDrift(shift-k, h+siegmundOvershoot), nil
}

// CUSUMARL returns Siegmund's ARL for a TWO-sided tabular CUSUM (the scheme
// nexus's cusumChangepoints implements: C+ and C- against the same h, reset on
// signal) with reference value k and decision interval h, for a standardised mean
// shift of shift sigma. The two one-sided charts combine as competing risks:
//
//	1/ARL = 1/ARL_upper(Delta = shift - k) + 1/ARL_lower(Delta = -shift - k).
//
// With shift = 0 this is the in-control ARL0 (the mean samples between false
// alarms), and equals the one-sided ARL0 halved. Requires k > 0 and h > 0.
//
// Precision: exact evaluation of Siegmund's closed form; see the package doc for
// the approximation's regime of validity.
func CUSUMARL(shift, k, h float64) (float64, error) {
	if k <= 0 {
		return 0, ErrNonPositiveK
	}
	if h <= 0 {
		return 0, ErrNonPositiveH
	}
	b := h + siegmundOvershoot
	upper := cusumARLOneSidedDrift(shift-k, b)
	lower := cusumARLOneSidedDrift(-shift-k, b)
	return 1.0 / (1.0/upper + 1.0/lower), nil
}

// CUSUMThresholdForARL solves the inverse calibration problem: given a target
// in-control ARL0 (the desired mean number of samples between false alarms) and a
// reference value k, it returns the decision interval h such that the two-sided
// CUSUM's ARL0 equals arl0. This replaces a folklore h (e.g. the ubiquitous h=5.0)
// with "one false alarm per arl0 points".
//
// The two-sided ARL0 is strictly increasing in h, so the root is unique; it is
// found by bracket-expansion plus optim.BisectionMethod. Requires k > 0 and an
// arl0 greater than the ARL0 attainable as h -> 0 (else ErrARLTarget).
//
// Precision: h to ~1e-8 (bisection tolerance); the mapping itself carries
// Siegmund's approximation error.
func CUSUMThresholdForARL(arl0, k float64) (float64, error) {
	if k <= 0 {
		return 0, ErrNonPositiveK
	}
	b0 := siegmundOvershoot // the boundary at h = 0
	minARL := 1.0 / (2.0 / cusumARLOneSidedDrift(-k, b0))
	if arl0 <= minARL {
		return 0, ErrARLTarget
	}
	f := func(h float64) float64 {
		bb := h + siegmundOvershoot
		up := cusumARLOneSidedDrift(-k, bb)
		twoSided := 1.0 / (2.0 / up)
		return twoSided - arl0
	}
	// Expand an upper bracket until the ARL0 exceeds the target (f > 0).
	hi := 1.0
	for f(hi) < 0 {
		hi *= 2
		if hi > 1e6 {
			return 0, ErrARLTarget
		}
	}
	return optim.BisectionMethod(f, 0, hi, 1e-8), nil
}

// EWMALimit holds the exact EWMA control limits at a given time index, expressed
// as offsets from the target mu0: the absolute limits are mu0 + UCL and mu0 + LCL.
type EWMALimit struct {
	T            int     // observation index this applies to (1-based)
	VarInflation float64 // (lambda/(2-lambda)) * (1 - (1-lambda)^(2t)), the variance-inflation factor
	SigmaZ       float64 // standard deviation of the EWMA statistic at time t (= sigma*sqrt(VarInflation))
	HalfWidth    float64 // L * SigmaZ, the distance from the centre line to each limit
	UCL          float64 // upper limit as an offset from mu0 (= +HalfWidth)
	LCL          float64 // lower limit as an offset from mu0 (= -HalfWidth)
}

// EWMALimits returns the EXACT (time-varying) EWMA control limits at observation
// index t, for smoothing weight lambda, width multiplier L and process standard
// deviation sigma. The variance of the EWMA statistic at time t is
//
//	Var(Z_t) = sigma^2 * (lambda/(2-lambda)) * (1 - (1-lambda)^(2t)),
//
// which starts narrow (lambda^2*sigma^2 at t=1) and widens to the steady-state
// sigma^2*lambda/(2-lambda). Using the exact factor (rather than the asymptote)
// tightens the limits for early observations and cuts start-up false alarms.
// Requires lambda in (0,1], L > 0, sigma > 0 and t >= 1.
//
// Precision: exact to float64; a single pow, a multiply and a sqrt.
func EWMALimits(lambda, L, sigma float64, t int) (EWMALimit, error) {
	if lambda <= 0 || lambda > 1 {
		return EWMALimit{}, ErrLambdaRange
	}
	if L <= 0 {
		return EWMALimit{}, ErrNonPositiveL
	}
	if sigma <= 0 {
		return EWMALimit{}, ErrNonPositiveSigmaARL
	}
	if t < 1 {
		return EWMALimit{}, ErrTimeIndex
	}
	steady := lambda / (2 - lambda)
	inflation := steady * (1 - math.Pow(1-lambda, 2*float64(t)))
	sigmaZ := sigma * math.Sqrt(inflation)
	half := L * sigmaZ
	return EWMALimit{
		T:            t,
		VarInflation: inflation,
		SigmaZ:       sigmaZ,
		HalfWidth:    half,
		UCL:          half,
		LCL:          -half,
	}, nil
}

// EWMASteadyStateSigma returns the steady-state standard deviation of the EWMA
// statistic, sigma*sqrt(lambda/(2-lambda)) — the width the time-varying limits
// converge to. Requires lambda in (0,1] and sigma > 0.
func EWMASteadyStateSigma(lambda, sigma float64) (float64, error) {
	if lambda <= 0 || lambda > 1 {
		return 0, ErrLambdaRange
	}
	if sigma <= 0 {
		return 0, ErrNonPositiveSigmaARL
	}
	return sigma * math.Sqrt(lambda/(2-lambda)), nil
}

// EWMAARL returns the average run length of a two-sided EWMA control chart with
// smoothing weight lambda and steady-state width multiplier L, for a standardised
// mean shift of shift sigma. With shift = 0 it is the in-control ARL0. It uses the
// default Markov-chain resolution (defaultEWMAStates cells); see EWMAARLGrid to
// control accuracy. Requires lambda in (0,1] and L > 0.
//
// Precision: Brook-Evans discretisation; ARL0 within ~1% of Lucas & Saccucci
// (1990) at the default resolution, exact in the lambda -> 1 Shewhart limit.
func EWMAARL(lambda, L, shift float64) (float64, error) {
	return EWMAARLGrid(lambda, L, shift, defaultEWMAStates)
}

// EWMAARLGrid is EWMAARL with an explicit Markov-chain cell count m (odd, >= 11).
// Larger m increases accuracy at O(m^3) cost (the linear solve). The in-control
// band [-H, H], H = L*sqrt(lambda/(2-lambda)), is partitioned into m equal cells;
// P[j][i] is the probability that one EWMA update from cell j's midpoint lands in
// cell i, and the ARL vector solves (I - P)*mu = 1, read at the centre cell.
//
// Precision: as EWMAARL; convergence in m is monotone and ~O(1/m^2).
func EWMAARLGrid(lambda, L, shift float64, m int) (float64, error) {
	if lambda <= 0 || lambda > 1 {
		return 0, ErrLambdaRange
	}
	if L <= 0 {
		return 0, ErrNonPositiveL
	}
	if m < 11 {
		return 0, ErrStates
	}
	if m%2 == 0 {
		m++ // force odd so the centre cell straddles Z_0 = 0 exactly
	}
	sigmaZ := math.Sqrt(lambda / (2 - lambda)) // standardised: sigma = 1
	H := L * sigmaZ
	w := 2 * H / float64(m) // cell width
	mid := make([]float64, m)
	for i := range mid {
		mid[i] = -H + (float64(i)+0.5)*w
	}
	// Build A = I - P (row j -> next state) and rhs b = 1. A update from state j's
	// midpoint s_j is Z' = (1-lambda)*s_j + lambda*x, x ~ N(shift, 1); Z' lands in
	// cell i's interval (mid_i - w/2, mid_i + w/2) with probability
	//   Phi((mid_i + w/2 - (1-lambda)s_j)/lambda - shift)
	// - Phi((mid_i - w/2 - (1-lambda)s_j)/lambda - shift).
	A := make([]float64, m*m)
	rhs := make([]float64, m)
	for j := 0; j < m; j++ {
		base := (1 - lambda) * mid[j]
		for i := 0; i < m; i++ {
			hi := (mid[i] + w/2 - base) / lambda
			lo := (mid[i] - w/2 - base) / lambda
			p := normalCDF(hi-shift) - normalCDF(lo-shift)
			A[j*m+i] = -p
		}
		A[j*m+j] += 1.0
		rhs[j] = 1.0
	}
	mu := solveDense(A, rhs, m)
	return mu[m/2], nil
}

// solveDense solves the dense n-by-n system A*x = b (A in row-major order) by
// Gaussian elimination with partial pivoting. A and b are overwritten. This is a
// standard first-principles direct solve, self-contained so spc carries no
// linear-algebra dependency for its small (m-by-m) ARL system.
func solveDense(A, b []float64, n int) []float64 {
	for c := 0; c < n; c++ {
		// Partial pivot: largest magnitude in column c at or below the diagonal.
		p := c
		best := math.Abs(A[c*n+c])
		for r := c + 1; r < n; r++ {
			if v := math.Abs(A[r*n+c]); v > best {
				best, p = v, r
			}
		}
		if p != c {
			for cc := 0; cc < n; cc++ {
				A[c*n+cc], A[p*n+cc] = A[p*n+cc], A[c*n+cc]
			}
			b[c], b[p] = b[p], b[c]
		}
		piv := A[c*n+c]
		for r := c + 1; r < n; r++ {
			f := A[r*n+c] / piv
			if f == 0 {
				continue
			}
			for cc := c; cc < n; cc++ {
				A[r*n+cc] -= f * A[c*n+cc]
			}
			b[r] -= f * b[c]
		}
	}
	x := make([]float64, n)
	for r := n - 1; r >= 0; r-- {
		sum := b[r]
		for cc := r + 1; cc < n; cc++ {
			sum -= A[r*n+cc] * x[cc]
		}
		x[r] = sum / A[r*n+r]
	}
	return x
}
