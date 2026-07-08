package prob

import "math"

// ---------------------------------------------------------------------------
// Trend first-passage: time-to-threshold with prediction intervals.
//
// The forward ("inverse forecast") problem. An ordinary least-squares trend
// answers "what value at time t?"; the ops question is the inverse: "at what
// time does the series cross a limit L, and how uncertain is that time?".
// This file fits an OLS linear trend on an equally-spaced series, forms the
// textbook Student-t prediction interval for a future observation, and solves
// for the first-passage time of the trend and of the PI band through L.
//
// Two named consumers motivate this (see reality review REALITY_FABLE_PLAY
// 2026-07-04, item "trend first-passage"):
//   - Nexus forecast_budget: projects "periods until budget exhausted" from a
//     bare OLS slope with zero uncertainty, so a noisy 3-point spend series and
//     a rock-steady 60-point series produce equally confident answers.
//   - Nexus disk monitoring: persists a disk_percent series but alerts only on
//     static instantaneous thresholds; nothing computes "earliest plausible
//     date the disk fills".
//
// The series is assumed equally spaced; time is the sample index i = 0..n-1.
// All crossing times are returned in the same units, measured FORWARD from the
// last sample (index n-1): a return of 3.0 means "3 periods after the most
// recent observation".
//
// Zero dependencies, deterministic, pure. References cited per function.
// ---------------------------------------------------------------------------

// StudentTQuantile returns the inverse CDF (quantile) of the Student's
// t-distribution with df degrees of freedom for probability p: the value t
// such that P(T <= t) = p.
//
// The t-distribution is symmetric about 0, so the lower tail is obtained by
// negating the upper tail: q(p) = -q(1-p). The upper tail is inverted by
// bisection on the monotone CDF (studentTCDF, itself the regularized
// incomplete beta function).
//
// Valid range: p in (0, 1), df > 0. Returns NaN if p <= 0, p >= 1, df <= 0,
// or either input is NaN.
// Precision: ~1e-10 absolute (bisection to convergence over studentTCDF, whose
// own precision is ~1e-12 for moderate df).
// Reference: Abramowitz & Stegun (1964), formula 26.7.5; inversion by
// bisection on the CDF (Press et al., Numerical Recipes, 3rd ed., §6.14).
func StudentTQuantile(p, df float64) float64 {
	if math.IsNaN(p) || math.IsNaN(df) || df <= 0 || p <= 0 || p >= 1 {
		return math.NaN()
	}
	if p == 0.5 {
		return 0
	}
	if p < 0.5 {
		return -StudentTQuantile(1-p, df)
	}
	// p in (0.5, 1): quantile is positive. Bracket [0, hi] then bisect.
	lo := 0.0
	hi := 1.0
	for studentTCDF(hi, df) < p {
		hi *= 2
		if hi > 1e12 {
			break
		}
	}
	for i := 0; i < 200; i++ {
		mid := 0.5 * (lo + hi)
		if studentTCDF(mid, df) < p {
			lo = mid
		} else {
			hi = mid
		}
	}
	return 0.5 * (lo + hi)
}

// olsFit fits y = intercept + slope*x over x = 0..n-1 and returns the fit plus
// the mean of x, Sxx = sum (x-xbar)^2, and the residual standard error
// s = sqrt(SS_res / (n-2)). Internal helper shared by the public functions.
func olsFit(data []float64) (slope, intercept, xbar, sxx, s float64, ok bool) {
	n := len(data)
	if n < 3 {
		return math.NaN(), math.NaN(), math.NaN(), math.NaN(), math.NaN(), false
	}
	nf := float64(n)
	var sx, sy, sxx2, sxy float64
	for i := 0; i < n; i++ {
		x := float64(i)
		sx += x
		sy += data[i]
		sxx2 += x * x
		sxy += x * data[i]
	}
	xbar = sx / nf
	sxx = sxx2 - sx*sx/nf // sum (x - xbar)^2 > 0 for n >= 2 distinct indices
	if sxx == 0 {
		return math.NaN(), math.NaN(), math.NaN(), math.NaN(), math.NaN(), false
	}
	slope = (sxy - sx*sy/nf) / sxx
	intercept = (sy - slope*sx) / nf
	var ssres float64
	for i := 0; i < n; i++ {
		r := data[i] - (intercept + slope*float64(i))
		ssres += r * r
	}
	s = math.Sqrt(ssres / (nf - 2)) // residual standard error, df = n-2
	return slope, intercept, xbar, sxx, s, true
}

// TrendPredictionInterval fits an OLS linear trend on an equally-spaced series
// and returns the forecast plus the two-sided Student-t prediction interval for
// a single future observation at horizon h periods after the last sample.
//
// The forecast point is x0 = (n-1) + h. The prediction interval for a NEW
// observation (not the mean response) is:
//
//	yhat(x0) +/- t_{(1+conf)/2, n-2} * s * sqrt(1 + 1/n + (x0 - xbar)^2 / Sxx)
//
// where s = sqrt(SS_res/(n-2)) is the residual standard error and
// Sxx = sum (x_i - xbar)^2. The "1 +" term (absent from a mean-response
// confidence interval) accounts for the variance of a single new observation.
//
// Valid range: len(data) >= 3 (df = n-2 >= 1 for the variance estimate),
// conf in (0, 1). h may be any real value (negative interpolates). Returns
// (NaN, NaN, NaN) outside the valid range.
// Precision: ~1e-10 (limited by StudentTQuantile).
// Reference: Weisberg, "Applied Linear Regression," 4th ed. (2014), §2.5
// (prediction of a future observation); Montgomery, Peck & Vining (2012), §2.4.2.
func TrendPredictionInterval(data []float64, h, conf float64) (yhat, lower, upper float64) {
	if conf <= 0 || conf >= 1 {
		return math.NaN(), math.NaN(), math.NaN()
	}
	slope, intercept, xbar, sxx, s, ok := olsFit(data)
	if !ok {
		return math.NaN(), math.NaN(), math.NaN()
	}
	nf := float64(len(data))
	x0 := (nf - 1) + h
	yhat = intercept + slope*x0
	tCrit := StudentTQuantile(0.5*(1+conf), nf-2)
	d := x0 - xbar
	se := s * math.Sqrt(1 + 1/nf + d*d/sxx)
	half := tCrit * se
	return yhat, yhat - half, yhat + half
}

// TrendCrossingResult holds the first-passage analysis of a linear trend and
// its prediction band through a threshold. All times are periods measured
// forward from the last sample (index n-1).
type TrendCrossingResult struct {
	Slope     float64 // OLS slope (units per period)
	Intercept float64 // OLS intercept at x = 0
	Sigma     float64 // residual standard error, sqrt(SS_res/(n-2))
	THat      float64 // point-estimate crossing time; +Inf if the trend never crosses
	TEarliest float64 // earliest plausible crossing (near PI bound reaches threshold)
	TLatest   float64 // latest plausible crossing (far PI bound reaches threshold); +Inf if unbounded
	Bounded   bool    // true iff TLatest is finite (slope significant at conf: |slope|/SE(slope) > t_crit)
	OK        bool    // true iff the point trend crosses the threshold in the future (THat finite and > 0)
}

// TrendCrossing fits an OLS linear trend on an equally-spaced series and
// computes the first-passage time through the threshold L for the point trend
// and for the two-sided Student-t prediction band at confidence conf.
//
// The point crossing solves yhat(x) = L. The band crossings solve the near and
// far prediction-interval bounds equal to L, i.e. yhat(x) +/- w(x) = L where
// w(x) = t_crit * s * sqrt(1 + 1/n + (x-xbar)^2/Sxx) is the PI half-width.
// Because the band widens with horizon, these are the roots of the quadratic
// g(x)^2 = w(x)^2 with g(x) = yhat(x) - L (Weisberg §2.5). The near bound
// (the band edge facing L) always crosses, giving a finite TEarliest; the far
// bound crosses in finite time iff the slope is significant at level conf,
// equivalently a2 = slope^2 - t_crit^2 * s^2 / Sxx > 0 (this is exactly the
// slope t-statistic exceeding its critical value). When the slope is not
// significant the far bound never reaches L: TLatest = +Inf, Bounded = false —
// the honest "too noisy / too short to bound" verdict.
//
// L may be approached from below (increasing slope) or above (decreasing
// slope); direction is handled symmetrically.
//
// Valid range: len(data) >= 3, conf in (0, 1). Returns a zero-value-ish result
// with NaN fields outside the valid range.
// Precision: ~1e-9 on the crossing times (bisection over the monotone bounds).
// Reference: Weisberg, "Applied Linear Regression," 4th ed. (2014), §2.5;
// first-passage / inverse-prediction of a regression band (Draper & Smith,
// "Applied Regression Analysis," 3rd ed., §3.2, inverse prediction).
func TrendCrossing(data []float64, threshold, conf float64) TrendCrossingResult {
	res := TrendCrossingResult{
		Slope: math.NaN(), Intercept: math.NaN(), Sigma: math.NaN(),
		THat: math.NaN(), TEarliest: math.NaN(), TLatest: math.NaN(),
	}
	if conf <= 0 || conf >= 1 {
		return res
	}
	slope, intercept, xbar, sxx, s, ok := olsFit(data)
	if !ok {
		return res
	}
	nf := float64(len(data))
	xLast := nf - 1
	res.Slope = slope
	res.Intercept = intercept
	res.Sigma = s

	current := intercept + slope*xLast

	// Point crossing.
	if slope != 0 {
		res.THat = (threshold-intercept)/slope - xLast
	} else {
		res.THat = math.Inf(1)
	}
	res.OK = !math.IsInf(res.THat, 0) && res.THat > 0

	// Degenerate: perfect fit (zero residuals) => the band collapses to the
	// point estimate; earliest = latest = point crossing.
	if s == 0 {
		res.TEarliest = res.THat
		res.TLatest = res.THat
		res.Bounded = !math.IsInf(res.THat, 0)
		return res
	}

	tCrit := StudentTQuantile(0.5*(1+conf), nf-2)

	// Direction of approach: sign of the trend movement toward the threshold.
	// For a flat trend, fall back to the side the threshold sits on so the
	// widening band still yields an "earliest" crossing.
	dir := 1.0
	switch {
	case slope > 0:
		dir = 1.0
	case slope < 0:
		dir = -1.0
	default:
		if threshold < current {
			dir = -1.0
		}
	}

	w := func(x float64) float64 {
		d := x - xbar
		return tCrit * s * math.Sqrt(1+1/nf+d*d/sxx)
	}
	yhat := func(x float64) float64 { return intercept + slope*x }
	// near bound: the PI edge leading toward the threshold; far bound: trailing.
	near := func(x float64) float64 { return yhat(x) + dir*w(x) }
	far := func(x float64) float64 { return yhat(x) - dir*w(x) }

	// reached reports whether a bound value has reached/passed the threshold in
	// the direction of approach (monotone in x over [xLast, inf) for x >= xbar).
	reached := func(v float64) bool { return dir*(v-threshold) >= 0 }

	// findCross returns the forward time (periods after xLast) at which f first
	// reaches the threshold, searching x >= xLast. ok=false if no crossing is
	// found within the horizon cap (treat as unbounded).
	findCross := func(f func(float64) float64) (float64, bool) {
		if reached(f(xLast)) {
			return 0, true // band already spans the threshold now
		}
		hiSpan := 1.0
		found := false
		for k := 0; k < 200; k++ {
			if reached(f(xLast + hiSpan)) {
				found = true
				break
			}
			hiSpan *= 2
			if hiSpan > 1e15 {
				break
			}
		}
		if !found {
			return 0, false
		}
		loSpan := 0.0
		for i := 0; i < 100; i++ {
			mid := 0.5 * (loSpan + hiSpan)
			if reached(f(xLast + mid)) {
				hiSpan = mid
			} else {
				loSpan = mid
			}
		}
		return 0.5 * (loSpan + hiSpan), true
	}

	// Earliest: the near bound always eventually reaches the threshold (its
	// asymptotic slope is dir*(|slope| + sqrt(D)), reinforced away from zero).
	if te, teOK := findCross(near); teOK {
		res.TEarliest = te
	}

	// Latest: finite iff the slope is significant at level conf. Equivalently
	// a2 = slope^2 - t_crit^2 s^2 / Sxx > 0; then the far bound is monotone and
	// crosses. Otherwise the far bound never reaches the threshold.
	a2 := slope*slope - tCrit*tCrit*s*s/sxx
	if a2 > 0 {
		if tl, tlOK := findCross(far); tlOK {
			res.TLatest = tl
			res.Bounded = true
		} else {
			res.TLatest = math.Inf(1)
		}
	} else {
		res.TLatest = math.Inf(1)
	}
	return res
}
