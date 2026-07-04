package evt

import (
	"math"
	"sort"
)

// GPDParams holds the two shape/scale parameters of the Generalized Pareto
// Distribution of exceedances over a threshold (the peaks-over-threshold
// model).  The location of a GPD of exceedances is 0 by construction; the
// physical threshold u is carried separately in POTModel.
//
//	Sigma - scale (> 0)
//	Xi    - shape.  Xi > 0 heavy (Pareto-type) tail, Xi == 0 exponential
//	        tail, Xi < 0 short tail with finite upper endpoint -Sigma/Xi.
type GPDParams struct {
	Sigma float64
	Xi    float64
}

// GPDCDF is the GPD cumulative distribution function of an exceedance y >= 0:
//
//	F(y) = 1 - [1 + xi y/sigma]^(-1/xi),   xi != 0
//	F(y) = 1 - exp(-y/sigma),              xi == 0.
//
// For xi < 0 the support is [0, -sigma/xi]; above the upper endpoint F = 1.
//
// Reference: Coles (2001) eq. (4.2); Pickands (1975).
// Precision: exact to machine eps; the xi != 0 branch uses log1p to stay
// stable as xi -> 0.
func GPDCDF(y float64, p GPDParams) float64 {
	if p.Sigma <= 0 || math.IsNaN(p.Sigma) {
		return math.NaN()
	}
	if y <= 0 {
		return 0
	}
	if p.Xi == 0 {
		return -math.Expm1(-y / p.Sigma)
	}
	arg := 1 + p.Xi*y/p.Sigma
	if arg <= 0 { // above the finite upper endpoint (xi < 0)
		return 1
	}
	return -math.Expm1((-1 / p.Xi) * math.Log(arg))
}

// GPDPDF is the GPD density of an exceedance y >= 0:
//
//	f(y) = (1/sigma) [1 + xi y/sigma]^(-1/xi - 1),   xi != 0
//	f(y) = (1/sigma) exp(-y/sigma),                  xi == 0.
//
// Density is 0 for y < 0 and above the finite upper endpoint (xi < 0).
//
// Reference: Coles (2001) §4.2.  Precision: exact to machine eps.
func GPDPDF(y float64, p GPDParams) float64 {
	if p.Sigma <= 0 || math.IsNaN(p.Sigma) {
		return math.NaN()
	}
	if y < 0 {
		return 0
	}
	if p.Xi == 0 {
		return math.Exp(-y/p.Sigma) / p.Sigma
	}
	arg := 1 + p.Xi*y/p.Sigma
	if arg <= 0 {
		return 0
	}
	return math.Exp((-1/p.Xi-1)*math.Log(arg)) / p.Sigma
}

// GPDQuantile is the inverse CDF of the GPD: the exceedance y with F(y) = pr
// for pr in [0,1),
//
//	y = (sigma/xi) [ (1-pr)^{-xi} - 1 ],   xi != 0
//	y = -sigma ln(1-pr),                   xi == 0.
//
// Reference: Coles (2001) eq. (4.2) inverted.  The xi != 0 branch uses expm1
// so it degrades smoothly to the exponential form as xi -> 0.
func GPDQuantile(pr float64, p GPDParams) float64 {
	if p.Sigma <= 0 || math.IsNaN(p.Sigma) || math.IsNaN(pr) || pr < 0 || pr >= 1 {
		if pr == 1 {
			if p.Xi < 0 {
				return -p.Sigma / p.Xi // finite upper endpoint
			}
			return math.Inf(1)
		}
		return math.NaN()
	}
	if p.Xi == 0 {
		return -p.Sigma * math.Log1p(-pr)
	}
	// (sigma/xi)( (1-pr)^{-xi} - 1 ) = (sigma/xi) expm1(-xi ln(1-pr))
	return (p.Sigma / p.Xi) * math.Expm1(-p.Xi*math.Log1p(-pr))
}

// POTModel is a fitted peaks-over-threshold model: a GPD for the exceedances
// above a threshold u, plus the bookkeeping needed to convert conditional
// tail probabilities into unconditional ones.
//
//	Threshold     - u
//	Params        - fitted GPD of the exceedances (y = x - u)
//	NumExceed     - number of observations strictly above u
//	NumTotal      - total sample size
//	ExceedanceRate - NumExceed / NumTotal = empirical P(X > u)
type POTModel struct {
	Threshold      float64
	Params         GPDParams
	NumExceed      int
	NumTotal       int
	ExceedanceRate float64
}

// Exceedances returns the excesses (x - threshold) of every value strictly
// above threshold, in input order.
func Exceedances(data []float64, threshold float64) []float64 {
	out := make([]float64, 0, len(data))
	for _, x := range data {
		if x > threshold {
			out = append(out, x-threshold)
		}
	}
	return out
}

// ThresholdAtRate returns the data value at the upper tail fraction rate,
// i.e. a threshold u such that approximately rate*len(data) observations lie
// strictly above it.  rate in (0,1); the u chosen is the (1-rate) empirical
// quantile using the sorted order statistic just below the tail.
func ThresholdAtRate(data []float64, rate float64) float64 {
	if len(data) == 0 || rate <= 0 || rate >= 1 {
		return math.NaN()
	}
	sorted := append([]float64(nil), data...)
	sort.Float64s(sorted)
	// Index of the smallest tail element: the top ceil(rate*n) values are the tail.
	k := int(float64(len(sorted)) * rate)
	if k < 1 {
		k = 1
	}
	if k > len(sorted)-1 {
		k = len(sorted) - 1
	}
	// Threshold is the largest value NOT in the tail.
	return sorted[len(sorted)-1-k]
}

// FitPOT builds a peaks-over-threshold model: it extracts exceedances above
// threshold and fits a GPD to them by probability-weighted moments
// (FitGPDPWM).  Returns ok == false if fewer than 5 exceedances are present
// or the PWM fit is degenerate.
//
// Reference: Coles (2001) §4.3 (the POT / GPD workflow).
func FitPOT(data []float64, threshold float64) (POTModel, bool) {
	exc := Exceedances(data, threshold)
	if len(exc) < 5 {
		return POTModel{}, false
	}
	params, ok := FitGPDPWM(exc)
	if !ok {
		return POTModel{}, false
	}
	return POTModel{
		Threshold:      threshold,
		Params:         params,
		NumExceed:      len(exc),
		NumTotal:       len(data),
		ExceedanceRate: float64(len(exc)) / float64(len(data)),
	}, true
}

// EvtVaR is the peaks-over-threshold Value-at-Risk at confidence conf
// (e.g. 0.99).  Writing zeta_u = P(X > u) (the exceedance rate) and
// p = 1 - conf,
//
//	VaR = u + (sigma/xi) [ (zeta_u / p)^{xi} - 1 ],   xi != 0
//	VaR = u + sigma ln(zeta_u / p),                   xi == 0.
//
// Valid only for conf above the threshold's own coverage (p < zeta_u); for
// less extreme conf the empirical quantile should be used instead.
//
// Reference: McNeil & Frey (2000) eq. (11); McNeil, Frey & Embrechts (2005)
// §7.2.3.
func EvtVaR(m POTModel, conf float64) float64 {
	if conf <= 0 || conf >= 1 || m.ExceedanceRate <= 0 {
		return math.NaN()
	}
	p := 1 - conf
	xi := m.Params.Xi
	sigma := m.Params.Sigma
	u := m.Threshold
	ratio := m.ExceedanceRate / p
	if xi == 0 {
		return u + sigma*math.Log(ratio)
	}
	return u + (sigma/xi)*math.Expm1(xi*math.Log(ratio))
}

// EvtES is the peaks-over-threshold Expected Shortfall (conditional VaR) at
// confidence conf.  With v = EvtVaR(m, conf) and xi < 1,
//
//	ES = v/(1-xi) + (sigma - xi*u)/(1-xi).
//
// For xi >= 1 the tail mean is infinite and NaN is returned.  ES is clamped
// to be at least VaR (coherence).
//
// Reference: McNeil & Frey (2000) eq. (12); McNeil, Frey & Embrechts (2005)
// §7.2.3.
func EvtES(m POTModel, conf float64) float64 {
	v := EvtVaR(m, conf)
	if math.IsNaN(v) {
		return math.NaN()
	}
	xi := m.Params.Xi
	if xi >= 1 {
		return math.NaN() // infinite mean excess
	}
	es := v/(1-xi) + (m.Params.Sigma-xi*m.Threshold)/(1-xi)
	if es < v {
		return v
	}
	return es
}

// EvtReturnLevel is the POT return level: the loss magnitude exceeded on
// average once per every mObs observations.  With zeta_u the exceedance rate,
//
//	x_m = u + (sigma/xi) [ (m * zeta_u)^{xi} - 1 ],   xi != 0
//	x_m = u + sigma ln(m * zeta_u),                   xi == 0.
//
// Reference: Coles (2001) eq. (4.12)-(4.13).
func EvtReturnLevel(m POTModel, mObs float64) float64 {
	if mObs <= 0 || m.ExceedanceRate <= 0 {
		return math.NaN()
	}
	xi := m.Params.Xi
	sigma := m.Params.Sigma
	u := m.Threshold
	prod := mObs * m.ExceedanceRate
	if prod <= 0 {
		return math.NaN()
	}
	if xi == 0 {
		return u + sigma*math.Log(prod)
	}
	return u + (sigma/xi)*math.Expm1(xi*math.Log(prod))
}

// EvtReturnPeriod is the inverse of EvtReturnLevel: the expected number of
// observations between exceedances of magnitude x (>= threshold).  It equals
// 1 / (zeta_u * S(x-u)) where S is the GPD survival function.
//
// Reference: Coles (2001) §4.3.3.
func EvtReturnPeriod(m POTModel, x float64) float64 {
	if x <= m.Threshold || m.ExceedanceRate <= 0 {
		return math.NaN()
	}
	surv := 1 - GPDCDF(x-m.Threshold, m.Params)
	exceedProb := m.ExceedanceRate * surv
	if exceedProb <= 0 {
		return math.Inf(1)
	}
	return 1 / exceedProb
}
