package evt

import (
	"math"

	"github.com/davly/reality/optim"
)

// GEVLogLik returns the GEV log-likelihood of data under params.  It is
// -Inf when any observation falls outside the support (which is how the
// support constraint 1 + xi (x-mu)/sigma > 0 enters maximum likelihood).
//
// Reference: Coles (2001) eq. (3.7)-(3.8).
func GEVLogLik(data []float64, p GEVParams) float64 {
	if p.Sigma <= 0 {
		return math.Inf(-1)
	}
	n := float64(len(data))
	ll := -n * math.Log(p.Sigma)
	if p.Xi == 0 {
		for _, x := range data {
			z := (x - p.Mu) / p.Sigma
			ll += -z - math.Exp(-z)
		}
		return ll
	}
	for _, x := range data {
		t := 1 + p.Xi*(x-p.Mu)/p.Sigma
		if t <= 0 {
			return math.Inf(-1)
		}
		logT := math.Log(t)
		ll += -(1+1/p.Xi)*logT - math.Exp((-1/p.Xi)*logT)
	}
	return ll
}

// GPDLogLik returns the GPD log-likelihood of non-negative exceedances under
// params, -Inf outside the support.
//
// Reference: Coles (2001) eq. (4.10).
func GPDLogLik(exceedances []float64, p GPDParams) float64 {
	if p.Sigma <= 0 {
		return math.Inf(-1)
	}
	n := float64(len(exceedances))
	ll := -n * math.Log(p.Sigma)
	if p.Xi == 0 {
		for _, y := range exceedances {
			ll += -y / p.Sigma
		}
		return ll
	}
	for _, y := range exceedances {
		arg := 1 + p.Xi*y/p.Sigma
		if arg <= 0 {
			return math.Inf(-1)
		}
		ll += -(1 + 1/p.Xi) * math.Log(arg)
	}
	return ll
}

// numGrad fills g with a central-difference approximation of the gradient of
// f at x.  Used to drive L-BFGS without a hand-coded analytic gradient (the
// GEV/GPD scores are error-prone near the support boundary; central
// differences of the barrier-guarded objective are robust and deterministic).
func numGrad(f func([]float64) float64, x, g []float64) {
	const h = 1e-6
	tmp := make([]float64, len(x))
	copy(tmp, x)
	for i := range x {
		step := h * (1 + math.Abs(x[i]))
		tmp[i] = x[i] + step
		fp := f(tmp)
		tmp[i] = x[i] - step
		fm := f(tmp)
		tmp[i] = x[i]
		g[i] = (fp - fm) / (2 * step)
	}
}

// bigPenalty is a large finite objective value returned for infeasible
// parameters, so the L-BFGS line search can back away without meeting a
// non-finite value.
const bigPenalty = 1e12

// FitGEVMLE refines a GEV fit by maximum likelihood, deterministically: it
// starts from the L-moment estimate (FitGEVLMoments) and minimises the
// negative log-likelihood with L-BFGS over the reparameterisation
// (mu, log sigma, xi) so that sigma stays positive.  If L-BFGS fails to
// improve on the L-moment start (or wanders infeasible), the L-moment
// estimate is returned unchanged, so the result is never worse than the
// deterministic closed-form fit.
//
// ok == false only if the L-moment start itself cannot be formed.
//
// Reference: Coles (2001) §3.3.2 (numerical MLE of the GEV).  Fixed starts
// from L-moments make the optimisation reproducible (no random restarts).
func FitGEVMLE(blockMaxima []float64) (GEVParams, bool) {
	start, ok := FitGEVLMoments(blockMaxima)
	if !ok {
		return GEVParams{}, false
	}
	obj := func(v []float64) float64 {
		p := GEVParams{Mu: v[0], Sigma: math.Exp(v[1]), Xi: v[2]}
		ll := GEVLogLik(blockMaxima, p)
		if math.IsInf(ll, -1) || math.IsNaN(ll) {
			return bigPenalty
		}
		return -ll
	}
	x0 := []float64{start.Mu, math.Log(start.Sigma), start.Xi}
	res := optim.LBFGS(obj, func(x, g []float64) { numGrad(obj, x, g) }, x0, 6, 200, 1e-8)

	cand := GEVParams{Mu: res[0], Sigma: math.Exp(res[1]), Xi: res[2]}
	if cand.Sigma > 0 && !math.IsNaN(cand.Xi) &&
		GEVLogLik(blockMaxima, cand) > GEVLogLik(blockMaxima, start) {
		return cand, true
	}
	return start, true
}

// FitGPDMLE refines a GPD fit by maximum likelihood, deterministically: it
// starts from the PWM estimate (FitGPDPWM) and minimises the negative
// log-likelihood with L-BFGS over (log sigma, xi).  Falls back to the PWM
// estimate whenever L-BFGS does not strictly improve the likelihood.
//
// ok == false only if the PWM start cannot be formed.
//
// Reference: Coles (2001) §4.3.2; Grimshaw (1993) for the well-posedness of
// GPD MLE.  Fixed PWM start => reproducible optimisation.
func FitGPDMLE(exceedances []float64) (GPDParams, bool) {
	start, ok := FitGPDPWM(exceedances)
	if !ok {
		return GPDParams{}, false
	}
	obj := func(v []float64) float64 {
		p := GPDParams{Sigma: math.Exp(v[0]), Xi: v[1]}
		ll := GPDLogLik(exceedances, p)
		if math.IsInf(ll, -1) || math.IsNaN(ll) {
			return bigPenalty
		}
		return -ll
	}
	x0 := []float64{math.Log(start.Sigma), start.Xi}
	res := optim.LBFGS(obj, func(x, g []float64) { numGrad(obj, x, g) }, x0, 6, 200, 1e-8)

	cand := GPDParams{Sigma: math.Exp(res[0]), Xi: res[1]}
	if cand.Sigma > 0 && !math.IsNaN(cand.Xi) &&
		GPDLogLik(exceedances, cand) > GPDLogLik(exceedances, start) {
		return cand, true
	}
	return start, true
}
