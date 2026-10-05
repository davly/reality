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

// bigPenalty is the large objective value reported for parameters outside the
// model's domain (an observation outside the support, or a shape at or below
// xiFloor), so the L-BFGS line search can back away from them without meeting
// a non-finite value.
const bigPenalty = 1e12

// xiFloor is the lower edge of the admissible shape range.  For xi <= -1 the
// GEV and GPD likelihoods are unbounded (they grow without limit as the
// upper endpoint approaches the largest observation), so no maximum exists
// there; the search is confined to xi > xiFloor.
//
// Reference: Smith (1985) "Maximum likelihood estimation in a class of
// nonregular cases", Biometrika 72: 67-90; Coles (2001) §3.3.2.
const xiFloor = -1.0

// numGrad fills g with a finite-difference approximation of the gradient of
// f at x.  Used to drive L-BFGS without a hand-coded analytic gradient (the
// GEV/GPD scores are error-prone near the support boundary).
//
// f reports an infeasible point as a value >= bigPenalty.  Each partial
// derivative is a central difference when both neighbours are feasible, a
// one-sided difference toward the feasible neighbour when only one is, and is
// retried with a halved step when neither is.  A central difference that
// straddled the edge of the support would mix the penalty value into the
// quotient and return a gradient of order 1e17.
func numGrad(f func([]float64) float64, x, g []float64) {
	tmp := make([]float64, len(x))
	copy(tmp, x)
	f0 := f(x)
	for i := range x {
		g[i] = partialDiff(f, x, tmp, i, f0)
	}
}

// partialDiff is the partial derivative of f at x along coordinate i, see
// numGrad.  f0 = f(x); tmp is scratch space equal to x on entry and exit.
func partialDiff(f func([]float64) float64, x, tmp []float64, i int, f0 float64) float64 {
	const h = 1e-6
	step := h * (1 + math.Abs(x[i]))
	for try := 0; try < 40; try++ {
		tmp[i] = x[i] + step
		fp := f(tmp)
		tmp[i] = x[i] - step
		fm := f(tmp)
		tmp[i] = x[i]
		if f0 >= bigPenalty {
			// x is itself outside the domain, where the objective is a
			// smooth bowl (see refineMLE): a central difference is right.
			return (fp - fm) / (2 * step)
		}
		okPlus, okMinus := fp < bigPenalty, fm < bigPenalty
		switch {
		case okPlus && okMinus:
			return (fp - fm) / (2 * step)
		case okPlus:
			return (fp - f0) / step
		case okMinus:
			return (f0 - fm) / step
		}
		step *= 0.5
	}
	return 0
}

// refinement is the outcome of one L-BFGS run: the best feasible point the
// objective was evaluated at, its log-likelihood (-Inf if there is none), and
// whether that point is a stationary point of the likelihood.
type refinement struct {
	v          []float64
	ll         float64
	stationary bool
}

// stationaryTol is the largest gradient norm of the negative log-likelihood
// accepted as "stationary".  A converged fit has a norm of order 1e-6 or less
// (the finite-difference gradient has a noise floor of that order), whereas a
// run that ends against the xi = -1 limit, where the likelihood is still
// rising, has a norm of tens.  Both scale with the number of observations, so
// the tolerance grows with n.
func stationaryTol(n int) float64 {
	return 1e-3 * math.Max(1, float64(n)/1000)
}

func finiteLL(l float64) bool { return !math.IsNaN(l) && !math.IsInf(l, 0) }

// refineMLE maximises the log-likelihood ll from v0 with L-BFGS and returns
// the best feasible point it evaluated.  inDomain restricts the parameters
// beyond what ll already does (a point with a non-finite log-likelihood is
// always infeasible).
//
// optim.LBFGS returns only its last iterate, and that iterate can be worse
// than one it passed through: after converging, the noise in the finite
// -difference gradient can produce a degenerate curvature pair and a wild
// step.  The best feasible point is therefore tracked inside the objective.
//
// Outside the domain the objective is bigPenalty plus the squared distance to
// v0, a bowl rather than a flat plateau: on a plateau the numerical gradient
// is exactly zero and L-BFGS reports convergence wherever it lands.
func refineMLE(ll func([]float64) float64, inDomain func([]float64) bool, v0 []float64, n int) refinement {
	bowl := func(v []float64) float64 {
		d := 0.0
		for i := range v {
			e := v[i] - v0[i]
			d += e * e
		}
		if !(d <= bigPenalty) { // also catches NaN
			d = bigPenalty
		}
		return bigPenalty + d
	}
	value := func(v []float64) (obj, l float64) {
		if inDomain(v) {
			if l = ll(v); finiteLL(l) {
				return -l, l
			}
		}
		return bowl(v), math.Inf(-1)
	}

	best := refinement{ll: math.Inf(-1)}
	tracked := func(v []float64) float64 {
		obj, l := value(v)
		if l > best.ll {
			best.ll = l
			best.v = append(best.v[:0], v...)
		}
		return obj
	}
	optim.LBFGS(tracked, func(x, g []float64) { numGrad(tracked, x, g) }, v0, 6, 200, 1e-8)
	if best.v == nil {
		return best
	}

	pure := func(v []float64) float64 { obj, _ := value(v); return obj }
	g := make([]float64, len(v0))
	numGrad(pure, best.v, g)
	norm := 0.0
	for _, gi := range g {
		norm += gi * gi
	}
	best.stationary = math.Sqrt(norm) <= stationaryTol(n)
	return best
}

// mleSearch maximises the log-likelihood ll over a packed parameter vector,
// confined to inDomain.  start is the closed-form estimate and fallback is a
// fit that is valid for any sample (the Gumbel or exponential limit), or nil
// if none can be formed.  It returns the vector to report, or nil when the
// closed-form estimate start itself is the answer (so the caller can return it
// unchanged, without a round trip through the parameterisation).
//
// In order of preference the answer is:
//  1. the stationary point reached from start, when start is a valid fit
//     (inside the domain, finite likelihood);
//  2. the stationary point reached from fallback, unless a valid start has
//     at least its likelihood;
//  3. start, when it is a valid fit (no interior maximum was found, so the
//     closed-form estimate stands);
//  4. the best finite-likelihood point the run from fallback evaluated.
func mleSearch(ll func([]float64) float64, inDomain func([]float64) bool, n int, start, fallback []float64) []float64 {
	startLL, startValid := math.Inf(-1), false
	if inDomain(start) {
		if l := ll(start); finiteLL(l) {
			startLL, startValid = l, true
		}
	}
	if startValid {
		if r := refineMLE(ll, inDomain, start, n); r.stationary {
			return r.v
		}
	}
	var r2 refinement
	if fallback != nil {
		r2 = refineMLE(ll, inDomain, fallback, n)
		if r2.stationary && !(startValid && startLL >= r2.ll) {
			return r2.v
		}
	}
	if startValid {
		return nil
	}
	return r2.v
}

// FitGEVMLE refines a GEV fit by maximum likelihood, deterministically: it
// starts from the L-moment estimate (FitGEVLMoments) and maximises the
// log-likelihood with L-BFGS over the reparameterisation (mu, log sigma, xi)
// so that sigma stays positive.  The shape is confined to xi > -1, because
// for xi <= -1 the likelihood is unbounded and no maximum exists.
//
// The refinement is guarded so that it cannot quietly return a worse or
// meaningless fit:
//   - the numerical gradient is one-sided where a central difference would
//     straddle the edge of the support;
//   - the best feasible point the optimiser evaluates is kept, since L-BFGS
//     can wander off after converging and returns only its last iterate;
//   - a refined point counts only if it is a stationary point of the
//     likelihood, so a run that ends against the xi = -1 limit is not taken
//     for a maximum;
//   - if the L-moment estimate has zero likelihood on the data (an
//     observation lies outside its support), or refinement from it does not
//     converge, the search is repeated from the Gumbel (xi = 0) L-moment
//     fit, which is valid for any sample.
//
// The result is the converged maximum-likelihood estimate whenever one is
// found.  When the likelihood has no interior maximum (it keeps rising toward
// xi = -1, which can happen for a small sample from a short-tailed
// distribution), the L-moment estimate is returned unchanged if it is a valid
// fit, and otherwise the best finite-likelihood point found.  The result
// therefore never has zero likelihood on the data and is never worse than a
// valid L-moment estimate.  A caller that must tell the L-moment fallback
// from a maximum-likelihood fit can compare the result with FitGEVLMoments:
// equality means the fallback was returned.
//
// ok == false only if the L-moment start itself cannot be formed.
//
// Reference: Coles (2001) §3.3.2 (numerical MLE of the GEV); Smith (1985) for
// the unbounded likelihood at xi <= -1.  Fixed starts from L-moments make the
// optimisation reproducible (no random restarts).
func FitGEVMLE(blockMaxima []float64) (GEVParams, bool) {
	start, ok := FitGEVLMoments(blockMaxima)
	if !ok {
		return GEVParams{}, false
	}
	ll := func(v []float64) float64 {
		return GEVLogLik(blockMaxima, GEVParams{Mu: v[0], Sigma: math.Exp(v[1]), Xi: v[2]})
	}
	inDomain := func(v []float64) bool { return v[2] > xiFloor }

	// The Gumbel L-moment fit: valid for every sample (its support is the
	// whole real line).
	var fallback []float64
	if l1, l2, _, ok := LMoments3(blockMaxima); ok && l2 > 0 {
		sigma := l2 / math.Ln2
		fallback = []float64{l1 - eulerMascheroni*sigma, math.Log(sigma), 0}
	}

	v := mleSearch(ll, inDomain, len(blockMaxima), []float64{start.Mu, math.Log(start.Sigma), start.Xi}, fallback)
	if v == nil {
		return start, true
	}
	return GEVParams{Mu: v[0], Sigma: math.Exp(v[1]), Xi: v[2]}, true
}

// FitGPDMLE refines a GPD fit by maximum likelihood, deterministically: it
// starts from the PWM estimate (FitGPDPWM) and maximises the log-likelihood
// with L-BFGS over (log sigma, xi).  The shape is confined to xi > -1,
// because for xi <= -1 the likelihood is unbounded and no maximum exists.
//
// The refinement is guarded exactly as in FitGEVMLE: a one-sided numerical
// gradient at the edge of the support, the best feasible point kept, a
// refined point accepted only if it is a stationary point of the likelihood,
// and a second search from the exponential (xi = 0) fit when the PWM estimate
// has zero likelihood on the data (an exceedance lies beyond its finite upper
// endpoint -sigma/xi) or refinement from it does not converge.
//
// The result is the converged maximum-likelihood estimate whenever one is
// found.  When the likelihood has no interior maximum, the PWM estimate is
// returned unchanged if it is a valid fit, and otherwise the best
// finite-likelihood point found; the result never has zero likelihood on the
// data and is never worse than a valid PWM estimate.  A caller that must tell
// the PWM fallback from a maximum-likelihood fit can compare the result with
// FitGPDPWM: equality means the fallback was returned.
//
// ok == false only if the PWM start cannot be formed.
//
// Reference: Coles (2001) §4.3.2; Grimshaw (1993) for the well-posedness of
// GPD MLE; Smith (1985) for the unbounded likelihood at xi <= -1.  Fixed PWM
// start => reproducible optimisation.
func FitGPDMLE(exceedances []float64) (GPDParams, bool) {
	start, ok := FitGPDPWM(exceedances)
	if !ok {
		return GPDParams{}, false
	}
	ll := func(v []float64) float64 {
		return GPDLogLik(exceedances, GPDParams{Sigma: math.Exp(v[0]), Xi: v[1]})
	}
	inDomain := func(v []float64) bool { return v[1] > xiFloor }

	// The exponential fit (xi = 0, sigma = mean): valid for every sample of
	// non-negative exceedances.
	var fallback []float64
	mean := 0.0
	for _, y := range exceedances {
		mean += y
	}
	mean /= float64(len(exceedances))
	if mean > 0 {
		fallback = []float64{math.Log(mean), 0}
	}

	v := mleSearch(ll, inDomain, len(exceedances), []float64{math.Log(start.Sigma), start.Xi}, fallback)
	if v == nil {
		return start, true
	}
	return GPDParams{Sigma: math.Exp(v[0]), Xi: v[1]}, true
}
