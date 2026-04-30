package changepoint

import (
	"errors"
	"math"
)

// DefaultRMax is the default truncation length for the run-length posterior.
// 500 timesteps is sufficient for most finance-cadence applications (daily
// data covers ~2 years, intraday covers ~hour at 1-min cadence) and bounds
// per-step compute at O(R_max). Increase for very long historical regimes.
const DefaultRMax = 500

// DefaultLambda is the default constant hazard rate. With H(r) = 1/lambda,
// lambda = 250 corresponds to one expected change-point per ~year of daily
// data. Caller should override this for their cadence.
const DefaultLambda = 250.0

// NigPrior holds the four hyper-parameters of the Normal-Inverse-Gamma prior
// on (mu, sigma^2):
//
//	mu | sigma^2 ~ Normal(Mu0, sigma^2 / Kappa0)
//	sigma^2      ~ InverseGamma(Alpha0, Beta0)
//
// The posterior predictive p(x | hyperparams) is Student's-t with df = 2*alpha,
// location mu, and scale sqrt(beta * (kappa + 1) / (alpha * kappa)).
//
// # Choosing priors
//
// Mu0    — prior expected mean. For zero-mean returns: 0. For a known
//
//	regime mean: that mean.
//
// Kappa0 — prior precision on Mu0 in units of effective observations.
//
//	Small (e.g., 1.0) means weak prior. Large means strong prior.
//
// Alpha0 — prior shape on sigma^2.   Small (e.g., 1.0) means weak prior.
// Beta0  — prior rate on sigma^2.    With Alpha0 = 1, prior expected
//
//	sigma^2 = Beta0. Pick to match your data scale.
//
// For weakly-informative defaults on standardised data: Mu0=0, Kappa0=1,
// Alpha0=1, Beta0=1.
type NigPrior struct {
	Mu0    float64
	Kappa0 float64
	Alpha0 float64
	Beta0  float64
}

// DefaultNigPrior returns a weakly-informative prior suitable for unit-variance
// data: Mu0=0, Kappa0=1, Alpha0=1, Beta0=1.
func DefaultNigPrior() NigPrior {
	return NigPrior{Mu0: 0.0, Kappa0: 1.0, Alpha0: 1.0, Beta0: 1.0}
}

// Validate returns an error if any hyper-parameter is non-positive (where
// positivity is required) or NaN/Inf.
func (p NigPrior) Validate() error {
	if math.IsNaN(p.Mu0) || math.IsInf(p.Mu0, 0) {
		return errors.New("changepoint: NigPrior.Mu0 must be finite")
	}
	if !(p.Kappa0 > 0) || math.IsInf(p.Kappa0, 0) {
		return errors.New("changepoint: NigPrior.Kappa0 must be positive and finite")
	}
	if !(p.Alpha0 > 0) || math.IsInf(p.Alpha0, 0) {
		return errors.New("changepoint: NigPrior.Alpha0 must be positive and finite")
	}
	if !(p.Beta0 > 0) || math.IsInf(p.Beta0, 0) {
		return errors.New("changepoint: NigPrior.Beta0 must be positive and finite")
	}
	return nil
}

// Bocpd is the streaming Bayesian Online Change-Point Detection state.
// It maintains a run-length posterior P[r] and the per-run-length sufficient
// statistics (mu_r, kappa_r, alpha_r, beta_r) of the Normal-Inverse-Gamma
// observation model.
//
// Use New to construct. Call Update for each observation. Call RunLengthPosterior
// or ChangePointProbability to query state.
//
// Bocpd is not safe for concurrent use. Wrap in a mutex if shared.
type Bocpd struct {
	prior    NigPrior
	rMax     int
	lambda   float64 // constant hazard rate; H(r) = 1/lambda
	t        int     // observations seen so far

	// Run-length posterior. p[r] = P(r_t = r | x_{1:t}).
	// Length is min(t+1, rMax+1). Always sums to 1 after Update.
	p []float64

	// Per-run-length Normal-Inverse-Gamma posterior parameters. Index r holds
	// the parameters that produced the predictive density used at the most
	// recent step for hypothesis r.
	mu, kappa, alpha, beta []float64
}

// Config configures a Bocpd. RMax bounds the run-length truncation; Lambda is
// the constant hazard rate (one expected change every ~Lambda timesteps).
type Config struct {
	Prior  NigPrior
	RMax   int
	Lambda float64
}

// DefaultConfig returns a sensible default configuration for unit-variance
// daily-cadence financial data with one expected regime change per year.
func DefaultConfig() Config {
	return Config{
		Prior:  DefaultNigPrior(),
		RMax:   DefaultRMax,
		Lambda: DefaultLambda,
	}
}

// New constructs a Bocpd with the given configuration. Returns an error if
// any field is invalid.
func New(cfg Config) (*Bocpd, error) {
	if err := cfg.Prior.Validate(); err != nil {
		return nil, err
	}
	if cfg.RMax < 1 {
		return nil, errors.New("changepoint: Config.RMax must be >= 1")
	}
	if !(cfg.Lambda > 0) || math.IsInf(cfg.Lambda, 0) {
		return nil, errors.New("changepoint: Config.Lambda must be positive and finite")
	}
	b := &Bocpd{
		prior:  cfg.Prior,
		rMax:   cfg.RMax,
		lambda: cfg.Lambda,
	}
	// Initial state: P(r_0 = 0) = 1 with prior hyper-parameters.
	b.p = []float64{1.0}
	b.mu = []float64{cfg.Prior.Mu0}
	b.kappa = []float64{cfg.Prior.Kappa0}
	b.alpha = []float64{cfg.Prior.Alpha0}
	b.beta = []float64{cfg.Prior.Beta0}
	return b, nil
}

// hazard returns H(r), the prior probability of a change-point given run-
// length r. Constant-hazard model: H(r) = 1/lambda for all r.
func (b *Bocpd) hazard(r int) float64 {
	_ = r
	return 1.0 / b.lambda
}

// studentT computes the log-density of x under a Student's-t with the given
// degrees-of-freedom, location, and scale parameters. Working in log-space
// avoids underflow when the predictive density is small (common at the start
// of a new regime).
func studentTLogPDF(x, df, loc, scale float64) float64 {
	z := (x - loc) / scale
	// log Gamma((df+1)/2) - log Gamma(df/2) - 0.5*log(df*pi) - log(scale)
	//   - ((df+1)/2) * log(1 + z^2/df)
	logCoef := lgamma((df+1)*0.5) - lgamma(df*0.5) - 0.5*math.Log(df*math.Pi) - math.Log(scale)
	logTail := -0.5 * (df + 1) * math.Log1p(z*z/df)
	return logCoef + logTail
}

func lgamma(x float64) float64 {
	v, _ := math.Lgamma(x)
	return v
}

// Update ingests a new observation x_t and returns the resulting run-length
// posterior P (a fresh slice; caller may retain). The slice has length
// min(t+1, rMax+1) where t is the number of observations seen including x.
//
// Update returns an error if x is not finite.
func (b *Bocpd) Update(x float64) ([]float64, error) {
	if math.IsNaN(x) || math.IsInf(x, 0) {
		return nil, errors.New("changepoint: observation x must be finite")
	}

	// Step 1: compute predictive log-density log pi_r = log p(x | hyperparams_r)
	// for each currently-tracked run-length r. The predictive is Student's-t
	// with df = 2*alpha_r, location mu_r, scale^2 = beta_r * (kappa_r+1) /
	// (alpha_r * kappa_r).
	n := len(b.p)
	logPi := make([]float64, n)
	for r := 0; r < n; r++ {
		df := 2 * b.alpha[r]
		scale2 := b.beta[r] * (b.kappa[r] + 1.0) / (b.alpha[r] * b.kappa[r])
		scale := math.Sqrt(scale2)
		logPi[r] = studentTLogPDF(x, df, b.mu[r], scale)
	}

	// Step 2: compute the joint distribution over r_t after seeing x.
	// Working in log-space until renormalisation to avoid underflow.
	//   growth: log P(r+1, x) = log P(r) + log pi_r + log(1 - H(r))
	//   reset:  log P(0, x)   = logsumexp_r [log P(r) + log pi_r + log H(r)]
	newLen := n + 1
	if newLen > b.rMax+1 {
		newLen = b.rMax + 1
	}
	newLogP := make([]float64, newLen)
	for i := range newLogP {
		newLogP[i] = math.Inf(-1)
	}

	// Growth.
	for r := 0; r < n; r++ {
		dst := r + 1
		if dst >= newLen {
			continue
		}
		h := b.hazard(r)
		logGrowth := math.Log(b.p[r]) + logPi[r] + math.Log(1.0-h)
		newLogP[dst] = logSumExp(newLogP[dst], logGrowth)
	}

	// Reset to r = 0 (a change-point happened just before x).
	for r := 0; r < n; r++ {
		h := b.hazard(r)
		logReset := math.Log(b.p[r]) + logPi[r] + math.Log(h)
		newLogP[0] = logSumExp(newLogP[0], logReset)
	}

	// Step 3: renormalise. Convert to linear space subtracting the max.
	maxLogP := math.Inf(-1)
	for _, v := range newLogP {
		if v > maxLogP {
			maxLogP = v
		}
	}
	if math.IsInf(maxLogP, -1) {
		return nil, errors.New("changepoint: numerical underflow in run-length posterior")
	}
	newP := make([]float64, newLen)
	var sum float64
	for i, v := range newLogP {
		newP[i] = math.Exp(v - maxLogP)
		sum += newP[i]
	}
	if !(sum > 0) {
		return nil, errors.New("changepoint: zero total mass after update")
	}
	for i := range newP {
		newP[i] /= sum
	}

	// Step 4: update Normal-Inverse-Gamma sufficient statistics.
	// For run-length r > 0 (continuation of an existing regime), update
	// (mu_{r-1}, kappa_{r-1}, alpha_{r-1}, beta_{r-1}) to (mu_r, kappa_r,
	// alpha_r, beta_r) by absorbing observation x:
	//   mu'    = (kappa*mu + x) / (kappa + 1)
	//   kappa' = kappa + 1
	//   alpha' = alpha + 0.5
	//   beta'  = beta + 0.5 * kappa * (x - mu)^2 / (kappa + 1)
	// For r = 0 (just had a change-point), reset to the prior.
	newMu := make([]float64, newLen)
	newKappa := make([]float64, newLen)
	newAlpha := make([]float64, newLen)
	newBeta := make([]float64, newLen)
	newMu[0] = b.prior.Mu0
	newKappa[0] = b.prior.Kappa0
	newAlpha[0] = b.prior.Alpha0
	newBeta[0] = b.prior.Beta0
	for r := 1; r < newLen; r++ {
		src := r - 1
		if src >= n {
			break
		}
		mu := b.mu[src]
		kappa := b.kappa[src]
		alpha := b.alpha[src]
		beta := b.beta[src]
		newMu[r] = (kappa*mu + x) / (kappa + 1.0)
		newKappa[r] = kappa + 1.0
		newAlpha[r] = alpha + 0.5
		dx := x - mu
		newBeta[r] = beta + 0.5*kappa*dx*dx/(kappa+1.0)
	}

	b.p = newP
	b.mu = newMu
	b.kappa = newKappa
	b.alpha = newAlpha
	b.beta = newBeta
	b.t++

	out := make([]float64, len(newP))
	copy(out, newP)
	return out, nil
}

// logSumExp returns log(exp(a) + exp(b)) computed in a numerically stable
// way. Defines log(exp(-inf) + exp(b)) = b.
func logSumExp(a, b float64) float64 {
	if math.IsInf(a, -1) {
		return b
	}
	if math.IsInf(b, -1) {
		return a
	}
	if a > b {
		return a + math.Log1p(math.Exp(b-a))
	}
	return b + math.Log1p(math.Exp(a-b))
}

// RunLengthPosterior returns a copy of the current run-length posterior. The
// returned slice has length min(t+1, rMax+1) and sums to 1.0 within float64
// precision.
func (b *Bocpd) RunLengthPosterior() []float64 {
	out := make([]float64, len(b.p))
	copy(out, b.p)
	return out
}

// ChangePointProbability returns P(r_t = 0 | x_{1:t}), the posterior
// probability that a change-point occurred at the most recent observation.
//
// Note: under a constant hazard rate H = 1/lambda, this quantity is
// algebraically equal to H at every step — the predictive likelihood cancels
// in the numerator and denominator. It is therefore *not* a useful alarm
// signal on its own; use ChangePointProbabilityWithin or MapRunLength to
// detect regime shifts. ChangePointProbability is exposed for reference and
// for non-constant-hazard variants where the cancellation does not occur.
func (b *Bocpd) ChangePointProbability() float64 {
	if len(b.p) == 0 {
		return 0
	}
	return b.p[0]
}

// ChangePointProbabilityWithin returns sum_{r < window} P(r_t = r), the
// posterior probability that a change-point occurred within the last `window`
// observations. This is the canonical alarm signal: when mass shifts down
// from high run-lengths to low ones (regime reset), this quantity spikes.
//
// Window must be >= 1. Window = 1 is equivalent to ChangePointProbability.
// Window > current posterior length returns 1.0.
func (b *Bocpd) ChangePointProbabilityWithin(window int) float64 {
	if window < 1 {
		return 0
	}
	if window > len(b.p) {
		window = len(b.p)
	}
	var s float64
	for r := 0; r < window; r++ {
		s += b.p[r]
	}
	return s
}

// MapRunLength returns the maximum-a-posteriori run-length argmax_r P(r_t = r).
func (b *Bocpd) MapRunLength() int {
	best := 0
	bestP := -1.0
	for r, p := range b.p {
		if p > bestP {
			bestP = p
			best = r
		}
	}
	return best
}

// ExpectedRunLength returns the posterior expected run-length E[r_t | x_{1:t}].
func (b *Bocpd) ExpectedRunLength() float64 {
	var ev float64
	for r, p := range b.p {
		ev += float64(r) * p
	}
	return ev
}

// CurrentRegimeMean returns the posterior mean of the latent mu under the
// MAP run-length hypothesis. Useful as a smoothed point estimate.
func (b *Bocpd) CurrentRegimeMean() float64 {
	r := b.MapRunLength()
	if r >= len(b.mu) {
		return b.prior.Mu0
	}
	return b.mu[r]
}

// CurrentRegimeVariance returns the posterior expected sigma^2 under the MAP
// run-length hypothesis. Posterior is Inverse-Gamma(alpha_r, beta_r) so
// E[sigma^2] = beta_r / (alpha_r - 1) for alpha_r > 1.
func (b *Bocpd) CurrentRegimeVariance() float64 {
	r := b.MapRunLength()
	if r >= len(b.alpha) {
		return b.prior.Beta0 / (b.prior.Alpha0 - 1.0)
	}
	if !(b.alpha[r] > 1.0) {
		// Posterior variance not yet defined; fall back to the rate.
		return b.beta[r]
	}
	return b.beta[r] / (b.alpha[r] - 1.0)
}

// Step returns the number of observations consumed so far.
func (b *Bocpd) Step() int { return b.t }
