package mdl

import "math"

// GaussianCodeLength returns the codelength in nats of a sample
// vector under a fixed Gaussian hypothesis with the supplied
// hypothesis mean and standard deviation:
//
//	L(x | mu, sigma) = sum_i (-log phi(x_i | mu, sigma))
//	                 = (n/2) * log(2*pi*sigma^2)
//	                 + (1 / (2*sigma^2)) * sum_i (x_i - mu)^2
//
// where phi is the Gaussian density.  This is the "data given the
// model" half of the two-part MDL codelength; for the full
// codelength under a parametric class with model selection, add the
// parametric-complexity NML regret + the universal-integer-coded
// description of the parameter precision.
//
// hypothesisStdev must be strictly positive.  For zero or negative
// stdev the function returns +Inf (the data has zero codelength
// only if it lies exactly on the mean; a regularising codelength
// model never emits zero stdev).
//
// References:
//   - Cover, T. M. & Thomas, J. A. (2006).  Elements of Information
//     Theory, 2nd ed., John Wiley & Sons, §8.
//   - Grünwald, P. D. (2007).  The Minimum Description Length
//     Principle.  MIT Press, §6.
func GaussianCodeLength(samples []float64, hypothesisMean, hypothesisStdev float64) float64 {
	if hypothesisStdev <= 0 || math.IsNaN(hypothesisStdev) || math.IsInf(hypothesisStdev, 0) {
		return math.Inf(1)
	}
	if math.IsNaN(hypothesisMean) || math.IsInf(hypothesisMean, 0) {
		return math.Inf(1)
	}
	n := len(samples)
	if n == 0 {
		return 0
	}
	variance := hypothesisStdev * hypothesisStdev
	ssr := 0.0
	for _, x := range samples {
		d := x - hypothesisMean
		ssr += d * d
	}
	const logTwoPi = 1.8378770664093454836 // math.Log(2*math.Pi)
	return 0.5*float64(n)*(logTwoPi+math.Log(variance)) + ssr/(2*variance)
}

// ModelCodeLength returns the BIC-shape codelength term for the
// model itself in nats:
//
//	L(model) = (numParams / 2) * log(sampleSize)
//
// This is the asymptotic Laplace approximation that BIC uses;
// equivalent to NML's parametric-complexity regret to leading order
// when the regularity conditions hold (Grünwald 2007 §8).  Use this
// as a fast falsifiable adapter over BIC scores; use NMLMultinomial
// for the principled non-asymptotic value.
//
// Returns 0 if numParams <= 0 or sampleSize <= 1 (the BIC-term is
// asymptotic and degenerates on tiny samples).
//
// References:
//   - Schwarz, G. (1978).  Estimating the dimension of a model.
//     Annals of Statistics 6(2): 461-464.
//   - Grünwald, P. D. (2007).  The Minimum Description Length
//     Principle.  MIT Press, §8.
func ModelCodeLength(numParams int, sampleSize int) float64 {
	if numParams <= 0 || sampleSize <= 1 {
		return 0
	}
	return 0.5 * float64(numParams) * math.Log(float64(sampleSize))
}

// BICShape returns the full BIC score in nats:
//
//	BIC = -2 * log L(x | hat-theta) + numParams * log(n)
//
// scaled to the codelength convention (BIC / 2).  This is the
// adapter for legacy AIC/BIC scalar consumers — RubberDuck's
// `OptimalLag`, Simulacra's calibration `bic`, Causal-engine's
// `local_score_BIC`, etc.  The recommendation when porting these
// sites is to swap BICShape -> NMLMultinomial / NMLBernoulli for
// the principled non-asymptotic value when regularity is suspect
// (boundary case, finite n, multi-modal likelihood); BICShape is
// the same-shape replacement when callers want a 1-to-1 swap.
//
// negLogLikelihood is the negative log-likelihood at the MLE in
// nats.  numParams is the number of free parameters in the model.
// sampleSize is the number of observations.
//
// Returns +Inf if negLogLikelihood is non-finite.
func BICShape(negLogLikelihood float64, numParams, sampleSize int) float64 {
	if math.IsNaN(negLogLikelihood) || math.IsInf(negLogLikelihood, 0) {
		return math.Inf(1)
	}
	return negLogLikelihood + ModelCodeLength(numParams, sampleSize)
}

// AICShape returns the full AIC score in nats:
//
//	AIC = -2 * log L(x | hat-theta) + 2 * numParams
//
// scaled to the codelength convention (AIC / 2).  Akaike's
// information criterion has no asymptotic-Laplace shape — it is
// asymptotically equivalent to leave-one-out cross-validation for
// the squared-error case (Stone 1977).  As a model-selection adapter
// AIC is the canonical alternative to BIC; both are asymptotic
// approximations that NML supersedes in the finite-sample regime.
//
// Returns +Inf if negLogLikelihood is non-finite.
//
// References:
//   - Akaike, H. (1974).  A new look at the statistical model
//     identification.  IEEE Trans. Automatic Control 19(6): 716-723.
//   - Stone, M. (1977).  An asymptotic equivalence of choice of model
//     by cross-validation and Akaike's criterion.  J. Royal Statistical
//     Society B 39(1): 44-47.
func AICShape(negLogLikelihood float64, numParams int) float64 {
	if math.IsNaN(negLogLikelihood) || math.IsInf(negLogLikelihood, 0) {
		return math.Inf(1)
	}
	if numParams < 0 {
		numParams = 0
	}
	return negLogLikelihood + float64(numParams)
}
