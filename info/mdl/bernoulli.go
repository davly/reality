package mdl

import "math"

// NMLBernoulli returns the Normalised Maximum Likelihood (NML)
// regret in nats for a Bernoulli model with `successes` successes
// out of `trials` total observations.
//
// The NML regret of a parametric model class M for an observed
// sample x^n is
//
//	C(M, n) = log sum_{x'^n} P(x'^n | hat-theta(x'^n))
//
// where hat-theta is the maximum-likelihood estimator on the
// hypothetical sample x'^n.  For the Bernoulli class this reduces
// to a finite sum over k in {0, ..., n}:
//
//	C(Bern, n) = log sum_{k=0}^{n} C(n, k) * (k/n)^k * ((n-k)/n)^{n-k}
//
// The regret is independent of the actual observed sample — it is
// the *worst-case* coding-redundancy that NML guarantees against
// any other code over the same model class (Shtarkov 1987).  It is
// equivalent to the multinomial-NML at k = 2 categories (cf.
// NMLMultinomial) and is provided as a special case for the most
// common application.
//
// Returns ErrInvalidTrials if trials < 1 or successes > trials.
// Returns ErrNegativeInput if successes < 0.
//
// Reference: Shtarkov, Y. M. (1987).  Universal sequential coding
// of single messages.  Problems Inform. Transmission 23(3): 3-17.
// Grünwald, P. D. (2007).  The Minimum Description Length
// Principle.  MIT Press, §11.
func NMLBernoulli(successes, trials int) (float64, error) {
	if successes < 0 {
		return 0, ErrNegativeInput
	}
	if trials < 1 || successes > trials {
		return 0, ErrInvalidTrials
	}
	// Delegate to the multinomial NML with k = 2 categories.
	// The regret is independent of the actual (successes, trials)
	// values — only `trials` (the total count = sample size) is
	// needed.  We pass dummy counts that sum to `trials` so the
	// recursion can run; the value of the counts does not affect
	// the regret.
	counts := []int{successes, trials - successes}
	return NMLMultinomial(counts)
}

// BernoulliCodeLength returns the *total* MDL codelength in nats
// for a Bernoulli model fit to `successes` successes out of
// `trials` observations:
//
//	L(x^n) = -log P(x^n | hat-theta) + C(Bern, n)
//
// the negative-log-likelihood at the MLE plus the NML regret.
// This is the codelength a buyer-language regulator wants when
// comparing "Bernoulli with this success rate" against any
// alternative model (e.g. against a uniform-noise null).
//
// Returns ErrInvalidTrials if trials < 1 or successes > trials.
// Returns ErrNegativeInput if successes < 0.
func BernoulliCodeLength(successes, trials int) (float64, error) {
	if successes < 0 {
		return 0, ErrNegativeInput
	}
	if trials < 1 || successes > trials {
		return 0, ErrInvalidTrials
	}

	// Negative log-likelihood at the MLE.  The MLE for Bernoulli
	// is hat-p = successes / trials; the NLL at MLE is
	//
	//   -log L = -successes*log(p) - (trials-successes)*log(1-p)
	//
	// Edge cases: 0/n or n/n -> NLL = 0 (the data is perfectly
	// predicted by p = 0 or p = 1).
	nll := 0.0
	if successes > 0 && successes < trials {
		p := float64(successes) / float64(trials)
		nll = -float64(successes)*math.Log(p) - float64(trials-successes)*math.Log(1-p)
	}

	regret, err := NMLBernoulli(successes, trials)
	if err != nil {
		return 0, err
	}
	return nll + regret, nil
}
