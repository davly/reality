package garch

import (
	"errors"
	"math"
)

// FitConfig configures the MLE calibration of GARCH(1,1) parameters.
type FitConfig struct {
	// MaxIter caps the gradient-descent iteration count.  Default 500.
	MaxIter int

	// LearningRate is the step size in unconstrained coordinate space.
	// Default 0.05.
	LearningRate float64

	// AbsTol terminates when the infinity-norm of the parameter update is
	// below AbsTol.  Default 1e-7.
	AbsTol float64

	// TikhonovLambda is the L2 regularisation strength on the
	// unconstrained reparameterisation.  Default 1e-4.  Per PLAN_RISKS.md
	// R3 mitigation: GARCH calibration is ill-posed; small Tikhonov
	// regularisation stabilises convergence.
	TikhonovLambda float64
}

// FitResult reports the outcome of MLE calibration.
type FitResult struct {
	Iter        int
	Converged   bool
	FinalLogLik float64
}

// Fit calibrates GARCH(1,1) parameters by Tikhonov-regularised maximum
// likelihood estimation against a residual series eps.  Uses unconstrained
// reparameterisation
//
//	omega = exp(theta_omega)
//	(alpha, beta, slack) = softmax(theta_a, theta_b, theta_s)
//
// to enforce omega > 0 and alpha + beta < 1 at every step.  Gradients
// are computed analytically (closed form for the GARCH recursion) and
// stabilised by adding lambda * theta^T theta to the negative log-
// likelihood objective.
//
// Initial guess defaults to a typical financial-returns calibration
// (omega = 1e-6, alpha = 0.05, beta = 0.90) when the receiver model has
// invalid parameters; otherwise the receiver is the warm-start.
//
// Reference: Bollerslev 1986; Francq-Zakoian 2010 ch. 7 (consistency of
// QMLE for GARCH).  Tikhonov stabilisation per Engle 2009 (Anticipating
// Correlations) §3.4.
func Fit(eps []float64, init Model, cfg FitConfig) (Model, FitResult, error) {
	if len(eps) < 50 {
		return Model{}, FitResult{}, errors.New("garch: Fit requires at least 50 residuals")
	}
	maxIter := cfg.MaxIter
	if maxIter == 0 {
		maxIter = 500
	}
	lr := cfg.LearningRate
	if !(lr > 0) {
		lr = 0.05
	}
	tol := cfg.AbsTol
	if !(tol > 0) {
		tol = 1e-7
	}
	tikh := cfg.TikhonovLambda
	if tikh == 0 {
		tikh = 1e-4
	}

	// Default warm-start if init is invalid.
	if err := init.Validate(); err != nil {
		init = Model{Omega: 1e-6, Alpha: 0.05, Beta: 0.90}
		init.UncondVar = init.Omega / (1 - init.Alpha - init.Beta)
	}

	// Reparameterise to unconstrained theta = (theta_omega, theta_a,
	// theta_b, theta_s).  Inverse map:
	//   theta_omega = log(omega)
	//   alpha = softmax_1; beta = softmax_2; slack = softmax_3.
	// Choose theta_a, theta_b, theta_s so that the softmax matches init
	// up to a free additive constant; we set theta_s = 0 and back out
	// theta_a = log(alpha / slack), theta_b = log(beta / slack).
	slack := 1.0 - init.Alpha - init.Beta
	if slack <= 0 {
		slack = 0.05
	}
	theta := [4]float64{
		math.Log(init.Omega),
		math.Log(init.Alpha / slack),
		math.Log(init.Beta / slack),
		0.0,
	}

	var converged bool
	var finalLL float64
	var lastDelta float64
	iters := maxIter
	for k := 0; k < maxIter; k++ {
		m := unpack(theta)
		ll, grad, err := negLogLikGrad(eps, m, theta, tikh)
		if err != nil {
			return Model{}, FitResult{}, err
		}
		finalLL = -ll
		// theta update.
		var delta [4]float64
		for i := 0; i < 4; i++ {
			delta[i] = lr * grad[i]
			theta[i] -= delta[i]
		}
		// L-inf norm of the update.
		var d float64
		for i := 0; i < 4; i++ {
			if a := math.Abs(delta[i]); a > d {
				d = a
			}
		}
		lastDelta = d
		if d < tol {
			converged = true
			iters = k + 1
			break
		}
	}
	_ = lastDelta
	final := unpack(theta)
	final.UncondVar = final.Omega / (1.0 - final.Alpha - final.Beta)
	return final, FitResult{Iter: iters, Converged: converged, FinalLogLik: finalLL}, nil
}

// unpack maps unconstrained theta to a GARCH Model.
func unpack(theta [4]float64) Model {
	omega := math.Exp(theta[0])
	// Softmax over (theta_a, theta_b, theta_s).
	maxT := theta[1]
	for _, v := range theta[2:] {
		if v > maxT {
			maxT = v
		}
	}
	ea := math.Exp(theta[1] - maxT)
	eb := math.Exp(theta[2] - maxT)
	es := math.Exp(theta[3] - maxT)
	z := ea + eb + es
	alpha := ea / z
	beta := eb / z
	m := Model{Omega: omega, Alpha: alpha, Beta: beta}
	m.UncondVar = omega / (1.0 - alpha - beta)
	return m
}

// negLogLikGrad returns the negative log-likelihood plus the Tikhonov
// penalty (function value) and its gradient with respect to theta.
//
// Gradient is computed by the standard analytic GARCH recursion:
//
//	d sigma^2_t / d omega = 1 + beta * d sigma^2_{t-1} / d omega
//	d sigma^2_t / d alpha = eps^2_{t-1} + beta * d sigma^2_{t-1} / d alpha
//	d sigma^2_t / d beta  = sigma^2_{t-1} + beta * d sigma^2_{t-1} / d beta
//
// then chain-ruled through the softmax + exp reparameterisation.
func negLogLikGrad(eps []float64, m Model, theta [4]float64, tikh float64) (float64, [4]float64, error) {
	n := len(eps)
	const log2pi = 1.8378770664093454835606594728112
	if err := m.Validate(); err != nil {
		// Return a high penalty + zero gradient so the next step
		// effectively ignores this point and the previous theta is
		// retained (gradient descent's update will go nowhere).  This is
		// crude but sufficient for the Tikhonov-regularised MLE.
		return math.Inf(1), [4]float64{}, nil
	}
	// Forward filter + accumulate MEAN log-likelihood + its gradient
	// w.r.t. (omega, alpha, beta).  Working with the mean (rather than
	// the sum) makes the learning rate independent of the sample size.
	var nll float64
	var dOmega, dAlpha, dBeta float64
	prevS2 := m.UncondVar
	prevEps := 0.0
	dS2_dOmega := 0.0
	dS2_dAlpha := 0.0
	dS2_dBeta := 0.0
	for _, e := range eps {
		s2 := m.Omega + m.Alpha*prevEps*prevEps + m.Beta*prevS2
		// Sensitivities propagate through the recursion.
		newDOmega := 1.0 + m.Beta*dS2_dOmega
		newDAlpha := prevEps*prevEps + m.Beta*dS2_dAlpha
		newDBeta := prevS2 + m.Beta*dS2_dBeta

		nll += 0.5 * (log2pi + math.Log(s2) + e*e/s2)
		dNllDs2 := 0.5 * (1.0/s2 - e*e/(s2*s2))
		dOmega += dNllDs2 * newDOmega
		dAlpha += dNllDs2 * newDAlpha
		dBeta += dNllDs2 * newDBeta

		dS2_dOmega = newDOmega
		dS2_dAlpha = newDAlpha
		dS2_dBeta = newDBeta
		prevS2 = s2
		prevEps = e
	}
	invN := 1.0 / float64(n)
	nll *= invN
	dOmega *= invN
	dAlpha *= invN
	dBeta *= invN

	// Now chain-rule (omega, alpha, beta) -> theta.
	// d omega / d theta_omega = omega.
	// d (alpha, beta, slack) / d theta_{a,b,s} = softmax derivatives.
	//   d alpha / d theta_a = alpha (1 - alpha)
	//   d alpha / d theta_b = -alpha * beta
	//   d alpha / d theta_s = -alpha * slack
	//   etc.
	slack := 1.0 - m.Alpha - m.Beta
	gOmega := m.Omega * dOmega
	gA := dAlpha*m.Alpha*(1.0-m.Alpha) - dBeta*m.Alpha*m.Beta
	gB := dAlpha*(-m.Alpha*m.Beta) + dBeta*m.Beta*(1.0-m.Beta)
	gS := dAlpha*(-m.Alpha*slack) + dBeta*(-m.Beta*slack)

	// Tikhonov: 0.5 lambda ||theta||^2 -> gradient = lambda * theta.
	gOmega += tikh * theta[0]
	gA += tikh * theta[1]
	gB += tikh * theta[2]
	gS += tikh * theta[3]

	// Likelihood penalty: also add the Tikhonov term to nll for monitoring.
	nll += 0.5 * tikh * (theta[0]*theta[0] + theta[1]*theta[1] + theta[2]*theta[2] + theta[3]*theta[3])
	return nll, [4]float64{gOmega, gA, gB, gS}, nil
}
