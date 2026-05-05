package garch

import (
	"math"
	"testing"

	"github.com/davly/reality/autodiff"
)

// TestNegLogLikGrad_AutodiffEquivalence pins the hand-rolled analytic
// gradient in negLogLikGrad against the same gradient computed by
// reverse-mode autodiff over the same forward graph. The two MUST agree
// to within numerical precision (1e-9) at the same theta point.
//
// This is the first-consumer push for github.com/davly/reality/autodiff
// AND github.com/davly/reality/timeseries/garch. autodiff/doc.go names
// "GARCH / DCC-GARCH calibration (ill-posed; needs Tikhonov-regularised
// Newton-CG with adjoint gradients)" as the first candidate use case;
// garch/doc.go names "Tikhonov-regularised MLE using autodiff-supplied
// gradients" as the Fit method's contract. Until this test landed, both
// packages were substrate-first with zero ecosystem consumers — the
// closed-form gradient in negLogLikGrad and the autodiff package were
// independently correct but not pinned to each other.
//
// Pinning them here:
//   - falsifies the "autodiff is broken on real inputs" hypothesis
//   - falsifies the "negLogLikGrad has a sign error" hypothesis
//   - establishes the substrate equivalence claimed in fit.go's doc-comment
//   - opens the path to replacing the analytic derivation with the
//     autodiff version when a future calibration target (DCC-GARCH,
//     Heston, SABR) needs gradients that are too painful to derive by
//     hand
//
// The analytic gradient stays in production for speed (no tape allocation
// per Fit iteration); the autodiff path stays in tests as the parity
// witness.
func TestNegLogLikGrad_AutodiffEquivalence(t *testing.T) {
	// Synthesize a deterministic residual series via a Lehmer LCG.
	const n = 200
	eps := make([]float64, n)
	prevEps2 := 0.0
	prevS2 := 1.0
	rng := uint64(0xdeadbeef)
	for i := range eps {
		rng = rng*6364136223846793005 + 1442695040888963407
		// Uniform in [-1, 1].
		u := float64(rng>>32) / 4294967296.0
		z := 2*u - 1
		s2 := 1e-6 + 0.05*prevEps2 + 0.90*prevS2
		eps[i] = z * math.Sqrt(s2)
		prevEps2 = eps[i] * eps[i]
		prevS2 = s2
	}

	// Choose theta such that unpack(theta) gives a known (omega, alpha, beta).
	// Want omega = 1e-6, alpha = 0.05, beta = 0.90, slack = 0.05.
	theta := [4]float64{
		math.Log(1e-6),       // theta_omega
		math.Log(0.05 / 0.05), // theta_a (alpha / slack)
		math.Log(0.90 / 0.05), // theta_b (beta / slack)
		0.0,                   // theta_s (slack reference, set to 0)
	}
	const tikh = 1e-4

	// (1) Hand-rolled analytic path.
	mHand := unpack(theta)
	llHand, gradHand, err := negLogLikGrad(eps, mHand, theta, tikh)
	if err != nil {
		t.Fatalf("negLogLikGrad: %v", err)
	}

	// (2) Autodiff-built path.
	llAuto, gradAuto := negLogLikGradAutodiff(eps, theta, tikh)

	// Function-value parity.
	if d := math.Abs(llHand - llAuto); d > 1e-9 {
		t.Errorf("nll mismatch: hand=%g auto=%g |delta|=%g", llHand, llAuto, d)
	}

	// Gradient parity per coordinate.
	names := [4]string{"theta_omega", "theta_a", "theta_b", "theta_s"}
	for i := 0; i < 4; i++ {
		d := math.Abs(gradHand[i] - gradAuto[i])
		if d > 1e-9 {
			t.Errorf("grad[%s] mismatch: hand=%g auto=%g |delta|=%g",
				names[i], gradHand[i], gradAuto[i], d)
		}
	}
}

// negLogLikGradAutodiff is the autodiff-built equivalent of negLogLikGrad.
// Constructs the GARCH(1,1) negative-mean-log-likelihood plus Tikhonov
// penalty as an autodiff tape over the unconstrained theta parameters,
// then reverses to extract the gradient. Returns the function value and
// the gradient with respect to (theta_omega, theta_a, theta_b, theta_s) —
// the same coordinate basis as the analytic path.
//
// Reparameterisation matches Fit / unpack:
//
//	omega                       = exp(theta_omega)
//	(alpha, beta, slack)        = softmax(theta_a, theta_b, theta_s)
//	uncondVar                   = omega / (1 - alpha - beta)
//
// The forward filter is the same recursion as negLogLikGrad's:
//
//	sigma^2_t = omega + alpha * eps^2_{t-1} + beta * sigma^2_{t-1}
//
// with sigma^2_0 = uncondVar and eps_0 = 0, accumulating the mean of
// 0.5 * (log(2*pi) + log(sigma^2_t) + eps^2_t / sigma^2_t).
func negLogLikGradAutodiff(eps []float64, theta [4]float64, tikh float64) (float64, [4]float64) {
	tape := autodiff.NewTape()

	tOmega := tape.Var(theta[0])
	tA := tape.Var(theta[1])
	tB := tape.Var(theta[2])
	tS := tape.Var(theta[3])

	// Numerically-stable softmax: subtract the max before exp.
	maxT := theta[1]
	if theta[2] > maxT {
		maxT = theta[2]
	}
	if theta[3] > maxT {
		maxT = theta[3]
	}
	eA := autodiff.Exp(autodiff.AddConst(tA, -maxT))
	eB := autodiff.Exp(autodiff.AddConst(tB, -maxT))
	eS := autodiff.Exp(autodiff.AddConst(tS, -maxT))
	z := autodiff.Add(autodiff.Add(eA, eB), eS)
	alpha := autodiff.Div(eA, z)
	beta := autodiff.Div(eB, z)
	omega := autodiff.Exp(tOmega)

	// Initial unconditional variance is treated as a CONSTANT (not
	// differentiated through), matching negLogLikGrad's choice on lines
	// 184-186 of fit.go where dS2_dOmega/dAlpha/dBeta start at 0. This
	// simplification ignores the gradient path through the initial
	// condition, which is standard for QMLE on long series since the
	// initial-condition contribution decays geometrically with beta < 1.
	mForUncond := unpack(theta)
	uncondVarConst := tape.Constant(mForUncond.UncondVar)

	// Filter loop: s2_t = omega + alpha * prevEpsSq + beta * prevS2.
	const log2pi = 1.8378770664093454835606594728112
	prevS2 := uncondVarConst
	prevEpsSq := tape.Constant(0.0)
	nllSum := tape.Constant(0.0)
	for _, e := range eps {
		alphaTerm := autodiff.Mul(alpha, prevEpsSq)
		betaTerm := autodiff.Mul(beta, prevS2)
		s2 := autodiff.Add(autodiff.Add(omega, alphaTerm), betaTerm)

		logS2 := autodiff.Log(s2)
		eSq := tape.Constant(e * e)
		quad := autodiff.Div(eSq, s2)
		// nllT = 0.5 * (log2pi + log(s2) + eSq/s2)
		inner := autodiff.AddConst(autodiff.Add(logS2, quad), log2pi)
		nllT := autodiff.MulConst(inner, 0.5)
		nllSum = autodiff.Add(nllSum, nllT)

		prevS2 = s2
		prevEpsSq = tape.Constant(e * e)
	}

	invN := 1.0 / float64(len(eps))
	nllMean := autodiff.MulConst(nllSum, invN)

	// Tikhonov: 0.5 * tikh * sum(theta_i^2).
	tOmegaSq := autodiff.Mul(tOmega, tOmega)
	tASq := autodiff.Mul(tA, tA)
	tBSq := autodiff.Mul(tB, tB)
	tSSq := autodiff.Mul(tS, tS)
	tikhSum := autodiff.Add(autodiff.Add(autodiff.Add(tOmegaSq, tASq), tBSq), tSSq)
	tikhTerm := autodiff.MulConst(tikhSum, 0.5*tikh)

	out := autodiff.Add(nllMean, tikhTerm)

	gradAll := tape.Backward(out)

	return out.Val, [4]float64{
		gradAll[tOmega.ID],
		gradAll[tA.ID],
		gradAll[tB.ID],
		gradAll[tS.ID],
	}
}
