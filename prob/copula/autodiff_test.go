package copula

import (
	"math"
	"testing"

	"github.com/davly/reality/autodiff"
)

// TestClaytonLogPDF_AutodiffGradientMatchesAnalytic closes the third
// consumer for the R-CLOSED-FORM-PINNED-TO-AUTODIFF pattern (after
// F2.a's autodiff × garch and S62-overnight's infogeo × autodiff).
// Saturates the R-rule to 3/3 — promotable to STANDARD.
//
// The Clayton bivariate copula log-density is
//
//	log c(u, v; θ) = log(1+θ)
//	               + (−1 − θ)·(log u + log v)
//	               + (−2 − 1/θ)·log S, with  S = u^(−θ) + v^(−θ) − 1
//
// Differentiating with respect to θ yields the closed form
//
//	∂ log c / ∂θ = 1/(1+θ)
//	             − (log u + log v)
//	             + (1/θ²)·log S
//	             + (−2 − 1/θ)·(d log S / dθ)
//
//	d log S / dθ = (−u^(−θ)·log u − v^(−θ)·log v) / S
//
// This test pins reverse-mode autodiff against that analytic gradient
// at multiple (u, v, θ) points with 1e-9 tolerance.
//
// Why it matters:
//
//   - autodiff's doc-comment cites "Heston / SABR / rough-vol calibration
//     (same reason as GARCH)" as a use case. The Clayton copula is a
//     simpler member of the same calibration family — a 1-parameter
//     log-density, but with the same structural complexity (log of a
//     transformed variable raised to a function-of-the-parameter power).
//     Pinning Clayton's analytic gradient against autodiff is the smallest
//     non-trivial demonstration that the autodiff machinery handles
//     copula calibration correctly.
//
//   - L13 Reality copula is one half of the Solvency II SCR cohort
//     (the other half is risk-measures / VaR / ES). Solvency II
//     calibration ultimately needs autodiff-supplied gradients for
//     QMLE on copula parameters. This test is the substrate-internal
//     witness that the composition will work when relic-insurance
//     wires it.
//
// The test exercises Pow-with-variable-exponent via the
// x^θ = exp(θ · log x) identity (autodiff has Pow only with constant
// exponent; θ here is a Variable).
func TestClaytonLogPDF_AutodiffGradientMatchesAnalytic(t *testing.T) {
	cases := []struct {
		name        string
		u, v, theta float64
	}{
		{"interior_mid_theta", 0.4, 0.6, 0.8},
		{"interior_low_theta", 0.3, 0.7, 0.2},
		{"interior_high_theta", 0.5, 0.5, 4.0},
		{"asymmetric", 0.1, 0.9, 1.5},
		{"near_independence", 0.5, 0.5, 0.05},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// Direct value via the package's own log-density closure.
			fn, err := ClaytonLogPDFFn(tc.theta)
			if err != nil {
				t.Fatalf("ClaytonLogPDFFn: %v", err)
			}
			direct := fn(tc.u, tc.v)
			if math.IsInf(direct, 0) || math.IsNaN(direct) {
				t.Fatalf("direct log c is not finite: %g", direct)
			}

			// Analytic gradient by hand-derivation (see doc-comment).
			analytic := claytonLogPDFGradAnalytic(tc.u, tc.v, tc.theta)

			// Autodiff value + gradient.
			autoVal, autoGrad := buildClaytonLogPDFAutodiff(tc.u, tc.v, tc.theta)

			// Value parity: direct vs autodiff must agree to 1e-12.
			if d := math.Abs(direct - autoVal); d > 1e-12 {
				t.Errorf("log c value mismatch: direct=%g auto=%g |delta|=%g",
					direct, autoVal, d)
			}

			// Gradient parity: analytic vs autodiff to 1e-9.
			if d := math.Abs(analytic - autoGrad); d > 1e-9 {
				t.Errorf("∂ log c / ∂θ mismatch: analytic=%g auto=%g |delta|=%g",
					analytic, autoGrad, d)
			}
		})
	}
}

// claytonLogPDFGradAnalytic computes ∂ log c(u, v; θ) / ∂θ by direct
// closed-form differentiation. See the test's doc-comment for the
// derivation.
func claytonLogPDFGradAnalytic(u, v, theta float64) float64 {
	uNegT := math.Pow(u, -theta)
	vNegT := math.Pow(v, -theta)
	S := uNegT + vNegT - 1.0
	logS := math.Log(S)
	logU := math.Log(u)
	logV := math.Log(v)

	dSdTheta := -uNegT*logU - vNegT*logV
	dLogSdTheta := dSdTheta / S

	return 1.0/(1.0+theta) - (logU + logV) + (1.0/(theta*theta))*logS + (-2.0-1.0/theta)*dLogSdTheta
}

// buildClaytonLogPDFAutodiff constructs log c(u, v; θ) on an autodiff
// Tape with θ as the only Variable, and returns (value, ∂/∂θ). u and v
// are constants in this calibration scenario.
//
// Implementation note: autodiff has Pow only with constant exponent, but
// here we need x^θ where x is a known constant and θ is a Variable. Use
// the identity x^θ = exp(θ · log x).
func buildClaytonLogPDFAutodiff(u, v, theta float64) (float64, float64) {
	tape := autodiff.NewTape()

	t := tape.Var(theta)

	// log(1 + θ) — autodiff has no Log1p; use Log(Add(1, θ)).
	one := tape.Constant(1.0)
	onePlusT := autodiff.Add(one, t)
	term1 := autodiff.Log(onePlusT)

	// (−1 − θ)·(log u + log v). u, v are constants; precompute log u + log v.
	logUV := math.Log(u) + math.Log(v)
	negOneMinusT := autodiff.AddConst(autodiff.Neg(t), -1.0) // −1 − θ
	term2 := autodiff.MulConst(negOneMinusT, logUV)

	// u^(−θ) = exp(−θ · log u);  v^(−θ) = exp(−θ · log v)
	negT := autodiff.Neg(t)
	uNegT := autodiff.Exp(autodiff.MulConst(negT, math.Log(u)))
	vNegT := autodiff.Exp(autodiff.MulConst(negT, math.Log(v)))
	S := autodiff.AddConst(autodiff.Add(uNegT, vNegT), -1.0)
	logS := autodiff.Log(S)

	// (−2 − 1/θ) · log S. Compute −2 − 1/θ as Variable too (1/θ is variable).
	invT := autodiff.Div(one, t)
	negInvT := autodiff.Neg(invT)
	negTwoMinusInvT := autodiff.AddConst(negInvT, -2.0)
	term3 := autodiff.Mul(negTwoMinusInvT, logS)

	// log c = term1 + term2 + term3.
	out := autodiff.Add(autodiff.Add(term1, term2), term3)

	gradAll := tape.Backward(out)
	return out.Val, gradAll[t.ID]
}
