package infogeo

import (
	"math"
	"testing"

	"github.com/davly/reality/autodiff"
)

// TestKL_AutodiffGradientMatchesQMinusP closes a second substrate-internal
// first-consumer push for github.com/davly/reality/autodiff (now: garch +
// infogeo) and adds a second verified consumer for infogeo (now:
// changepoint + autodiff). It pins the gradient of KL(p || softmax(θ))
// computed by reverse-mode autodiff against the analytic closed form
//
//	∇_θ KL(p || softmax(θ)) = softmax(θ) − p
//
// at multiple θ points, with 1e-9 tolerance on each coordinate.
//
// This is the second consumer of the R-CLOSED-FORM-PINNED-TO-AUTODIFF
// pattern (after F2.a's autodiff × garch parity test, S62 2026-05-05).
// One more consumer saturates the R-rule to 3/3.
//
// Why the closed form is q − p:
//
//	q_j = exp(θ_j) / Σ_k exp(θ_k)
//	log q_j = θ_j − log Σ_k exp(θ_k)
//	KL(p || q) = Σ_i p_i log(p_i / q_i) = const(p) − Σ_i p_i · log q_i
//	∂(log q_i)/∂θ_j = δ_{ij} − q_j
//	∂KL/∂θ_j = −Σ_i p_i (δ_{ij} − q_j) = −p_j + q_j · Σ_i p_i = q_j − p_j
//
// (The simplex constraint Σ_i p_i = 1 collapses the second term cleanly.)
// This is the gradient that natural-gradient methods, variational
// inference, and policy gradients all use — it's the canonical example.
//
// Test structure mirrors F2.a (autodiff_test.go in timeseries/garch):
// build the loss via the autodiff Tape, call Backward, compare per-
// coordinate gradient against the analytic baseline.
func TestKL_AutodiffGradientMatchesQMinusP(t *testing.T) {
	cases := []struct {
		name  string
		p     []float64
		theta []float64
	}{
		{
			name:  "uniform_p_zero_theta",
			p:     []float64{0.25, 0.25, 0.25, 0.25},
			theta: []float64{0.0, 0.0, 0.0, 0.0},
		},
		{
			name:  "skewed_p_skewed_theta",
			p:     []float64{0.5, 0.3, 0.1, 0.1},
			theta: []float64{0.7, -0.4, 1.2, 0.1},
		},
		{
			name:  "near_independence",
			p:     []float64{0.4, 0.3, 0.2, 0.1},
			theta: []float64{0.4, 0.3, 0.2, 0.1},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			n := len(tc.p)
			if len(tc.theta) != n {
				t.Fatalf("|p|=%d != |theta|=%d", n, len(tc.theta))
			}

			// Analytic baseline: q = softmax(theta); ∇KL = q − p.
			q := softmax(tc.theta)
			analytic := make([]float64, n)
			for i := range q {
				analytic[i] = q[i] - tc.p[i]
			}

			// Pin: KL value via direct infogeo.KL on the resolved q must
			// match the autodiff-built value within 1e-12. (Same input,
			// two implementations.)
			klDirect, err := KL(tc.p, q)
			if err != nil {
				t.Fatalf("infogeo.KL: %v", err)
			}

			// Autodiff path: build KL(p || softmax(theta)) over a Tape.
			klAuto, gradAuto := buildKLAutodiff(tc.p, tc.theta)

			if d := math.Abs(klDirect - klAuto); d > 1e-12 {
				t.Errorf("KL value mismatch: infogeo.KL=%g autodiff=%g |delta|=%g",
					klDirect, klAuto, d)
			}

			// Per-coordinate gradient parity.
			for j := 0; j < n; j++ {
				if d := math.Abs(analytic[j] - gradAuto[j]); d > 1e-9 {
					t.Errorf("∂KL/∂θ[%d] mismatch: analytic=%g autodiff=%g |delta|=%g",
						j, analytic[j], gradAuto[j], d)
				}
			}
		})
	}
}

// buildKLAutodiff constructs KL(p || softmax(theta)) on an autodiff Tape
// and returns the value plus the gradient with respect to theta. The
// formulation matches the analytic derivation in the test's doc-comment.
//
// Numerical-stability detail: softmax is computed via subtract-the-max
// before exp (matching the F2.a garch reparameterisation pattern), so
// the gradient is finite at large |theta|.
func buildKLAutodiff(p, theta []float64) (float64, []float64) {
	tape := autodiff.NewTape()
	n := len(theta)

	// Register theta as Tape Variables.
	thetaVars := make([]*autodiff.Variable, n)
	for i, t := range theta {
		thetaVars[i] = tape.Var(t)
	}

	// Softmax with max-subtraction for stability.
	maxT := theta[0]
	for _, v := range theta[1:] {
		if v > maxT {
			maxT = v
		}
	}
	expShifted := make([]*autodiff.Variable, n)
	for i, tv := range thetaVars {
		expShifted[i] = autodiff.Exp(autodiff.AddConst(tv, -maxT))
	}
	z := expShifted[0]
	for i := 1; i < n; i++ {
		z = autodiff.Add(z, expShifted[i])
	}
	q := make([]*autodiff.Variable, n)
	for i := range expShifted {
		q[i] = autodiff.Div(expShifted[i], z)
	}

	// KL(p || q) = Σ p_i · log(p_i / q_i).
	// p is a constant, so log(p_i / q_i) = log(p_i) − log(q_i).
	// Accumulate −Σ p_i · log(q_i); add the constant term Σ p_i log(p_i)
	// at the end (it does not contribute to the gradient).
	negSum := tape.Constant(0.0)
	for i, qi := range q {
		// p_i is constant; multiply by it via MulConst.
		logQi := autodiff.Log(qi)
		term := autodiff.MulConst(logQi, p[i])
		negSum = autodiff.Add(negSum, term)
	}
	// negSum = Σ p_i log(q_i). KL = −negSum + const.
	negKLNoConst := negSum

	// const(p) = Σ p_i log(p_i).
	var constP float64
	for _, pi := range p {
		if pi > 0 {
			constP += pi * math.Log(pi)
		}
	}

	// We want the gradient of KL = constP − negSum.
	// Equivalent: gradient of (−negSum) is what we need; constP is a
	// constant w.r.t. theta. Build out = MulConst(negKLNoConst, -1) and
	// add the constant for the value match.
	out := autodiff.MulConst(negKLNoConst, -1.0)
	out = autodiff.AddConst(out, constP)

	gradAll := tape.Backward(out)

	gradTheta := make([]float64, n)
	for i, tv := range thetaVars {
		gradTheta[i] = gradAll[tv.ID]
	}
	return out.Val, gradTheta
}

// softmax(theta) returns the simplex projection
// q_i = exp(theta_i - max) / sum(exp(theta_j - max)).
func softmax(theta []float64) []float64 {
	n := len(theta)
	if n == 0 {
		return nil
	}
	maxT := theta[0]
	for _, v := range theta[1:] {
		if v > maxT {
			maxT = v
		}
	}
	exps := make([]float64, n)
	var s float64
	for i, t := range theta {
		exps[i] = math.Exp(t - maxT)
		s += exps[i]
	}
	for i := range exps {
		exps[i] /= s
	}
	return exps
}
