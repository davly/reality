package trust

import (
	"math"
	"testing"
)

const eps = 1e-12   // exact rational fractions
const epsAcc = 1e-9 // accumulated fusion/combination arithmetic

func approx(t *testing.T, name string, got, want, tol float64) {
	t.Helper()
	if diff := got - want; diff > tol || diff < -tol {
		t.Errorf("%s = %.15g, want %.15g (|diff|=%.3g > %.3g)", name, got, want, math.Abs(diff), tol)
	}
}

// TestOpinionFromEvidence_JosangBetaMapping reproduces the canonical
// Beta-to-opinion mapping of Jøsang, Subjective Logic (2016) §3.3, with
// prior weight W=2. The fractions below are exact and hand-derived from
// b=r/(r+s+2), d=s/(r+s+2), u=2/(r+s+2).
func TestOpinionFromEvidence_JosangBetaMapping(t *testing.T) {
	// r=6 positive, s=1 negative, base rate a=0.5 → denom = 6+1+2 = 9.
	o, err := OpinionFromEvidence(6, 1, 0.5)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	approx(t, "b", o.B, 6.0/9.0, eps)
	approx(t, "d", o.D, 1.0/9.0, eps)
	approx(t, "u", o.U, 2.0/9.0, eps)
	approx(t, "b+d+u", o.B+o.D+o.U, 1.0, eps)
	// Projected probability P = b + a·u = 6/9 + 0.5·2/9 = 7/9.
	approx(t, "P", o.ProbabilityProjection(), 7.0/9.0, eps)
	if err := o.Validate(); err != nil {
		t.Errorf("evidence opinion should validate: %v", err)
	}
}

// TestOpinionFromEvidence_Vacuous is the load-bearing property: NO evidence
// maps to the wholly-uncertain opinion (u=1), whose projected probability is
// exactly the base rate — never certainty. This is the belief-calculus fix
// for hive-forge-credibility's "n<2 → 1.0" vacuous gate.
func TestOpinionFromEvidence_Vacuous(t *testing.T) {
	o, err := OpinionFromEvidence(0, 0, 0.3)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	approx(t, "b", o.B, 0, eps)
	approx(t, "d", o.D, 0, eps)
	approx(t, "u", o.U, 1, eps)
	if !o.IsVacuous() {
		t.Errorf("no-evidence opinion must be vacuous")
	}
	// Projects to the base rate, NOT to 1.0.
	approx(t, "P", o.ProbabilityProjection(), 0.3, eps)
}

// TestOpinionFromEvidence_ConvergesToCertainty checks that as positive
// evidence dominates, belief → 1 and uncertainty → 0 (Jøsang §3.3 limit).
func TestOpinionFromEvidence_ConvergesToCertainty(t *testing.T) {
	o, _ := OpinionFromEvidence(1000, 0, 0.5)
	if o.B <= 0.99 || o.U >= 0.01 {
		t.Errorf("with r=1000,s=0 expected b>0.99,u<0.01, got b=%v u=%v", o.B, o.U)
	}
}

// TestEvidence_RoundTrip verifies Evidence() inverts OpinionFromEvidence.
func TestEvidence_RoundTrip(t *testing.T) {
	o, _ := OpinionFromEvidence(6, 1, 0.5)
	r, s := o.Evidence()
	approx(t, "r", r, 6, epsAcc)
	approx(t, "s", s, 1, epsAcc)

	dogmatic := Opinion{B: 1, D: 0, U: 0, A: 0.5}
	r, s = dogmatic.Evidence()
	if !math.IsInf(r, 1) || !math.IsInf(s, 1) {
		t.Errorf("dogmatic opinion should imply infinite evidence, got r=%v s=%v", r, s)
	}
}

// TestProbabilityProjection_Manual checks b + a·u on a hand-built opinion.
func TestProbabilityProjection_Manual(t *testing.T) {
	o := Opinion{B: 0.7, D: 0.1, U: 0.2, A: 0.5}
	approx(t, "P", o.ProbabilityProjection(), 0.8, eps) // 0.7 + 0.5·0.2
}

func TestNewOpinion_Validation(t *testing.T) {
	cases := []struct {
		name       string
		b, d, u, a float64
		wantErr    bool
	}{
		{"valid", 0.5, 0.3, 0.2, 0.5, false},
		{"valid vacuous", 0, 0, 1, 0.5, false},
		{"additivity broken", 0.5, 0.5, 0.5, 0.5, true},
		{"negative mass", -0.1, 0.6, 0.5, 0.5, true},
		{"mass over 1", 1.1, 0, -0.1, 0.5, true},
		{"base rate over 1", 0.5, 0.3, 0.2, 1.5, true},
		{"NaN", math.NaN(), 0.3, 0.2, 0.5, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			_, err := NewOpinion(c.b, c.d, c.u, c.a)
			if (err != nil) != c.wantErr {
				t.Errorf("NewOpinion(%v,%v,%v,%v) err=%v, wantErr=%v", c.b, c.d, c.u, c.a, err, c.wantErr)
			}
		})
	}
}

func TestOpinionFromEvidence_NegativeRejected(t *testing.T) {
	if _, err := OpinionFromEvidence(-1, 0, 0.5); err != ErrNegativeEvidence {
		t.Errorf("err = %v, want ErrNegativeEvidence", err)
	}
	if _, err := OpinionFromEvidence(0, -5, 0.5); err != ErrNegativeEvidence {
		t.Errorf("err = %v, want ErrNegativeEvidence", err)
	}
}
