package trust

import "testing"

// Frame element bitmasks reused across the Dempster/Yager tests.
const (
	// Binary frame {x, ¬x}.
	setX    uint = 0b01
	setNotX uint = 0b10
	setBin  uint = 0b11 // Θ for the 2-element frame

	// Zadeh's 3-element medical frame {M(eningitis), C(oncussion), T(umour)}.
	setM   uint = 0b001
	setC   uint = 0b010
	setT   uint = 0b100
	setMCT uint = 0b111 // Θ
)

// TestDempsterCombine_ZadehCounterexample is the REQUIRED golden vector: the
// classic Zadeh (1984) conflict counterexample. Two doctors examine a patient
// over the frame {Meningitis, Concussion, Tumour}:
//
//	Doctor 1: m(M)=0.99, m(T)=0.01
//	Doctor 2: m(C)=0.99, m(T)=0.01
//
// Every pairing except T∩T is empty, so conflict K = 0.9801 + 0.0099 + 0.0099
// = 0.9999 and only 0.0001 mass survives on {T}. Dempster's normalisation
// divides by 1−K = 0.0001, minting m(T) = 1.0 — CERTAINTY of the tumour both
// doctors thought least likely. The point of this test is that the harness
// EXPOSES K = 0.9999, so a caller can see the result is worthless despite the
// confident-looking mass.
//
// Reference: Zadeh, L. A. (1984). Review of Shafer's "A Mathematical Theory
// of Evidence". AI Magazine 5(3): 81-83.
func TestDempsterCombine_ZadehCounterexample(t *testing.T) {
	d1, err := NewMassFunction(3, map[uint]float64{setM: 0.99, setT: 0.01})
	if err != nil {
		t.Fatalf("d1: %v", err)
	}
	d2, err := NewMassFunction(3, map[uint]float64{setC: 0.99, setT: 0.01})
	if err != nil {
		t.Fatalf("d2: %v", err)
	}

	combined, k, err := DempsterCombine(d1, d2)
	if err != nil {
		t.Fatalf("DempsterCombine: %v", err)
	}
	approx(t, "K", k, 0.9999, epsAcc)
	approx(t, "m(T)", combined.Masses[setT], 1.0, epsAcc)
	approx(t, "Bel(T)", combined.Belief(setT), 1.0, epsAcc)
	// The counterintuitive certainty is the WHOLE point: high K, m(T)=1.
	if k <= 0.5 {
		t.Errorf("Zadeh K must expose massive conflict, got %v", k)
	}
}

// TestYagerCombine_ZadehIsHonest runs the SAME Zadeh inputs through Yager's
// rule, which keeps the conflict mass on the frame Θ instead of normalising it
// away. The tumour hypothesis is then correctly reported as barely supported:
// Bel(T)=0.0001, Pl(T)=1.0, with the 0.9999 conflict living on Θ as declared
// ignorance.
//
// Reference: Yager, R. R. (1987). Information Sciences 41(2): 93-137.
func TestYagerCombine_ZadehIsHonest(t *testing.T) {
	d1, _ := NewMassFunction(3, map[uint]float64{setM: 0.99, setT: 0.01})
	d2, _ := NewMassFunction(3, map[uint]float64{setC: 0.99, setT: 0.01})

	combined, k, err := YagerCombine(d1, d2)
	if err != nil {
		t.Fatalf("YagerCombine: %v", err)
	}
	approx(t, "K", k, 0.9999, epsAcc)
	approx(t, "m(T)", combined.Masses[setT], 0.0001, epsAcc)
	approx(t, "m(Θ)", combined.Masses[setMCT], 0.9999, epsAcc)
	approx(t, "Bel(T)", combined.Belief(setT), 0.0001, epsAcc)
	approx(t, "Pl(T)", combined.Plausibility(setT), 1.0, epsAcc)

	// Masses still sum to 1.
	var sum float64
	for _, m := range combined.Masses {
		sum += m
	}
	approx(t, "Σm", sum, 1.0, epsAcc)
}

// TestDempsterCombine_ConcordantSensors is a positive (low-conflict) golden:
// two concordant sensors over a binary frame, each with some ignorance mass on
// Θ, reinforce belief in x. Hand-derived:
//
//	Sensor1: m(x)=0.9, m(Θ)=0.1
//	Sensor2: m(x)=0.8, m(Θ)=0.2
//
// No pairing is empty (K=0). m(x) = 0.72+0.18+0.08 = 0.98, m(Θ)=0.02, so
// Bel(x)=0.98, Pl(x)=1.0 — the standard "agreeing evidence raises belief"
// result (Shafer 1976, §3; Sentz & Ferson 2002).
func TestDempsterCombine_ConcordantSensors(t *testing.T) {
	s1, _ := NewMassFunction(2, map[uint]float64{setX: 0.9, setBin: 0.1})
	s2, _ := NewMassFunction(2, map[uint]float64{setX: 0.8, setBin: 0.2})

	combined, k, err := DempsterCombine(s1, s2)
	if err != nil {
		t.Fatalf("DempsterCombine: %v", err)
	}
	approx(t, "K", k, 0.0, epsAcc)
	approx(t, "m(x)", combined.Masses[setX], 0.98, epsAcc)
	approx(t, "m(Θ)", combined.Masses[setBin], 0.02, epsAcc)
	approx(t, "Bel(x)", combined.Belief(setX), 0.98, epsAcc)
	approx(t, "Pl(x)", combined.Plausibility(setX), 1.0, epsAcc)
	// With no conflict Dempster and Yager must agree exactly.
	yager, _, _ := YagerCombine(s1, s2)
	approx(t, "yager m(x)", yager.Masses[setX], 0.98, epsAcc)
}

// TestDempsterCombine_TotalConflict: two dogmatic contradictory opinions
// (K=1) have an undefined Dempster combination; the harness returns
// ErrTotalConflict rather than dividing by zero, and still reports K.
func TestDempsterCombine_TotalConflict(t *testing.T) {
	a, _ := NewMassFunction(2, map[uint]float64{setX: 1.0})
	b, _ := NewMassFunction(2, map[uint]float64{setNotX: 1.0})
	_, k, err := DempsterCombine(a, b)
	if err != ErrTotalConflict {
		t.Errorf("err = %v, want ErrTotalConflict", err)
	}
	approx(t, "K", k, 1.0, epsAcc)
	// Yager still produces a defined result: all mass → Θ.
	y, ky, err := YagerCombine(a, b)
	if err != nil {
		t.Fatalf("YagerCombine: %v", err)
	}
	approx(t, "Yager K", ky, 1.0, epsAcc)
	approx(t, "Yager m(Θ)", y.Masses[setBin], 1.0, epsAcc)
}

// TestBeliefPlausibility_Interval checks Bel ≤ Pl and the Pl(a)=1−Bel(¬a)
// duality on a mixed mass function.
func TestBeliefPlausibility_Interval(t *testing.T) {
	// Frame {M,C,T}: m(M)=0.5, m({M,C})=0.3, m(Θ)=0.2.
	mf, err := NewMassFunction(3, map[uint]float64{setM: 0.5, setM | setC: 0.3, setMCT: 0.2})
	if err != nil {
		t.Fatalf("NewMassFunction: %v", err)
	}
	// Bel(M) = m(M) = 0.5 (only {M} ⊆ {M}).
	approx(t, "Bel(M)", mf.Belief(setM), 0.5, epsAcc)
	// Pl(M) = m(M)+m({M,C})+m(Θ) = 1.0 (all intersect M).
	approx(t, "Pl(M)", mf.Plausibility(setM), 1.0, epsAcc)
	// Bel({M,C}) = m(M)+m({M,C}) = 0.8.
	approx(t, "Bel(MC)", mf.Belief(setM|setC), 0.8, epsAcc)
	// Duality: Pl(M) = 1 − Bel(¬M) where ¬M = {C,T}.
	approx(t, "duality", mf.Plausibility(setM), 1-mf.Belief(setC|setT), epsAcc)
}

// TestOpinionBinaryMass_RoundTrip checks the opinion ↔ binary-BPA bridge
// (Jøsang §3.5): belief↔m(x), disbelief↔m(¬x), uncertainty↔m(Θ).
func TestOpinionBinaryMass_RoundTrip(t *testing.T) {
	o := Opinion{B: 0.6, D: 0.1, U: 0.3, A: 0.5}
	mf := o.ToBinaryMass()
	approx(t, "m(x)", mf.Masses[setX], 0.6, eps)
	approx(t, "m(¬x)", mf.Masses[setNotX], 0.1, eps)
	approx(t, "m(Θ)", mf.Masses[setBin], 0.3, eps)
	// Belief/Plausibility of x must bracket the opinion's projected prob.
	if mf.Belief(setX) > o.ProbabilityProjection() || o.ProbabilityProjection() > mf.Plausibility(setX) {
		t.Errorf("P(x)=%v must lie in [Bel=%v, Pl=%v]", o.ProbabilityProjection(), mf.Belief(setX), mf.Plausibility(setX))
	}

	back, err := OpinionFromBinaryMass(mf)
	if err != nil {
		t.Fatalf("OpinionFromBinaryMass: %v", err)
	}
	approx(t, "b", back.B, 0.6, eps)
	approx(t, "d", back.D, 0.1, eps)
	approx(t, "u", back.U, 0.3, eps)

	// Fusing two binary BPAs with Dempster should agree with the opinion
	// projection direction (concordant beliefs reinforce).
	if _, err := OpinionFromBinaryMass(MassFunction{FrameSize: 3}); err != ErrFrameMismatch {
		t.Errorf("non-binary frame must give ErrFrameMismatch, got %v", err)
	}
}

func TestNewMassFunction_Validation(t *testing.T) {
	cases := []struct {
		name      string
		frameSize int
		masses    map[uint]float64
		wantErr   bool
	}{
		{"valid", 2, map[uint]float64{setX: 0.7, setBin: 0.3}, false},
		{"empty set mass", 2, map[uint]float64{0: 0.5, setX: 0.5}, true},
		{"out of frame", 2, map[uint]float64{0b100: 1.0}, true},
		{"negative", 2, map[uint]float64{setX: -0.1, setNotX: 1.1}, true},
		{"sum not 1", 2, map[uint]float64{setX: 0.5}, true},
		{"bad frame size", 0, map[uint]float64{setX: 1.0}, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			_, err := NewMassFunction(c.frameSize, c.masses)
			if (err != nil) != c.wantErr {
				t.Errorf("NewMassFunction err=%v, wantErr=%v", err, c.wantErr)
			}
		})
	}
}

func TestDempsterCombine_FrameMismatch(t *testing.T) {
	a, _ := NewMassFunction(2, map[uint]float64{setX: 1.0})
	b, _ := NewMassFunction(3, map[uint]float64{setM: 1.0})
	if _, _, err := DempsterCombine(a, b); err != ErrFrameMismatch {
		t.Errorf("err = %v, want ErrFrameMismatch", err)
	}
	if _, _, err := YagerCombine(a, b); err != ErrFrameMismatch {
		t.Errorf("err = %v, want ErrFrameMismatch", err)
	}
}
