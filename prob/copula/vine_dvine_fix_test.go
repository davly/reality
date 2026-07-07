package copula

import (
	"math"
	"testing"
)

// Regression for the D-vine LogPDF h-direction fix. For a 3-D D-vine the
// conditional edge c_{13|2} must couple h(u1|u2) and h(u3|u2) — both conditioned
// on the SHARED middle variable u2. The old code used h(u2|u3) for the second
// pseudo-observation, making the joint density wrong at every non-symmetric point
// (the only test checked finiteness, so it never caught the value).

func TestDVine_LogPDF_Dim3_CorrectAssembly(t *testing.T) {
	trees := [][]VineEdge{
		{{FamilyClayton, 2.0}, {FamilyClayton, 1.5}},
		{{FamilyClayton, 1.0}},
	}
	v, err := NewDVine(3, trees)
	if err != nil {
		t.Fatalf("NewDVine: %v", err)
	}

	got, err := v.LogPDF([]float64{0.3, 0.5, 0.7})
	if err != nil {
		t.Fatalf("LogPDF: %v", err)
	}

	// Independent reassembly from the separately-verified primitives, per the
	// correct Aas-Czado formula: log c_12(u1,u2) + log c_23(u2,u3)
	//                           + log c_13|2( h(u1|u2), h(u3|u2) ).
	lp12, _ := LogPDFFnForFamily(FamilyClayton, 2.0)
	lp23, _ := LogPDFFnForFamily(FamilyClayton, 1.5)
	lp13c2, _ := LogPDFFnForFamily(FamilyClayton, 1.0)
	h12, _ := HFnForFamily(FamilyClayton, 2.0)
	h23, _ := HFnForFamily(FamilyClayton, 1.5)
	p0 := h12(0.3, 0.5) // h(u1|u2)
	p1 := h23(0.7, 0.5) // h(u3|u2) — conditioned on the shared middle u2
	want := lp12(0.3, 0.5) + lp23(0.5, 0.7) + lp13c2(p0, p1)
	if math.Abs(got-want) > 1e-12 {
		t.Errorf("LogPDF=%.12f want %.12f (correct assembly)", got, want)
	}

	// The buggy assembly (h(u2|u3) for the 2nd pseudo-obs) must DIFFER — proving
	// the fix actually changed the value at this non-symmetric point.
	buggy := lp12(0.3, 0.5) + lp23(0.5, 0.7) + lp13c2(p0, h23(0.5, 0.7))
	if math.Abs(got-buggy) < 1e-6 {
		t.Errorf("LogPDF still equals the buggy h(u2|u3) assembly (%.12f)", buggy)
	}
}

func TestDVine_LogPDF_Dim4_HonestError(t *testing.T) {
	trees := [][]VineEdge{
		{{FamilyClayton, 2.0}, {FamilyClayton, 1.5}, {FamilyClayton, 1.2}},
		{{FamilyClayton, 1.0}, {FamilyClayton, 1.1}},
		{{FamilyClayton, 1.3}},
	}
	v, err := NewDVine(4, trees)
	if err != nil {
		t.Fatalf("NewDVine(4): %v", err)
	}
	if _, err := v.LogPDF([]float64{0.3, 0.5, 0.7, 0.4}); err == nil {
		t.Error("dim=4 LogPDF should return an honest not-implemented error (was silently wrong)")
	}
}
