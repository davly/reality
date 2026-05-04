package copula

import (
	"errors"
	"math"
	"testing"
)

// Tests for the D-vine substrate (a241 — vine copula stepping-stone).

func TestVineEdge_ValidatesClayton(t *testing.T) {
	cases := []struct {
		theta float64
		ok    bool
	}{
		{2.0, true},
		{0.1, true},
		{0, false},
		{-1, false},
	}
	for _, c := range cases {
		err := VineEdge{Family: FamilyClayton, Theta: c.theta}.Validate()
		if (err == nil) != c.ok {
			t.Errorf("Clayton θ=%v Validate ok=%v err=%v", c.theta, c.ok, err)
		}
	}
}

func TestVineEdge_ValidatesGumbel(t *testing.T) {
	cases := []struct {
		theta float64
		ok    bool
	}{
		{1.0, true},
		{2.5, true},
		{0.5, false},
		{0, false},
	}
	for _, c := range cases {
		err := VineEdge{Family: FamilyGumbel, Theta: c.theta}.Validate()
		if (err == nil) != c.ok {
			t.Errorf("Gumbel θ=%v Validate ok=%v err=%v", c.theta, c.ok, err)
		}
	}
}

func TestVineEdge_RejectsUnknownFamily(t *testing.T) {
	err := VineEdge{Family: ArchimedeanFamily(99), Theta: 1.0}.Validate()
	if !errors.Is(err, ErrVineEdgeInvalid) {
		t.Errorf("expected ErrVineEdgeInvalid, got %v", err)
	}
}

func TestNewDVine_RejectsBadDim(t *testing.T) {
	for _, dim := range []int{0, 1, -1} {
		if _, err := NewDVine(dim, nil); !errors.Is(err, ErrVineInvalidDimension) {
			t.Errorf("NewDVine(%d): expected ErrVineInvalidDimension, got %v", dim, err)
		}
	}
}

func TestNewDVine_RejectsBadTreeCount(t *testing.T) {
	// dim=3 expects 2 trees. Pass 1 tree to fail.
	trees := [][]VineEdge{
		{{Family: FamilyClayton, Theta: 1.0}, {Family: FamilyClayton, Theta: 1.0}},
	}
	if _, err := NewDVine(3, trees); !errors.Is(err, ErrVineEdgeMismatch) {
		t.Errorf("expected ErrVineEdgeMismatch on tree count, got %v", err)
	}
}

func TestNewDVine_RejectsBadEdgeCount(t *testing.T) {
	// dim=3 expects T_1 to have 2 edges. Pass 1 edge in T_1.
	trees := [][]VineEdge{
		{{Family: FamilyClayton, Theta: 1.0}}, // T_1: should have 2
		{{Family: FamilyClayton, Theta: 1.0}}, // T_2: 1 OK
	}
	if _, err := NewDVine(3, trees); !errors.Is(err, ErrVineEdgeMismatch) {
		t.Errorf("expected ErrVineEdgeMismatch on edge count, got %v", err)
	}
}

func TestNewDVine_RejectsBadEdgeTheta(t *testing.T) {
	// Clayton θ=0 is invalid — should be rejected.
	trees := [][]VineEdge{
		{{Family: FamilyClayton, Theta: 0}, {Family: FamilyClayton, Theta: 1.0}},
		{{Family: FamilyClayton, Theta: 1.0}},
	}
	if _, err := NewDVine(3, trees); !errors.Is(err, ErrVineEdgeInvalid) {
		t.Errorf("expected ErrVineEdgeInvalid on theta, got %v", err)
	}
}

func TestNewDVine_AcceptsValid3DVine(t *testing.T) {
	trees := [][]VineEdge{
		{{Family: FamilyClayton, Theta: 2.0}, {Family: FamilyGumbel, Theta: 1.5}},
		{{Family: FamilyClayton, Theta: 1.0}},
	}
	v, err := NewDVine(3, trees)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if v.Dim() != 3 {
		t.Errorf("Dim = %d, want 3", v.Dim())
	}
	if v.EdgeCount() != 3 {
		t.Errorf("EdgeCount = %d, want 3", v.EdgeCount())
	}
}

func TestNewDVine_AcceptsValid4DVine(t *testing.T) {
	// dim=4: T_1 has 3 edges, T_2 has 2, T_3 has 1 = 6 total
	trees := [][]VineEdge{
		{{FamilyClayton, 2.0}, {FamilyClayton, 1.5}, {FamilyClayton, 1.2}},
		{{FamilyGumbel, 1.5}, {FamilyGumbel, 1.3}},
		{{FamilyClayton, 0.8}},
	}
	v, err := NewDVine(4, trees)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if v.EdgeCount() != 6 {
		t.Errorf("EdgeCount = %d, want 6 (=4*3/2)", v.EdgeCount())
	}
}

func TestDVine_HFunctionPass_ProducesCorrectShape(t *testing.T) {
	trees := [][]VineEdge{
		{{FamilyClayton, 2.0}, {FamilyGumbel, 1.5}},
		{{FamilyClayton, 1.0}},
	}
	v, err := NewDVine(3, trees)
	if err != nil {
		t.Fatalf("NewDVine err: %v", err)
	}

	// T_1 input has length dim - 0 = 3; output has length 2.
	u := []float64{0.3, 0.5, 0.7}
	out, err := v.HFunctionPass(0, u)
	if err != nil {
		t.Fatalf("HFunctionPass(0) err: %v", err)
	}
	if len(out) != 2 {
		t.Errorf("T_1 output length = %d, want 2", len(out))
	}

	// Each output should be in [0, 1] (it's a conditional CDF).
	for i, o := range out {
		if o < 0 || o > 1 {
			t.Errorf("T_1 output[%d] = %v, expected in [0, 1]", i, o)
		}
	}

	// T_2 input has length dim - 1 = 2; output has length 1.
	out2, err := v.HFunctionPass(1, out)
	if err != nil {
		t.Fatalf("HFunctionPass(1) err: %v", err)
	}
	if len(out2) != 1 {
		t.Errorf("T_2 output length = %d, want 1", len(out2))
	}
}

func TestDVine_HFunctionPass_RejectsBadTreeIdx(t *testing.T) {
	trees := [][]VineEdge{
		{{FamilyClayton, 2.0}, {FamilyClayton, 1.5}},
		{{FamilyClayton, 1.0}},
	}
	v, _ := NewDVine(3, trees)
	if _, err := v.HFunctionPass(-1, []float64{0.3, 0.5, 0.7}); err == nil {
		t.Error("expected error on negative treeIdx")
	}
	if _, err := v.HFunctionPass(2, []float64{0.3, 0.5}); err == nil {
		t.Error("expected error on treeIdx >= dim-1")
	}
}

func TestDVine_HFunctionPass_RejectsBadInputLength(t *testing.T) {
	trees := [][]VineEdge{
		{{FamilyClayton, 2.0}, {FamilyClayton, 1.5}},
		{{FamilyClayton, 1.0}},
	}
	v, _ := NewDVine(3, trees)
	// T_1 expects length 3, give 2.
	if _, err := v.HFunctionPass(0, []float64{0.3, 0.5}); err == nil {
		t.Error("expected error on short input")
	}
}

func TestDVine_LogPDF_ProducesFiniteValueOnInteriorPoint(t *testing.T) {
	// Phase 15: LogPDF now wires the PDF closures. Interior points
	// must return finite log-densities; boundary points return -∞.
	trees := [][]VineEdge{
		{{FamilyClayton, 2.0}, {FamilyClayton, 1.5}},
		{{FamilyClayton, 1.0}},
	}
	v, _ := NewDVine(3, trees)
	got, err := v.LogPDF([]float64{0.3, 0.5, 0.7})
	if err != nil {
		t.Fatalf("LogPDF interior point unexpected err: %v", err)
	}
	if math.IsNaN(got) || math.IsInf(got, 0) {
		t.Errorf("LogPDF returned non-finite %v on interior point", got)
	}
}

func TestDVine_LogPDF_RejectsBoundaryPoint(t *testing.T) {
	trees := [][]VineEdge{
		{{FamilyClayton, 2.0}, {FamilyClayton, 1.5}},
		{{FamilyClayton, 1.0}},
	}
	v, _ := NewDVine(3, trees)
	got, err := v.LogPDF([]float64{0.0, 0.5, 0.7})
	if err == nil {
		t.Errorf("expected boundary error from LogPDF, got nil (value=%v)", got)
	}
	if !math.IsInf(got, -1) {
		t.Errorf("expected -∞ on boundary, got %v", got)
	}
}

func TestDVine_LogPDF_RejectsLengthMismatch(t *testing.T) {
	trees := [][]VineEdge{
		{{FamilyClayton, 2.0}, {FamilyClayton, 1.5}},
		{{FamilyClayton, 1.0}},
	}
	v, _ := NewDVine(3, trees)
	if _, err := v.LogPDF([]float64{0.3, 0.5}); err == nil {
		t.Error("expected length-mismatch error from LogPDF")
	}
}
