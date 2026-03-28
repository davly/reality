package testutil

import (
	"math"
	"testing"
)

// TestLoadGolden verifies that LoadGolden correctly parses a sample golden file.
func TestLoadGolden(t *testing.T) {
	gf := LoadGolden(t, "testdata/sample_golden.json")

	if gf.Function != "TestUtil.SampleAdd" {
		t.Errorf("Function = %q, want %q", gf.Function, "TestUtil.SampleAdd")
	}

	if len(gf.Cases) != 5 {
		t.Fatalf("got %d cases, want 5", len(gf.Cases))
	}

	// Verify first case was parsed correctly.
	tc := gf.Cases[0]
	if tc.Description != "simple addition 1 + 2 = 3" {
		t.Errorf("case 0 description = %q", tc.Description)
	}
	if tc.Tolerance != 0.0 {
		t.Errorf("case 0 tolerance = %v, want 0", tc.Tolerance)
	}

	a := InputFloat64(t, tc, "a")
	b := InputFloat64(t, tc, "b")
	if a != 1.0 || b != 2.0 {
		t.Errorf("case 0 inputs: a=%v, b=%v", a, b)
	}
}

// TestAssertFloat64_Pass verifies that matching values pass.
func TestAssertFloat64_Pass(t *testing.T) {
	gf := LoadGolden(t, "testdata/sample_golden.json")

	for _, tc := range gf.Cases {
		a := InputFloat64(t, tc, "a")
		b := InputFloat64(t, tc, "b")
		got := a + b
		AssertFloat64(t, tc, got)
	}
}

// TestAssertFloat64_NaN verifies that NaN == NaN passes in golden checks.
func TestAssertFloat64_NaN(t *testing.T) {
	tc := TestCase{
		Description: "NaN equality",
		Expected:    math.NaN(),
		Tolerance:   0,
	}
	// This should not fail — NaN matches NaN in our system.
	AssertFloat64(t, tc, math.NaN())
}

// TestAssertFloat64_PosInf verifies that +Inf == +Inf passes.
func TestAssertFloat64_PosInf(t *testing.T) {
	tc := TestCase{
		Description: "+Inf equality",
		Expected:    math.Inf(1),
		Tolerance:   0,
	}
	AssertFloat64(t, tc, math.Inf(1))
}

// TestAssertFloat64_NegInf verifies that -Inf == -Inf passes.
func TestAssertFloat64_NegInf(t *testing.T) {
	tc := TestCase{
		Description: "-Inf equality",
		Expected:    math.Inf(-1),
		Tolerance:   0,
	}
	AssertFloat64(t, tc, math.Inf(-1))
}

// TestAssertFloat64Slice_Pass verifies element-wise slice comparison.
func TestAssertFloat64Slice_Pass(t *testing.T) {
	gf := LoadGolden(t, "testdata/sample_vector_golden.json")

	for _, tc := range gf.Cases {
		vec := InputFloat64Slice(t, tc, "vec")
		scale := InputFloat64(t, tc, "scale")

		got := make([]float64, len(vec))
		for i, v := range vec {
			got[i] = v * scale
		}

		AssertFloat64Slice(t, tc, got)
	}
}

// TestInputFloat64_Extraction verifies input extraction works for various types.
func TestInputFloat64_Extraction(t *testing.T) {
	tc := TestCase{
		Description: "input extraction",
		Inputs: map[string]any{
			"float":   3.14,
			"integer": float64(42), // JSON numbers always decode as float64
		},
	}

	f := InputFloat64(t, tc, "float")
	if f != 3.14 {
		t.Errorf("float input = %v, want 3.14", f)
	}

	i := InputFloat64(t, tc, "integer")
	if i != 42.0 {
		t.Errorf("integer input = %v, want 42", i)
	}
}

// TestToFloat64_Conversions verifies internal type conversion.
func TestToFloat64_Conversions(t *testing.T) {
	tests := []struct {
		name string
		val  any
		want float64
		ok   bool
	}{
		{"float64", float64(3.14), 3.14, true},
		{"int", int(42), 42.0, true},
		{"int64", int64(100), 100.0, true},
		{"string", "hello", 0, false},
		{"nil", nil, 0, false},
	}

	for _, tt := range tests {
		got, ok := toFloat64(tt.val)
		if ok != tt.ok {
			t.Errorf("toFloat64(%v): ok = %v, want %v", tt.val, ok, tt.ok)
		}
		if ok && got != tt.want {
			t.Errorf("toFloat64(%v) = %v, want %v", tt.val, got, tt.want)
		}
	}
}

// TestToFloat64Slice_Conversions verifies slice conversion.
func TestToFloat64Slice_Conversions(t *testing.T) {
	// Valid slice
	arr := []any{float64(1), float64(2), float64(3)}
	got, ok := toFloat64Slice(arr)
	if !ok {
		t.Fatal("toFloat64Slice: expected ok=true for valid slice")
	}
	if len(got) != 3 || got[0] != 1 || got[1] != 2 || got[2] != 3 {
		t.Errorf("toFloat64Slice = %v, want [1 2 3]", got)
	}

	// Invalid: not a slice
	_, ok = toFloat64Slice("not a slice")
	if ok {
		t.Error("toFloat64Slice: expected ok=false for string")
	}

	// Invalid: slice with non-numeric element
	bad := []any{float64(1), "two", float64(3)}
	_, ok = toFloat64Slice(bad)
	if ok {
		t.Error("toFloat64Slice: expected ok=false for mixed slice")
	}
}

// TestLoadGolden_VectorFile verifies loading of vector golden files.
func TestLoadGolden_VectorFile(t *testing.T) {
	gf := LoadGolden(t, "testdata/sample_vector_golden.json")

	if gf.Function != "TestUtil.SampleVector" {
		t.Errorf("Function = %q, want %q", gf.Function, "TestUtil.SampleVector")
	}

	if len(gf.Cases) != 3 {
		t.Fatalf("got %d cases, want 3", len(gf.Cases))
	}
}
