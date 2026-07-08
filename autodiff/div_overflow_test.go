package autodiff

import (
	"math"
	"testing"
)

// TestDiv_LargeDenominatorGradientNoOverflow pins the fix. The Div pullback
// computed grad_b = -g*a/(b*b); forming b*b overflows to +Inf for
// |b| > sqrt(MaxFloat64) ~= 1.34e154, collapsing the gradient to 0 even when
// -a/b^2 is finite and representable. The reformulation (-g*a/b)/b avoids it.
func TestDiv_LargeDenominatorGradientNoOverflow(t *testing.T) {
	tape := NewTape()
	a := tape.Var(1e200)
	b := tape.Var(1e160)
	y := Div(a, b)
	grads := tape.Backward(y)
	got := grads[b.ID]
	want := -1e200 / 1e160 / 1e160 // -a/b^2 = -1e-120
	if got == 0 || math.IsNaN(got) || math.IsInf(got, 0) {
		t.Fatalf("grad_b = %v, want finite ~%v (b*b overflow collapsed it to 0)", got, want)
	}
	if math.Abs(got-want) > math.Abs(want)*1e-9 {
		t.Errorf("grad_b = %v, want %v", got, want)
	}
}
