package autodiff

import (
	"math"
	"testing"
)

// =========================================================================
// Expansion coverage for autodiff. Pre-this: 18 tests.
// Adds: Neg/AddConst/MulConst pullbacks, panics on cross-tape vector ops,
// Sum/Dot/MSE empty + length panics, gradient of constants is zero,
// Backward on a leaf gives unit grad, NewTape produces clean state,
// Pow at p=0 + p=1, transcendental composition determinism.
// =========================================================================

// --- Single-op pullbacks not in the existing suite ---------------------

func TestNeg_Gradient(t *testing.T) {
	tape := NewTape()
	x := tape.Var(3.0)
	out := Neg(x)
	if out.Val != -3.0 {
		t.Errorf("Neg(3).Val = %v, want -3", out.Val)
	}
	g := tape.Backward(out)
	if g[x.ID] != -1.0 {
		t.Errorf("d(-x)/dx = %v, want -1", g[x.ID])
	}
}

func TestAddConst_Gradient(t *testing.T) {
	tape := NewTape()
	x := tape.Var(2.0)
	out := AddConst(x, 5.0)
	if out.Val != 7.0 {
		t.Errorf("AddConst(2,5).Val = %v, want 7", out.Val)
	}
	g := tape.Backward(out)
	if g[x.ID] != 1.0 {
		t.Errorf("d(x+5)/dx = %v, want 1", g[x.ID])
	}
}

func TestMulConst_Gradient(t *testing.T) {
	tape := NewTape()
	x := tape.Var(4.0)
	out := MulConst(x, 3.0)
	if out.Val != 12.0 {
		t.Errorf("MulConst(4,3).Val = %v, want 12", out.Val)
	}
	g := tape.Backward(out)
	if g[x.ID] != 3.0 {
		t.Errorf("d(3x)/dx = %v, want 3", g[x.ID])
	}
}

// --- Pow special powers ---------------------------------------------------

func TestPow_PEqualsZero_GradientIsZero(t *testing.T) {
	// d/dx [x^0] = 0 for x != 0.
	tape := NewTape()
	x := tape.Var(2.5)
	out := Pow(x, 0.0)
	if out.Val != 1.0 {
		t.Errorf("Pow(x,0).Val = %v, want 1", out.Val)
	}
	g := tape.Backward(out)
	// Pullback formula: g * 0 * x^(-1) = 0 (since 0 * anything = 0).
	if math.Abs(g[x.ID]) > 1e-12 {
		t.Errorf("d(x^0)/dx = %v, want 0", g[x.ID])
	}
}

func TestPow_PEqualsOne_GradientIsOne(t *testing.T) {
	tape := NewTape()
	x := tape.Var(7.0)
	out := Pow(x, 1.0)
	if out.Val != 7.0 {
		t.Errorf("Pow(x,1).Val = %v, want 7", out.Val)
	}
	g := tape.Backward(out)
	if math.Abs(g[x.ID]-1.0) > 1e-12 {
		t.Errorf("d(x^1)/dx = %v, want 1", g[x.ID])
	}
}

// --- Backward on a leaf --------------------------------------------------

func TestBackward_OnLeafItself_GradOne(t *testing.T) {
	tape := NewTape()
	x := tape.Var(3.0)
	g := tape.Backward(x)
	if g[x.ID] != 1.0 {
		t.Errorf("Backward(leaf)[leaf] = %v, want 1", g[x.ID])
	}
}

func TestBackward_PanicsOnNil(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("Backward(nil) should panic")
		}
	}()
	tape := NewTape()
	tape.Backward(nil)
}

// --- Cross-tape panics for vector ops ------------------------------------

func TestSum_CrossTape_Panics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("Sum across tapes should panic")
		}
	}()
	t1 := NewTape()
	t2 := NewTape()
	a := t1.Var(1.0)
	b := t2.Var(2.0)
	Sum([]*Variable{a, b})
}

func TestSum_Empty_Panics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("Sum([]) should panic")
		}
	}()
	Sum(nil)
}

func TestDot_CrossTape_Panics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("Dot across tapes should panic")
		}
	}()
	t1 := NewTape()
	t2 := NewTape()
	a := []*Variable{t1.Var(1.0)}
	b := []*Variable{t2.Var(2.0)}
	Dot(a, b)
}

func TestDot_LengthMismatch_Panics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("Dot length mismatch should panic")
		}
	}()
	tape := NewTape()
	Dot([]*Variable{tape.Var(1)}, []*Variable{tape.Var(2), tape.Var(3)})
}

func TestDot_Empty_Panics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("Dot([], []) should panic")
		}
	}()
	Dot(nil, nil)
}

func TestMSE_CrossTape_Panics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("MSE across tapes should panic")
		}
	}()
	t1 := NewTape()
	t2 := NewTape()
	pred := []*Variable{t1.Var(1.0), t2.Var(2.0)}
	MeanSquaredError(pred, []float64{1, 2})
}

func TestMSE_LengthMismatch_Panics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("MSE length mismatch should panic")
		}
	}()
	tape := NewTape()
	MeanSquaredError([]*Variable{tape.Var(1)}, []float64{1, 2})
}

func TestMSE_Empty_Panics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("MSE([], []) should panic")
		}
	}()
	MeanSquaredError(nil, nil)
}

// --- Constants don't accumulate gradient via downstream ops ------------

func TestConstant_GradientFlowsThrough(t *testing.T) {
	// Constant() and Var() are equivalent in the MVP — gradients still
	// accumulate on them. Document this so behaviour is locked.
	tape := NewTape()
	c := tape.Constant(3.0)
	x := tape.Var(2.0)
	out := Mul(c, x)
	if out.Val != 6.0 {
		t.Errorf("3 * 2 = %v, want 6", out.Val)
	}
	g := tape.Backward(out)
	if math.Abs(g[c.ID]-2.0) > 1e-12 {
		t.Errorf("d(c*x)/dc = %v, want 2 (Constant currently == Var)", g[c.ID])
	}
}

// --- Sum gradient is uniform 1.0 ----------------------------------------

func TestSum_AllGradientsAreOne(t *testing.T) {
	tape := NewTape()
	xs := []*Variable{tape.Var(1), tape.Var(2), tape.Var(3), tape.Var(4)}
	out := Sum(xs)
	if out.Val != 10 {
		t.Errorf("Sum=%v, want 10", out.Val)
	}
	g := tape.Backward(out)
	for _, x := range xs {
		if math.Abs(g[x.ID]-1.0) > 1e-12 {
			t.Errorf("d(sum)/dx[%d] = %v, want 1", x.ID, g[x.ID])
		}
	}
}

// --- Determinism + idempotency of values (Backward double-call docs) ---

func TestBackward_RecomputesGradientsFromScratch(t *testing.T) {
	// Two separate tapes evaluating the same function should give the
	// same gradient. Locks the per-Tape statelessness invariant.
	build := func() (*Tape, *Variable, *Variable) {
		tape := NewTape()
		x := tape.Var(2.0)
		out := Pow(x, 3.0) // d/dx = 3x^2 = 12
		return tape, x, out
	}
	t1, x1, o1 := build()
	t2, x2, o2 := build()
	g1 := t1.Backward(o1)
	g2 := t2.Backward(o2)
	if g1[x1.ID] != g2[x2.ID] {
		t.Errorf("non-reproducible: %v vs %v", g1[x1.ID], g2[x2.ID])
	}
	if math.Abs(g1[x1.ID]-12.0) > 1e-12 {
		t.Errorf("d(x^3)/dx at x=2 = %v, want 12", g1[x1.ID])
	}
}

// --- Composition golden values ----------------------------------------

func TestComposition_Tanh_Of_Mul(t *testing.T) {
	// f(x, y) = tanh(x * y) at (x=1, y=2): tanh(2) ≈ 0.96402758
	// df/dx = y * (1 - tanh(xy)^2)
	// df/dy = x * (1 - tanh(xy)^2)
	tape := NewTape()
	x := tape.Var(1.0)
	y := tape.Var(2.0)
	out := Tanh(Mul(x, y))
	want := math.Tanh(2.0)
	if math.Abs(out.Val-want) > 1e-12 {
		t.Errorf("tanh(2) = %v, want %v", out.Val, want)
	}
	g := tape.Backward(out)
	dxdy := 1.0 - math.Tanh(2.0)*math.Tanh(2.0)
	if math.Abs(g[x.ID]-2.0*dxdy) > 1e-12 {
		t.Errorf("df/dx = %v, want %v", g[x.ID], 2.0*dxdy)
	}
	if math.Abs(g[y.ID]-1.0*dxdy) > 1e-12 {
		t.Errorf("df/dy = %v, want %v", g[y.ID], 1.0*dxdy)
	}
}

func TestComposition_LogExp_Identity_Gradient(t *testing.T) {
	// log(exp(x)) = x → d/dx = 1 (analytically).
	tape := NewTape()
	x := tape.Var(3.7)
	out := Log(Exp(x))
	g := tape.Backward(out)
	if math.Abs(g[x.ID]-1.0) > 1e-9 {
		t.Errorf("d log(exp(x))/dx = %v, want 1", g[x.ID])
	}
}

// --- Tape state inspection (basic invariants) -------------------------

func TestNewTape_StartsEmpty(t *testing.T) {
	tape := NewTape()
	if len(tape.nodes) != 0 {
		t.Errorf("new tape should have 0 nodes, got %d", len(tape.nodes))
	}
}

func TestVar_AssignsSequentialIDs(t *testing.T) {
	tape := NewTape()
	a := tape.Var(1.0)
	b := tape.Var(2.0)
	c := tape.Var(3.0)
	if a.ID != 0 || b.ID != 1 || c.ID != 2 {
		t.Errorf("IDs not sequential: %d %d %d", a.ID, b.ID, c.ID)
	}
}
