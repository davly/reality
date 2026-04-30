package autodiff

import (
	"math"
	"testing"
)

// =========================================================================
// Helpers
// =========================================================================

// finiteDiff returns d f / d x at x via central differences with step h.
// Used to validate analytic gradients.
func finiteDiff(f func(float64) float64, x, h float64) float64 {
	return (f(x+h) - f(x-h)) / (2 * h)
}

// =========================================================================
// Elementary ops
// =========================================================================

func TestAdd_Gradient(t *testing.T) {
	tape := NewTape()
	a := tape.Var(3.0)
	b := tape.Var(4.0)
	out := Add(a, b)
	if out.Val != 7.0 {
		t.Errorf("forward: %v, want 7", out.Val)
	}
	g := tape.Backward(out)
	if g[a.ID] != 1.0 || g[b.ID] != 1.0 {
		t.Errorf("grads = %v, want [1, 1]", g)
	}
}

func TestSub_Gradient(t *testing.T) {
	tape := NewTape()
	a := tape.Var(7.0)
	b := tape.Var(2.0)
	out := Sub(a, b)
	g := tape.Backward(out)
	if g[a.ID] != 1.0 || g[b.ID] != -1.0 {
		t.Errorf("grads = %v, want [1, -1]", g)
	}
}

func TestMul_Gradient(t *testing.T) {
	tape := NewTape()
	a := tape.Var(3.0)
	b := tape.Var(4.0)
	out := Mul(a, b)
	g := tape.Backward(out)
	if g[a.ID] != 4.0 || g[b.ID] != 3.0 {
		t.Errorf("grads = %v, want [4, 3]", g)
	}
}

func TestDiv_Gradient(t *testing.T) {
	tape := NewTape()
	a := tape.Var(8.0)
	b := tape.Var(2.0)
	out := Div(a, b)
	g := tape.Backward(out)
	if g[a.ID] != 0.5 {
		t.Errorf("g_a = %v, want 0.5", g[a.ID])
	}
	if g[b.ID] != -2.0 {
		t.Errorf("g_b = %v, want -2", g[b.ID])
	}
}

// =========================================================================
// Transcendental ops
// =========================================================================

func TestExp_Gradient(t *testing.T) {
	tape := NewTape()
	x := tape.Var(1.5)
	out := Exp(x)
	g := tape.Backward(out)
	if math.Abs(g[x.ID]-math.Exp(1.5)) > 1e-12 {
		t.Errorf("g = %v, want e^1.5 = %v", g[x.ID], math.Exp(1.5))
	}
}

func TestLog_Gradient(t *testing.T) {
	tape := NewTape()
	x := tape.Var(2.5)
	out := Log(x)
	g := tape.Backward(out)
	if math.Abs(g[x.ID]-1.0/2.5) > 1e-12 {
		t.Errorf("g = %v, want 0.4", g[x.ID])
	}
}

func TestSqrt_Gradient(t *testing.T) {
	tape := NewTape()
	x := tape.Var(9.0)
	out := Sqrt(x)
	if out.Val != 3.0 {
		t.Errorf("sqrt forward = %v, want 3", out.Val)
	}
	g := tape.Backward(out)
	if math.Abs(g[x.ID]-1.0/6.0) > 1e-12 {
		t.Errorf("g = %v, want 1/6", g[x.ID])
	}
}

func TestPow_Gradient(t *testing.T) {
	tape := NewTape()
	x := tape.Var(2.0)
	out := Pow(x, 3.0)
	if out.Val != 8.0 {
		t.Errorf("pow forward = %v, want 8", out.Val)
	}
	g := tape.Backward(out)
	if math.Abs(g[x.ID]-12.0) > 1e-12 { // 3 * 2^2
		t.Errorf("g = %v, want 12", g[x.ID])
	}
}

func TestSinCos_Gradient(t *testing.T) {
	tape := NewTape()
	x := tape.Var(0.7)
	s := Sin(x)
	gS := tape.Backward(s)
	if math.Abs(gS[x.ID]-math.Cos(0.7)) > 1e-12 {
		t.Errorf("d sin / dx = %v, want cos(0.7)", gS[x.ID])
	}

	tape2 := NewTape()
	y := tape2.Var(0.7)
	c := Cos(y)
	gC := tape2.Backward(c)
	if math.Abs(gC[y.ID]+math.Sin(0.7)) > 1e-12 {
		t.Errorf("d cos / dx = %v, want -sin(0.7)", gC[y.ID])
	}
}

func TestTanh_Gradient(t *testing.T) {
	tape := NewTape()
	x := tape.Var(0.3)
	out := Tanh(x)
	g := tape.Backward(out)
	want := 1.0 - math.Tanh(0.3)*math.Tanh(0.3)
	if math.Abs(g[x.ID]-want) > 1e-12 {
		t.Errorf("g = %v, want %v", g[x.ID], want)
	}
}

// =========================================================================
// Composite — chain rule on f(x) = (x^2 + 1) / sin(x)
// =========================================================================

func TestComposite_AgreesWithFiniteDiff(t *testing.T) {
	x0 := 1.3
	tape := NewTape()
	x := tape.Var(x0)

	x2 := Mul(x, x)
	num := AddConst(x2, 1.0)
	den := Sin(x)
	out := Div(num, den)

	g := tape.Backward(out)

	f := func(z float64) float64 { return (z*z + 1.0) / math.Sin(z) }
	wantG := finiteDiff(f, x0, 1e-6)
	if math.Abs(g[x.ID]-wantG) > 1e-6 {
		t.Errorf("autodiff g = %v, finite diff = %v", g[x.ID], wantG)
	}
}

// =========================================================================
// Multi-input — gradient of f(a, b, c) = a*b + sin(c) - log(a+b)
// =========================================================================

func TestMultiInput_GradientsAgreeWithFiniteDiff(t *testing.T) {
	a0, b0, c0 := 2.0, 3.0, 0.5
	tape := NewTape()
	a := tape.Var(a0)
	b := tape.Var(b0)
	c := tape.Var(c0)

	ab := Mul(a, b)
	sc := Sin(c)
	apb := Add(a, b)
	logApb := Log(apb)
	tmp := Add(ab, sc)
	out := Sub(tmp, logApb)

	g := tape.Backward(out)

	f := func(av, bv, cv float64) float64 {
		return av*bv + math.Sin(cv) - math.Log(av+bv)
	}
	dfa := (f(a0+1e-6, b0, c0) - f(a0-1e-6, b0, c0)) / 2e-6
	dfb := (f(a0, b0+1e-6, c0) - f(a0, b0-1e-6, c0)) / 2e-6
	dfc := (f(a0, b0, c0+1e-6) - f(a0, b0, c0-1e-6)) / 2e-6

	if math.Abs(g[a.ID]-dfa) > 1e-6 {
		t.Errorf("d/da = %v, want %v", g[a.ID], dfa)
	}
	if math.Abs(g[b.ID]-dfb) > 1e-6 {
		t.Errorf("d/db = %v, want %v", g[b.ID], dfb)
	}
	if math.Abs(g[c.ID]-dfc) > 1e-6 {
		t.Errorf("d/dc = %v, want %v", g[c.ID], dfc)
	}
}

// =========================================================================
// Vector ops
// =========================================================================

func TestSum_Gradient(t *testing.T) {
	tape := NewTape()
	xs := make([]*Variable, 5)
	for i := range xs {
		xs[i] = tape.Var(float64(i))
	}
	out := Sum(xs)
	if out.Val != 0+1+2+3+4 {
		t.Errorf("sum = %v, want 10", out.Val)
	}
	g := tape.Backward(out)
	for i, x := range xs {
		if g[x.ID] != 1.0 {
			t.Errorf("g[%d] = %v, want 1", i, g[x.ID])
		}
	}
}

func TestDot_Gradient(t *testing.T) {
	tape := NewTape()
	a := []*Variable{tape.Var(1), tape.Var(2), tape.Var(3)}
	b := []*Variable{tape.Var(4), tape.Var(5), tape.Var(6)}
	out := Dot(a, b)
	if out.Val != 1*4+2*5+3*6 {
		t.Errorf("dot = %v, want 32", out.Val)
	}
	g := tape.Backward(out)
	// d (a.b) / d a_i = b_i; d (a.b) / d b_i = a_i.
	wantA := []float64{4, 5, 6}
	wantB := []float64{1, 2, 3}
	for i := range a {
		if g[a[i].ID] != wantA[i] {
			t.Errorf("g[a%d] = %v, want %v", i, g[a[i].ID], wantA[i])
		}
		if g[b[i].ID] != wantB[i] {
			t.Errorf("g[b%d] = %v, want %v", i, g[b[i].ID], wantB[i])
		}
	}
}

func TestMSE_Gradient(t *testing.T) {
	tape := NewTape()
	pred := []*Variable{tape.Var(1.0), tape.Var(2.0), tape.Var(3.0)}
	target := []float64{1.5, 2.5, 2.5}
	out := MeanSquaredError(pred, target)
	want := 0.5 * (0.25 + 0.25 + 0.25) / 3.0
	if math.Abs(out.Val-want) > 1e-12 {
		t.Errorf("MSE = %v, want %v", out.Val, want)
	}
	g := tape.Backward(out)
	// d MSE / d pred_i = (pred_i - target_i) / n.
	wantGrad := []float64{-0.5 / 3, -0.5 / 3, 0.5 / 3}
	for i, p := range pred {
		if math.Abs(g[p.ID]-wantGrad[i]) > 1e-12 {
			t.Errorf("g[%d] = %v, want %v", i, g[p.ID], wantGrad[i])
		}
	}
}

// =========================================================================
// Calibration end-to-end — fit y = a * x + b by gradient descent on MSE
// =========================================================================

func TestEndToEnd_LinearRegressionByGradientDescent(t *testing.T) {
	xs := []float64{0.0, 1.0, 2.0, 3.0, 4.0}
	ys := []float64{1.0, 3.1, 5.0, 7.1, 9.0} // y = 2x + 1 + small noise
	a := 0.0
	b := 0.0
	lr := 0.05
	for step := 0; step < 1000; step++ {
		tape := NewTape()
		aV := tape.Var(a)
		bV := tape.Var(b)
		preds := make([]*Variable, len(xs))
		for i, xi := range xs {
			ax := MulConst(aV, xi)
			preds[i] = Add(ax, bV)
		}
		loss := MeanSquaredError(preds, ys)
		g := tape.Backward(loss)
		a -= lr * g[aV.ID]
		b -= lr * g[bV.ID]
	}
	if math.Abs(a-2.0) > 0.05 {
		t.Errorf("a = %v, want close to 2", a)
	}
	if math.Abs(b-1.0) > 0.05 {
		t.Errorf("b = %v, want close to 1", b)
	}
}

// =========================================================================
// Cross-tape rejection
// =========================================================================

func TestCrossTape_Panics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("combining variables from different tapes should panic")
		}
	}()
	t1 := NewTape()
	t2 := NewTape()
	a := t1.Var(1)
	b := t2.Var(2)
	Add(a, b)
}

// =========================================================================
// Determinism
// =========================================================================

func TestBackward_Deterministic(t *testing.T) {
	run := func() float64 {
		tape := NewTape()
		x := tape.Var(0.7)
		out := Add(Mul(x, x), Sin(x))
		g := tape.Backward(out)
		return g[x.ID]
	}
	a := run()
	b := run()
	if a != b {
		t.Errorf("non-deterministic: %v vs %v", a, b)
	}
}
