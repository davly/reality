package autodiff

import "math"

// Add returns a + b.  Pullback: grad_a += g; grad_b += g.
func Add(a, b *Variable) *Variable {
	requireSame(a, b)
	aID, bID := a.ID, b.ID
	return a.Tape.register(a.Val+b.Val, func(g float64, grads []float64) {
		grads[aID] += g
		grads[bID] += g
	})
}

// Sub returns a - b.  Pullback: grad_a += g; grad_b -= g.
func Sub(a, b *Variable) *Variable {
	requireSame(a, b)
	aID, bID := a.ID, b.ID
	return a.Tape.register(a.Val-b.Val, func(g float64, grads []float64) {
		grads[aID] += g
		grads[bID] -= g
	})
}

// Mul returns a * b.  Pullback: grad_a += g*b; grad_b += g*a.
func Mul(a, b *Variable) *Variable {
	requireSame(a, b)
	aID, bID := a.ID, b.ID
	aVal, bVal := a.Val, b.Val
	return a.Tape.register(aVal*bVal, func(g float64, grads []float64) {
		grads[aID] += g * bVal
		grads[bID] += g * aVal
	})
}

// Div returns a / b.  Pullback: grad_a += g/b; grad_b += -g*a / b^2.
func Div(a, b *Variable) *Variable {
	requireSame(a, b)
	aID, bID := a.ID, b.ID
	aVal, bVal := a.Val, b.Val
	return a.Tape.register(aVal/bVal, func(g float64, grads []float64) {
		grads[aID] += g / bVal
		grads[bID] += -g * aVal / (bVal * bVal)
	})
}

// Neg returns -a.  Pullback: grad_a += -g.
func Neg(a *Variable) *Variable {
	aID := a.ID
	return a.Tape.register(-a.Val, func(g float64, grads []float64) {
		grads[aID] -= g
	})
}

// AddConst returns a + c.  Pullback: grad_a += g.
func AddConst(a *Variable, c float64) *Variable {
	aID := a.ID
	return a.Tape.register(a.Val+c, func(g float64, grads []float64) {
		grads[aID] += g
	})
}

// MulConst returns c * a.  Pullback: grad_a += g*c.
func MulConst(a *Variable, c float64) *Variable {
	aID := a.ID
	return a.Tape.register(c*a.Val, func(g float64, grads []float64) {
		grads[aID] += g * c
	})
}

// =========================================================================
// Transcendental ops
// =========================================================================

// Exp returns e^a.  Pullback: grad_a += g * e^a.
func Exp(a *Variable) *Variable {
	aID := a.ID
	v := math.Exp(a.Val)
	return a.Tape.register(v, func(g float64, grads []float64) {
		grads[aID] += g * v
	})
}

// Log returns log(a) (natural).  Pullback: grad_a += g / a.  Caller must
// ensure a.Val > 0; behaviour at non-positive a is undefined.
func Log(a *Variable) *Variable {
	aID := a.ID
	aVal := a.Val
	return a.Tape.register(math.Log(aVal), func(g float64, grads []float64) {
		grads[aID] += g / aVal
	})
}

// Sqrt returns sqrt(a).  Pullback: grad_a += g / (2 * sqrt(a)).  Caller
// must ensure a.Val >= 0.
func Sqrt(a *Variable) *Variable {
	aID := a.ID
	v := math.Sqrt(a.Val)
	return a.Tape.register(v, func(g float64, grads []float64) {
		grads[aID] += g / (2.0 * v)
	})
}

// Pow returns a ^ p where p is a scalar constant.  Pullback: grad_a +=
// g * p * a^(p-1).
func Pow(a *Variable, p float64) *Variable {
	aID := a.ID
	aVal := a.Val
	return a.Tape.register(math.Pow(aVal, p), func(g float64, grads []float64) {
		grads[aID] += g * p * math.Pow(aVal, p-1.0)
	})
}

// Sin returns sin(a).  Pullback: grad_a += g * cos(a).
func Sin(a *Variable) *Variable {
	aID := a.ID
	aVal := a.Val
	return a.Tape.register(math.Sin(aVal), func(g float64, grads []float64) {
		grads[aID] += g * math.Cos(aVal)
	})
}

// Cos returns cos(a).  Pullback: grad_a += -g * sin(a).
func Cos(a *Variable) *Variable {
	aID := a.ID
	aVal := a.Val
	return a.Tape.register(math.Cos(aVal), func(g float64, grads []float64) {
		grads[aID] -= g * math.Sin(aVal)
	})
}

// Tanh returns tanh(a).  Pullback: grad_a += g * (1 - tanh(a)^2).  Useful
// for neural-network style nonlinearities and for smoothing the regime-
// continuation hazard in BOCPD calibration.
func Tanh(a *Variable) *Variable {
	aID := a.ID
	v := math.Tanh(a.Val)
	return a.Tape.register(v, func(g float64, grads []float64) {
		grads[aID] += g * (1.0 - v*v)
	})
}
