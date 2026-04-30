package autodiff

// =========================================================================
// Vector ops over slices of *Variable
//
// These compose elementary ops but live as their own functions so that
// the pullback can be written once efficiently rather than via a chain of
// per-element Add/Mul registrations.
// =========================================================================

// Sum returns sum_i xs[i].  Pullback: grad_xs[i] += g for every i.
// Panics if xs is empty (use a Tape.Constant(0) instead).
func Sum(xs []*Variable) *Variable {
	if len(xs) == 0 {
		panic("autodiff: Sum requires at least one variable")
	}
	tape := xs[0].Tape
	ids := make([]int, len(xs))
	var v float64
	for i, x := range xs {
		if x.Tape != tape {
			panic("autodiff: Sum requires all Variables on the same Tape")
		}
		ids[i] = x.ID
		v += x.Val
	}
	return tape.register(v, func(g float64, grads []float64) {
		for _, id := range ids {
			grads[id] += g
		}
	})
}

// Dot returns sum_i a[i] * b[i].  a and b must have equal length and live
// on the same Tape.  Pullback: grad_a[i] += g * b[i]; grad_b[i] += g *
// a[i].
func Dot(a, b []*Variable) *Variable {
	if len(a) != len(b) {
		panic("autodiff: Dot requires equal-length slices")
	}
	if len(a) == 0 {
		panic("autodiff: Dot requires non-empty slices")
	}
	tape := a[0].Tape
	aIDs := make([]int, len(a))
	bIDs := make([]int, len(b))
	aVals := make([]float64, len(a))
	bVals := make([]float64, len(b))
	var v float64
	for i := range a {
		if a[i].Tape != tape || b[i].Tape != tape {
			panic("autodiff: Dot requires all Variables on the same Tape")
		}
		aIDs[i] = a[i].ID
		bIDs[i] = b[i].ID
		aVals[i] = a[i].Val
		bVals[i] = b[i].Val
		v += a[i].Val * b[i].Val
	}
	return tape.register(v, func(g float64, grads []float64) {
		for i := range aIDs {
			grads[aIDs[i]] += g * bVals[i]
			grads[bIDs[i]] += g * aVals[i]
		}
	})
}

// MeanSquaredError returns (1 / 2n) * sum_i (pred[i] - target[i])^2.
// pred is a slice of Variables; target is a slice of float64 constants.
// Useful as a calibration loss.
func MeanSquaredError(pred []*Variable, target []float64) *Variable {
	if len(pred) != len(target) {
		panic("autodiff: MeanSquaredError requires equal-length slices")
	}
	if len(pred) == 0 {
		panic("autodiff: MeanSquaredError requires non-empty slices")
	}
	tape := pred[0].Tape
	ids := make([]int, len(pred))
	resid := make([]float64, len(pred))
	var sse float64
	for i, p := range pred {
		if p.Tape != tape {
			panic("autodiff: MeanSquaredError requires all Variables on the same Tape")
		}
		ids[i] = p.ID
		r := p.Val - target[i]
		resid[i] = r
		sse += r * r
	}
	n := float64(len(pred))
	return tape.register(0.5*sse/n, func(g float64, grads []float64) {
		// d MSE / d pred_i = (pred_i - target_i) / n
		for i, id := range ids {
			grads[id] += g * resid[i] / n
		}
	})
}
