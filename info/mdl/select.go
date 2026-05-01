package mdl

import "math"

// SelectMDL returns the index of the model with the smallest
// codelength in the supplied slice — the canonical
// "pick the model that compresses best" model-selection helper.
//
// Returns ErrEmptyModelList if the slice is empty.  Returns
// ErrNonFiniteCodeLength if any codelength is NaN or ±Inf — a
// non-finite codelength signals an upstream numerical-instability
// bug (an underflowed likelihood, an overflow in the regret term,
// etc.) and rather than silently propagating NaN through the
// argmin we surface the typed signal.
//
// Ties are broken by lower index (the first model with the minimum
// codelength wins).
//
// Worked example: an AR(p) lag-selection problem evaluates the
// codelength of the data under AR(1), AR(2), ..., AR(maxLag).  Pass
// the codelength slice in lag order; the returned index + 1 is the
// MDL-optimal lag.
func SelectMDL(modelCodeLengths []float64) (int, error) {
	if len(modelCodeLengths) == 0 {
		return 0, ErrEmptyModelList
	}
	for _, v := range modelCodeLengths {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return 0, ErrNonFiniteCodeLength
		}
	}
	bestIdx := 0
	best := modelCodeLengths[0]
	for i := 1; i < len(modelCodeLengths); i++ {
		if modelCodeLengths[i] < best {
			best = modelCodeLengths[i]
			bestIdx = i
		}
	}
	return bestIdx, nil
}

// SelectMDLWithMargin returns both the argmin index and the gap
// in nats to the second-best model — the "how confident is the
// MDL choice?" diagnostic.  A narrow margin (e.g. < 1 nat) means
// the codelength surface is shallow at the MDL minimum and the
// caller should report uncertainty rather than commit to the
// argmin selection.
//
// For len(modelCodeLengths) == 1, returns (0, +Inf, nil) — a
// single-model list is unambiguously its own minimum, with infinite
// margin to any (non-existent) alternative.
//
// Returns the same error shape as SelectMDL.
func SelectMDLWithMargin(modelCodeLengths []float64) (int, float64, error) {
	idx, err := SelectMDL(modelCodeLengths)
	if err != nil {
		return 0, 0, err
	}
	if len(modelCodeLengths) == 1 {
		return idx, math.Inf(1), nil
	}
	best := modelCodeLengths[idx]
	secondBest := math.Inf(1)
	for i, v := range modelCodeLengths {
		if i == idx {
			continue
		}
		if v < secondBest {
			secondBest = v
		}
	}
	return idx, secondBest - best, nil
}
