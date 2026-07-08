package agreement

// CohenKappa returns Cohen's (1960) kappa coefficient of chance-corrected
// agreement between two raters' categorical ratings of the same n units:
//
//	kappa = (po - pe) / (1 - pe)
//
// where po is the observed proportion of agreement and pe is the
// proportion of agreement expected if each rater assigned categories
// independently at their own observed marginal rate. kappa = 1 for perfect
// agreement, kappa = 0 for agreement no better than chance, and kappa can
// go negative for agreement systematically worse than chance.
//
// ratings1 and ratings2 must be the same length (one entry per unit,
// positionally paired) and non-empty. Category codes are arbitrary
// distinct ints treated as NOMINAL labels — order carries no meaning; use
// WeightedKappa when categories are ordinal and near-miss disagreement
// should count for less than a full miss.
//
// Returns ErrLengthMismatch if the inputs differ in length, ErrEmptyInput
// if they are empty, and ErrDegenerateChanceAgreement if pe == 1 (no
// variability across categories to correct for chance against — the ratio
// is 0/0).
//
// Golden vector — Cohen (1960), reproduced in the Wikipedia "Cohen's
// kappa" worked example: 50 grant proposals, two readers, Yes/No:
//
//	                Reader B: Yes   Reader B: No
//	Reader A: Yes        20              5
//	Reader A: No         10             15
//
// po = (20+15)/50 = 0.70
// pe = (25/50)*(30/50) + (25/50)*(20/50) = 0.30 + 0.20 = 0.50
// kappa = (0.70 - 0.50) / (1 - 0.50) = 0.40
//
// Reference: Cohen, J. (1960). A coefficient of agreement for nominal
// scales. Educational and Psychological Measurement 20(1): 37-46.
func CohenKappa(ratings1, ratings2 []int) (float64, error) {
	_, matrix, err := confusionMatrix(ratings1, ratings2)
	if err != nil {
		return 0, err
	}
	po, pe := observedExpectedAgreement(matrix)
	if pe == 1 {
		return 0, ErrDegenerateChanceAgreement
	}
	return (po - pe) / (1 - pe), nil
}

// observedExpectedAgreement computes po (observed agreement proportion)
// and pe (chance-expected agreement proportion) from a k x k contingency
// matrix — the shared core of CohenKappa.
func observedExpectedAgreement(matrix [][]float64) (po, pe float64) {
	rowSum, colSum, n := marginals(matrix)
	for i := range matrix {
		po += matrix[i][i]
	}
	po /= n
	for i := range rowSum {
		pe += (rowSum[i] / n) * (colSum[i] / n)
	}
	return po, pe
}
