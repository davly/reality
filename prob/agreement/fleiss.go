package agreement

// FleissKappa returns Fleiss' (1971) kappa, a chance-corrected measure of
// agreement among m >= 2 raters classifying N subjects into k nominal
// categories — the generalization of Cohen's kappa beyond two raters (the
// raters classifying each subject need not be the same individuals across
// subjects, only the same COUNT).
//
// ratings[i][j] is the number of raters who assigned subject i to
// category j. Every row must sum to the same total m (a fixed number of
// raters per subject, m >= 2); a ragged panel size (a variable number of
// raters per subject) is rejected with ErrRaggedRows because Fleiss'
// formula assumes a constant m.
//
//	P_i    = (sum_j n_ij^2 - m) / (m*(m-1))   -- agreement on subject i
//	Pbar   = mean_i(P_i)                       -- observed agreement
//	p_j    = (sum_i n_ij) / (N*m)              -- category j's marginal rate
//	Pbar_e = sum_j p_j^2                       -- chance-expected agreement
//	kappa  = (Pbar - Pbar_e) / (1 - Pbar_e)
//
// Returns ErrEmptyInput if ratings (or its first row) has zero length,
// ErrTooFewRaters if m < 2, ErrNegativeCount if any count is negative,
// ErrRaggedRows if rows have differing lengths or row totals, and
// ErrDegenerateChanceAgreement if Pbar_e == 1.
//
// Golden vector — Fleiss (1971) Table 1: 10 psychiatric patients each
// diagnosed by the same 14 psychiatrists into 5 categories (Depression,
// Personality Disorder, Schizophrenia, Neurosis, Other), reproduced in
// Wikipedia's "Fleiss' kappa" article. Every P_i below was independently
// re-derived from the table and matches the article's stated per-subject
// values and Pbar=0.378, Pbar_e=0.213, kappa=0.210 (see fleiss_test.go):
//
//	           Cat1 Cat2 Cat3 Cat4 Cat5    P_i
//	Patient1:    0    0    0    0   14    1.000
//	Patient2:    0    2    6    4    2    0.253
//	Patient3:    0    0    3    5    6    0.308
//	Patient4:    0    3    9    2    0    0.440
//	Patient5:    2    2    8    1    1    0.330
//	Patient6:    7    7    0    0    0    0.462
//	Patient7:    3    2    6    3    0    0.242
//	Patient8:    2    5    3    2    2    0.176
//	Patient9:    6    5    2    1    0    0.286
//	Patient10:   0    2    2    3    7    0.286
//
// kappa = 4211/20059 ~= 0.209910 (rounds to the published 0.210).
//
// Reference: Fleiss, J. L. (1971). Measuring nominal scale agreement among
// many raters. Psychological Bulletin 76(5): 378-382.
func FleissKappa(ratings [][]int) (float64, error) {
	n := len(ratings)
	if n == 0 || len(ratings[0]) == 0 {
		return 0, ErrEmptyInput
	}
	k := len(ratings[0])

	m := 0
	for _, cnt := range ratings[0] {
		m += cnt
	}
	if m < 2 {
		return 0, ErrTooFewRaters
	}

	colSum := make([]float64, k)
	var pBarSum float64
	for _, row := range ratings {
		if len(row) != k {
			return 0, ErrRaggedRows
		}
		rowTotal := 0
		var sumSq float64
		for j, nij := range row {
			if nij < 0 {
				return 0, ErrNegativeCount
			}
			rowTotal += nij
			sumSq += float64(nij) * float64(nij)
			colSum[j] += float64(nij)
		}
		if rowTotal != m {
			return 0, ErrRaggedRows
		}
		pBarSum += (sumSq - float64(m)) / (float64(m) * float64(m-1))
	}
	pBar := pBarSum / float64(n)

	var pBarE float64
	nm := float64(n) * float64(m)
	for _, cs := range colSum {
		p := cs / nm
		pBarE += p * p
	}

	if pBarE == 1 {
		return 0, ErrDegenerateChanceAgreement
	}
	return (pBar - pBarE) / (1 - pBarE), nil
}
