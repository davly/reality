package agreement

// WeightScheme selects the disagreement-weighting scheme for WeightedKappa.
type WeightScheme int

const (
	// Linear weights disagreement proportionally to rank distance:
	// w(i,k) = |rank(i) - rank(k)|. Also known as Cicchetti-Allison
	// weights.
	Linear WeightScheme = iota
	// Quadratic weights disagreement by squared rank distance:
	// w(i,k) = (rank(i) - rank(k))^2. Also known as Fleiss-Cohen weights;
	// penalizes large misses more, small misses less, than Linear.
	Quadratic
)

// WeightedKappa returns Cohen's (1968) weighted kappa for two raters'
// ORDINAL ratings, crediting a near-miss disagreement (adjacent categories)
// less than a full miss (categories far apart) — the statistic CohenKappa
// cannot express, because it treats every non-exact-match as an equally
// total disagreement.
//
//	kappa_w = 1 - Do/De
//	Do      = sum_ck w(c,k) * o_ck
//	De      = sum_ck w(c,k) * e_ck
//
// where o_ck is the observed count of (rater1=c, rater2=k) pairs, e_ck =
// rowSum(c)*colSum(k)/n is the chance-expected count under the raters'
// marginals, and w(c,k) is 0 on the diagonal (exact agreement never
// contributes disagreement) and Linear/Quadratic rank distance off it.
//
// Category codes need only be in ascending rank order — rank distance is
// measured by POSITION in the sorted set of distinct codes observed across
// both raters, not by the raw numeric gap between codes. Unevenly-spaced
// ordinal codes (e.g. 1, 2, 5 used to mean three consecutive grades) still
// weight as ranks 0, 1, 2 — adjacent, not far apart.
//
// Returns ErrLengthMismatch / ErrEmptyInput as CohenKappa, and
// ErrDegenerateChanceAgreement if the weighted chance-expected disagreement
// De is 0 (fewer than 2 distinct categories are in play, so no weight is
// ever nonzero).
//
// # Golden vector 1: the k=2 reduction identity
//
// With exactly two categories there is only one possible rank distance
// (adjacent ranks 0 and 1), so BOTH Linear (|0-1|=1) and Quadratic
// ((0-1)^2=1) weighting collapse to the same weight CohenKappa implicitly
// uses (0 on the diagonal, 1 off it). WeightedKappa on Cohen's (1960)
// 50-grant-proposal example (see CohenKappa's doc comment) must therefore
// equal the unweighted kappa exactly, under either scheme: 0.40.
//
// # Golden vector 2: hand-derived 3-category example
//
// Confusion matrix (Never/Sometimes/Often, N=55), with the standard
// Cohen (1968) weighted-kappa formula applied directly and cross-checked
// by the algebraically-equivalent agreement-weight form kappa = (po(w) -
// pe(w)) / (1 - pe(w)) with w'(c,k) = 1 - w(c,k)/w_max (see
// weighted_test.go, which performs both derivations and asserts they
// agree with each other and with WeightedKappa's output):
//
//	           Never  Sometimes  Often
//	Never        5        3        2
//	Sometimes    2       11        4
//	Often        2        4       22
//
// linear kappa    = 1222/2377 ~= 0.514093
// quadratic kappa = 1846/3441 ~= 0.536472
//
// (quadratic > linear here because the disagreements that occur are
// concentrated near the diagonal — Quadratic forgives adjacent-category
// misses more than Linear does.)
//
// Reference: Cohen, J. (1968). Weighted kappa: Nominal scale agreement
// with provision for scaled disagreement or partial credit. Psychological
// Bulletin 70(4): 213-220.
func WeightedKappa(ratings1, ratings2 []int, scheme WeightScheme) (float64, error) {
	_, matrix, err := confusionMatrix(ratings1, ratings2)
	if err != nil {
		return 0, err
	}
	rowSum, colSum, n := marginals(matrix)

	weight := func(i, j int) float64 {
		d := float64(i - j)
		if d < 0 {
			d = -d
		}
		if scheme == Quadratic {
			return d * d
		}
		return d
	}

	var do, de float64
	for i := range matrix {
		for j := range matrix[i] {
			w := weight(i, j)
			if w == 0 {
				continue
			}
			do += w * matrix[i][j]
			de += w * (rowSum[i] * colSum[j] / n)
		}
	}
	if de == 0 {
		return 0, ErrDegenerateChanceAgreement
	}
	return 1 - do/de, nil
}
