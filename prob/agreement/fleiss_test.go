package agreement

import "testing"

// TestFleissKappa_Fleiss1971Table1 reproduces Fleiss (1971) Table 1 — 10
// psychiatric patients, each diagnosed by 14 psychiatrists into 5
// categories — as given in Wikipedia's "Fleiss' kappa" article.
//
// Each row's P_i below was independently re-derived by hand from the
// n_ij matrix ((sum_j n_ij^2 - 14) / (14*13)) and matches the article's
// stated per-subject values exactly:
//
//	Patient1:  (0,0,0,0,14) -> P_i = 1.000
//	Patient2:  (0,2,6,4,2)  -> P_i = 0.253
//	Patient3:  (0,0,3,5,6)  -> P_i = 0.308
//	Patient4:  (0,3,9,2,0)  -> P_i = 0.440
//	Patient5:  (2,2,8,1,1)  -> P_i = 0.330
//	Patient6:  (7,7,0,0,0)  -> P_i = 0.462
//	Patient7:  (3,2,6,3,0)  -> P_i = 0.242
//	Patient8:  (2,5,3,2,2)  -> P_i = 0.176
//	Patient9:  (6,5,2,1,0)  -> P_i = 0.286
//	Patient10: (0,2,2,3,7)  -> P_i = 0.286
//
// Pbar = 0.378, Pbar_e = 0.213 (published), kappa = 4211/20059 ~= 0.209910
// (rounds to the published 0.210).
func TestFleissKappa_Fleiss1971Table1(t *testing.T) {
	ratings := [][]int{
		{0, 0, 0, 0, 14},
		{0, 2, 6, 4, 2},
		{0, 0, 3, 5, 6},
		{0, 3, 9, 2, 0},
		{2, 2, 8, 1, 1},
		{7, 7, 0, 0, 0},
		{3, 2, 6, 3, 0},
		{2, 5, 3, 2, 2},
		{6, 5, 2, 1, 0},
		{0, 2, 2, 3, 7},
	}
	kappa, err := FleissKappa(ratings)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	want := 4211.0 / 20059.0
	if diff := kappa - want; diff > 1e-6 || diff < -1e-6 {
		t.Errorf("kappa = %v, want %v (Fleiss 1971 Table 1)", kappa, want)
	}
	// Sanity-check against the published rounded headline figure too.
	if diff := kappa - 0.210; diff > 5e-4 || diff < -5e-4 {
		t.Errorf("kappa = %v, does not round to the published 0.210", kappa)
	}
}

func TestFleissKappa_PerfectAgreement(t *testing.T) {
	// 3 raters, 3 subjects, all raters agree every time.
	ratings := [][]int{
		{3, 0},
		{0, 3},
		{3, 0},
	}
	kappa, err := FleissKappa(ratings)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if diff := kappa - 1.0; diff > 1e-12 || diff < -1e-12 {
		t.Errorf("kappa = %v, want 1.0 for perfect agreement", kappa)
	}
}

func TestFleissKappa_EmptyInput(t *testing.T) {
	_, err := FleissKappa(nil)
	if err != ErrEmptyInput {
		t.Errorf("err = %v, want ErrEmptyInput", err)
	}
	_, err = FleissKappa([][]int{{}})
	if err != ErrEmptyInput {
		t.Errorf("err = %v, want ErrEmptyInput", err)
	}
}

func TestFleissKappa_TooFewRaters(t *testing.T) {
	_, err := FleissKappa([][]int{{1, 0}, {0, 1}})
	if err != ErrTooFewRaters {
		t.Errorf("err = %v, want ErrTooFewRaters", err)
	}
}

func TestFleissKappa_RaggedRows(t *testing.T) {
	// Row 2 sums to 3 raters, not 4 like row 1.
	_, err := FleissKappa([][]int{{2, 2}, {1, 2}})
	if err != ErrRaggedRows {
		t.Errorf("err = %v, want ErrRaggedRows", err)
	}
	// Row 2 has a different number of categories.
	_, err = FleissKappa([][]int{{2, 2}, {2, 1, 1}})
	if err != ErrRaggedRows {
		t.Errorf("err = %v, want ErrRaggedRows", err)
	}
}

func TestFleissKappa_NegativeCount(t *testing.T) {
	_, err := FleissKappa([][]int{{3, -1, 2}})
	if err != ErrNegativeCount {
		t.Errorf("err = %v, want ErrNegativeCount", err)
	}
}
