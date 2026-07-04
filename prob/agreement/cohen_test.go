package agreement

import "testing"

// buildFromCounts expands a 2x2 (or k x k) count table into paired rating
// slices, so tests can express golden-file contingency tables directly as
// counts instead of hand-enumerating unit-by-unit ratings.
func buildFromCounts(categories []int, counts [][]int) (a, b []int) {
	for i, row := range counts {
		for j, c := range row {
			for n := 0; n < c; n++ {
				a = append(a, categories[i])
				b = append(b, categories[j])
			}
		}
	}
	return a, b
}

// TestCohenKappa_Wikipedia50GrantProposals reproduces Cohen (1960)'s
// worked example as given in Wikipedia's "Cohen's kappa" article: 50 grant
// proposals read by two readers (Yes/No). po=0.70, pe=0.50, kappa=0.40.
func TestCohenKappa_Wikipedia50GrantProposals(t *testing.T) {
	// categories: 0 = Yes, 1 = No
	a, b := buildFromCounts([]int{0, 1}, [][]int{
		{20, 5},  // Reader A = Yes: 20 both-yes, 5 A-yes/B-no
		{10, 15}, // Reader A = No:  10 A-no/B-yes, 15 both-no
	})
	if len(a) != 50 {
		t.Fatalf("expected 50 proposals, got %d", len(a))
	}
	kappa, err := CohenKappa(a, b)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if diff := kappa - 0.40; diff > 1e-9 || diff < -1e-9 {
		t.Errorf("kappa = %v, want 0.40 (Cohen 1960 worked example)", kappa)
	}
}

func TestCohenKappa_PerfectAgreement(t *testing.T) {
	a := []int{1, 2, 3, 1, 2, 3, 1, 2, 3}
	b := []int{1, 2, 3, 1, 2, 3, 1, 2, 3}
	kappa, err := CohenKappa(a, b)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if diff := kappa - 1.0; diff > 1e-12 || diff < -1e-12 {
		t.Errorf("kappa = %v, want 1.0 for perfect agreement", kappa)
	}
}

func TestCohenKappa_LengthMismatch(t *testing.T) {
	_, err := CohenKappa([]int{1, 2, 3}, []int{1, 2})
	if err != ErrLengthMismatch {
		t.Errorf("err = %v, want ErrLengthMismatch", err)
	}
}

func TestCohenKappa_EmptyInput(t *testing.T) {
	_, err := CohenKappa(nil, nil)
	if err != ErrEmptyInput {
		t.Errorf("err = %v, want ErrEmptyInput", err)
	}
}

// TestCohenKappa_DegenerateChanceAgreement covers the case where both
// raters only ever use a single category: po = pe = 1, so kappa is 0/0.
func TestCohenKappa_DegenerateChanceAgreement(t *testing.T) {
	a := []int{7, 7, 7, 7}
	b := []int{7, 7, 7, 7}
	_, err := CohenKappa(a, b)
	if err != ErrDegenerateChanceAgreement {
		t.Errorf("err = %v, want ErrDegenerateChanceAgreement", err)
	}
}
