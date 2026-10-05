package prob

import (
	"math"
	"testing"
)

// fisherTieTolerance mirrors the documented tie rule of FisherExactTest:
// tables at most 1 + 1e-7 times as probable as the observed one count as at
// least as extreme. The reference applies it in exact integer arithmetic as
// num*10^7 <= obs*(10^7 + 1).
const (
	fisherTieScale = 10_000_000
	fisherMaxN     = 40
)

// binomTable40 holds C(m, j) for 0 <= j <= m <= 40, built by addition.
func binomTable40() [fisherMaxN + 1][fisherMaxN + 1]uint64 {
	var c [fisherMaxN + 1][fisherMaxN + 1]uint64
	for m := 0; m <= fisherMaxN; m++ {
		c[m][0] = 1
		for j := 1; j <= m; j++ {
			c[m][j] = c[m-1][j-1] + c[m-1][j]
		}
	}
	return c
}

// fisherReference returns the exact two-sided Fisher p-value of the table
// (a, b, c, d) as num/den, where den = C(n, a+c) and num sums the numerators
// C(a+b, x) * C(c+d, a+c-x) of the tables at least as extreme as the observed
// one: rTies under the 1 + 1e-7 tie rule, strict counting only exact ties.
// Every quantity is an integer below 2^53 for n <= 40, so num and den convert
// to float64 exactly.
func fisherReference(c *[fisherMaxN + 1][fisherMaxN + 1]uint64, a, b, cc, d int) (rTies, strict, den uint64) {
	r1, r2, c1 := a+b, cc+d, a+cc
	lo, hi := max(0, c1-r2), min(r1, c1)
	obs := c[r1][a] * c[r2][cc]
	for x := lo; x <= hi; x++ {
		v := c[r1][x] * c[r2][c1-x]
		if v*fisherTieScale <= obs*(fisherTieScale+1) {
			rTies += v
		}
		if v <= obs {
			strict += v
		}
	}
	return rTies, strict, c[a+b+cc+d][c1]
}

// TestFisherExactTestExhaustive compares FisherExactTest with the exact
// p-value of every 2x2 table with n = a+b+c+d <= 40 (135,751 tables). The
// previous absolute tolerance of 1e-14 on the probability comparison was
// smaller than the rounding error of the log-gamma evaluation, so equally
// likely tables were dropped: (9, 11, 10, 10) gave 0.7636 instead of 1.
func TestFisherExactTestExhaustive(t *testing.T) {
	const tol = 1e-12
	maxN := fisherMaxN
	if testing.Short() {
		maxN = 20
	}
	c := binomTable40()
	var worst float64
	var worstTable [4]int
	tables, strictDiffers, bad := 0, 0, 0
	for n := 0; n <= maxN; n++ {
		for a := 0; a <= n; a++ {
			for b := 0; a+b <= n; b++ {
				for cc := 0; a+b+cc <= n; cc++ {
					d := n - a - b - cc
					tables++
					num, strict, den := fisherReference(&c, a, b, cc, d)
					if strict != num {
						strictDiffers++
					}
					want := float64(num) / float64(den)
					got := FisherExactTest(a, b, cc, d)
					rel := math.Abs(got-want) / want
					if rel > worst {
						worst, worstTable = rel, [4]int{a, b, cc, d}
					}
					if rel > tol {
						bad++
						if bad <= 10 {
							t.Errorf("FisherExactTest(%d, %d, %d, %d) = %.17g, want %.17g (relative error %.3g)", a, b, cc, d, got, want, rel)
						}
					}
				}
			}
		}
	}
	if bad > 0 {
		t.Errorf("%d of %d tables exceed relative error %g", bad, tables, tol)
	}
	t.Logf("%d tables with n <= %d: worst relative error %.3g at %v; the tie rule changes the exact p-value of %d tables", tables, maxN, worst, worstTable, strictDiffers)
}

// TestFisherExactTestTies pins tables whose probability exactly equals that
// of another table in the same margins: they must count as equally extreme.
func TestFisherExactTestTies(t *testing.T) {
	// Exact values from rational arithmetic (Python fractions), rounded once.
	cases := []struct {
		a, b, c, d int
		want       float64
	}{
		{9, 11, 10, 10, 1},                     // the two most likely tables tie
		{10, 10, 9, 11, 1},                     // the same margins, the other mode
		{10, 10, 10, 10, 1},                    // a single mode
		{5, 5, 5, 5, 1},                        // a single mode
		{8, 2, 1, 5, 0.03496503496503497},      // 5/143, no ties
		{1, 9, 11, 3, 0.0027594561852200836},   // 41/14858, no ties
		{0, 10, 10, 0, 1.082508822446903e-05},  // 1/92378: the two extreme tables tie
		{20, 0, 0, 20, 1.4508889103849688e-11}, // 1/68923264410
		{3, 1, 1, 3, 0.4857142857142857},       // 17/35: the two tails tie pairwise
	}
	for _, c := range cases {
		got := FisherExactTest(c.a, c.b, c.c, c.d)
		if rel := math.Abs(got-c.want) / c.want; rel > 1e-12 {
			t.Errorf("FisherExactTest(%d, %d, %d, %d) = %.17g, want %.17g", c.a, c.b, c.c, c.d, got, c.want)
		}
	}
}
