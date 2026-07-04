package agreement

import (
	"math"
	"testing"
)

// TestMcNemarExact_Fagerland2013 reproduces the exact conditional p-value of
// Fagerland, Lydersen & Laake (2013), BMC Med. Res. Methodol. 13:91 — the
// airway hyper-responsiveness matched-pairs data (Bentur et al. 2009) with
// discordant counts (b, c) = (1, 7), n = 8:
//
//	tail = C(8,0) + C(8,1) = 9;  p = 2*9/256 = 18/256 = 0.0703125.
func TestMcNemarExact_Fagerland2013(t *testing.T) {
	p, err := McNemarExact(1, 7)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if want := 18.0 / 256.0; math.Abs(p-want) > 1e-12 {
		t.Errorf("McNemarExact(1,7) = %v, want %v (Fagerland et al. 2013 exact = 0.0703)", p, want)
	}
}

// TestMcNemarMidP_Fagerland2013 reproduces the mid-p value from the same
// worked example: p_mid = p_exact - C(8,1)/2^8 = 0.0703125 - 0.03125 =
// 0.0390625 (Fagerland et al. 2013 report mid-p = 0.039).
func TestMcNemarMidP_Fagerland2013(t *testing.T) {
	p, err := McNemarMidP(1, 7)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if want := 0.0390625; math.Abs(p-want) > 1e-12 {
		t.Errorf("McNemarMidP(1,7) = %v, want %v (Fagerland et al. 2013 mid-p = 0.039)", p, want)
	}
}

// TestMcNemarExact_SmallHandChecked verifies a second table by direct hand
// computation of the exact binomial tail. (b, c) = (2, 8), n = 10, m = 2:
//
//	tail = C(10,0)+C(10,1)+C(10,2) = 1+10+45 = 56;  p = 2*56/1024 = 0.109375.
func TestMcNemarExact_SmallHandChecked(t *testing.T) {
	p, err := McNemarExact(2, 8)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if want := 112.0 / 1024.0; math.Abs(p-want) > 1e-12 {
		t.Errorf("McNemarExact(2,8) = %v, want %v", p, want)
	}
}

// TestMcNemarExact_Symmetric: the test is invariant to swapping the two
// discordant cells (b and c play symmetric roles under two-sided testing).
func TestMcNemarExact_Symmetric(t *testing.T) {
	p1, _ := McNemarExact(3, 11)
	p2, _ := McNemarExact(11, 3)
	if math.Abs(p1-p2) > 1e-15 {
		t.Errorf("McNemarExact not symmetric: (3,11)=%v (11,3)=%v", p1, p2)
	}
}

// TestMcNemarExact_EqualDiscordantIsOne: when b == c the two tails are equal
// and each >= 1/2, so the doubled tail caps at exactly 1.
func TestMcNemarExact_EqualDiscordantIsOne(t *testing.T) {
	p, err := McNemarExact(5, 5)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if p != 1.0 {
		t.Errorf("McNemarExact(5,5) = %v, want exactly 1.0", p)
	}
}

// TestMcNemarExact_NoDiscordantPairs: b+c == 0 means no evidence about
// marginal homogeneity; p = 1 by convention.
func TestMcNemarExact_NoDiscordantPairs(t *testing.T) {
	p, err := McNemarExact(0, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if p != 1.0 {
		t.Errorf("McNemarExact(0,0) = %v, want 1.0", p)
	}
	pm, err := McNemarMidP(0, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if pm != 1.0 {
		t.Errorf("McNemarMidP(0,0) = %v, want 1.0", pm)
	}
}

// TestMcNemarExact_TailIsProperProbability: for a range of discordant tables,
// the exact p is in (0, 1], the mid-p is in [0, 1], and mid-p <= exact
// (mid-p removes a positive point mass).
func TestMcNemarExact_TailIsProperProbability(t *testing.T) {
	cases := [][2]int{{0, 1}, {1, 4}, {2, 9}, {6, 6}, {0, 20}, {7, 13}}
	for _, bc := range cases {
		ex, err := McNemarExact(bc[0], bc[1])
		if err != nil {
			t.Fatalf("McNemarExact%v: %v", bc, err)
		}
		mp, err := McNemarMidP(bc[0], bc[1])
		if err != nil {
			t.Fatalf("McNemarMidP%v: %v", bc, err)
		}
		if ex <= 0 || ex > 1.0000000001 {
			t.Errorf("exact%v = %v out of (0,1]", bc, ex)
		}
		if mp < 0 || mp > 1 {
			t.Errorf("midp%v = %v out of [0,1]", bc, mp)
		}
		if mp > ex+1e-12 {
			t.Errorf("midp%v = %v exceeds exact %v", bc, mp, ex)
		}
	}
}

// TestMcNemarExact_LargeNNoUnderflow: a large discordant count where a naive
// 2^-n PMF summation would underflow. Exact big.Rat arithmetic keeps the tiny
// tail probability finite and positive. (b, c) = (0, 200): p = 2/2^200, a
// representable subnormal-free small double.
func TestMcNemarExact_LargeNNoUnderflow(t *testing.T) {
	p, err := McNemarExact(0, 200)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	want := 2.0 * math.Pow(2, -200)
	if p <= 0 || math.Abs(p-want)/want > 1e-12 {
		t.Errorf("McNemarExact(0,200) = %v, want %v", p, want)
	}
}

func TestMcNemarExact_NegativeCount(t *testing.T) {
	if _, err := McNemarExact(-1, 3); err != ErrNegativeCount {
		t.Errorf("err = %v, want ErrNegativeCount", err)
	}
	if _, err := McNemarMidP(3, -2); err != ErrNegativeCount {
		t.Errorf("err = %v, want ErrNegativeCount", err)
	}
}

// TestDiscordantCounts derives (b, c) from paired 0/1 slices and feeds them to
// McNemarExact, reproducing the Fagerland (1, 7) table from raw pairs.
func TestDiscordantCounts(t *testing.T) {
	// 1 pair x=1,y=0 (b); 7 pairs x=0,y=1 (c); plus concordant noise.
	x := []int{1, 0, 0, 0, 0, 0, 0, 0, 1, 0}
	y := []int{0, 1, 1, 1, 1, 1, 1, 1, 1, 0}
	b, c, err := DiscordantCounts(x, y)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if b != 1 || c != 7 {
		t.Fatalf("DiscordantCounts = (%d,%d), want (1,7)", b, c)
	}
	p, _ := McNemarExact(b, c)
	if want := 18.0 / 256.0; math.Abs(p-want) > 1e-12 {
		t.Errorf("exact from derived counts = %v, want %v", p, want)
	}
}

func TestDiscordantCounts_Errors(t *testing.T) {
	if _, _, err := DiscordantCounts([]int{1, 0}, []int{1}); err != ErrLengthMismatch {
		t.Errorf("err = %v, want ErrLengthMismatch", err)
	}
	if _, _, err := DiscordantCounts(nil, nil); err != ErrEmptyInput {
		t.Errorf("err = %v, want ErrEmptyInput", err)
	}
	if _, _, err := DiscordantCounts([]int{2}, []int{0}); err != ErrNotBinary {
		t.Errorf("err = %v, want ErrNotBinary", err)
	}
}

// TestBinomialBig spot-checks the exact big-integer binomial coefficients used
// by the tail sums against known values, including the symmetry branch.
func TestBinomialBig(t *testing.T) {
	cases := []struct {
		n, k int
		want int64
	}{
		{0, 0, 1}, {8, 0, 1}, {8, 1, 8}, {10, 2, 45}, {10, 8, 45},
		{20, 10, 184756}, {5, 6, 0}, {5, -1, 0},
	}
	for _, c := range cases {
		got := binomialBig(c.n, c.k).Int64()
		if got != c.want {
			t.Errorf("binomialBig(%d,%d) = %d, want %d", c.n, c.k, got, c.want)
		}
	}
}
