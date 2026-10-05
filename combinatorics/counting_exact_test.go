package combinatorics

// Exactness tests for BinomialCoeff and Permutations. The oracles are exact
// integers from math/big (Pascal's triangle by addition, running products,
// and big.Int.Binomial), rounded to float64 by math/big; the function under
// test is never its own oracle.

import (
	"math"
	"math/big"
	"testing"
)

// mismatchReporter counts mismatches and prints only the first few.
type mismatchReporter struct {
	t     *testing.T
	count int
}

func (r *mismatchReporter) add(format string, args ...any) {
	r.t.Helper()
	r.count++
	if r.count <= 10 {
		r.t.Errorf(format, args...)
	}
}

// TestBinomialCoeffExhaustive compares BinomialCoeff(n, k) with the float64
// nearest to the exact C(n, k) for every 0 <= k <= n <= 300. The exact values
// come from Pascal's triangle built by big.Int addition. The previous
// exp(lgamma) evaluation returned 242 wrong values for n <= 60 alone, the
// first at C(49, 20).
func TestBinomialCoeffExhaustive(t *testing.T) {
	const maxN = 300
	rep := &mismatchReporter{t: t}
	wrongUpTo60 := 0
	row := []*big.Int{big.NewInt(1)} // row n of Pascal's triangle
	for n := 0; n <= maxN; n++ {
		if n > 0 {
			next := make([]*big.Int, n+1)
			next[0], next[n] = big.NewInt(1), big.NewInt(1)
			for k := 1; k < n; k++ {
				next[k] = new(big.Int).Add(row[k-1], row[k])
			}
			row = next
		}
		for k := 0; k <= n; k++ {
			want := nearestFloat64(t, row[k])
			if got := BinomialCoeff(n, k); got != want {
				if n <= 60 {
					wrongUpTo60++
				}
				rep.add("BinomialCoeff(%d, %d) = %v, want %v (nearest to %v)", n, k, got, want, row[k])
			}
		}
		for _, k := range []int{-1, n + 1, math.MinInt, math.MaxInt} {
			if got := BinomialCoeff(n, k); got != 0 {
				rep.add("BinomialCoeff(%d, %d) = %v, want 0", n, k, got)
			}
		}
	}
	if rep.count > 0 {
		t.Errorf("%d mismatches for n <= %d (%d of them with n <= 60)", rep.count, maxN, wrongUpTo60)
	}
}

// TestPermutationsExhaustive compares Permutations(n, k) with the float64
// nearest to the exact n!/(n-k)! for every 0 <= k <= n <= 300, including the
// overflow to +Inf. The exact values are running big.Int products.
func TestPermutationsExhaustive(t *testing.T) {
	const maxN = 300
	rep := &mismatchReporter{t: t}
	for n := 0; n <= maxN; n++ {
		exact := big.NewInt(1)
		for k := 0; k <= n; k++ {
			if k > 0 {
				exact.Mul(exact, big.NewInt(int64(n-k+1)))
			}
			want := nearestFloat64(t, exact)
			if got := Permutations(n, k); got != want {
				rep.add("Permutations(%d, %d) = %v, want %v", n, k, got, want)
			}
		}
		for _, k := range []int{-1, n + 1, math.MinInt, math.MaxInt} {
			if got := Permutations(n, k); got != 0 {
				rep.add("Permutations(%d, %d) = %v, want 0", n, k, got)
			}
		}
	}
	if rep.count > 0 {
		t.Errorf("%d mismatches for n <= %d", rep.count, maxN)
	}
}

// TestBinomialCoeffLargeN spot-checks large and extreme arguments against
// big.Int.Binomial, including whole rows that straddle the float64 overflow
// threshold (C(1029, 514) is finite, C(1030, 515) is not) and n near
// math.MaxInt64.
func TestBinomialCoeffLargeN(t *testing.T) {
	rep := &mismatchReporter{t: t}
	check := func(n, k int64) {
		t.Helper()
		var want float64
		if k < 0 || k > n {
			want = 0
		} else {
			want = nearestFloat64(t, new(big.Int).Binomial(n, k))
		}
		if got := BinomialCoeff(int(n), int(k)); got != want {
			rep.add("BinomialCoeff(%d, %d) = %v, want %v", n, k, got, want)
		}
	}
	for n := int64(1020); n <= 1040; n++ {
		for k := int64(0); k <= n; k++ {
			check(n, k)
		}
	}
	const maxInt64 = math.MaxInt64
	pairs := [][2]int64{
		{1000, 500}, {1100, 400}, {2000, 150}, {5000, 120}, {100000, 50000},
		{1000000, 3}, {1000000, 50}, {1 << 31, 2}, {1 << 40, 20}, {1 << 40, 30},
		{1000000000, 33}, {1000000000000, 80}, {1 << 53, 2}, {1<<53 + 1, 2},
		{maxInt64, 1}, {maxInt64, 2}, {maxInt64, 3}, {maxInt64, 15}, {maxInt64, 16},
		{maxInt64, 17}, {maxInt64 - 1, maxInt64 - 3},
	}
	for _, p := range pairs {
		check(p[0], p[1])
		check(p[0], p[0]-p[1])
	}
	if rep.count > 0 {
		t.Errorf("%d mismatches", rep.count)
	}
}

// TestPermutationsLargeN spot-checks Permutations at large n against running
// big.Int products, and P(n, n) against the exact n!.
func TestPermutationsLargeN(t *testing.T) {
	rep := &mismatchReporter{t: t}
	check := func(n, k int64) {
		t.Helper()
		exact := big.NewInt(1)
		for i := int64(0); i < k; i++ {
			exact.Mul(exact, big.NewInt(n-i))
			if exact.BitLen() > 1100 {
				break // beyond float64 range: the nearest float64 is +Inf either way
			}
		}
		want := nearestFloat64(t, exact)
		if got := Permutations(int(n), int(k)); got != want {
			rep.add("Permutations(%d, %d) = %v, want %v", n, k, got, want)
		}
	}
	const maxInt64 = math.MaxInt64
	pairs := [][2]int64{
		{170, 170}, {171, 171}, {1000, 100}, {1000, 120}, {2000, 100},
		{1000000, 50}, {1000000, 52}, {1 << 40, 25}, {1000000000, 34},
		{maxInt64, 1}, {maxInt64, 2}, {maxInt64, 16}, {maxInt64, 17}, {maxInt64, 1000},
	}
	for _, p := range pairs {
		check(p[0], p[1])
	}
	if rep.count > 0 {
		t.Errorf("%d mismatches", rep.count)
	}
}
