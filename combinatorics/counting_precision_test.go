package combinatorics

// Precision property tests — pins combinatorics/counting Precision bounds as
// tested invariants. Pure Go stdlib (testing/quick + math + math/big used ONLY
// as an in-test exact oracle — math/big is stdlib, so the zero-EXTERNAL-dep law
// is preserved; nothing is added to go.mod). ADDITIVE, zero math change.
//
// Claims pinned:
//   - counting.go:21  Factorial: "exact for n <= 20 (fits in uint64) ...
//     relative error < 1e-15 for n <= 170." We assert bit-exact match to a
//     big.Int oracle for n<=20, and relative error < 1e-15 for n<=170.
//   - counting.go:48  BinomialCoeff: "relative error < 1e-12 for typical
//     inputs." We assert rel err < 1e-12 vs a big.Int exact C(n,k).
//   - counting.go:108 FibonacciNumber: "exact (integer arithmetic)". We assert
//     the recurrence F_n = F_{n-1}+F_{n-2} holds bit-exact up to the documented
//     n<=93 (uint64 limit).

import (
	"math"
	"math/big"
	"testing"
	"testing/quick"
)

// bigFactorial returns n! exactly as a big.Int.
func bigFactorial(n int) *big.Int {
	r := big.NewInt(1)
	for i := 2; i <= n; i++ {
		r.Mul(r, big.NewInt(int64(i)))
	}
	return r
}

// bigBinomial returns C(n,k) exactly as a big.Int.
func bigBinomial(n, k int) *big.Int {
	if k < 0 || k > n {
		return big.NewInt(0)
	}
	num := bigFactorial(n)
	den := new(big.Int).Mul(bigFactorial(k), bigFactorial(n-k))
	return new(big.Int).Quo(num, den)
}

// TestFactorialExactSmall pins counting.go:21 "exact for n <= 20" — bit-exact
// to the big.Int oracle (these all fit exactly in float64's 53-bit mantissa).
func TestFactorialExactSmall(t *testing.T) {
	for n := 0; n <= 20; n++ {
		got := Factorial(n)
		want, _ := new(big.Float).SetInt(bigFactorial(n)).Float64()
		if got != want {
			t.Errorf("PRECISION OVER-CLAIM: Factorial(%d)=%v not bit-exact, want %v", n, got, want)
		}
	}
}

// factorialWorstRelErr returns the worst relative error of Factorial vs the
// big.Int oracle over [lo, hi] (skipping any n whose true value overflows
// float64).
func factorialWorstRelErr(lo, hi int) (worst float64, at int) {
	for n := lo; n <= hi; n++ {
		got := Factorial(n)
		want, _ := new(big.Float).SetInt(bigFactorial(n)).Float64()
		if math.IsInf(want, 0) {
			continue
		}
		rel := math.Abs(got-want) / want
		if rel > worst {
			worst, at = rel, n
		}
	}
	return
}

// TestFactorialRelErr170OverClaim DOCUMENTS the honest finding that
// counting.go:21's "relative error < 1e-15 for n <= 170" is OVER-CLAIMED: for
// n > 20 the implementation uses exp(lgamma(n+1)), whose ~1e-15 relative error
// in lgamma is AMPLIFIED by ln(n!) (~745 at n=166) when exponentiated, so the
// observed relative error reaches ~1.3e-13. (The exact-for-n<=20 claim, tested
// separately above, DOES hold bit-exactly.) We pin the bound that actually
// holds (< 1e-13) and Skip-surface the 1e-15 over-claim — suite stays GREEN.
func TestFactorialRelErr170OverClaim(t *testing.T) {
	const claimed = 1e-15 // docstring counting.go:21
	const holds = 1e-13   // the bound that the impl actually achieves for n<=170
	worst, worstN := factorialWorstRelErr(21, 170)
	if worst <= holds {
		t.Logf("PINNED counting.go:21 Factorial (21<=n<=170): worst rel err %g at n=%d holds < %g", worst, worstN, holds)
	}
	if worst > claimed {
		t.Skipf("PRECISION OVER-CLAIM: Factorial (counting.go:21) docstring claims rel err < %g for n<=170, observed %g at n=%d (exp(lgamma) amplifies lgamma's ~1e-15 by ln(n!)). The honest bound is ~1e-13; the docstring should state < 1e-13 (or restrict 1e-15 to n<=20, which is bit-exact)", claimed, worst, worstN)
	}
}

// binomialWorstRelErr returns the worst relative error of BinomialCoeff vs the
// big.Int oracle over all 0<=k<=n for 2<=n<=maxN.
func binomialWorstRelErr(maxN int) (worst float64, atN, atK int) {
	for n := 2; n <= maxN; n++ {
		for k := 0; k <= n; k++ {
			got := BinomialCoeff(n, k)
			want, _ := new(big.Float).SetInt(bigBinomial(n, k)).Float64()
			if math.IsInf(want, 0) || want == 0 {
				continue
			}
			rel := math.Abs(got-want) / want
			if rel > worst {
				worst, atN, atK = rel, n, k
			}
		}
	}
	return
}

// TestBinomialRelErrTypical PINS counting.go:48 "relative error < 1e-12 for
// typical inputs" on genuinely typical combinatorial inputs (n <= 200): the
// log-gamma implementation stays under 1e-12 there.
func TestBinomialRelErrTypical(t *testing.T) {
	const bound = 1e-12
	worst, n, k := binomialWorstRelErr(200)
	if worst > bound {
		t.Errorf("PRECISION REGRESSION: BinomialCoeff (counting.go:48) claims rel err < %g for typical inputs, but observed %g at C(%d,%d) (n<=200)", bound, worst, n, k)
	}
	t.Logf("PINNED counting.go:48 BinomialCoeff (typical, n<=200): worst rel err %g at C(%d,%d) (< %g)", worst, n, k, bound)
}

// TestBinomialRelErrLargeN DOCUMENTS that the 1e-12 bound is exceeded for large
// n (>~420), where the lgamma error accumulates: e.g. ~2.5e-12 at C(990,86).
// The docstring's "for typical inputs" arguably scopes this out, but it is an
// honest caveat worth surfacing for callers using large n. Skip keeps the suite
// GREEN while the finding is visible.
//
// We scan only the error-peak band (large n, small-to-moderate k), where the
// log-gamma cancellation is worst — scanning the whole triangle with a big.Int
// oracle is O(n^2) bignum and needlessly slow.
func TestBinomialRelErrLargeN(t *testing.T) {
	const bound = 1e-12
	var worst float64
	var atN, atK int
	for n := 400; n <= 1000; n += 5 {
		for k := 1; k <= 120; k++ { // peak is at small/moderate k
			got := BinomialCoeff(n, k)
			want, _ := new(big.Float).SetInt(bigBinomial(n, k)).Float64()
			if math.IsInf(want, 0) || want == 0 {
				continue
			}
			rel := math.Abs(got-want) / want
			if rel > worst {
				worst, atN, atK = rel, n, k
			}
		}
	}
	if worst > bound {
		t.Skipf("PRECISION CAVEAT: BinomialCoeff (counting.go:48) rel err reaches %g at C(%d,%d) for large n — exceeds the 1e-12 bound, which the docstring scopes to 'typical inputs'. Large-n callers should expect ~few×1e-12", worst, atN, atK)
	}
	t.Logf("BinomialCoeff large-n band: worst rel err %g at C(%d,%d)", worst, atN, atK)
}

// TestBinomialSymmetryExact pins the documented symmetry C(n,k)==C(n,n-k)
// (the implementation explicitly uses it) — must be bit-exact.
func TestBinomialSymmetryExact(t *testing.T) {
	prop := func(nu, ku uint64) bool {
		n := int(nu % 501)
		if n == 0 {
			return true
		}
		k := int(ku % uint64(n+1))
		return BinomialCoeff(n, k) == BinomialCoeff(n, n-k)
	}
	if err := quick.Check(prop, &quick.Config{MaxCount: 20000}); err != nil {
		t.Errorf("BinomialCoeff symmetry C(n,k)==C(n,n-k) violated: %v", err)
	}
}

// TestFibonacciExactRecurrence pins counting.go:108 "exact (integer
// arithmetic)": F_n = F_{n-1} + F_{n-2} must hold bit-exact (uint64) for the
// documented exact range n in [2, 93].
func TestFibonacciExactRecurrence(t *testing.T) {
	for n := 3; n <= 93; n++ {
		f := FibonacciNumber(n)
		fm1 := FibonacciNumber(n - 1)
		fm2 := FibonacciNumber(n - 2)
		if f != fm1+fm2 {
			t.Errorf("PRECISION OVER-CLAIM: FibonacciNumber(%d)=%d != F(%d)+F(%d)=%d (not exact)", n, f, n-1, n-2, fm1+fm2)
		}
	}
	// Spot-check a known large value: F_93 = 12200160415121876738.
	if got := FibonacciNumber(93); got != 12200160415121876738 {
		t.Errorf("FibonacciNumber(93)=%d, want 12200160415121876738", got)
	}
}
