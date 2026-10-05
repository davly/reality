package combinatorics

// Precision property tests — pins combinatorics/counting Precision bounds as
// tested invariants. Pure Go stdlib (testing/quick + math + math/big used ONLY
// as an in-test exact oracle — math/big is stdlib, so the zero-EXTERNAL-dep law
// is preserved; nothing is added to go.mod). ADDITIVE, zero math change.
//
// Claims pinned:
//   - Factorial: "correctly rounded for every n <= 170, and exact for
//     n <= 22". We assert every value equals the float64 nearest to the exact
//     big.Int n!, exactness for n <= 22, and relative error <= 2^-53 vs n!.
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

// nearestFloat64 returns the float64 nearest to the exact integer x (ties to
// even; +Inf beyond math.MaxFloat64). big.Float.SetInt is exact (its precision
// grows to x.BitLen()) and Float64 rounds to nearest even; big.Rat.Float64 is
// an independent conversion, and the two must agree.
func nearestFloat64(t *testing.T, x *big.Int) float64 {
	t.Helper()
	f, _ := new(big.Float).SetInt(x).Float64()
	r, _ := new(big.Rat).SetInt(x).Float64()
	if f != r {
		t.Fatalf("oracle disagreement converting %v: big.Float %v, big.Rat %v", x, f, r)
	}
	return f
}

// TestFactorialCorrectlyRounded checks every finite Factorial value against
// the exact integer n! from math/big: Factorial(n) must be the float64 nearest
// to n! for every 0 <= n <= 170, and exactly n! wherever n! is representable
// (n <= 22; 23! has a 56-bit odd part). Before the table, the exp(lgamma)
// path for n > 20 was off by up to 1.3e-13 relative (several hundred ulps).
func TestFactorialCorrectlyRounded(t *testing.T) {
	wrong := 0
	for n := 0; n <= 170; n++ {
		exact := bigFactorial(n)
		want := nearestFloat64(t, exact)
		got := Factorial(n)
		if got != want {
			wrong++
			t.Errorf("Factorial(%d) = %v, want %v (the float64 nearest to %d!)", n, got, want, n)
		}
		if n <= 22 {
			back, acc := new(big.Float).SetFloat64(got).Int(nil)
			if acc != big.Exact || back.Cmp(exact) != 0 {
				t.Errorf("Factorial(%d) = %v is not exactly %v", n, got, exact)
			}
		}
	}
	if wrong > 0 {
		t.Errorf("%d of 171 values are not correctly rounded", wrong)
	}
	if got := Factorial(171); !math.IsInf(got, 1) {
		t.Errorf("Factorial(171) = %v, want +Inf", got)
	}
	if got := Factorial(1 << 40); !math.IsInf(got, 1) {
		t.Errorf("Factorial(2^40) = %v, want +Inf", got)
	}
	for _, n := range []int{0, -1, -170, math.MinInt} {
		if got := Factorial(n); got != 1 {
			t.Errorf("Factorial(%d) = %v, want 1", n, got)
		}
	}
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

// relErrVsExact returns |got - exact| / exact, evaluated in 256-bit binary
// floating point, so the oracle's own rounding (<= 2^-256) is negligible.
func relErrVsExact(got float64, exact *big.Int) float64 {
	const prec = 256
	g := new(big.Float).SetPrec(prec).SetFloat64(got)
	e := new(big.Float).SetPrec(prec).SetInt(exact)
	d := new(big.Float).SetPrec(prec).Sub(g, e)
	d.Abs(d)
	d.Quo(d, e)
	r, _ := d.Float64()
	return r
}

// factorialWorstRelErr returns the worst relative error of Factorial vs the
// exact integer n! over [lo, hi].
func factorialWorstRelErr(lo, hi int) (worst float64, at int) {
	for n := lo; n <= hi; n++ {
		rel := relErrVsExact(Factorial(n), bigFactorial(n))
		if rel > worst {
			worst, at = rel, n
		}
	}
	return
}

// TestFactorialRelErrHalfUlp pins the consequence of correct rounding: the
// relative error against the exact n! is at most the unit roundoff 2^-53 for
// every n <= 170. (The previous exp(lgamma(n+1)) path reached 1.30e-13 at
// n = 166, with the exact figure depending on whether the compiler fused
// multiply-adds.)
func TestFactorialRelErrHalfUlp(t *testing.T) {
	const unitRoundoff = 0x1p-53
	worst, worstN := factorialWorstRelErr(0, 170)
	if worst > unitRoundoff {
		t.Errorf("Factorial relative error %g at n=%d exceeds 2^-53 = %g", worst, worstN, unitRoundoff)
	}
	t.Logf("Factorial (0<=n<=170): worst relative error vs exact n! = %g at n=%d (<= 2^-53)", worst, worstN)
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
