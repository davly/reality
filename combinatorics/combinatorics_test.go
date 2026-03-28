package combinatorics

import (
	"math"
	"sort"
	"testing"

	"github.com/davly/reality/testutil"
)

// ---------------------------------------------------------------------------
// Factorial
// ---------------------------------------------------------------------------

func TestFactorial_Zero(t *testing.T) {
	if got := Factorial(0); got != 1 {
		t.Errorf("Factorial(0) = %v, want 1", got)
	}
}

func TestFactorial_One(t *testing.T) {
	if got := Factorial(1); got != 1 {
		t.Errorf("Factorial(1) = %v, want 1", got)
	}
}

func TestFactorial_Five(t *testing.T) {
	if got := Factorial(5); got != 120 {
		t.Errorf("Factorial(5) = %v, want 120", got)
	}
}

func TestFactorial_Ten(t *testing.T) {
	if got := Factorial(10); math.Abs(got-3628800) > 0.5 {
		t.Errorf("Factorial(10) = %v, want 3628800", got)
	}
}

func TestFactorial_Twenty(t *testing.T) {
	// 20! = 2432902008176640000
	want := 2432902008176640000.0
	got := Factorial(20)
	if math.Abs(got-want)/want > 1e-12 {
		t.Errorf("Factorial(20) = %v, want %v", got, want)
	}
}

func TestFactorial_Negative(t *testing.T) {
	if got := Factorial(-5); got != 1 {
		t.Errorf("Factorial(-5) = %v, want 1", got)
	}
}

func TestFactorial_Large_170(t *testing.T) {
	got := Factorial(170)
	if math.IsInf(got, 0) || math.IsNaN(got) {
		t.Errorf("Factorial(170) should be finite, got %v", got)
	}
}

func TestFactorial_Overflow_171(t *testing.T) {
	got := Factorial(171)
	if !math.IsInf(got, 1) {
		t.Errorf("Factorial(171) should be +Inf, got %v", got)
	}
}

// ---------------------------------------------------------------------------
// BinomialCoeff
// ---------------------------------------------------------------------------

func TestBinomialCoeff_10_5(t *testing.T) {
	got := BinomialCoeff(10, 5)
	if got != 252 {
		t.Errorf("BinomialCoeff(10,5) = %v, want 252", got)
	}
}

func TestBinomialCoeff_Symmetry(t *testing.T) {
	// C(n,k) = C(n, n-k)
	pairs := [][2]int{{10, 3}, {15, 7}, {20, 8}, {52, 5}}
	for _, p := range pairs {
		n, k := p[0], p[1]
		a := BinomialCoeff(n, k)
		b := BinomialCoeff(n, n-k)
		if a != b {
			t.Errorf("C(%d,%d) = %v != C(%d,%d) = %v", n, k, a, n, n-k, b)
		}
	}
}

func TestBinomialCoeff_Edges(t *testing.T) {
	// C(n,0) = C(n,n) = 1
	for _, n := range []int{0, 1, 5, 50, 100} {
		if got := BinomialCoeff(n, 0); got != 1 {
			t.Errorf("C(%d,0) = %v, want 1", n, got)
		}
		if got := BinomialCoeff(n, n); got != 1 {
			t.Errorf("C(%d,%d) = %v, want 1", n, n, got)
		}
	}
}

func TestBinomialCoeff_Invalid(t *testing.T) {
	if got := BinomialCoeff(5, -1); got != 0 {
		t.Errorf("C(5,-1) = %v, want 0", got)
	}
	if got := BinomialCoeff(5, 6); got != 0 {
		t.Errorf("C(5,6) = %v, want 0", got)
	}
}

func TestBinomialCoeff_100_50(t *testing.T) {
	got := BinomialCoeff(100, 50)
	// C(100,50) ≈ 1.00891344545564e+29
	want := 1.00891344545564e+29
	if math.Abs(got-want)/want > 1e-10 {
		t.Errorf("C(100,50) = %v, want ~%v", got, want)
	}
}

func TestBinomialCoeff_PascalRow(t *testing.T) {
	// Row 5 of Pascal's triangle: 1 5 10 10 5 1
	expected := []float64{1, 5, 10, 10, 5, 1}
	for k := 0; k <= 5; k++ {
		got := BinomialCoeff(5, k)
		if got != expected[k] {
			t.Errorf("C(5,%d) = %v, want %v", k, got, expected[k])
		}
	}
}

// ---------------------------------------------------------------------------
// BinomialCoeff Golden File
// ---------------------------------------------------------------------------

func TestBinomialCoeff_Golden(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/combinatorics/binomial_coeff.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			n := testutil.InputInt(t, tc, "n")
			k := testutil.InputInt(t, tc, "k")
			got := BinomialCoeff(n, k)
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

// ---------------------------------------------------------------------------
// Permutations (count)
// ---------------------------------------------------------------------------

func TestPermutations_Count(t *testing.T) {
	tests := []struct {
		n, k int
		want float64
	}{
		{5, 0, 1},
		{5, 1, 5},
		{5, 2, 20},
		{5, 5, 120},
		{10, 3, 720},
	}
	for _, tt := range tests {
		got := Permutations(tt.n, tt.k)
		if got != tt.want {
			t.Errorf("P(%d,%d) = %v, want %v", tt.n, tt.k, got, tt.want)
		}
	}
}

func TestPermutations_Invalid(t *testing.T) {
	if got := Permutations(5, -1); got != 0 {
		t.Errorf("P(5,-1) = %v, want 0", got)
	}
	if got := Permutations(5, 6); got != 0 {
		t.Errorf("P(5,6) = %v, want 0", got)
	}
}

// ---------------------------------------------------------------------------
// CatalanNumber
// ---------------------------------------------------------------------------

func TestCatalan_Known(t *testing.T) {
	// C_0=1, C_1=1, C_2=2, C_3=5, C_4=14, C_5=42, C_10=16796
	known := map[int]float64{
		0: 1, 1: 1, 2: 2, 3: 5, 4: 14, 5: 42, 10: 16796,
	}
	for n, want := range known {
		got := CatalanNumber(n)
		if got != want {
			t.Errorf("Catalan(%d) = %v, want %v", n, got, want)
		}
	}
}

func TestCatalan_Negative(t *testing.T) {
	if got := CatalanNumber(-1); got != 1 {
		t.Errorf("Catalan(-1) = %v, want 1", got)
	}
}

// ---------------------------------------------------------------------------
// FibonacciNumber
// ---------------------------------------------------------------------------

func TestFibonacci_BaseCase(t *testing.T) {
	if got := FibonacciNumber(0); got != 0 {
		t.Errorf("F(0) = %d, want 0", got)
	}
	if got := FibonacciNumber(1); got != 1 {
		t.Errorf("F(1) = %d, want 1", got)
	}
	if got := FibonacciNumber(2); got != 1 {
		t.Errorf("F(2) = %d, want 1", got)
	}
}

func TestFibonacci_Ten(t *testing.T) {
	if got := FibonacciNumber(10); got != 55 {
		t.Errorf("F(10) = %d, want 55", got)
	}
}

func TestFibonacci_Twenty(t *testing.T) {
	if got := FibonacciNumber(20); got != 6765 {
		t.Errorf("F(20) = %d, want 6765", got)
	}
}

func TestFibonacci_Fifty(t *testing.T) {
	if got := FibonacciNumber(50); got != 12586269025 {
		t.Errorf("F(50) = %d, want 12586269025", got)
	}
}

func TestFibonacci_Negative(t *testing.T) {
	if got := FibonacciNumber(-1); got != 0 {
		t.Errorf("F(-1) = %d, want 0", got)
	}
}

func TestFibonacci_Sequence(t *testing.T) {
	// Verify F(n) = F(n-1) + F(n-2) for n = 2..20
	for n := 2; n <= 20; n++ {
		got := FibonacciNumber(n)
		want := FibonacciNumber(n-1) + FibonacciNumber(n-2)
		if got != want {
			t.Errorf("F(%d) = %d, want F(%d)+F(%d) = %d", n, got, n-1, n-2, want)
		}
	}
}

// ---------------------------------------------------------------------------
// DerangementCount
// ---------------------------------------------------------------------------

func TestDerangement_Known(t *testing.T) {
	// !0=1, !1=0, !2=1, !3=2, !4=9, !5=44, !6=265
	known := map[int]float64{
		0: 1, 1: 0, 2: 1, 3: 2, 4: 9, 5: 44, 6: 265,
	}
	for n, want := range known {
		got := DerangementCount(n)
		if got != want {
			t.Errorf("D(%d) = %v, want %v", n, got, want)
		}
	}
}

func TestDerangement_Ratio(t *testing.T) {
	// !n / n! → 1/e as n → ∞. For n=10, should be close.
	ratio := DerangementCount(10) / Factorial(10)
	invE := 1.0 / math.E
	if math.Abs(ratio-invE) > 1e-6 {
		t.Errorf("D(10)/10! = %v, want ~%v", ratio, invE)
	}
}

// ---------------------------------------------------------------------------
// GeneratePermutations
// ---------------------------------------------------------------------------

func TestGeneratePermutations_Empty(t *testing.T) {
	result := GeneratePermutations(nil)
	if result != nil {
		t.Errorf("GeneratePermutations(nil) should return nil, got %v", result)
	}
}

func TestGeneratePermutations_Single(t *testing.T) {
	result := GeneratePermutations([]int{42})
	if len(result) != 1 || result[0][0] != 42 {
		t.Errorf("GeneratePermutations([42]) = %v, want [[42]]", result)
	}
}

func TestGeneratePermutations_Three(t *testing.T) {
	result := GeneratePermutations([]int{1, 2, 3})
	if len(result) != 6 {
		t.Errorf("GeneratePermutations([1,2,3]) produced %d perms, want 6", len(result))
	}

	// All permutations should be distinct.
	seen := make(map[[3]int]bool)
	for _, p := range result {
		var key [3]int
		copy(key[:], p)
		if seen[key] {
			t.Errorf("duplicate permutation: %v", p)
		}
		seen[key] = true
	}
}

func TestGeneratePermutations_DoesNotModifyInput(t *testing.T) {
	input := []int{3, 1, 2}
	orig := make([]int, len(input))
	copy(orig, input)
	GeneratePermutations(input)
	for i := range input {
		if input[i] != orig[i] {
			t.Errorf("input modified: input[%d] = %d, was %d", i, input[i], orig[i])
		}
	}
}

// ---------------------------------------------------------------------------
// GenerateCombinations
// ---------------------------------------------------------------------------

func TestGenerateCombinations_5_2(t *testing.T) {
	result := GenerateCombinations(5, 2)
	// C(5,2) = 10
	if len(result) != 10 {
		t.Errorf("GenerateCombinations(5,2) produced %d, want 10", len(result))
	}
	// Each combination should have 2 elements.
	for _, c := range result {
		if len(c) != 2 {
			t.Errorf("combination has %d elements, want 2: %v", len(c), c)
		}
	}
}

func TestGenerateCombinations_Edges(t *testing.T) {
	// C(n,0) = {[]}
	result := GenerateCombinations(5, 0)
	if len(result) != 1 || len(result[0]) != 0 {
		t.Errorf("GenerateCombinations(5,0) = %v, want [[]]", result)
	}

	// C(n,n) = {[0,1,...,n-1]}
	result = GenerateCombinations(3, 3)
	if len(result) != 1 {
		t.Errorf("GenerateCombinations(3,3) produced %d, want 1", len(result))
	}
}

func TestGenerateCombinations_Invalid(t *testing.T) {
	if got := GenerateCombinations(5, -1); got != nil {
		t.Errorf("GenerateCombinations(5,-1) should be nil")
	}
	if got := GenerateCombinations(5, 6); got != nil {
		t.Errorf("GenerateCombinations(5,6) should be nil")
	}
}

func TestGenerateCombinations_Sorted(t *testing.T) {
	// Every combination should be in ascending order.
	for _, c := range GenerateCombinations(6, 3) {
		for i := 1; i < len(c); i++ {
			if c[i] <= c[i-1] {
				t.Errorf("combination not sorted: %v", c)
				break
			}
		}
	}
}

// ---------------------------------------------------------------------------
// NextPermutation
// ---------------------------------------------------------------------------

func TestNextPermutation_123(t *testing.T) {
	perm := []int{1, 2, 3}
	if !NextPermutation(perm) {
		t.Fatal("NextPermutation([1,2,3]) returned false")
	}
	want := []int{1, 3, 2}
	for i := range perm {
		if perm[i] != want[i] {
			t.Errorf("after NextPermutation: got %v, want %v", perm, want)
			break
		}
	}
}

func TestNextPermutation_Last(t *testing.T) {
	perm := []int{3, 2, 1}
	if NextPermutation(perm) {
		t.Error("NextPermutation([3,2,1]) should return false")
	}
}

func TestNextPermutation_AllPerms(t *testing.T) {
	// Walk through all 24 permutations of [1,2,3,4].
	perm := []int{1, 2, 3, 4}
	count := 1
	for NextPermutation(perm) {
		count++
	}
	if count != 24 {
		t.Errorf("NextPermutation walked %d permutations, want 24", count)
	}
	// Final permutation should be [4,3,2,1].
	want := []int{4, 3, 2, 1}
	for i := range perm {
		if perm[i] != want[i] {
			t.Errorf("final perm: got %v, want %v", perm, want)
			break
		}
	}
}

func TestNextPermutation_Empty(t *testing.T) {
	if NextPermutation(nil) {
		t.Error("NextPermutation(nil) should return false")
	}
	if NextPermutation([]int{}) {
		t.Error("NextPermutation([]) should return false")
	}
}

func TestNextPermutation_Single(t *testing.T) {
	if NextPermutation([]int{1}) {
		t.Error("NextPermutation([1]) should return false")
	}
}

// ---------------------------------------------------------------------------
// RandomSubset
// ---------------------------------------------------------------------------

type fixedRNG struct{ vals []int; idx int }

func (r *fixedRNG) Intn(n int) int {
	v := r.vals[r.idx%len(r.vals)]
	r.idx++
	return v % n
}

func TestRandomSubset_Basic(t *testing.T) {
	rng := &fixedRNG{vals: []int{0, 0, 0, 0, 0}}
	result := RandomSubset(5, 3, rng)
	if len(result) != 3 {
		t.Errorf("RandomSubset(5,3) returned %d elements, want 3", len(result))
	}
	// All elements should be in [0,5).
	for _, v := range result {
		if v < 0 || v >= 5 {
			t.Errorf("element %d out of range [0,5)", v)
		}
	}
}

func TestRandomSubset_Distinct(t *testing.T) {
	rng := &fixedRNG{vals: []int{2, 1, 0, 3}}
	result := RandomSubset(10, 4, rng)
	seen := make(map[int]bool)
	for _, v := range result {
		if seen[v] {
			t.Errorf("duplicate element %d in subset %v", v, result)
		}
		seen[v] = true
	}
}

func TestRandomSubset_Invalid(t *testing.T) {
	rng := &fixedRNG{vals: []int{0}}
	if got := RandomSubset(5, -1, rng); got != nil {
		t.Errorf("RandomSubset(5,-1) should be nil")
	}
	if got := RandomSubset(5, 6, rng); got != nil {
		t.Errorf("RandomSubset(5,6) should be nil")
	}
}

func TestRandomSubset_ZeroK(t *testing.T) {
	rng := &fixedRNG{vals: []int{0}}
	result := RandomSubset(5, 0, rng)
	if len(result) != 0 {
		t.Errorf("RandomSubset(5,0) should return empty slice, got %v", result)
	}
}

func TestRandomSubset_FullN(t *testing.T) {
	rng := &fixedRNG{vals: []int{0, 0, 0, 0, 0}}
	result := RandomSubset(5, 5, rng)
	if len(result) != 5 {
		t.Errorf("RandomSubset(5,5) returned %d elements, want 5", len(result))
	}
	// All elements of {0..4} should be present.
	sort.Ints(result)
	for i := 0; i < 5; i++ {
		if result[i] != i {
			t.Errorf("RandomSubset(5,5) missing element %d, got %v", i, result)
			break
		}
	}
}
