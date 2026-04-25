// commit-conscience: SKIP_FNV_CHECK reason=dual-form-library-test (tests for both FNV1a32 and FNV1a64 in a dual-form library)

package crypto

import (
	"encoding/json"
	"math"
	"os"
	"strconv"
	"testing"

	"github.com/davly/reality/testutil"
)

// =========================================================================
// Primality tests
// =========================================================================

func TestIsPrime_KnownPrimes(t *testing.T) {
	primes := []uint64{2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43,
		97, 101, 997, 7919, 104729, 1299709}
	for _, p := range primes {
		if !IsPrime(p) {
			t.Errorf("IsPrime(%d) = false, want true", p)
		}
	}
}

func TestIsPrime_KnownComposites(t *testing.T) {
	composites := []uint64{4, 6, 8, 9, 10, 12, 15, 21, 25, 100, 1000, 561, 1105, 1729}
	for _, c := range composites {
		if IsPrime(c) {
			t.Errorf("IsPrime(%d) = true, want false", c)
		}
	}
}

func TestIsPrime_EdgeCases(t *testing.T) {
	// 0 and 1 are not prime.
	if IsPrime(0) {
		t.Error("IsPrime(0) = true, want false")
	}
	if IsPrime(1) {
		t.Error("IsPrime(1) = true, want false")
	}
	// 2 is the smallest prime.
	if !IsPrime(2) {
		t.Error("IsPrime(2) = false, want true")
	}
}

func TestIsPrime_LargePrime(t *testing.T) {
	// 2^61 - 1 = 2305843009213693951 is a Mersenne prime.
	mersennePrime := uint64(1)<<61 - 1
	if !IsPrime(mersennePrime) {
		t.Errorf("IsPrime(2^61-1) = false, want true (Mersenne prime)")
	}
}

func TestIsPrime_LargeComposite(t *testing.T) {
	// 2^61 - 1 + 2 is even, so composite.
	n := (uint64(1)<<61 - 1) + 1
	if IsPrime(n) {
		t.Errorf("IsPrime(2^61) = true, want false")
	}
}

// =========================================================================
// Miller-Rabin tests
// =========================================================================

func TestMillerRabin_Primes(t *testing.T) {
	primes := []uint64{2, 3, 997, 7919, 104729}
	for _, p := range primes {
		if !MillerRabin(p, 5) {
			t.Errorf("MillerRabin(%d, 5) = false, want true", p)
		}
	}
}

func TestMillerRabin_CarmichaelNumbers(t *testing.T) {
	// Carmichael numbers: composites that satisfy Fermat's little theorem
	// for all bases coprime to them. Miller-Rabin should catch them.
	carmichaels := []uint64{561, 1105, 1729, 2465, 2821, 6601, 8911}
	for _, c := range carmichaels {
		if MillerRabin(c, 3) {
			t.Errorf("MillerRabin(%d, 3) = true, want false (Carmichael number)", c)
		}
	}
}

func TestMillerRabin_EdgeCases(t *testing.T) {
	if MillerRabin(0, 5) {
		t.Error("MillerRabin(0, 5) = true, want false")
	}
	if MillerRabin(1, 5) {
		t.Error("MillerRabin(1, 5) = true, want false")
	}
	if !MillerRabin(2, 1) {
		t.Error("MillerRabin(2, 1) = false, want true")
	}
}

// =========================================================================
// Golden-file Miller-Rabin test
// =========================================================================

func TestGoldenMillerRabin(t *testing.T) {
	gf := testutil.LoadGolden(t, "../testdata/crypto/miller_rabin.json")

	if gf.Function != "Crypto.MillerRabin" {
		t.Errorf("golden file function = %q, want %q", gf.Function, "Crypto.MillerRabin")
	}

	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			nF, ok := toUint64FromAny(tc.Inputs["n"])
			if !ok {
				t.Fatalf("input 'n' is not a number: %v", tc.Inputs["n"])
			}
			kF, ok := toUint64FromAny(tc.Inputs["k"])
			if !ok {
				t.Fatalf("input 'k' is not a number: %v", tc.Inputs["k"])
			}
			expectedBool, ok := tc.Expected.(bool)
			if !ok {
				t.Fatalf("expected is not a bool: %v", tc.Expected)
			}

			got := MillerRabin(nF, int(kF))
			if got != expectedBool {
				t.Errorf("MillerRabin(%d, %d) = %v, want %v", nF, kF, got, expectedBool)
			}
		})
	}
}

// =========================================================================
// Factorization tests
// =========================================================================

func TestPrimeFactors_Known(t *testing.T) {
	tests := []struct {
		n    uint64
		want []uint64
	}{
		{12, []uint64{2, 2, 3}},
		{100, []uint64{2, 2, 5, 5}},
		{60, []uint64{2, 2, 3, 5}},
		{2, []uint64{2}},
		{997, []uint64{997}},
		{1729, []uint64{7, 13, 19}}, // Hardy-Ramanujan number
	}

	for _, tt := range tests {
		got := PrimeFactors(tt.n)
		if !uint64SliceEqual(got, tt.want) {
			t.Errorf("PrimeFactors(%d) = %v, want %v", tt.n, got, tt.want)
		}
	}
}

func TestPrimeFactors_EdgeCases(t *testing.T) {
	if got := PrimeFactors(0); len(got) != 0 {
		t.Errorf("PrimeFactors(0) = %v, want []", got)
	}
	if got := PrimeFactors(1); len(got) != 0 {
		t.Errorf("PrimeFactors(1) = %v, want []", got)
	}
}

func TestPrimeFactors_ProductEqualsN(t *testing.T) {
	// Verify that the product of factors equals n for various inputs.
	testNs := []uint64{12, 100, 360, 65537, 1000000}
	for _, n := range testNs {
		factors := PrimeFactors(n)
		product := uint64(1)
		for _, f := range factors {
			product *= f
		}
		if product != n {
			t.Errorf("PrimeFactors(%d) = %v, product = %d, want %d", n, factors, product, n)
		}
	}
}

// =========================================================================
// NextPrime tests
// =========================================================================

func TestNextPrime(t *testing.T) {
	tests := []struct {
		n    uint64
		want uint64
	}{
		{0, 2},
		{1, 2},
		{2, 2},
		{3, 3},
		{4, 5},
		{10, 11},
		{14, 17},
		{100, 101},
		{997, 997},  // 997 is already prime
		{998, 1009},
	}

	for _, tt := range tests {
		got := NextPrime(tt.n)
		if got != tt.want {
			t.Errorf("NextPrime(%d) = %d, want %d", tt.n, got, tt.want)
		}
	}
}

// =========================================================================
// GCD / LCM tests
// =========================================================================

func TestGCD_Known(t *testing.T) {
	tests := []struct {
		a, b, want uint64
	}{
		{12, 8, 4},
		{100, 75, 25},
		{17, 13, 1},    // coprime
		{48, 48, 48},   // identical
		{0, 5, 5},
		{5, 0, 5},
		{0, 0, 0},
		{1, 1, 1},
		{14, 21, 7},
	}

	for _, tt := range tests {
		got := GCD(tt.a, tt.b)
		if got != tt.want {
			t.Errorf("GCD(%d, %d) = %d, want %d", tt.a, tt.b, got, tt.want)
		}
	}
}

func TestLCM_Known(t *testing.T) {
	tests := []struct {
		a, b, want uint64
	}{
		{4, 6, 12},
		{12, 8, 24},
		{7, 13, 91},    // coprime
		{0, 5, 0},
		{5, 0, 0},
		{0, 0, 0},
		{1, 1, 1},
	}

	for _, tt := range tests {
		got := LCM(tt.a, tt.b)
		if got != tt.want {
			t.Errorf("LCM(%d, %d) = %d, want %d", tt.a, tt.b, got, tt.want)
		}
	}
}

func TestExtendedGCD(t *testing.T) {
	tests := []struct {
		a, b    int64
		wantGCD int64
	}{
		{12, 8, 4},
		{35, 15, 5},
		{17, 13, 1},
		{100, 75, 25},
		{0, 5, 5},
		{5, 0, 5},
		{-12, 8, 4},
	}

	for _, tt := range tests {
		gcd, x, y := ExtendedGCD(tt.a, tt.b)
		if gcd != tt.wantGCD {
			t.Errorf("ExtendedGCD(%d, %d): gcd = %d, want %d", tt.a, tt.b, gcd, tt.wantGCD)
		}
		// Verify Bezout's identity: a*x + b*y = gcd.
		if tt.a*x+tt.b*y != gcd {
			t.Errorf("ExtendedGCD(%d, %d): Bezout's identity failed: %d*%d + %d*%d = %d, want %d",
				tt.a, tt.b, tt.a, x, tt.b, y, tt.a*x+tt.b*y, gcd)
		}
	}
}

// =========================================================================
// Modular arithmetic tests
// =========================================================================

func TestModPow_Known(t *testing.T) {
	tests := []struct {
		base, exp, mod, want uint64
	}{
		{2, 10, 1000, 24},      // 2^10 = 1024 mod 1000 = 24
		{3, 4, 100, 81},        // 3^4 = 81
		{2, 0, 100, 1},         // x^0 = 1
		{0, 5, 100, 0},         // 0^x = 0
		{5, 3, 13, 8},          // 5^3 = 125 mod 13 = 8
		{7, 1, 13, 7},          // x^1 = x
		{2, 10, 1, 0},          // anything mod 1 = 0
	}

	for _, tt := range tests {
		got := ModPow(tt.base, tt.exp, tt.mod)
		if got != tt.want {
			t.Errorf("ModPow(%d, %d, %d) = %d, want %d", tt.base, tt.exp, tt.mod, got, tt.want)
		}
	}
}

func TestModPow_FermatsLittleTheorem(t *testing.T) {
	// Fermat's little theorem: a^(p-1) ≡ 1 (mod p) for prime p and gcd(a,p) = 1.
	primes := []uint64{7, 11, 13, 17, 19, 23, 997}
	for _, p := range primes {
		for a := uint64(2); a < 6; a++ {
			got := ModPow(a, p-1, p)
			if got != 1 {
				t.Errorf("Fermat: %d^(%d-1) mod %d = %d, want 1", a, p, p, got)
			}
		}
	}
}

func TestModInverse(t *testing.T) {
	tests := []struct {
		a, mod uint64
		want   uint64
		ok     bool
	}{
		{3, 7, 5, true},    // 3*5 = 15 ≡ 1 (mod 7)
		{2, 5, 3, true},    // 2*3 = 6 ≡ 1 (mod 5)
		{6, 7, 6, true},    // 6*6 = 36 ≡ 1 (mod 7)
		{2, 4, 0, false},   // gcd(2,4) = 2, no inverse
		{0, 5, 0, false},   // 0 has no inverse
	}

	for _, tt := range tests {
		got, ok := ModInverse(tt.a, tt.mod)
		if ok != tt.ok {
			t.Errorf("ModInverse(%d, %d): ok = %v, want %v", tt.a, tt.mod, ok, tt.ok)
			continue
		}
		if ok && got != tt.want {
			t.Errorf("ModInverse(%d, %d) = %d, want %d", tt.a, tt.mod, got, tt.want)
		}
	}
}

func TestModInverse_Verification(t *testing.T) {
	// For each inverse, verify a * inv ≡ 1 (mod m).
	pairs := [][2]uint64{{3, 7}, {2, 5}, {6, 7}, {10, 13}, {7, 11}}
	for _, pair := range pairs {
		a, m := pair[0], pair[1]
		inv, ok := ModInverse(a, m)
		if !ok {
			t.Errorf("ModInverse(%d, %d) returned not ok", a, m)
			continue
		}
		if (a*inv)%m != 1 {
			t.Errorf("ModInverse(%d, %d) = %d, but %d*%d mod %d = %d, want 1",
				a, m, inv, a, inv, m, (a*inv)%m)
		}
	}
}

func TestChineseRemainder(t *testing.T) {
	// x ≡ 2 (mod 3), x ≡ 3 (mod 5), x ≡ 2 (mod 7) → x = 23.
	result, err := ChineseRemainder([]uint64{2, 3, 2}, []uint64{3, 5, 7})
	if err != nil {
		t.Fatalf("ChineseRemainder: %v", err)
	}
	if result != 23 {
		t.Errorf("ChineseRemainder([2,3,2], [3,5,7]) = %d, want 23", result)
	}

	// Verify result satisfies all congruences.
	if result%3 != 2 || result%5 != 3 || result%7 != 2 {
		t.Errorf("CRT result %d doesn't satisfy congruences: %d mod 3 = %d, %d mod 5 = %d, %d mod 7 = %d",
			result, result, result%3, result, result%5, result, result%7)
	}
}

func TestChineseRemainder_Simple(t *testing.T) {
	// x ≡ 1 (mod 2), x ≡ 2 (mod 3) → x = 5.
	result, err := ChineseRemainder([]uint64{1, 2}, []uint64{2, 3})
	if err != nil {
		t.Fatalf("ChineseRemainder: %v", err)
	}
	if result != 5 {
		t.Errorf("ChineseRemainder([1,2], [2,3]) = %d, want 5", result)
	}
}

func TestChineseRemainder_Errors(t *testing.T) {
	// Empty inputs.
	_, err := ChineseRemainder(nil, nil)
	if err == nil {
		t.Error("ChineseRemainder(nil, nil) expected error")
	}

	// Mismatched lengths.
	_, err = ChineseRemainder([]uint64{1}, []uint64{2, 3})
	if err == nil {
		t.Error("ChineseRemainder with mismatched lengths expected error")
	}

	// Zero modulus.
	_, err = ChineseRemainder([]uint64{1, 2}, []uint64{3, 0})
	if err == nil {
		t.Error("ChineseRemainder with zero modulus expected error")
	}

	// Non-coprime moduli.
	_, err = ChineseRemainder([]uint64{1, 2}, []uint64{4, 6})
	if err == nil {
		t.Error("ChineseRemainder with non-coprime moduli expected error")
	}
}

// =========================================================================
// Hash function tests
// =========================================================================

func TestFNV1a32_Reference(t *testing.T) {
	// Verified against Go stdlib hash/fnv.
	tests := []struct {
		data string
		want uint32
	}{
		{"", 2166136261},
		{"hello", 1335831723},
		{"foobar", 3214735720},
	}

	for _, tt := range tests {
		got := FNV1a32([]byte(tt.data))
		if got != tt.want {
			t.Errorf("FNV1a32(%q) = %d, want %d", tt.data, got, tt.want)
		}
	}
}

func TestFNV1a64_Reference(t *testing.T) {
	// Verified against Go stdlib hash/fnv.
	tests := []struct {
		data string
		want uint64
	}{
		{"", 14695981039346656037},
		{"hello", 11831194018420276491},
		{"foobar", 9625390261332436968},
	}

	for _, tt := range tests {
		got := FNV1a64([]byte(tt.data))
		if got != tt.want {
			t.Errorf("FNV1a64(%q) = %d, want %d", tt.data, got, tt.want)
		}
	}
}

func TestFNV1a_Deterministic(t *testing.T) {
	data := []byte("the quick brown fox jumps over the lazy dog")
	h1 := FNV1a64(data)
	h2 := FNV1a64(data)
	if h1 != h2 {
		t.Errorf("FNV1a64 not deterministic: %d != %d", h1, h2)
	}
}

func TestMurmurHash3_32_Reference(t *testing.T) {
	tests := []struct {
		data string
		seed uint32
		want uint32
	}{
		{"", 0, 0},
		{"hello", 0, 613153351},
		{"hello", 42, 3806057185},
		{"foobar", 0, 2764362941},
	}

	for _, tt := range tests {
		got := MurmurHash3_32([]byte(tt.data), tt.seed)
		if got != tt.want {
			t.Errorf("MurmurHash3_32(%q, %d) = %d, want %d", tt.data, tt.seed, got, tt.want)
		}
	}
}

func TestMurmurHash3_32_Deterministic(t *testing.T) {
	data := []byte("repeatability matters")
	h1 := MurmurHash3_32(data, 0)
	h2 := MurmurHash3_32(data, 0)
	if h1 != h2 {
		t.Errorf("MurmurHash3_32 not deterministic: %d != %d", h1, h2)
	}
}

func TestMurmurHash3_32_DifferentSeeds(t *testing.T) {
	data := []byte("same data different seeds")
	h1 := MurmurHash3_32(data, 0)
	h2 := MurmurHash3_32(data, 1)
	if h1 == h2 {
		t.Error("MurmurHash3_32 with different seeds produced same hash — unlikely collision")
	}
}

func TestMurmurHash3_32_AllLengths(t *testing.T) {
	// Test all tail lengths (0, 1, 2, 3 bytes after last 4-byte block).
	for length := 0; length <= 8; length++ {
		data := make([]byte, length)
		for i := range data {
			data[i] = byte(i + 1)
		}
		// Should not panic.
		_ = MurmurHash3_32(data, 0)
	}
}

// =========================================================================
// Consistent hash tests
// =========================================================================

func TestConsistentHash_Deterministic(t *testing.T) {
	b1 := ConsistentHash(42, 10)
	b2 := ConsistentHash(42, 10)
	if b1 != b2 {
		t.Errorf("ConsistentHash not deterministic: %d != %d", b1, b2)
	}
}

func TestConsistentHash_InRange(t *testing.T) {
	for key := uint64(0); key < 100; key++ {
		bucket := ConsistentHash(key, 10)
		if bucket < 0 || bucket >= 10 {
			t.Errorf("ConsistentHash(%d, 10) = %d, want [0, 10)", key, bucket)
		}
	}
}

func TestConsistentHash_Monotonicity(t *testing.T) {
	// When adding a bucket, keys should only move to the new bucket,
	// never between existing buckets.
	const numKeys = 1000
	const oldBuckets = 10
	const newBuckets = 11

	for key := uint64(0); key < numKeys; key++ {
		oldBucket := ConsistentHash(key, oldBuckets)
		newBucket := ConsistentHash(key, newBuckets)

		// Key should stay in same bucket OR move to the new bucket (bucket 10).
		if newBucket != oldBucket && newBucket != oldBuckets {
			t.Errorf("ConsistentHash(%d): moved from bucket %d to %d (expected same or %d)",
				key, oldBucket, newBucket, oldBuckets)
		}
	}
}

func TestConsistentHash_EdgeCases(t *testing.T) {
	if got := ConsistentHash(42, 0); got != 0 {
		t.Errorf("ConsistentHash(42, 0) = %d, want 0", got)
	}
	if got := ConsistentHash(42, -1); got != 0 {
		t.Errorf("ConsistentHash(42, -1) = %d, want 0", got)
	}
	if got := ConsistentHash(42, 1); got != 0 {
		t.Errorf("ConsistentHash(42, 1) = %d, want 0", got)
	}
}

// =========================================================================
// PRNG tests — Mersenne Twister
// =========================================================================

func TestMersenneTwister_Deterministic(t *testing.T) {
	mt1 := NewMersenneTwister(42)
	mt2 := NewMersenneTwister(42)

	for i := 0; i < 100; i++ {
		v1, v2 := mt1.Uint64(), mt2.Uint64()
		if v1 != v2 {
			t.Errorf("MT seed=42, index %d: %d != %d", i, v1, v2)
		}
	}
}

func TestGoldenMersenneTwister(t *testing.T) {
	gf := testutil.LoadGolden(t, "../testdata/crypto/mersenne_twister.json")

	if gf.Function != "Crypto.MersenneTwister" {
		t.Errorf("golden file function = %q, want %q", gf.Function, "Crypto.MersenneTwister")
	}

	mt := NewMersenneTwister(42)
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			got := mt.Uint64()
			// Golden file stores uint64 as strings to preserve precision
			// (float64 loses precision above 2^53).
			expectedStr, ok := tc.Expected.(string)
			if !ok {
				t.Fatalf("expected is not a string: %v (type %T)", tc.Expected, tc.Expected)
			}
			expected, err := strconv.ParseUint(expectedStr, 10, 64)
			if err != nil {
				t.Fatalf("cannot parse expected value %q as uint64: %v", expectedStr, err)
			}
			if got != expected {
				t.Errorf("got %d, want %d", got, expected)
			}
		})
	}
}

func TestMersenneTwister_Float64Range(t *testing.T) {
	mt := NewMersenneTwister(12345)
	for i := 0; i < 1000; i++ {
		f := mt.Float64()
		if f < 0 || f >= 1 {
			t.Errorf("MT Float64() = %v, want [0, 1)", f)
		}
	}
}

func TestMersenneTwister_DifferentSeeds(t *testing.T) {
	mt1 := NewMersenneTwister(1)
	mt2 := NewMersenneTwister(2)
	if mt1.Uint64() == mt2.Uint64() {
		t.Error("MT with different seeds produced same first output — unlikely")
	}
}

// =========================================================================
// PRNG tests — PCG
// =========================================================================

func TestPCG_Deterministic(t *testing.T) {
	p1 := NewPCG(42, 54)
	p2 := NewPCG(42, 54)

	for i := 0; i < 100; i++ {
		v1, v2 := p1.Uint32(), p2.Uint32()
		if v1 != v2 {
			t.Errorf("PCG seed=(42,54), index %d: %d != %d", i, v1, v2)
		}
	}
}

func TestPCG_KnownSequence(t *testing.T) {
	pcg := NewPCG(42, 54)
	expected := []uint32{210066564, 199112357, 1239240105, 2463922947, 72149789}
	for i, want := range expected {
		got := pcg.Uint32()
		if got != want {
			t.Errorf("PCG(42,54)[%d] = %d, want %d", i, got, want)
		}
	}
}

func TestPCG_Float64Range(t *testing.T) {
	pcg := NewPCG(12345, 67890)
	for i := 0; i < 1000; i++ {
		f := pcg.Float64()
		if f < 0 || f >= 1 {
			t.Errorf("PCG Float64() = %v, want [0, 1)", f)
		}
	}
}

func TestPCG_DifferentSeeds(t *testing.T) {
	p1 := NewPCG(1, 1)
	p2 := NewPCG(2, 1)
	if p1.Uint32() == p2.Uint32() {
		t.Error("PCG with different seeds produced same first output — unlikely")
	}
}

func TestPCG_DifferentSequences(t *testing.T) {
	p1 := NewPCG(42, 1)
	p2 := NewPCG(42, 2)
	// Different sequences should produce different outputs (at least eventually).
	same := true
	for i := 0; i < 10; i++ {
		if p1.Uint32() != p2.Uint32() {
			same = false
			break
		}
	}
	if same {
		t.Error("PCG with different sequences produced identical first 10 outputs — unlikely")
	}
}

// =========================================================================
// PRNG tests — Xoshiro256**
// =========================================================================

func TestXoshiro256_Deterministic(t *testing.T) {
	x1 := NewXoshiro256(42)
	x2 := NewXoshiro256(42)

	for i := 0; i < 100; i++ {
		v1, v2 := x1.Uint64(), x2.Uint64()
		if v1 != v2 {
			t.Errorf("Xoshiro256 seed=42, index %d: %d != %d", i, v1, v2)
		}
	}
}

func TestXoshiro256_KnownSequence(t *testing.T) {
	xo := NewXoshiro256(42)
	expected := []uint64{
		1546998764402558742,
		6990951692964543102,
		12544586762248559009,
		17057574109182124193,
		18295552978065317476,
	}
	for i, want := range expected {
		got := xo.Uint64()
		if got != want {
			t.Errorf("Xoshiro256(42)[%d] = %d, want %d", i, got, want)
		}
	}
}

func TestXoshiro256_Float64Range(t *testing.T) {
	xo := NewXoshiro256(12345)
	for i := 0; i < 1000; i++ {
		f := xo.Float64()
		if f < 0 || f >= 1 {
			t.Errorf("Xoshiro256 Float64() = %v, want [0, 1)", f)
		}
	}
}

func TestXoshiro256_DifferentSeeds(t *testing.T) {
	x1 := NewXoshiro256(1)
	x2 := NewXoshiro256(2)
	if x1.Uint64() == x2.Uint64() {
		t.Error("Xoshiro256 with different seeds produced same first output — unlikely")
	}
}

// =========================================================================
// PRNG distribution uniformity
// =========================================================================

func TestMersenneTwister_Uniformity(t *testing.T) {
	mt := NewMersenneTwister(42)
	const n = 10000
	const bins = 10
	counts := make([]int, bins)

	for i := 0; i < n; i++ {
		f := mt.Float64()
		bin := int(f * float64(bins))
		if bin >= bins {
			bin = bins - 1
		}
		counts[bin]++
	}

	// Chi-squared test: expect ~1000 per bin. Allow generous threshold.
	expected := float64(n) / float64(bins)
	chi2 := 0.0
	for _, c := range counts {
		diff := float64(c) - expected
		chi2 += (diff * diff) / expected
	}

	// For 9 degrees of freedom, chi2 > 27.88 has p < 0.001.
	if chi2 > 27.88 {
		t.Errorf("MT uniformity chi2 = %.2f (threshold 27.88), counts = %v", chi2, counts)
	}
}

func TestPCG_Uniformity(t *testing.T) {
	pcg := NewPCG(42, 54)
	const n = 10000
	const bins = 10
	counts := make([]int, bins)

	for i := 0; i < n; i++ {
		f := pcg.Float64()
		bin := int(f * float64(bins))
		if bin >= bins {
			bin = bins - 1
		}
		counts[bin]++
	}

	expected := float64(n) / float64(bins)
	chi2 := 0.0
	for _, c := range counts {
		diff := float64(c) - expected
		chi2 += (diff * diff) / expected
	}

	if chi2 > 27.88 {
		t.Errorf("PCG uniformity chi2 = %.2f (threshold 27.88), counts = %v", chi2, counts)
	}
}

func TestXoshiro256_Uniformity(t *testing.T) {
	xo := NewXoshiro256(42)
	const n = 10000
	const bins = 10
	counts := make([]int, bins)

	for i := 0; i < n; i++ {
		f := xo.Float64()
		bin := int(f * float64(bins))
		if bin >= bins {
			bin = bins - 1
		}
		counts[bin]++
	}

	expected := float64(n) / float64(bins)
	chi2 := 0.0
	for _, c := range counts {
		diff := float64(c) - expected
		chi2 += (diff * diff) / expected
	}

	if chi2 > 27.88 {
		t.Errorf("Xoshiro256 uniformity chi2 = %.2f (threshold 27.88), counts = %v", chi2, counts)
	}
}

// =========================================================================
// Helpers
// =========================================================================

// toUint64FromAny converts a JSON-decoded value to uint64.
func toUint64FromAny(v any) (uint64, bool) {
	switch val := v.(type) {
	case float64:
		if val < 0 || val != math.Trunc(val) {
			return 0, false
		}
		return uint64(val), true
	case int:
		return uint64(val), true
	case int64:
		return uint64(val), true
	case json.Number:
		n, err := val.Int64()
		if err != nil {
			return 0, false
		}
		return uint64(n), true
	default:
		return 0, false
	}
}

// uint64SliceEqual compares two uint64 slices for equality.
func uint64SliceEqual(a, b []uint64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// Ensure os is used (for golden-file tests).
var _ = os.ReadFile
