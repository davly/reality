package crypto

import (
	"math/big"
	"testing"
)

// Regression tests for the uint64-native ModInverse fix and the ChineseRemainder
// overflow guard. Before the fix, ModInverse cast its uint64 operands to int64,
// so for operands >= 2^63 it returned a wrong inverse with ok=true (e.g.
// ModInverse(2^63+7, 1000003) = (619542, true) which is NOT an inverse) or a
// false "no inverse"; ChineseRemainder silently overflowed M and returned a
// value congruent to none of the inputs with a nil error.

func bigInv(a, mod uint64) (uint64, bool) {
	r := new(big.Int).ModInverse(new(big.Int).SetUint64(a), new(big.Int).SetUint64(mod))
	if r == nil {
		return 0, false
	}
	return r.Uint64(), true
}

func mulModBig(a, b, mod uint64) uint64 {
	p := new(big.Int).Mul(new(big.Int).SetUint64(a), new(big.Int).SetUint64(b))
	return p.Mod(p, new(big.Int).SetUint64(mod)).Uint64()
}

func TestModInverse_LargeOperands_MatchBigInt(t *testing.T) {
	cases := []struct{ a, mod uint64 }{
		{1<<63 + 7, 1000003},
		{3, 9223372036854775837},     // mod is prime > 2^63
		{1<<63 + 1, 9223372036854775837},
		{18446744073709551557, 9223372036854775783}, // both near 2^64 / 2^63, prime mod
		{12345678901234567, 9999999999999999961},
		{2, 9223372036854775837},
		{7, 13}, {3, 7}, {10, 17}, // small (unchanged-behavior regression)
	}
	for _, c := range cases {
		got, ok := ModInverse(c.a, c.mod)
		want, wantOK := bigInv(c.a, c.mod)
		if ok != wantOK {
			t.Errorf("ModInverse(%d,%d) ok=%v want %v", c.a, c.mod, ok, wantOK)
			continue
		}
		if ok {
			if got != want {
				t.Errorf("ModInverse(%d,%d)=%d want %d", c.a, c.mod, got, want)
			}
			// The defining property: (a*inv) mod m == 1.
			if mulModBig(c.a, got, c.mod) != 1 {
				t.Errorf("ModInverse(%d,%d)=%d is NOT an inverse ((a*inv)%%m=%d)", c.a, c.mod, got, mulModBig(c.a, got, c.mod))
			}
		}
	}
}

func TestModInverse_NonInvertible(t *testing.T) {
	// gcd(a,mod) != 1 must return (0,false).
	for _, c := range []struct{ a, mod uint64 }{{4, 8}, {6, 9}, {0, 5}, {10, 15}} {
		if got, ok := ModInverse(c.a, c.mod); ok {
			t.Errorf("ModInverse(%d,%d)=%d,true; want 0,false (gcd!=1)", c.a, c.mod, got)
		}
	}
	if v, ok := ModInverse(0, 1); !ok || v != 0 {
		t.Errorf("ModInverse(0,1)=%d,%v; want 0,true", v, ok)
	}
}

func TestChineseRemainder_OverflowGuarded(t *testing.T) {
	// Product of these five ~16-bit moduli exceeds 2^64 -> must error, not return
	// a wrong value with nil error.
	_, err := ChineseRemainder([]uint64{1, 2, 3, 4, 5}, []uint64{65521, 65519, 65497, 65479, 65449})
	if err == nil {
		t.Error("ChineseRemainder with overflowing modulus product returned nil error; want overflow error")
	}

	// Small valid systems still work (classic Sunzi: x≡2(3), x≡3(5), x≡2(7) -> 23).
	got, err := ChineseRemainder([]uint64{2, 3, 2}, []uint64{3, 5, 7})
	if err != nil || got != 23 {
		t.Errorf("ChineseRemainder(Sunzi)=%d,%v; want 23,nil", got, err)
	}
}
