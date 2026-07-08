package crypto

import (
	"math/big"
	"testing"
)

// TestModInverse_FullUint64Range pins the fix for moduli/values >= 2^63, where the
// uint64->int64 reinterpretation used to silently return a corrupt inverse with
// ok=true. Verifies a*inv ≡ 1 (mod m) via big.Int across the high-bit domain.
func TestModInverse_FullUint64Range(t *testing.T) {
	cases := []struct{ a, mod uint64 }{
		{3, (1 << 63) + 9},
		{5, 9223372036854775837},
		{7, (1 << 63) + 1},
		{12345, (1 << 63) + 99},
		{(1 << 63) + 7, 11}, // large a, small mod
	}
	for _, c := range cases {
		inv, ok := ModInverse(c.a, c.mod)
		if !ok {
			t.Fatalf("ModInverse(%d, %d): expected ok=true", c.a, c.mod)
		}
		prod := new(big.Int).Mod(
			new(big.Int).Mul(new(big.Int).SetUint64(c.a), new(big.Int).SetUint64(inv)),
			new(big.Int).SetUint64(c.mod),
		)
		if prod.Cmp(big.NewInt(1)) != 0 {
			t.Errorf("ModInverse(%d, %d)=%d: a*inv mod mod = %s, want 1", c.a, c.mod, inv, prod)
		}
	}
	// fast path (<2^63) still correct
	if inv, ok := ModInverse(3, 7); !ok || inv != 5 {
		t.Errorf("ModInverse(3,7)=%d,%v want 5,true", inv, ok)
	}
}
