package crypto

import (
	"math"
	"math/big"
	"testing"
)

// Least common multiples whose size matters: some fit in a uint64, some do
// not. The exact values (lcm) and their residues modulo 2^64 (wrapped) were
// computed with arbitrary-precision integers (Python int, equivalent to
// math/big), independently of LCM and LCMChecked.
var lcmCases = []struct {
	name    string
	a, b    uint64
	lcm     string // exact decimal value
	wrapped uint64 // lcm mod 2^64: what LCM returns
	fits    bool   // lcm <= math.MaxUint64
}{
	{"smallest operands that wrap: 2^32, 2^32+1", 4294967296, 4294967297, "18446744078004518912", 4294967296, false},
	{"largest exact fit: 2^32-1, 2^32+1", 4294967295, 4294967297, "18446744073709551615", math.MaxUint64, true},
	{"2^63, 3", 9223372036854775808, 3, "27670116110564327424", 9223372036854775808, false},
	{"2^63, 2^62 (one divides the other)", 9223372036854775808, 4611686018427387904, "9223372036854775808", 9223372036854775808, true},
	{"MaxUint64, MaxUint64-1 (coprime)", math.MaxUint64, math.MaxUint64 - 1, "340282366920938463408034375210639556610", 2, false},
	{"MaxUint64, 1", math.MaxUint64, 1, "18446744073709551615", math.MaxUint64, true},
	{"MaxUint64, MaxUint64", math.MaxUint64, math.MaxUint64, "18446744073709551615", math.MaxUint64, true},
	{"2^63, 2^63-1 (coprime)", 9223372036854775808, 9223372036854775807, "85070591730234615856620279821087277056", 9223372036854775808, false},
	{"3^40, 2^40 (coprime)", 12157665459056928801, 1099511627776, "13367494538843734067838845976576", 2299123893656354816, false},
	{"two primes near 2^32 (fits)", 4294967291, 4294967279, "18446743979220271189", 18446743979220271189, true},
	{"3e9, 7000000001", 3000000000, 7000000001, "21000000003000000000", 2553255929290448384, false},
	{"2^40, 2^41+1", 1099511627776, 2199023255553, "2417851639230357861040128", 1099511627776, false},
	{"6, 4", 6, 4, "12", 12, true},
	{"0, 5", 0, 5, "0", 0, true},
	{"5, 0", 5, 0, "0", 0, true},
	{"0, 0", 0, 0, "0", 0, true},
}

func TestLCMChecked_KnownCases(t *testing.T) {
	for _, c := range lcmCases {
		t.Run(c.name, func(t *testing.T) {
			got, ok := LCMChecked(c.a, c.b)
			if ok != c.fits {
				t.Fatalf("LCMChecked(%d, %d) ok = %v, want %v (exact lcm %s)", c.a, c.b, ok, c.fits, c.lcm)
			}
			want := uint64(0)
			if c.fits {
				want = c.wrapped // equal to the exact value when it fits
			}
			if got != want {
				t.Errorf("LCMChecked(%d, %d) = %d, want %d", c.a, c.b, got, want)
			}
		})
	}
}

// LCM wraps modulo 2^64 when the result does not fit (documented behaviour).
func TestLCM_WrapsModulo2Pow64(t *testing.T) {
	for _, c := range lcmCases {
		if got := LCM(c.a, c.b); got != c.wrapped {
			t.Errorf("%s: LCM(%d, %d) = %d, want %d (exact lcm %s mod 2^64)", c.name, c.a, c.b, got, c.wrapped, c.lcm)
		}
	}
}

// Every pair from a set of operands around the powers of two, the word and
// half-word boundaries, and a few primes, checked against math/big.
func TestLCMChecked_AgainstBigInt(t *testing.T) {
	vals := []uint64{0, 1, 2, 3, 4, 5, 6, 7, 10, 12, 255, 256, 65535, 65536, 65537,
		1_000_000_007, 4294967291, 1_000_000_000_000_000_000, math.MaxUint64 - 1, math.MaxUint64}
	for k := uint(1); k < 64; k++ {
		p := uint64(1) << k
		vals = append(vals, p-1, p, p+1)
	}
	mask := new(big.Int).Sub(new(big.Int).Lsh(big.NewInt(1), 64), big.NewInt(1))
	checked := 0
	for _, a := range vals {
		for _, b := range vals {
			A := new(big.Int).SetUint64(a)
			B := new(big.Int).SetUint64(b)
			l := new(big.Int)
			if a != 0 && b != 0 {
				g := new(big.Int).GCD(nil, nil, A, B)
				l.Div(A, g).Mul(l, B)
			}
			fits := l.IsUint64()
			got, ok := LCMChecked(a, b)
			if ok != fits {
				t.Fatalf("LCMChecked(%d, %d) ok = %v, exact lcm %s", a, b, ok, l)
			}
			if fits && got != l.Uint64() {
				t.Fatalf("LCMChecked(%d, %d) = %d, exact lcm %s", a, b, got, l)
			}
			if !fits && got != 0 {
				t.Fatalf("LCMChecked(%d, %d) = %d on overflow, want 0", a, b, got)
			}
			if w := new(big.Int).And(l, mask).Uint64(); LCM(a, b) != w {
				t.Fatalf("LCM(%d, %d) = %d, exact lcm %s mod 2^64 = %d", a, b, LCM(a, b), l, w)
			}
			checked++
		}
	}
	t.Logf("%d operand pairs checked against math/big", checked)
}
