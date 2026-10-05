package constants

import (
	"math/big"
	"testing"
)

// stefanBoltzmannExact is the Stefan-Boltzmann constant of the 2019 SI,
//
//	sigma = 2 pi^5 k^4 / (15 h^3 c^2),  k = 1.380649e-23, h = 6.62607015e-34, c = 299792458,
//
// to 40 significant digits, computed with mpmath at 80 digits. It is a
// non-terminating decimal: NIST prints it as "5.670 374 419..." and the
// ellipsis is part of the value.
const stefanBoltzmannExact = "5.670374419184429453970996731889230875840e-8"

func parseBig(t *testing.T, s string) *big.Float {
	t.Helper()
	v, _, err := big.ParseFloat(s, 10, 300, big.ToNearestEven)
	if err != nil {
		t.Fatalf("parsing %q: %v", s, err)
	}
	return v
}

// The float64 constant must be the correctly rounded exact value. The former
// 5.670374419e-8 truncated the digits after the ninth and was low by 3.3e-11
// relative, 5.7e6 times the rounding error of a float64.
func TestStefanBoltzmann_IsTheCorrectlyRoundedExactValue(t *testing.T) {
	want, _ := parseBig(t, stefanBoltzmannExact).Float64() // nearest float64, ties to even
	if StefanBoltzmann != want {
		t.Errorf("StefanBoltzmann = %.17g, want %.17g (the exact value rounded to float64)", StefanBoltzmann, want)
	}
}

// The 40-digit value above is itself checked against the definition, derived
// here with 300-bit arithmetic from the defining constants.
func TestStefanBoltzmann_ExactValueMatchesItsDefinition(t *testing.T) {
	pi := parseBig(t, "3.14159265358979323846264338327950288419716939937510582097494459")
	k := parseBig(t, "1.380649e-23")
	h := parseBig(t, "6.62607015e-34")
	c := parseBig(t, "299792458")

	pow := func(x *big.Float, n int) *big.Float {
		r := new(big.Float).SetPrec(300).SetInt64(1)
		for i := 0; i < n; i++ {
			r.Mul(r, x)
		}
		return r
	}
	num := new(big.Float).SetPrec(300).Mul(big.NewFloat(2), pow(pi, 5))
	num.Mul(num, pow(k, 4))
	den := new(big.Float).SetPrec(300).Mul(big.NewFloat(15), pow(h, 3))
	den.Mul(den, pow(c, 2))
	sigma := new(big.Float).SetPrec(300).Quo(num, den)

	exact := parseBig(t, stefanBoltzmannExact)
	diff := new(big.Float).SetPrec(300).Sub(sigma, exact)
	diff.Quo(diff, exact)
	limit := parseBig(t, "1e-39") // the stored value is rounded to 40 digits
	if diff.Abs(diff).Cmp(limit) > 0 {
		t.Errorf("definition and stored exact value differ by a relative %v", diff)
	}
}
