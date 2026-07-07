package crypto

import (
	"errors"
	"math/big"
	"math/bits"
)

// ---------------------------------------------------------------------------
// Modular exponentiation
// ---------------------------------------------------------------------------

// ModPow computes (base^exp) mod m using binary exponentiation (repeated
// squaring). Returns 0 if mod is 0.
//
// This is the standard algorithm used in RSA, Diffie-Hellman, and all
// modular arithmetic systems. The implementation avoids overflow by using
// modular multiplication at each step.
//
// Time complexity: O(log(exp) · cost_of_mulmod)
// Reference: Knuth, The Art of Computer Programming, Vol. 2, §4.6.3
func ModPow(base, exp, mod uint64) uint64 {
	if mod == 0 {
		return 0
	}
	if mod == 1 {
		return 0
	}

	result := uint64(1)
	base = base % mod

	for exp > 0 {
		if exp%2 == 1 {
			result = mulmod(result, base, mod)
		}
		exp /= 2
		base = mulmod(base, base, mod)
	}

	return result
}

// ---------------------------------------------------------------------------
// Modular inverse
// ---------------------------------------------------------------------------

// ModInverse returns the modular multiplicative inverse of a modulo mod,
// i.e., the value x such that (a * x) ≡ 1 (mod m). Returns (0, false)
// if gcd(a, mod) != 1 (inverse does not exist).
//
// Uses the extended Euclidean algorithm internally.
//
// Time complexity: O(log(min(a, mod)))
// Reference: Bezout's identity applied to modular arithmetic
func ModInverse(a, mod uint64) (uint64, bool) {
	if mod == 0 {
		return 0, false
	}
	if mod == 1 {
		return 0, true // everything ≡ 0 (mod 1)
	}

	// Below 2^63 the int64 extended-Euclid path is exact and fast. At/above
	// 2^63 a uint64 has its high bit set, so int64(a)/int64(mod) reinterpret it
	// as a NEGATIVE two's-complement value — the extended-Euclid run and the
	// final reduction then operate on the wrong magnitude and silently return a
	// corrupt inverse with ok=true. Use math/big there, which is correct across
	// the full uint64 range (the package doc advertises RSA/DH-style use with no
	// <2^63 restriction).
	if a < 1<<63 && mod < 1<<63 {
		gcd, x, _ := ExtendedGCD(int64(a), int64(mod))
		if gcd != 1 {
			return 0, false
		}
		// x may be negative; bring it into [0, mod).
		result := x % int64(mod)
		if result < 0 {
			result += int64(mod)
		}
		return uint64(result), true
	}

	inv := new(big.Int).ModInverse(new(big.Int).SetUint64(a), new(big.Int).SetUint64(mod))
	if inv == nil {
		return 0, false // gcd(a, mod) != 1, inverse does not exist
	}
	return inv.Uint64(), true
}

// ---------------------------------------------------------------------------
// Chinese Remainder Theorem
// ---------------------------------------------------------------------------

// ChineseRemainder solves a system of simultaneous congruences using the
// Chinese Remainder Theorem:
//
//   x ≡ residues[0] (mod moduli[0])
//   x ≡ residues[1] (mod moduli[1])
//   ...
//
// Returns the unique solution x in [0, M) where M = product of all moduli,
// provided all moduli are pairwise coprime. Returns an error if:
//   - inputs are empty or mismatched in length
//   - any modulus is 0
//   - moduli are not pairwise coprime
//
// Time complexity: O(n · log(M)) where n is the number of congruences
// Reference: Chinese Remainder Theorem (Sunzi, ~3rd century CE)
func ChineseRemainder(residues, moduli []uint64) (uint64, error) {
	if len(residues) == 0 || len(moduli) == 0 {
		return 0, errors.New("crypto.ChineseRemainder: empty inputs")
	}
	if len(residues) != len(moduli) {
		return 0, errors.New("crypto.ChineseRemainder: mismatched lengths")
	}

	for i, m := range moduli {
		if m == 0 {
			return 0, errors.New("crypto.ChineseRemainder: zero modulus")
		}
		_ = i
	}

	// Compute M = product of all moduli, detecting uint64 overflow. The previous
	// code accumulated M *= m and silently wrapped once the product exceeded
	// 2^64, after which every modular step used the wrong M and the function
	// returned a value congruent to NONE of the inputs with a nil error. We now
	// fail honestly; callers needing a larger M must use a big-integer CRT.
	M := uint64(1)
	for _, m := range moduli {
		hi, lo := bits.Mul64(M, m)
		if hi != 0 {
			return 0, errors.New("crypto.ChineseRemainder: product of moduli overflows uint64 (M must be < 2^64); use a big-integer CRT for larger systems")
		}
		M = lo
	}

	var result uint64

	for i := range residues {
		// Mi = M / moduli[i]
		mi := M / moduli[i]

		// Find Mi^(-1) mod moduli[i]
		inv, ok := ModInverse(mi, moduli[i])
		if !ok {
			return 0, errors.New("crypto.ChineseRemainder: moduli are not pairwise coprime")
		}

		// result += residues[i] * Mi * Mi_inv (all mod M)
		term := mulmod(residues[i]%moduli[i], mulmod(mi, inv, M), M)
		result = addmod(result, term, M)
	}

	return result, nil
}
