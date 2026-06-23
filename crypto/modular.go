package crypto

import (
	"errors"
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
// i.e., the value x in [0, mod) such that (a * x) ≡ 1 (mod m). Returns
// (0, false) if gcd(a, mod) != 1 (inverse does not exist).
//
// The extended Euclidean algorithm is run entirely in uint64 arithmetic so the
// result is correct across the WHOLE uint64 range. (The previous implementation
// cast a and mod to int64; for operands >= 2^63 that cast wrapped to a negative
// value, so it silently returned a wrong inverse — reporting ok=true — or a
// false "no inverse". This is the foundation primitive for RSA/DH/CRT, where
// 64-bit moduli are routine.)
//
// Time complexity: O(log(mod) · cost_of_mulmod)
// Reference: Bezout's identity applied to modular arithmetic
func ModInverse(a, mod uint64) (uint64, bool) {
	if mod == 0 {
		return 0, false
	}
	if mod == 1 {
		return 0, true // everything ≡ 0 (mod 1)
	}

	// Iterative extended Euclidean. The Bezout coefficient t is maintained in
	// [0, mod) via modular arithmetic (mulmod is overflow-safe for any uint64
	// operands); the remainders r strictly decrease so plain uint64 subtraction
	// in "r - q*newR" never underflows or overflows.
	var t, newT uint64 = 0, 1
	r, newR := mod, a%mod
	for newR != 0 {
		q := r / newR
		t, newT = newT, subMod(t, mulmod(q, newT, mod), mod)
		r, newR = newR, r-q*newR
	}
	if r != 1 {
		return 0, false // gcd(a, mod) != 1 -> not invertible
	}
	return t, true
}

// subMod returns (a - b) mod m for inputs a, b already reduced into [0, m).
func subMod(a, b, m uint64) uint64 {
	if a >= b {
		return a - b
	}
	return a + (m - b)
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
