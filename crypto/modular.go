package crypto

import (
	"errors"
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

	// Compute M = product of all moduli.
	M := uint64(1)
	for _, m := range moduli {
		M *= m
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
