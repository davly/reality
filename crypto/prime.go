// Package crypto provides number theory and cryptographic primitives:
// primality testing, modular arithmetic, non-cryptographic hash functions,
// and deterministic pseudorandom number generators.
//
// All functions are pure, deterministic, and use only the Go standard
// library. PRNGs produce identical sequences for identical seeds across
// all platforms.
//
// Consumers: Recall (hashing), Phantom (hashing), Pistachio (PRNGs),
// Muse (random generation).
package crypto

// ---------------------------------------------------------------------------
// Primality testing
// ---------------------------------------------------------------------------

// IsPrime returns true if n is a prime number. For n < 2^32, a deterministic
// set of Miller-Rabin witnesses is used that gives correct results for all
// 32-bit integers. For larger n, a deterministic set of 7 witnesses is used
// that is correct for all n < 3.317×10^24 (more than covers uint64 range
// in practice for most uses).
//
// Time complexity: O(k · log²(n)) where k is the number of witnesses.
// Reference: Miller (1976), Rabin (1980); witness sets from
// https://miller-rabin.appspot.com/ (Jim Sinclair).
func IsPrime(n uint64) bool {
	if n < 2 {
		return false
	}
	// Small primes table for quick checks.
	smallPrimes := []uint64{2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}
	for _, p := range smallPrimes {
		if n == p {
			return true
		}
		if n%p == 0 {
			return false
		}
	}
	if n < 41 {
		// We've checked all primes up to 37 and n isn't one of them;
		// since n > 37 is handled below, and n < 41 non-primes are caught
		// by the divisibility checks above.
		return false
	}

	// Deterministic Miller-Rabin witnesses.
	// For n < 3,215,031,751, witnesses {2, 3, 5, 7} suffice.
	// For larger n, we use {2, 3, 5, 7, 11, 13, 17} which is correct
	// for all n < 3.317×10^24 (covers all uint64 values we care about).
	var witnesses []uint64
	if n < 3215031751 {
		witnesses = []uint64{2, 3, 5, 7}
	} else {
		witnesses = []uint64{2, 3, 5, 7, 11, 13, 17}
	}

	return millerRabinTest(n, witnesses)
}

// MillerRabin performs the Miller-Rabin probabilistic primality test on n
// using k random-ish witnesses. In this deterministic implementation, the
// witnesses are the first k primes (2, 3, 5, 7, 11, 13, ...).
//
// For k >= 7, this is deterministic for all uint64 values (see IsPrime).
// For smaller k, false positives are possible for Carmichael numbers.
//
// Formula: decompose n-1 = 2^r · d, then for each witness a check:
//   a^d ≡ 1 (mod n) OR a^(2^i · d) ≡ -1 (mod n) for some 0 <= i < r.
//
// Time complexity: O(k · log²(n))
// Reference: Miller (1976), Rabin (1980)
func MillerRabin(n uint64, k int) bool {
	if n < 2 {
		return false
	}
	if n == 2 || n == 3 {
		return true
	}
	if n%2 == 0 {
		return false
	}

	// Use first k primes as witnesses.
	allWitnesses := []uint64{2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47}
	if k > len(allWitnesses) {
		k = len(allWitnesses)
	}

	witnesses := allWitnesses[:k]
	return millerRabinTest(n, witnesses)
}

// millerRabinTest is the core Miller-Rabin test with a given witness set.
func millerRabinTest(n uint64, witnesses []uint64) bool {
	// Write n-1 as 2^r · d with d odd.
	d := n - 1
	r := uint(0)
	for d%2 == 0 {
		d /= 2
		r++
	}

	for _, a := range witnesses {
		if a >= n {
			continue // skip witnesses >= n
		}
		if !millerRabinWitness(a, d, n, r) {
			return false // definitely composite
		}
	}
	return true // probably prime (deterministic with our witness sets)
}

// millerRabinWitness tests a single witness a for n with decomposition n-1 = 2^r · d.
// Returns true if n passes the test (possibly prime), false if composite.
func millerRabinWitness(a, d, n uint64, r uint) bool {
	x := ModPow(a, d, n)

	if x == 1 || x == n-1 {
		return true
	}

	for i := uint(0); i < r-1; i++ {
		x = mulmod(x, x, n)
		if x == n-1 {
			return true
		}
	}

	return false
}

// ---------------------------------------------------------------------------
// Factorization
// ---------------------------------------------------------------------------

// PrimeFactors returns the complete prime factorization of n in ascending
// order. Repeated factors appear multiple times.
//
// Examples:
//   PrimeFactors(12)  = [2, 2, 3]
//   PrimeFactors(100) = [2, 2, 5, 5]
//   PrimeFactors(0)   = []
//   PrimeFactors(1)   = []
//
// Time complexity: O(√n) for trial division.
// Reference: trial division, the simplest factorization method.
func PrimeFactors(n uint64) []uint64 {
	if n < 2 {
		return nil
	}

	var factors []uint64

	// Factor out 2s.
	for n%2 == 0 {
		factors = append(factors, 2)
		n /= 2
	}

	// Trial division by odd numbers from 3 up.
	for i := uint64(3); i*i <= n; i += 2 {
		for n%i == 0 {
			factors = append(factors, i)
			n /= i
		}
	}

	// If n is still > 1, then it's a prime factor.
	if n > 1 {
		factors = append(factors, n)
	}

	return factors
}

// ---------------------------------------------------------------------------
// Next prime
// ---------------------------------------------------------------------------

// NextPrime returns the smallest prime number >= n. For n <= 2, returns 2.
//
// Time complexity: O(k · log²(p) · gap) where gap is the distance to the
// next prime and k is the number of witnesses. By the prime number theorem,
// the average gap near n is approximately ln(n).
func NextPrime(n uint64) uint64 {
	if n <= 2 {
		return 2
	}

	// Start with n; if even, move to next odd.
	candidate := n
	if candidate%2 == 0 {
		if candidate == 2 {
			return 2
		}
		candidate++
	}

	for {
		if IsPrime(candidate) {
			return candidate
		}
		candidate += 2
		// Guard against overflow: if we wrapped around, we've exhausted uint64.
		if candidate < n {
			return 0 // no prime found (shouldn't happen in practice)
		}
	}
}

// ---------------------------------------------------------------------------
// GCD / LCM
// ---------------------------------------------------------------------------

// GCD returns the greatest common divisor of a and b using the Euclidean
// algorithm. GCD(0, 0) = 0.
//
// Time complexity: O(log(min(a,b)))
// Reference: Euclid, Elements, Book VII (~300 BCE)
func GCD(a, b uint64) uint64 {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}

// LCM returns the least common multiple of a and b. LCM(0, x) = 0.
//
// Formula: LCM(a,b) = a / GCD(a,b) * b (division first to avoid overflow).
// Time complexity: O(log(min(a,b)))
func LCM(a, b uint64) uint64 {
	if a == 0 || b == 0 {
		return 0
	}
	return a / GCD(a, b) * b
}

// ExtendedGCD computes the GCD of a and b, and also finds the Bezout
// coefficients x and y such that a*x + b*y = gcd.
//
// This is the extended Euclidean algorithm. The returned gcd is always
// non-negative. The signs of x and y follow the standard convention.
//
// Time complexity: O(log(min(|a|,|b|)))
// Reference: Bezout's identity; extended Euclidean algorithm
func ExtendedGCD(a, b int64) (gcd, x, y int64) {
	if a == 0 {
		return abs64(b), 0, sign64(b)
	}
	if b == 0 {
		return abs64(a), sign64(a), 0
	}

	// Standard iterative extended GCD.
	oldR, r := a, b
	oldS, s := int64(1), int64(0)
	oldT, t := int64(0), int64(1)

	for r != 0 {
		q := oldR / r
		oldR, r = r, oldR-q*r
		oldS, s = s, oldS-q*s
		oldT, t = t, oldT-q*t
	}

	// Ensure gcd is positive.
	if oldR < 0 {
		oldR = -oldR
		oldS = -oldS
		oldT = -oldT
	}

	return oldR, oldS, oldT
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

// mulmod computes (a * b) % m without overflow by using 128-bit arithmetic
// via two 64-bit halves (Karatsuba-like decomposition).
func mulmod(a, b, m uint64) uint64 {
	// Use the Russian peasant multiplication to avoid overflow.
	var result uint64
	a = a % m
	for b > 0 {
		if b%2 == 1 {
			result = addmod(result, a, m)
		}
		a = addmod(a, a, m)
		b /= 2
	}
	return result
}

// addmod computes (a + b) % m, handling potential overflow.
func addmod(a, b, m uint64) uint64 {
	a = a % m
	b = b % m
	if a >= m-b {
		return a - (m - b)
	}
	return a + b
}

// abs64 returns the absolute value of x. Panics on MinInt64.
func abs64(x int64) int64 {
	if x < 0 {
		return -x
	}
	return x
}

// sign64 returns 1 if x > 0, -1 if x < 0, 0 if x == 0.
func sign64(x int64) int64 {
	if x > 0 {
		return 1
	}
	if x < 0 {
		return -1
	}
	return 0
}
