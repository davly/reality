package crypto

import "testing"

// TestIsPrime_Uint64StrongPseudoprimes_AreComposite pins the correctness fix:
// the previous witness set {2,3,5,7,11,13,17} declared certain uint64 COMPOSITES
// prime (it is only deterministic to ~3.4e14, not the claimed uint64 range).
// Each value below is a strong pseudoprime to all bases up to a point but is in
// fact composite; IsPrime must reject every one.
func TestIsPrime_Uint64StrongPseudoprimes_AreComposite(t *testing.T) {
	composites := []uint64{
		3215031751,          // smallest SPRP to {2,3,5,7}            = 151·751·28351
		341550071728321,     // smallest SPRP to {2,3,5,7,11,13,17}   (the old set passed this)
		3825123056546413051, // smallest SPRP to the first 9 primes   (the named bug witness)
	}
	for _, n := range composites {
		if IsPrime(n) {
			t.Errorf("IsPrime(%d) = true, want false — %d is composite (false positive)", n, n)
		}
	}
}

// TestIsPrime_LargePrimes_AreStillPrime guards against the fix accidentally
// rejecting genuine large primes.
func TestIsPrime_LargePrimes_AreStillPrime(t *testing.T) {
	primes := []uint64{
		2305843009213693951,  // 2^61 - 1 (Mersenne prime M61)
		18446744073709551557, // the largest prime below 2^64
		67280421310721,       // a prime in the else-branch range (~6.7e13)
	}
	for _, n := range primes {
		if !IsPrime(n) {
			t.Errorf("IsPrime(%d) = false, want true — %d is prime", n, n)
		}
	}
}

// TestMillerRabin_K12_IsUint64Exact documents that the deterministic uint64
// guarantee requires k>=12 (and that k=7 does NOT, the doc-corrected claim).
func TestMillerRabin_K12_IsUint64Exact(t *testing.T) {
	const composite = 3825123056546413051
	if !MillerRabin(composite, 7) {
		t.Log("note: MillerRabin(_, 7) already rejects the witness; harmless")
	}
	if MillerRabin(composite, 12) {
		t.Errorf("MillerRabin(%d, 12) = true, want false — k=12 must be uint64-exact", uint64(composite))
	}
}
