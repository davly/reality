package session40

import (
	"testing"

	"github.com/davly/reality/crypto"
	"github.com/davly/reality/prob"
)

// TestAssertBakedGreen is the explicit entry-point for the init-time guard.
// It re-runs AssertBaked() from the test harness to document in test output
// that the R73 proof-of-bake ran and did not panic.
func TestAssertBakedGreen(t *testing.T) {
	// If any constant has drifted this call panics; the test harness
	// surfaces the panic as a failed test. This re-asserts the same
	// property init() already asserted at package load, but makes it
	// visible in test output (init() side effects are silent on green).
	AssertBaked()
}

// TestBakedConstants_TypeWidth verifies the compile-time declared types
// are what we think they are. Goes beyond simple value check: a future
// drive-by edit that widened `CanonicalFnvOffset` from uint64 to uint
// (platform-sized) would silently change the ABI at cross-language
// integration points. Pin the types explicitly.
func TestBakedConstants_TypeWidth(t *testing.T) {
	// Explicit typed-const assignments to same-name local variables
	// exercise the compiler's type checker — if any constant is not
	// assignable to the named type, this fails at compile time, not at
	// test time.
	var off uint64 = CanonicalFnvOffset
	var prime uint64 = CanonicalFnvPrime
	var alphaBps uint16 = CanonicalJeffreysAlphaBps
	var betaBps uint16 = CanonicalJeffreysBetaBps
	var timeoutMs int = CanonicalConduitTimeoutMs
	var marker uint16 = SubstrateBakedForge
	var floorBps uint16 = CanonicalVerdictConvergedFloor
	var ceilingBps uint16 = CanonicalVerdictNotConvergedCeiling
	var minObs int = CanonicalMinObservations

	if off == 0 || prime == 0 {
		t.Fatal("FNV constants must be non-zero")
	}
	if alphaBps == 0 || betaBps == 0 {
		t.Fatal("Jeffreys prior constants must be non-zero")
	}
	if timeoutMs <= 0 {
		t.Fatal("Conduit timeout must be positive")
	}
	if marker != 10000 {
		t.Fatalf("R73 marker should be 10000 bps; got %d", marker)
	}
	if floorBps <= ceilingBps {
		t.Fatalf("verdict floor (%d bps) must exceed ceiling (%d bps)", floorBps, ceilingBps)
	}
	if minObs < 3 {
		t.Fatalf("canonical min observations must be >= 3; got %d", minObs)
	}
}

// TestBakedConstants_MatchRealityCrypto is the R61 "cross-module const
// parity lock" enforced at test time against Reality's authoritative
// crypto package. Reality owns the canonical FNV-1a implementation; if
// anyone edits reality/crypto.FNV1a64 in a way that changes the empty-
// string output, Reality is internally inconsistent. Fail the test so
// the mismatch surfaces immediately.
//
// This is the Go analog of the Rust pattern in L5.a (limitless-browser):
// cross-module const assertion catching drift between two independently
// declared canonical copies.
func TestBakedConstants_MatchRealityCrypto(t *testing.T) {
	// FNV-1a of the empty byte slice is exactly the offset basis.
	if got := crypto.FNV1a64(nil); got != CanonicalFnvOffset {
		t.Fatalf("R73/R61 parity drift: reality/crypto.FNV1a64(nil) = %d; session40.CanonicalFnvOffset = %d",
			got, CanonicalFnvOffset)
	}
	if got := crypto.FNV1a64([]byte{}); got != CanonicalFnvOffset {
		t.Fatalf("R73/R61 parity drift: reality/crypto.FNV1a64([]byte{}) = %d; session40.CanonicalFnvOffset = %d",
			got, CanonicalFnvOffset)
	}
}

// TestBakedConstants_MatchRealityProb is the R61 parity lock against
// Reality's authoritative probability package. Jeffreys(0,0) MUST
// equal 0.5 (the Beta(0.5, 0.5) prior mean at no observations), which
// is the exact value encoded by CanonicalJeffreysAlphaBps /
// CanonicalJeffreysBetaBps both being 5000 bps.
func TestBakedConstants_MatchRealityProb(t *testing.T) {
	// Jeffreys at no observations should return the prior mean = 0.5.
	got := prob.JeffreysConfidence(0, 0)
	want := float64(CanonicalJeffreysAlphaBps) / 10000.0
	const eps = 1e-12
	if got < want-eps || got > want+eps {
		t.Fatalf("R73/R61 parity drift: reality/prob.JeffreysConfidence(0,0) = %v; session40 bps-derived = %v",
			got, want)
	}

	// Symmetric check: (1 success, 1 failure) with Beta(0.5, 0.5)
	// prior gives posterior Beta(1.5, 1.5) whose mean is 0.5.
	got = prob.JeffreysConfidence(1, 1)
	if got < 0.4999 || got > 0.5001 {
		t.Fatalf("prob.JeffreysConfidence(1,1) = %v; expected ~0.5 for symmetric observations with symmetric prior",
			got)
	}
}

// TestBakedConstants_HexEncodedForm documents the hex forms of the two
// canonical FNV-1a constants. Cross-substrate concordance tests in peer
// flagships (Rust limitless-browser, C# RubberDuck, Python Horizon) use
// the same hex strings. The test fails if the decimal constants no
// longer correspond to the canonical FNV-1a hex values.
func TestBakedConstants_HexEncodedForm(t *testing.T) {
	const wantOffsetHex uint64 = 0xcbf29ce484222325
	const wantPrimeHex uint64 = 0x100000001b3
	if CanonicalFnvOffset != wantOffsetHex {
		t.Fatalf("FNV offset (decimal %d) does not equal hex 0xcbf29ce484222325 (=%d)",
			CanonicalFnvOffset, wantOffsetHex)
	}
	if CanonicalFnvPrime != wantPrimeHex {
		t.Fatalf("FNV prime (decimal %d) does not equal hex 0x100000001b3 (=%d)",
			CanonicalFnvPrime, wantPrimeHex)
	}
}

// TestBakedConstants_JeffreysBpsRoundTrip verifies the bps form of the
// Jeffreys prior matches the float form. This is the `jeffreys_prior`
// entry in R74's registry-primitive vocabulary — if the bps scaling
// factor ever changes (it shouldn't; 10000 bps per unit is canonical)
// this test catches the silent re-scale.
func TestBakedConstants_JeffreysBpsRoundTrip(t *testing.T) {
	const alphaFloat float64 = 0.5
	const betaFloat float64 = 0.5
	alphaScaled := alphaFloat*10000.0 + 0.5
	betaScaled := betaFloat*10000.0 + 0.5
	wantAlphaBps := uint16(alphaScaled)
	wantBetaBps := uint16(betaScaled)
	if CanonicalJeffreysAlphaBps != wantAlphaBps {
		t.Fatalf("Jeffreys alpha bps %d != canonical %f * 10000 = %d",
			CanonicalJeffreysAlphaBps, alphaFloat, wantAlphaBps)
	}
	if CanonicalJeffreysBetaBps != wantBetaBps {
		t.Fatalf("Jeffreys beta bps %d != canonical %f * 10000 = %d",
			CanonicalJeffreysBetaBps, betaFloat, wantBetaBps)
	}
}

// TestBakedConstants_RealityIsCanonicalSource is a Reality-specific test
// asserting that Reality's FNV output for a few fixed inputs is itself
// the canonical cross-substrate value. Downstream peer-flagship tests
// reference these same fixed values; if Reality's output changed, every
// peer would fail.
func TestBakedConstants_RealityIsCanonicalSource(t *testing.T) {
	cases := []struct {
		input []byte
		want  uint64
	}{
		// Empty -> offset basis.
		{[]byte(""), 14695981039346656037},
		// Canonical single-byte test vector across the ecosystem.
		// FNV-1a("a") = 0xaf63dc4c8601ec8c.
		{[]byte("a"), 0xaf63dc4c8601ec8c},
	}
	for _, c := range cases {
		got := crypto.FNV1a64(c.input)
		if got != c.want {
			t.Errorf("reality/crypto.FNV1a64(%q) = %x (decimal %d); canonical want %x (decimal %d)",
				c.input, got, got, c.want, c.want)
		}
	}
}
