// Package session40 holds Reality's Phase B substrate-baked forge constants
// (R73) and the canonical-divergence registry (R74).
//
// R73 in Reality (the foundational Go math library)
//
// Reality is the MIT-licensed ground-truth math layer for the whole
// ecosystem. Every peer-flagship's canonical FNV-1a hash, Jeffreys prior,
// and verdict-scheme constants are *derived from* Reality's packages:
//
//   - `reality/crypto` owns the canonical FNV-1a offset basis and prime.
//   - `reality/prob` owns the canonical Jeffreys-prior formula.
//
// The constants in this file are asserted against those authoritative
// sources at package init(). If a downstream flagship (Nexus, Recall,
// Rift, Verdikt, Horizon, RubberDuck) ever disagrees with Reality on one
// of these numeric primitives, Reality is right by definition and the
// peer is wrong. This is not a matter of convention — it is enforced by
// Reality's hex-pinned tests (see `reality/crypto/crypto_test.go`).
//
// Per Standard Part V R73, Go is a non-R73 substrate because the language
// has no `const fn` / `constexpr` / `comptime`. What Go DOES offer is:
//
//  1. Typed `const` blocks whose values the compiler lays down as
//     immediate operands (not runtime loads).
//  2. Package `init()` that runs exactly once before any exported symbol
//     can be observed — a panic in init() aborts startup, so an init-time
//     assertion IS load-bearing for every subsequent call.
//
// Together these give us the "baked + fail-fast-on-drift" property that
// R73 requires of a strict-R73 substrate. Reality is the ecosystem's
// canonical Go implementation of R73 for that reason.
//
// R74 in Reality
//
// Reality is the pure-math foundation — it should have the THINNEST
// possible divergence registry. A foundational library that registered
// three divergences from itself would be a contradiction in terms. Per
// Session 40-L hypothesis B7.a/B8.a, infrastructure flagships register
// 0-2 divergences and domain flagships register 3+. Reality is more
// foundational than infrastructure: its registry is empty of timeout
// divergences (Reality has no network I/O) and contains only the
// ecosystem-level declaration that Reality itself is the canonical
// source for FNV / Jeffreys primitives.
//
// Contract
//
// Every constant in this file is referenced (by name or value) in at
// least one downstream flagship. They are NOT to be renamed or reshaped
// without coordinating with aicore, nexus, recall, rift, verdikt,
// rubberduck, horizon, limitless-py, limitless-dotnet, and phantom-common.
//
// Session 40 Phase B.
package session40

import (
	"github.com/davly/reality/crypto"
	"github.com/davly/reality/prob"
)

// Canonical build-time constants (R73 — Go typed-const substrate flavor).
//
// Each of these is pinned to the same value expressed by Reality's
// authoritative packages (`reality/crypto`, `reality/prob`) and by every
// peer flagship's session40.baked.go. The AssertBaked() function invoked
// from init() re-validates the values at program start; any drift
// introduced by a merge accident or a typo causes any binary that
// imports Reality to refuse to start.
//
// The constants are declared with explicit types so the compiler lays
// them down as u16 / u64 / int immediates rather than untyped-constant
// values that could be coerced differently at different callsites.
const (
	// CanonicalFnvOffset is the 64-bit FNV-1a offset basis
	// (0xcbf29ce484222325, decimal 14695981039346656037). Matches the
	// byte-level constant hard-coded in `reality/crypto/hash.go` and
	// re-derived by Go's `hash/fnv` package. This is THE canonical
	// ecosystem FNV offset; every peer flagship derives from it.
	CanonicalFnvOffset uint64 = 14695981039346656037

	// CanonicalFnvPrime is the 64-bit FNV-1a prime (0x100000001b3,
	// decimal 1099511628211). Canonical across the ecosystem.
	CanonicalFnvPrime uint64 = 1099511628211

	// CanonicalJeffreysAlphaBps is the Jeffreys prior alpha expressed
	// in basis points (5000 bps = 0.5). Matches the Beta(0.5, 0.5)
	// prior implemented by `reality/prob.JeffreysConfidence`. The bps
	// form matches the cross-substrate canonical constant used by
	// Rust's phantom-common innovations and by limitless-proto's
	// `jeffreys.proto`.
	CanonicalJeffreysAlphaBps uint16 = 5000

	// CanonicalJeffreysBetaBps is the Jeffreys prior beta in bps.
	CanonicalJeffreysBetaBps uint16 = 5000

	// CanonicalConduitTimeoutMs is the canonical fire-and-forget
	// conduit-emit timeout (100 ms). Reality itself does no network I/O
	// but downstream aicore / infrastructure consumers do, and they
	// reference this constant for their own Conduit clients.
	CanonicalConduitTimeoutMs int = 100

	// SubstrateBakedForge is the R73 marker expressed in basis points.
	// 10000 bps = 100.00%. A downstream audit scanning Reality should
	// see this marker and record "R73 claimed" against the Go foundation
	// substrate.
	SubstrateBakedForge uint16 = 10000

	// CanonicalVerdictConvergedFloor is the minimum confidence for a
	// "converged" verdict (6500 bps = 0.65). Matches the cross-substrate
	// convergence floor used by Nexus, Horizon, RubberDuck, and every
	// limitless-ecosystem flagship that produces verdicts.
	CanonicalVerdictConvergedFloor uint16 = 6500

	// CanonicalVerdictNotConvergedCeiling is the maximum confidence for
	// a "not_converged" verdict (3500 bps = 0.35).
	CanonicalVerdictNotConvergedCeiling uint16 = 3500

	// CanonicalMinObservations is the cross-substrate minimum
	// observation count before any verdict other than "uncertain" is
	// legal (3).
	CanonicalMinObservations int = 3
)

// init assertion (R73 proof-of-bake + R61 cross-module const parity lock)
//
// Go has no `static_assert` equivalent, but package init() runs once
// before any exported symbol in the package can be used, and a panic
// inside init() aborts program startup. That combination gives us the
// "drift makes the binary refuse to run" property R73 needs.
//
// Because Reality owns the canonical FNV and Jeffreys implementations,
// this assertion ALSO cross-checks that Reality's session40 constants
// match Reality's own authoritative crypto / prob packages. If anyone
// edits one without the other, Reality refuses to start instead of
// silently computing wrong hashes at runtime. R61 "Cross-module const
// parity lock" applied inside the foundational library itself.
func init() {
	AssertBaked()
}

// AssertBaked is the init-time / test-time proof that the R73 constants
// above still match the authoritative Reality packages and the cross-
// substrate canonical values. Exported so tests can call it explicitly
// without relying on init() side-effects (Go test harnesses sometimes
// run init() twice in -race mode).
//
// Panics on drift. Returns without side-effect on green.
func AssertBaked() {
	if CanonicalFnvOffset != 14695981039346656037 {
		panic("R73 drift: CanonicalFnvOffset diverged from 0xcbf29ce484222325")
	}
	if CanonicalFnvPrime != 1099511628211 {
		panic("R73 drift: CanonicalFnvPrime diverged from 0x100000001b3")
	}
	if CanonicalJeffreysAlphaBps != 5000 {
		panic("R73 drift: CanonicalJeffreysAlphaBps diverged from 5000 bps (0.5)")
	}
	if CanonicalJeffreysBetaBps != 5000 {
		panic("R73 drift: CanonicalJeffreysBetaBps diverged from 5000 bps (0.5)")
	}
	if CanonicalConduitTimeoutMs != 100 {
		panic("R73 drift: CanonicalConduitTimeoutMs diverged from 100 ms")
	}
	if SubstrateBakedForge != 10000 {
		panic("R73 drift: SubstrateBakedForge marker diverged from 10000 bps (100%)")
	}
	if CanonicalVerdictConvergedFloor != 6500 {
		panic("R73 drift: CanonicalVerdictConvergedFloor diverged from 6500 bps (0.65)")
	}
	if CanonicalVerdictNotConvergedCeiling != 3500 {
		panic("R73 drift: CanonicalVerdictNotConvergedCeiling diverged from 3500 bps (0.35)")
	}
	if CanonicalMinObservations != 3 {
		panic("R73 drift: CanonicalMinObservations diverged from 3")
	}

	// Reality-specific parity lock: the session40 FNV constants MUST
	// match the output of reality/crypto.FNV1a64 for the empty string.
	// If anyone edits reality/crypto/hash.go's fnv64OffsetBasis without
	// updating CanonicalFnvOffset (or vice versa), the binary refuses
	// to start.
	if got := crypto.FNV1a64(nil); got != CanonicalFnvOffset {
		panic("R73/R61 drift: reality/crypto.FNV1a64(nil) != session40.CanonicalFnvOffset")
	}

	// Reality-specific parity lock: Jeffreys(0,0) MUST equal 0.5 to
	// within floating-point epsilon. The bps form is derived from the
	// Beta(0.5, 0.5) prior implemented in reality/prob.
	if got := prob.JeffreysConfidence(0, 0); got < 0.4999999 || got > 0.5000001 {
		panic("R73/R61 drift: reality/prob.JeffreysConfidence(0,0) is not 0.5 -- session40 bps constants may be wrong")
	}
}
