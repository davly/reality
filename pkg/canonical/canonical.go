// Package canonical declares reality as a foundation-tier canonical source
// of truth for ecosystem primitives per ECOSYSTEM_QUALITY_STANDARD.md R85.
//
// R85 (IsCanonicalSource marker) contract:
//  1. IsCanonicalSource = true.
//  2. CanonicalPrimitives() lists the primitives for which this flagship
//     owns the canonical value.
//  3. The flagship holds ZERO R74 CanonicalDivergence registry entries
//     for primitives in CanonicalPrimitives() — a canonical source cannot
//     diverge from itself. R74 entries for OTHER primitives are
//     legitimate domain documentation, not a self-divergence.
//
// Scope clarification (S62 audit, 2026-05-06):
//
//	The original R85 test was a strict "no CanonicalDivergence{} struct
//	literal anywhere in the repo" — even excluding the test file itself.
//	The test file forge/session40/divergence_test.go legitimately
//	constructs CanonicalDivergence values to exercise the registry's
//	API and was caught by the strict scan. The S62 audit rewrote the
//	test to (a) skip *_test.go and (b) only fail when a divergence's
//	primitive is in CanonicalPrimitives() — matching the contract's
//	semantic intent. Reality's production code has no R74 divergence
//	entries; this clarification handles the test-fixture false positive.
//
// See ECOSYSTEM_QUALITY_STANDARD.md §R85 for the full rationale + the
// Three-Consumer Rule evidence (reality + recall + aicore).
package canonical

// IsCanonicalSource declares reality as a foundation-tier canonical source
// per ECOSYSTEM_QUALITY_STANDARD.md R85. A flagship declaring this MUST
// hold zero R74 divergence entries (enforced by TestR85ZeroDivergences).
const IsCanonicalSource = true

// CanonicalPrimitives returns the primitives that reality owns the canonical
// values for. Downstream consumers' cached values MUST match these sources.
//
// Extend as new canonical primitives land in reality. Each entry should be
// a dotted name of the form <package>.<symbol> referring to an exported
// reality package symbol (e.g. "constants.Pi", "constants.SpeedOfLight").
func CanonicalPrimitives() []string {
	return []string{
		"math.pi",               // reality/constants.Pi (delegates to math.Pi)
		"physics.c",             // reality/constants.SpeedOfLight (299792458 m/s, SI 2019 exact)
		"color.srgb_to_linear",  // reality/color sRGB <-> linear transforms + golden vectors
		/* extend as needed */
	}
}
