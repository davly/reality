// Package canonical declares reality as a foundation-tier canonical source
// of truth for ecosystem primitives per ECOSYSTEM_QUALITY_STANDARD.md R85.
//
// R85 (IsCanonicalSource marker) contract:
//  1. IsCanonicalSource = true.
//  2. CanonicalPrimitives() lists the primitives for which this flagship
//     owns the canonical value.
//  3. The flagship holds ZERO R74 CanonicalDivergence registry entries —
//     a canonical source cannot diverge from itself.
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
