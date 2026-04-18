package session40

import (
	"fmt"
	"sort"
	"strings"
	"sync"
)

// Session 40 Phase B — Canonical Divergence Registry (R74) for Reality.
//
// Cross-substrate context
//
// Horizon's Python R74 exemplar (Session 40-L L6) uses file-top comment
// headers + a source-file scanner. RubberDuck's C# exemplar uses custom
// attributes + reflection. Nexus's Go exemplar (Phase B #2) uses a
// struct-literal Register pattern from init(). Reality follows Nexus's
// Go pattern.
//
// Reality-specific note: Reality is the *authoritative source* for the
// ecosystem's canonical numeric primitives (FNV-1a, Jeffreys prior,
// verdict-convergence thresholds). A divergence registry on Reality
// itself is almost a category error — if Reality diverges from the
// "canonical" primitive, then the primitive isn't canonical anymore.
//
// For that reason Reality's registry is deliberately empty of the
// common divergence categories (timeouts, priors, thresholds). Reality
// has no network I/O, so there are no bridge / conduit / workflow
// timeouts to register. Reality has no policy-specific priors — every
// Jeffreys invocation uses Beta(0.5, 0.5) by construction.
//
// Per Session 40-L hypothesis B7.a/B8.a, infrastructure flagships
// register 0-2 divergences and domain flagships register 3+. Reality
// is MORE foundational than infrastructure — it is the math itself.
// Registry size = 0 is not a bug; it is the correct signal that
// Reality IS the canonical source.
//
// Primitive vocabulary (B1.b)
//
// The closed-set of primitive names MUST match the cross-substrate
// vocabulary so peer flagships (Horizon / RubberDuck / Nexus / etc.)
// can round-trip any future registrations via the canonical comment
// format. Even if Reality registers zero divergences today, the
// vocabulary must be kept in sync for the day a genuine divergence
// surfaces (e.g., a ring-buffer default depth differs from the
// ecosystem default — that's a plausible future entry).

// Canonical primitives recognised by R74. Any registration against a
// primitive outside this set fails the registry invariant test.
//
// Kept in sync with:
//   - RubberDuck.Innovations `CanonicalPrimitive` consts (C#)
//   - Horizon `horizon.innovations.divergence.PRIMITIVES` (Python)
//   - Nexus `session40.Primitive*` (Go infrastructure)
//   - phantom-common / limitless-browser Rust innovations divergences
const (
	PrimitiveConduitTimeout  = "conduit_timeout"
	PrimitiveBridgeTimeout   = "bridge_timeout"
	PrimitiveFnvHash         = "fnv_hash"
	PrimitiveJeffreysPrior   = "jeffreys_prior"
	PrimitiveEscapeThreshold = "escape_threshold"
	PrimitiveVerdictScheme   = "verdict_scheme"
)

// CanonicalPrimitives returns the closed set of primitive names this
// registry will accept. Tests use it to enforce the vocabulary.
func CanonicalPrimitives() []string {
	return []string{
		PrimitiveConduitTimeout,
		PrimitiveBridgeTimeout,
		PrimitiveFnvHash,
		PrimitiveJeffreysPrior,
		PrimitiveEscapeThreshold,
		PrimitiveVerdictScheme,
	}
}

// CanonicalDivergence is the record of a single deliberate Type 1
// divergence. Reality's registry is typically empty, but the type must
// exist so that future divergences (if any) register against the same
// shape as peer flagships.
type CanonicalDivergence struct {
	// Primitive is the canonical primitive being diverged from. MUST
	// be one of the PrimitiveXxx constants above.
	Primitive string

	// Variant is a human-readable description of the divergent value
	// or range.
	Variant string

	// Rationale is a one-sentence explanation of why the divergence is
	// deliberate. Must be non-empty.
	Rationale string

	// SourceFile is the Go source file where the divergence originates
	// (package-relative path). Used by audit tooling to cross-reference
	// the registration with the site.
	SourceFile string

	// Line is the 1-based line number of the divergent value. Optional.
	Line int

	// Role is a short tag classifying the divergent role.
	Role string
}

// String returns the ecosystem-canonical comment-form header so cross-
// substrate tooling that pattern-matches this format across Go / Python
// / Rust / C# sees the same wire string.
func (d CanonicalDivergence) String() string {
	var b strings.Builder
	b.WriteString("// canonical-divergence: ")
	b.WriteString(d.Primitive)
	b.WriteString(" -> ")
	b.WriteString(d.Variant)
	if d.Rationale != "" {
		b.WriteString(" (")
		b.WriteString(d.Rationale)
		b.WriteString(")")
	}
	if d.Role != "" {
		b.WriteString(" [role=")
		b.WriteString(d.Role)
		b.WriteString("]")
	}
	if d.SourceFile != "" {
		if d.Line > 0 {
			fmt.Fprintf(&b, " [@%s:%d]", d.SourceFile, d.Line)
		} else {
			fmt.Fprintf(&b, " [@%s]", d.SourceFile)
		}
	}
	return b.String()
}

// Registry guard.
var (
	registryMu sync.RWMutex
	registry   []CanonicalDivergence
)

// Register adds a divergence to the Reality registry. Duplicate identical
// registrations are idempotent. Any registration with an empty Rationale
// panics — a misfiled R74 declaration should not compile.
func Register(d CanonicalDivergence) {
	if d.Primitive == "" {
		panic("R74 registry: Primitive is empty")
	}
	if d.Rationale == "" {
		panic("R74 registry: Rationale is empty (primitive=" + d.Primitive + ")")
	}
	if d.Variant == "" {
		panic("R74 registry: Variant is empty (primitive=" + d.Primitive + ")")
	}
	registryMu.Lock()
	defer registryMu.Unlock()
	for _, existing := range registry {
		if existing == d {
			return
		}
	}
	registry = append(registry, d)
}

// Registered returns a copy of the registry sorted by primitive then
// source file. Callers get a snapshot; mutating the returned slice does
// not mutate the registry.
func Registered() []CanonicalDivergence {
	registryMu.RLock()
	defer registryMu.RUnlock()
	out := make([]CanonicalDivergence, len(registry))
	copy(out, registry)
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Primitive != out[j].Primitive {
			return out[i].Primitive < out[j].Primitive
		}
		if out[i].SourceFile != out[j].SourceFile {
			return out[i].SourceFile < out[j].SourceFile
		}
		return out[i].Line < out[j].Line
	})
	return out
}

// RegisteredByPrimitive returns a sub-slice of the registry filtered
// to a single primitive. Convenience for audit queries.
func RegisteredByPrimitive(primitive string) []CanonicalDivergence {
	all := Registered()
	out := make([]CanonicalDivergence, 0, len(all))
	for _, d := range all {
		if d.Primitive == primitive {
			out = append(out, d)
		}
	}
	return out
}

// CanonicalPrimitiveSet returns a lookup map of the canonical primitives.
func CanonicalPrimitiveSet() map[string]bool {
	prims := CanonicalPrimitives()
	out := make(map[string]bool, len(prims))
	for _, p := range prims {
		out[p] = true
	}
	return out
}

// IsCanonicalSource returns true for flagships that ARE the canonical
// source for a primitive rather than a consumer of it. Reality is the
// canonical source for fnv_hash and jeffreys_prior — divergences AGAINST
// those primitives registered by Reality would be self-contradictory.
// Downstream audit tools use this to decide whether zero registrations
// against a primitive is a bug (consumer never declared) or the norm
// (Reality is authoritative).
func IsCanonicalSource(primitive string) bool {
	switch primitive {
	case PrimitiveFnvHash, PrimitiveJeffreysPrior:
		// Reality owns reality/crypto.FNV1a64 and
		// reality/prob.JeffreysConfidence — these are the ecosystem's
		// canonical implementations. A divergence against these from
		// within Reality would be incoherent.
		return true
	default:
		return false
	}
}

// -----------------------------------------------------------------------------
// Built-in Reality registrations
// -----------------------------------------------------------------------------
//
// Reality is the pure-math foundation. It legitimately has zero entries
// at Session 40-L. If a future Reality contributor introduces a
// deliberate divergence (e.g., an alternative Jeffreys concentration
// parameter for a specialised research branch of reality/prob), the
// registration goes here.
func init() {
	// Intentionally empty. Reality's R74 registry is zero entries by
	// design — it IS the canonical source, not a consumer.
	//
	// B7.a/B8.a hypothesis: infra flagships 0-2, domain flagships 3+.
	// Reality is MORE foundational than infra: its correct answer is 0.
}
