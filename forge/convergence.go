// Package forge holds the canonical three-way convergence decision shared
// across the limitless ecosystem (the "Type 1" forge verdict scheme).
//
// This is the single in-`reality` home for the canonical Decide() that ~56
// flagship repos currently carry as a drifted private copy in
// internal/forge/convergence.go. The thresholds here are pinned to the
// basis-point constants baked into reality/forge/session40 via the
// const-parity test (R61), giving ONE source of truth for the 0.65 floor and
// the min-observation count.
//
// Note: this Verdict (int, {Uncertain|Converged|Escape}) is intentionally
// distinct from prob.Verdict (string, Wilson {dominates|uncertain|dominated})
// — they are different decision schemes that coexist in the reality module.
package forge

import "math"

// Canonical thresholds (ecosystem Type 1), byte-identical to the ecosystem
// fork canonical (flagships/argus/internal/forge/convergence.go).
const (
	// ConvergedDominance is the canonical dominance floor above which a
	// pattern is considered Converged. Ecosystem Type 1 = 0.70.
	ConvergedDominance = 0.70

	// ConvergedConfidence is the canonical confidence floor required to
	// emit a converged verdict. Ecosystem Type 1 = 0.65.
	ConvergedConfidence = 0.65

	// EscapeThreshold is the canonical escape-to-UNCERTAIN floor: below this
	// dominance, the pattern escapes rather than converging.
	// Ecosystem Type 1 = 0.60.
	EscapeThreshold = 0.60

	// MinObservations is the minimum observation count before a pattern can
	// emit any verdict (either Converged or Escape).
	MinObservations = 3
)

// Verdict enumerates the canonical three-way forge decision.
type Verdict int

const (
	VerdictUncertain Verdict = iota
	VerdictConverged
	VerdictEscape
)

// String returns a canonical name for the Verdict.
func (v Verdict) String() string {
	switch v {
	case VerdictUncertain:
		return "uncertain"
	case VerdictConverged:
		return "converged"
	case VerdictEscape:
		return "escape"
	}
	return "unknown"
}

// Decide returns the canonical three-way verdict given dominance, confidence,
// and observation count.
//
// The decision rule follows ecosystem-wide canonical semantics:
//   - If dominance or confidence is non-finite (NaN/±Inf): Uncertain (fail
//     CLOSED — a poisoned/non-finite input must never converge)
//   - If total < MinObservations: Uncertain (insufficient data)
//   - If dominance >= ConvergedDominance AND confidence >= ConvergedConfidence: Converged
//   - If dominance < EscapeThreshold: Escape
//   - Otherwise: Uncertain
func Decide(dominance, confidence float64, total int) Verdict {
	// C1 fail-safe: a non-finite ingress (NaN or ±Inf) must fail CLOSED.
	// Without this guard +Inf >= ConvergedDominance && +Inf >= ConvergedConfidence
	// evaluates true and emits a fail-OPEN Converged verdict on poisoned input.
	// This is strictly more conservative and does not affect any finite input.
	if math.IsNaN(dominance) || math.IsInf(dominance, 0) ||
		math.IsNaN(confidence) || math.IsInf(confidence, 0) {
		return VerdictUncertain
	}
	if total < MinObservations {
		return VerdictUncertain
	}
	if dominance >= ConvergedDominance && confidence >= ConvergedConfidence {
		return VerdictConverged
	}
	if dominance < EscapeThreshold {
		return VerdictEscape
	}
	return VerdictUncertain
}
