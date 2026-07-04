// Package trust implements the closed-form belief calculus that trust
// aggregation needs but almost never has: an explicit UNCERTAINTY mass and
// an explicit CONFLICT mass. It provides Jøsang's subjective-logic binomial
// opinions with their canonical fusion operators, and Dempster-Shafer
// evidence combination that returns the conflict coefficient K as a
// first-class output rather than silently normalising it away.
//
// # Why this exists in Reality
//
// Two trust-aggregation primitives in the Limitless estate share the same
// epistemology hole — they carry no uncertainty dimension and emit no
// conflict signal:
//
//   - infrastructure/hive-forge-credibility (credibility.go:51-53) returns
//     credibility 1.0 (perfect trust) for fewer than 2 signals, so
//     no-evidence maps to MAX trust — the exact "vacuous verification gate"
//     class the estate's own trust-detectors suite polices — and its
//     StdDevScorer ignores observation counts entirely, so two engines with
//     one observation each score agreement 1.0.
//   - infrastructure/corroboration (corroboration.go:55-57) counts only
//     agreeing sources: dissenting verdicts "are recorded but do not
//     contribute", so a 3-agree / 30-dissent split mints "corroborated" with
//     zero conflict signal.
//
// Both defects are the same MISSING ALGEBRAIC DIMENSION. Subjective logic is
// the closed-form, zero-dependency formalism purpose-built for it: an opinion
// (b,d,u,a) reserves an uncertainty mass u that only shrinks as real evidence
// accumulates, so no-evidence yields u=1 (total uncertainty) instead of b=1
// (total belief). Dempster's rule makes the dissent that corroboration
// discards visible as the conflict mass K.
//
// This package does NOT wire either consumer — per Reality's golden-file
// cross-language model (CONTEXT.md), Go is the canonical reference that any
// consumer in any language validates its own arithmetic against. The
// consumers adopt it through their own interfaces (the credibility.go
// CredibilityScorer interface was designed for exactly this scorer swap).
//
// # API surface
//
// Subjective logic (binomial opinions over a binary frame {x, ¬x}):
//
//   - Opinion{B, D, U, A} — belief, disbelief, uncertainty, base rate, with
//     the additivity law B+D+U = 1.
//   - OpinionFromEvidence(r, s, a) — the canonical Beta evidence mapping
//     b=r/(r+s+2), d=s/(r+s+2), u=2/(r+s+2): positive/negative evidence
//     counts become an opinion whose uncertainty vanishes only in the
//     evidence limit.
//   - Opinion.ProbabilityProjection() = b + a·u — the single expected
//     probability the opinion projects to.
//   - CumulativeFusion(a, b) — aleatory (independent-evidence) fusion; the
//     belief-calculus equivalent of ADDING the two evidence tallies, so it
//     rewards agreement backed by more observation.
//   - AveragingFusion(a, b) — epistemic (dependent-source) fusion; averages
//     rather than accumulates evidence.
//   - Opinion.Discount(trust) — transitive trust discounting: weaken an
//     opinion by an opinion held ABOUT its source (quality-as-opinion).
//
// Dempster-Shafer evidence theory (arbitrary discrete frame):
//
//   - MassFunction — a basic probability assignment over subsets of a frame
//     of N elements (subsets encoded as bitmasks), with the empty set
//     pinned to zero mass.
//   - DempsterCombine(m1, m2) — Dempster's rule of combination, returning the
//     combined mass AND the conflict coefficient K explicitly; errors on
//     total conflict (K=1) rather than dividing by zero.
//   - YagerCombine(m1, m2) — Yager's (1987) alternative that assigns the
//     conflict mass K to the whole frame (ignorance) instead of normalising
//     it away — the honest-degradation counterpart to Dempster.
//   - MassFunction.Belief(A) / .Plausibility(A) — the lower/upper probability
//     bounds [Bel, Pl] a mass function places on a hypothesis.
//   - Opinion.ToBinaryMass() / OpinionFromBinaryMass(m) — the exact bridge
//     between a binomial opinion and a mass function on a 2-element frame.
//
// All functions are pure, deterministic, allocation-light closed-form
// computations over float64 masses and integer evidence counts — no I/O,
// zero non-stdlib dependencies (only "errors", "math", "sort").
//
// # Precision
//
// Every quantity is an elementary arithmetic combination of the inputs, so
// results are exact to floating-point rounding. Evidence-mapping and
// projection vectors are exact rational fractions (test tolerance 1e-12);
// fusion and Dempster/Yager outputs accumulate a handful of multiplies and
// one division (tolerance 1e-9). Additivity (B+D+U=1, Σm=1) is preserved to
// rounding and re-asserted by the tests.
//
// # Golden-file provenance
//
// Every function's test reproduces a PUBLISHED worked example, not a value
// only this package has produced:
//
//   - OpinionFromEvidence / ProbabilityProjection: Jøsang, Subjective Logic
//     (Springer 2016), §3.3 the Beta-to-opinion mapping and §3.2.3 the
//     projected-probability operator, with hand-derived exact fractions.
//   - CumulativeFusion: Jøsang §12.3 — the operator's defining identity that
//     fusing OpinionFromEvidence(r1,s1) with OpinionFromEvidence(r2,s2)
//     equals OpinionFromEvidence(r1+r2, s1+s2) (cumulative fusion IS evidence
//     addition), reproduced numerically.
//   - DempsterCombine: the classic ZADEH (1984) counterexample —
//     meningitis/concussion/tumour, two doctors — where Dempster's rule
//     renormalises 0.9999 conflict away to mint certainty in the rare
//     hypothesis (m(Tumour)=1.0); the test asserts both that headline and the
//     exposed K=0.9999 that makes the result auditable.
//   - YagerCombine: the same Zadeh inputs, where the conflict is preserved on
//     the frame (Bel(Tumour)=0.0001, Pl(Tumour)=1.0) instead of vanishing.
//
// # References
//
//   - Jøsang, A. (2016). Subjective Logic: A Formalism for Reasoning Under
//     Uncertainty. Springer. (Opinions §3, cumulative fusion §12.3,
//     discounting/trust transitivity §14.)
//   - Dempster, A. P. (1968). A generalization of Bayesian inference.
//     Journal of the Royal Statistical Society, Series B 30(2): 205-247.
//   - Shafer, G. (1976). A Mathematical Theory of Evidence. Princeton.
//   - Zadeh, L. A. (1984). Review of Shafer's "A Mathematical Theory of
//     Evidence". AI Magazine 5(3): 81-83. (The meningitis/concussion/tumour
//     conflict counterexample.)
//   - Yager, R. R. (1987). On the Dempster-Shafer framework and new
//     combination rules. Information Sciences 41(2): 93-137.
//   - Sentz, K. & Ferson, S. (2002). Combination of Evidence in
//     Dempster-Shafer Theory. Sandia SAND2002-0835.
package trust
