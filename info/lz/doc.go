// Package lz implements Lempel-Ziv 1976 sequence-complexity primitives
// — the model-free side of the codelength layer.  The LZ76 production
// count and its Kaspar-Schuster normalisation against the random-iid
// upper bound n / log_A(n) give a parameter-free quantitative answer
// to "how compressible is this sequence?" — low complexity indicates
// exploitable structure (periodic / structured); high complexity
// indicates randomness (no model class compresses noticeably better
// than the literal sequence).
//
// # Why this exists in Reality
//
// Per the Ecosystem Revolution Hunt 2026-04-28 entry L12 (lens
// `minimum-description-length`, verdict NOTABLE), LZ76 ships in
// production already in exactly one flagship — RubberDuck's
// 304-LoC `KolmogorovComplexity.cs` + 245-LoC test suite — but the
// math passed code-review without ever wiring up to the
// `DescriptionLengthDelta = 18` enum value at
// `RubberDuck.Core/Session37/QuantModuleDistance.cs:104,190`, which
// was forward-declared and never filled in.  The Kolmogorov code
// exists; the description-length-delta consumer is a self-orphaned
// integration point.
//
// This is the third RubberDuck math-shipped-without-wiring entry in
// the hunt (alongside L09 PersistentHomology 561 LoC and L11
// OptimalTransport 180 LoC); the `info/lz` package lifts the
// production C# implementation to Go with R80b cross-substrate
// precision parity, freeing the math from the flagship and making
// it available to the broader ecosystem.
//
// # Inverse-consumers
//
// The hunt identified seven cross-substrate sites where an MDL /
// codelength primitive would subsume an ad-hoc AIC/BIC scalar or
// hard-coded bin count:
//
//   - RubberDuck `KolmogorovComplexity.cs` — the orphaned source of
//     truth this package ports byte-for-byte.
//   - RubberDuck `GrangerCausality.OptimalLag` — hand-rolled BIC =
//     T*ln(RSS/T) + numParams*ln(T) (Hansen-Yu 2001).
//   - RubberDuck `GevMaxDrawdown.AIC` — 6.0 - 2.0 * ll for 3-param
//     GEV; AIC fails on the xi-boundary case (Grünwald 2007 §11).
//   - RubberDuck `CointegrationTests` — comment "simplified BIC".
//   - Simulacra octave `calibration.m:171-180` — AIC + BIC reported
//     to humans for "how many parameters should the twin have?"
//   - Causal-engine GES `discovery_ges.py:35` — `local_score_BIC`
//     (Bayesian-network structure learning).  fNML is the prior-art
//     answer (Roos-Silander-Kontkanen-Myllymäki 2008).
//   - Sensorhub `histogram.ex:20` — `@default_num_bins 50`.
//   - Nexus oracle `calibration.go:167` — "10 bins of width 0.1".
//
// # API surface
//
//   - LempelZivComplexity(symbols, alphabetHint) — LZ76 exhaustive
//     parsing.  Returns LzComplexityResult { WordCount,
//     NormalizedComplexity, SequenceLength, AlphabetSize,
//     Interpretation }.
//   - SymbolizeByQuantile(returns, numBins) — rank-based binning
//     into [0, numBins-1].
//   - SymbolizeByThreshold(returns, sigmaThreshold) — sigma-multiple
//     binning into {0, 1, 2}.
//   - ComplexityFromReturns(returns, numBins) — convenience
//     SymbolizeByQuantile -> LempelZivComplexity pipeline.
//   - RollingComplexity(returns, windowSize, stepSize, numBins) —
//     regime-detection rolling-window LZ76.
//
// # Cross-substrate output parity (R80b)
//
// TestCrossSubstratePrecision_RubberDuck_* in lz76_test.go replicates
// the named fixtures from
// `flagships/rubberduck/tests/RubberDuck.Core.Tests/Analysis/
// KolmogorovComplexityTests.cs`: ConstantSequence, PeriodicSequence,
// RandomSequence, KnownSequence, TooShort, Empty,
// SingleEffectiveSymbol, QuantileSorted, QuantileAllIdentical,
// ThresholdClear, FromReturnsManualPipeline, FromReturnsTooShort,
// FromReturnsFewNaN, FromReturnsManyNaN, RollingRegimeChange,
// RollingWindowTooSmall.  Manual-vs-convenience pipeline parity to
// ≤1e-12.
//
// # Phase A scope and v2 deferrals
//
// v1 is the cheapest cut of the LZ76 / Kolmogorov-complexity
// surface RubberDuck already ships:
//
//   - LZ76 exhaustive parsing.
//   - Quantile + threshold symbolisation.
//   - Rolling-window complexity for regime detection.
//
// Deferred to v2 per Pre-Mortem 007 founder-time discipline:
//
//   - Normalised Compression Distance (Cilibrasi-Vitányi 2005;
//     d(x, y) = (C(xy) - min(C(x), C(y))) / max(C(x), C(y))) — the
//     gzip + k-NN classifier baseline (Jiang-Yang 2023 ACL).
//   - Block Decomposition Method (Zenil et al. 2018) — algorithmic-
//     complexity for short strings via approximate-CTM lookup.
//   - Context Tree Weighting (Willems-Shtarkov-Tjalkens 1995) — the
//     universal predictor on tree-shaped sources.  See `info/mdl`
//     for the parametric-NML / AIC-BIC adapter side of the same
//     codelength-as-evaluator story.
//   - Solomonoff prior M.
//
// # Determinism and allocations
//
// Pure int / float64, deterministic given a fixed iteration order.
// Zero non-stdlib deps (only `math` + `errors`).  Allocations bounded
// by O(n) for the symbolisation paths and O(n log n) for the merge-
// sort in SymbolizeByQuantile; the LZ76 parsing path itself is
// O(n^2) time but allocation-free past the input slice.
//
// # References
//
//   - Lempel, A. & Ziv, J. (1976).  On the complexity of finite
//     sequences.  IEEE Trans. Inform. Theory 22(1): 75-81.
//   - Kaspar, F. & Schuster, H. G. (1987).  Easily calculable measure
//     for the complexity of spatiotemporal patterns.  Phys. Rev. A
//     36(2): 842-848.
//   - Cilibrasi, R. & Vitányi, P. M. B. (2005).  Clustering by
//     compression.  IEEE Trans. Inform. Theory 51(4): 1523-1545.
//   - RubberDuck reference impl: `flagships/rubberduck/RubberDuck.Core/
//     Analysis/KolmogorovComplexity.cs:1-304`.
//
// # Namespace boundary
//
// `reality/info/lz/` (this package) ships LZ76 and downstream
// algorithmic-information primitives.  `reality/info/mdl/` (sibling
// package) ships codelength + NML primitives for parametric model
// selection.  `reality/info/{capacity, fisher, ...}/` (L10 packages,
// to land separately) ship Fano / CRLB / DPI capacity bounds.  The
// three subpackages are co-located under `reality/info/` but disjoint
// in scope.
//
// Package layout:
//
//   - lz76.go     — LempelZivComplexity + symbolisation helpers.
//   - errors.go   — sentinel errors.
//   - doc.go      — this file.
package lz
