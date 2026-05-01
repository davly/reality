// Package transport implements optimal-transport primitives over
// real-valued distributions: closed-form Wasserstein-1D distance with
// IQR normalisation, log-domain entropic-regularised Sinkhorn for the
// general n-dimensional case, and pairwise / min-pairwise utilities
// for diversity and regime-similarity tasks.
//
// # Why this exists in Reality
//
// Per the Ecosystem Revolution Hunt 2026-04-28 entry L11 (lens
// `optimal-transport`, verdict NOTABLE), OT is the canonical
// geometry-aware metric on probability distributions: unlike KL
// (which goes infinite on disjoint supports) or cosine (which
// ignores ground-cost between dimensions), Wasserstein-p respects
// the metric on the underlying space — zone 3a vs zone 3b is closer
// than zone 1 vs zone 3a, the way a regulator expects.
//
// Six day-one inverse-consumers across three substrates ship
// scalar-only or Euclidean-only drift gates today:
//
//   - RubberDuck self — `OptimalTransport.cs` ships 180 LoC of
//     production-grade Wasserstein-1D + 218-LoC test suite, but the
//     four design-doc consumers (`RegimeContextService`,
//     `CorrelationService`, `EvolutionOrchestratorService`,
//     `PortfolioRebalancer`) never wired up.  The math passed code-
//     review; only the wiring failed.  This package's R63 cross-
//     substrate-precision contract is verified against that 218-LoC
//     C# corpus.
//
//   - FleetWorks BM25 schema-card subset — `AiDbDelveService.cs:1534`
//     greedy `Take(40)` on BM25 score; OT respects semantic ordering
//     between query terms (year/amount/date adjacency).
//
//   - Shadow-service composite-score promotion gate —
//     `infrastructure/shadow-service/pkg/shadow/shadow.go:135` hard
//     scalar 0.7 promote / 0.3 reject.  Wasserstein-1 between
//     shadow-vs-production accuracy distributions is the principled
//     replacement.
//
//   - Nexus DirectionalDrift — `infrastructure/nexus/src/api/internal/
//     innovations/directional_drift.go` represents drift as
//     (PriorWidthBps, CurrentWidthBps, Rate, Direction).  Width is a
//     point statistic; OT handles full prior-vs-current shape
//     comparison.  This package's first non-RubberDuck retrofit lands
//     here.
//
//   - Recall cosine HNSW — `infrastructure/recall/internal/cache/
//     fuzzy_matcher.go` cosine on 256-dim structural embedding;
//     sliced-Wasserstein over the embedding is the principled lift
//     for ordered categorical structure.
//
//   - Sentinel-AV ordered-tier bps — `flagships/sentinel-av/src/
//     ops.rs:50` `dominance_bps / confidence_bps / escape_score_bps`.
//     Tiers are ordinal; W_1 respects ordering in a way KL does not.
//
// # API surface
//
//   - Wasserstein1D(u, v, p)       — closed-form Wasserstein-p on 1D
//                                    empirical distributions.
//                                    p in [1, +Inf); p = 1 is the
//                                    default Earth-Mover's case.
//   - Wasserstein1DDetailed(u, v)  — W_1 + IQR-normalised distance +
//                                    sample sizes.  Mirrors
//                                    RubberDuck's
//                                    `Wasserstein1DDetailed`.
//   - IQRNormalise(samples)        — robust z-score-style
//                                    normalisation by inter-quartile
//                                    range (Tukey 1977).
//   - PairwiseWasserstein1D(d, p)  — symmetric K×K distance matrix.
//   - MinPairwiseWasserstein1D(d)  — smallest pairwise distance + the
//                                    achieving (i, j) index pair.
//   - Sinkhorn(a, b, C, eps, ...)  — log-domain entropic-regularised
//                                    OT for the general n×m case.
//                                    Returns transport plan + cost +
//                                    iteration count.
//
// # Cross-substrate output parity (R80b)
//
// Wasserstein1D matches RubberDuck's reference implementation byte-
// for-byte to ≤1e-12 on the existing 218-LoC `OptimalTransportTests.cs`
// corpus.  TestCrossSubstratePrecision_RubberDuck_* in
// transport_test.go replicates each named fixture (identical-array,
// shifted, scaled, single-element, NaN/Inf filtering, symmetry,
// triangle-inequality, non-negativity, unequal-size, all-NaN,
// detailed normalisation, pairwise symmetry, min-pairwise) and
// asserts the same numeric output.
//
// Sinkhorn is not yet shipped in RubberDuck (the C# `Wasserstein2DSinkhorn`
// design-doc spec is unwired); R80b parity for n-D OT will be added when
// a sister implementation lands.  The current parity contract is the
// closed-form Wasserstein1D path against RubberDuck's reference.
//
// # Phase A scope and v2 deferrals
//
// v1 ships the cheapest cut of OT identified by the L11 entry: 1D
// closed-form (~180 LoC C# port) + log-domain Sinkhorn (~250 LoC).
// Deferred to v2 per Pre-Mortem 007 founder-time discipline (gated
// on a second consumer pulling):
//
//   - Sinkhorn-divergence (Feydy 2019 debiased — interpolates
//     between W_2² and MMD).
//   - Unbalanced OT (Chizat-Peyré-Schmitzer-Vialard 2018 — drops
//     the equal-mass marginal constraint).
//   - Multi-marginal OT (Pass 2015 — k > 2 distributions).
//   - Gradient-of-OT (the Sinkhorn-as-loss case for ML pipelines).
//   - Sliced-Wasserstein (Kolouri-Pope-Martin-Rohde 2018 — random
//     1-D projections for high-dimensional embeddings; the Recall
//     cosine HNSW use case).
//   - Wasserstein gradient flow (JKO 1998 / Otto 2001 — proximal
//     step in W_2 metric; the L03-proximal-calculus compose path).
//   - Gromov-Wasserstein (Mémoli 2011 — distances between metric
//     spaces, not just measures).
//   - Wasserstein-DP (Cuturi-Genevay-Klein 2020 — differential
//     privacy that respects ground cost).
//
// # Determinism and allocations
//
// Pure float64, deterministic given a fixed iteration order.  Zero
// non-stdlib deps (only `math`, `sort`, `errors`).  All mutating
// helpers operate on freshly-allocated working buffers; input slices
// are never mutated.  Sinkhorn allocates O(n + m) potential vectors
// + an n×m plan matrix on success.
//
// # References
//
//   - Vaserstein, L. N. (1969).  Markov processes over denumerable
//     products of spaces, describing large systems of automata.
//     Problems Inform. Transmission 5(3): 47-52.
//   - Cuturi, M. (2013).  Sinkhorn distances: lightspeed computation
//     of optimal transportation distances.  NeurIPS 26: 2292-2300.
//   - Peyré, G. & Cuturi, M. (2019).  Computational Optimal
//     Transport.  Foundations and Trends in Machine Learning 11(5-6).
//   - Villani, C. (2009).  Optimal transport: old and new.  Springer.
//   - Tukey, J. W. (1977).  Exploratory Data Analysis.  Addison-Wesley.
//   - RubberDuck reference impl: `flagships/rubberduck/RubberDuck.Core/
//     Analysis/OptimalTransport.cs:1-180`.
//
// Package layout:
//
//   - errors.go        — sentinel errors.
//   - iqr_norm.go      — IQRNormalise + sorted-quantile helpers.
//   - wasserstein1d.go — Wasserstein1D + Wasserstein1DDetailed.
//   - sinkhorn.go      — log-domain entropic-regularised OT.
//   - pairwise.go      — PairwiseWasserstein1D + MinPairwise.
//   - doc.go           — this file.
package transport
