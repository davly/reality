// Package persistent implements persistent homology of a finite
// point cloud under the Vietoris-Rips filtration: the canonical
// shape-of-data primitive of topological data analysis.
//
// Given n points in R^d and a maximum scale parameter r_max, this
// package builds the Vietoris-Rips simplicial complex up to dimension
// maxDim in {0, 1}, reduces its boundary matrix, and emits the
// persistence barcode — a list of (Birth, Death) intervals indexed
// by homological dimension where Birth/Death are scalar filtration
// times in [0, r_max].  H_0 bars track connected components; H_1
// bars track loops (1-cycles that are not boundaries of triangles).
// Bottleneck distance gives a stability-guaranteed metric between
// barcodes (Cohen-Steiner-Edelsbrunner-Harer 2007).
//
// # Why this exists in Reality
//
// Per the Ecosystem Revolution Hunt 2026-04-28 entry L09 (lens
// `topological-data-analysis`, verdict NOTABLE) and the Reality Math
// Hunt §05/§06/§08/§15 (composite=80/60/100 across four prior hunt
// entries), TDA appears as a load-bearing primitive in six day-one
// inverse-consumers but ships in only one flagship today:
//
//   - RubberDuck: 561-LoC PersistentHomology.cs with full XUnit
//     coverage (the keystone consumer this Go package mirrors).
//     Wired into AdvancedRiskAnalyzer behind feature flag
//     TopologicalPortfolioAnalysis for Gidea-Katz 2018 crash
//     detection.
//   - Workshop ecosystem topology dashboard (Reality Math Hunt §05
//     composite=80; replaces the flat 297-row dependency table with
//     a Mapper-style topological visualisation).
//   - Witness death-of-cycle bit-stable fingerprints (Reality Math
//     Hunt §06; bridges to the L39 witness-judge-with-falsifier
//     story via d_B(today, yesterday) <= ||X - Y||_inf stability).
//   - Mirror morning report (named in §05:53 Three-Consumer-Rule
//     list).
//   - Tether import-graph topology (graph.go:54-79; surfaces broken
//     layering as topological holes in the dependency graph).
//   - Insights blast-radius topology of the runtime service graph
//     (internal/topology/blast_radius.go) — same graph-PH idea
//     applied to live service dependencies.  This package's first
//     non-RubberDuck retrofit lands here.
//
// Until this package, every consumer either rolled their own
// reduction (RubberDuck did, with the comment-noted O(m^2) greedy
// matching) or shipped scalar gates that miss topological signal.
// The R80b cross-substrate-precision contract — that this Go
// package and the C# original produce the same barcode for the
// same input — is enforced in barcode_test.go via the
// TestCrossSubstratePrecision_RubberDuck_* corpus replicated from
// PersistentHomologyTests.cs.
//
// # API surface (Phase A)
//
//   - VietorisRipsComplex(points, maxRadius, maxDim) — build the
//     filtration.  Returns sorted Filtration{Simplices, Times}.
//   - ComputeBarcode(filtration, maxDim) — column-reduce the
//     boundary matrix, emit []Bar { Dim, Birth, Death }.
//     math.Inf(+1) Death sentinels essential classes.
//   - BottleneckDistance(d1, d2, dim) — stable metric between
//     persistence diagrams.  Returns 0 for identical diagrams,
//     +Inf when essential-bar counts disagree.
//   - Bar.Persistence(), Bar.IsEssential() — convenience accessors.
//   - Simplex.Dim(), Simplex.Equal(other) — simplex helpers.
//
// # Phase A scope and v2 deferrals
//
// v1 supports maxDim in {0, 1} only.  Triangles (2-simplices) are
// built so that they kill H_1 classes via column reduction, but
// H_2 + barcode is deferred to v2 because the boundary-matrix
// reduction needs the (maxDim+1)-skeleton — at maxDim = 2 that is
// a O(n^4) tetrahedron count which would overflow the Phase-A
// consumer scale (n <= 50).  Persistent cohomology, persistence
// landscape (Bubenik 2015), p-Wasserstein distance for p < +Inf,
// and the Mapper algorithm are all v2 — the L09 entry §6 explicitly
// gates these on a second consumer pulling, per Pre-Mortem 007
// founder-time discipline.
//
// # Cross-substrate output parity (R80b)
//
// TestCrossSubstratePrecision_RubberDuck_* in barcode_test.go
// replicates fixtures from RubberDuck's PersistentHomologyTests.cs —
// the equidistant-4-point H_0 fixture (4 points pairwise distance 1;
// expect 3 finite H_0 bars dying at 1.0 and 1 essential H_0 bar) and
// the cyclic-4-point H_1 fixture (a square with diagonal 1.5; expect
// 1 H_1 bar born at 1.0 dying at 1.5).  Tolerance: ≤1e-9 absolute.
// R80b (output-parity, not strict-byte) is appropriate because the
// substrate differs (Go float64 vs C# double — both IEEE-754
// binary64 but with intermediate-rounding differences in transcendental
// functions, particularly sqrt in pairwise Euclidean distances).
//
// # Determinism and allocations
//
// Pure float64, deterministic given a fixed iteration order.  Zero
// non-stdlib deps (only `math`, `sort`, `errors`).  Allocation in the
// hot path is bounded by O(m^2) where m is the simplex count after
// the maxRadius cut; for Phase-A consumer scale (n <= 50, m <= ~21k)
// this is a few hundred KB of working memory.
//
// # References
//
//   - Edelsbrunner, H., Letscher, D. & Zomorodian, A. (2000).
//     Topological persistence and simplification.
//     Proc. 41st FOCS: 454-463.
//   - Carlsson, G. (2009).  Topology and Data.  Bull. AMS 46: 255-308.
//   - Cohen-Steiner, D., Edelsbrunner, H. & Harer, J. (2007).
//     Stability of persistence diagrams.
//     Discrete & Computational Geometry 37: 103-120.
//   - Edelsbrunner, H. & Harer, J. (2010).  Computational Topology:
//     An Introduction.  AMS.
//   - Mantegna, R. N. (1999).  Hierarchical structure in financial
//     markets.  European Phys. J. B 11: 193-197.
//   - Gidea, M. & Katz, Y. (2018).  Topological data analysis of
//     financial time series: landscapes of crashes.  Physica A 491:
//     820-834.
//   - RubberDuck reference impl: `flagships/rubberduck/RubberDuck.Core/
//     Analysis/PersistentHomology.cs:1-561`.
//
// Package layout:
//
//   - vr.go         — VietorisRipsComplex + Filtration type.
//   - barcode.go    — ComputeBarcode + matrix reduction.
//   - bottleneck.go — BottleneckDistance + Hopcroft-Karp matching.
//   - errors.go     — sentinel errors.
//   - doc.go        — this file.
package persistent
