// Package hrp implements Hierarchical Risk Parity (HRP) portfolio
// construction — Reality's first hierarchical-clustering primitive — as
// four composable, first-principles stages that mirror Marcos Lopez de
// Prado's 2016 algorithm exactly:
//
//	Stage 1  CorrelationDistance   correlation -> metric distance
//	Stage 2a SingleLinkage         agglomerative nearest-neighbour tree
//	Stage 2b QuasiDiagonalize      dendrogram -> quasi-diagonal leaf order
//	Stage 3  RecursiveBisection    inverse-variance top-down allocation
//	         HRPWeights            the whole pipeline, cov + corr -> weights
//
// # Why this exists in Reality
//
// HRP builds diversified portfolios without inverting the covariance
// matrix, so it is far more stable out-of-sample than mean-variance /
// critical-line optimization when the covariance estimate is noisy
// (Lopez de Prado 2016). It is a multi-stage algorithm in which every
// stage has a silent-failure mode that still emits a plausible-looking
// weight vector: a mis-scaled distance still clusters, a nondeterministic
// tie-break still returns a tree, a mis-seriated order still bisects to
// weights that sum to one. Property-only tests (sum-to-one, positivity)
// catch none of these. The RubberDuck flagship hand-rolls the full HRP
// pipeline as a live allocation output (RubberDuck.Core/Analysis/
// HierarchicalRiskParity.cs, driven by RubberDuck.Brain.Risk's
// PortfolioOptimizationService and the HrpRotationStrategy), and its
// SingleLinkageClustering finds the closest pair by iterating a
// HashSet<int> with a strict `<` — a genuinely nondeterministic
// tie-break. This package pins the whole pipeline with a documented,
// deterministic tie-break contract and golden vectors that assert the
// intermediate dendrogram and leaf order, not just the final weights.
//
// # Clustering primitive is reusable beyond finance
//
// SingleLinkage and QuasiDiagonalize operate on plain [][]float64
// matrices (a distance matrix and a linkage, respectively) with no
// finance-specific structure. They are the general single-linkage
// agglomerative clustering + dendrogram-seriation primitives and can be
// used anywhere a nearest-neighbour hierarchy over a distance matrix is
// wanted. Only CorrelationDistance, RecursiveBisection, and HRPWeights
// carry portfolio semantics.
//
// # Cross-substrate parity with RubberDuck
//
// This package is the Go twin of RubberDuck.Core/Analysis/
// HierarchicalRiskParity.cs, following the estate's established
// twin+golden-fixture pattern (cf. optim/transport/wasserstein1d.go
// mirroring OptimalTransport.cs, and topology/persistent mirroring the
// RD persistent-homology corpus). Two intentional differences from the
// current C# code, both documented at their call sites:
//
//   - Reality's CorrelationDistance is the canonical de Prado
//     d = sqrt((1-rho)/2); RubberDuck's MatrixMath.CorrelationDistance
//     is sqrt(2(1-rho)) = 2x. Clustering, seriation and weights are
//     invariant to the positive rescaling, so only linkage-distance
//     magnitudes differ.
//   - Reality's SingleLinkage has a deterministic ascending-cluster-id
//     tie-break; RubberDuck's HashSet iteration does not. Exact parity
//     fixtures must avoid tie cases (or the C# side must adopt this
//     rule). The tie-break golden here is deliberately Reality-only.
//
// # Source
//
// Marcos Lopez de Prado, "Building Diversified Portfolios that
// Outperform Out-of-Sample", The Journal of Portfolio Management 42(4),
// 2016, pp. 59-69. Single linkage: Sibson, "SLINK", The Computer Journal
// 16(1), 1973; Lance & Williams, The Computer Journal 9(4), 1967.
//
// Zero dependencies; standard library math only.
package hrp
