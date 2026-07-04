package hrp

import "math"

// LinkageStep is one agglomerative merge in a single-linkage dendrogram.
// It mirrors one row of the SciPy-style linkage matrix
// [A, B, Dist, Size] used by the reference HRP implementations, but as
// a typed record rather than a float row so the cluster ids in A and B
// are exact integers (a float linkage matrix stores cluster ids as
// float64, which silently loses precision past 2^53 — never an issue
// at portfolio scale, but the typed form removes the round-trip).
//
// A and B are the ids of the two clusters merged at this step. Original
// leaves (assets) have ids 0..n-1; the cluster created at merge step s
// (0-indexed) has id n+s. Dist is the single-linkage distance at which
// the merge occurred (the minimum inter-cluster pairwise distance).
// Size is the number of original leaves in the merged cluster.
type LinkageStep struct {
	A    int
	B    int
	Dist float64
	Size int
}

// CorrelationDistance converts a correlation matrix into the canonical
// Lopez de Prado (2016) correlation-distance matrix
//
//	d(i,j) = sqrt( (1 - rho(i,j)) / 2 )
//
// which is a proper metric on the unit sphere of standardized return
// series: d = 0 for perfectly correlated assets (rho = +1), d = sqrt(1/2)
// for uncorrelated (rho = 0), and d = 1 for perfectly anti-correlated
// (rho = -1). This is the metric HRP clusters on so that assets which
// co-move are pulled adjacent in the dendrogram.
//
// Source: Marcos Lopez de Prado, "Building Diversified Portfolios that
// Outperform Out-of-Sample", Journal of Portfolio Management 42(4),
// 2016, Stage 1 (Tree Clustering), Eq. for the correlation-based
// distance d_{i,j} = sqrt( (1 - rho_{i,j}) / 2 ).
//
// Note on the RubberDuck twin: RubberDuck.Core's MatrixMath.Correlation
// Distance uses d = sqrt( 2 * (1 - rho) ), which is exactly 2x this
// value. Single-linkage clustering, quasi-diagonalization, and recursive
// bisection are all invariant to a positive rescaling of every distance,
// so the two conventions produce identical leaf orders and identical HRP
// weights; only the recorded LinkageStep.Dist magnitudes differ by the
// factor 2. Reality ships the canonical de Prado metric.
//
// rho values are clamped to [-1, 1] before the transform (a cleaned or
// shrinkage-estimated correlation matrix can carry entries a few ulps
// outside the valid range; an unclamped negative radicand would produce
// a spurious NaN). Precision: each entry is one sqrt of an exactly
// representable affine combination, so the result is correct to within
// one ulp of the true distance.
//
// ErrEmptyMatrix is returned for a nil/empty input. ErrNotSquare is
// returned if any row length differs from the number of rows.
func CorrelationDistance(corr [][]float64) ([][]float64, error) {
	n := len(corr)
	if n == 0 {
		return nil, ErrEmptyMatrix
	}
	for i := range corr {
		if len(corr[i]) != n {
			return nil, ErrNotSquare
		}
	}
	dist := make([][]float64, n)
	for i := 0; i < n; i++ {
		dist[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			rho := corr[i][j]
			if rho > 1 {
				rho = 1
			} else if rho < -1 {
				rho = -1
			}
			dist[i][j] = math.Sqrt((1 - rho) / 2)
		}
	}
	return dist, nil
}

// SingleLinkage performs agglomerative single-linkage (nearest-neighbour)
// hierarchical clustering on a symmetric distance matrix and returns the
// n-1 merge steps of the resulting dendrogram, in merge order.
//
// Single linkage defines the distance between two clusters as the
// minimum pairwise distance between their members:
//
//	D(P, Q) = min_{i in P, j in Q} d(i, j)
//
// At each of the n-1 steps the two closest active clusters are merged;
// the Lance-Williams update for single linkage sets the new cluster's
// distance to any other cluster k to min(D(P,k), D(Q,k)).
//
// Deterministic tie-break contract (this is the load-bearing part):
// when two or more candidate pairs share the minimum distance, the pair
// with the lexicographically smallest (cluster-id) tuple is chosen —
// i.e. smallest first id, and among those the smallest second id. This
// is realized by iterating active cluster ids in ascending sorted order
// with a strict-less-than replacement rule, so the first-encountered
// minimum (which is the lexicographically smallest pair) wins. Because
// newly-created clusters receive strictly increasing ids (n, n+1, ...),
// original leaves are always preferred over composite clusters on a tie,
// and lower-indexed assets over higher-indexed ones. This makes the
// dendrogram — and therefore the downstream leaf order and HRP weights —
// a pure deterministic function of the input matrix.
//
// (The RubberDuck twin iterates a HashSet<int> of active clusters, whose
// enumeration order is unspecified; its tie-break is therefore
// nondeterministic. The Reality contract fixes that. Exact cross-
// substrate parity fixtures must consequently avoid tie cases, or the C#
// side must adopt this same ascending-id rule.)
//
// Source: Lopez de Prado (2016), Stage 2 (Quasi-Diagonalization is
// Stage 2; clustering is Stage 1). Single linkage / nearest neighbour:
// Sibson, "SLINK: an optimally efficient algorithm for the single-link
// cluster method", The Computer Journal 16(1), 1973. Lance & Williams,
// "A general theory of classificatory sorting strategies", The Computer
// Journal 9(4), 1967 (the update formula).
//
// The input must be square and symmetric; only the values are read, no
// symmetry check is enforced (an asymmetric matrix is treated by its
// upper-triangular reading via the i<j iteration). ErrEmptyMatrix is
// returned for n == 0; a single-asset matrix (n == 1) returns an empty,
// non-nil slice (no merges). ErrNotSquare is returned for ragged input.
func SingleLinkage(dist [][]float64) ([]LinkageStep, error) {
	n := len(dist)
	if n == 0 {
		return nil, ErrEmptyMatrix
	}
	for i := range dist {
		if len(dist[i]) != n {
			return nil, ErrNotSquare
		}
	}
	if n == 1 {
		return []LinkageStep{}, nil
	}

	// Working distance table indexed by cluster id in [0, 2n-1). We only
	// ever read D[a][b] for active a, b, so the extra rows created by
	// merges are appended lazily as a dense (2n-1)x(2n-1) table.
	size := 2*n - 1
	D := make([][]float64, size)
	for i := 0; i < size; i++ {
		D[i] = make([]float64, size)
	}
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			D[i][j] = dist[i][j]
		}
	}

	// active holds current cluster ids, kept sorted ascending so the
	// tie-break is the lexicographically-smallest pair.
	active := make([]int, n)
	for i := 0; i < n; i++ {
		active[i] = i
	}
	memberCount := make([]int, size)
	for i := 0; i < n; i++ {
		memberCount[i] = 1
	}

	linkage := make([]LinkageStep, 0, n-1)
	for step := 0; step < n-1; step++ {
		// Find the closest active pair. active is sorted ascending, and
		// strict `<` means the first (smallest-id) minimum pair wins ties.
		best := math.Inf(1)
		ci, cj := -1, -1
		for a := 0; a < len(active); a++ {
			for b := a + 1; b < len(active); b++ {
				p, q := active[a], active[b]
				if D[p][q] < best {
					best = D[p][q]
					ci, cj = p, q
				}
			}
		}

		newID := n + step
		memberCount[newID] = memberCount[ci] + memberCount[cj]
		linkage = append(linkage, LinkageStep{A: ci, B: cj, Dist: best, Size: memberCount[newID]})

		// Lance-Williams single-linkage update against remaining clusters.
		for _, k := range active {
			if k == ci || k == cj {
				continue
			}
			nd := math.Min(D[ci][k], D[cj][k])
			D[newID][k] = nd
			D[k][newID] = nd
		}

		// Replace ci, cj by newID, keeping active sorted ascending.
		next := make([]int, 0, len(active)-1)
		for _, k := range active {
			if k == ci || k == cj {
				continue
			}
			next = append(next, k)
		}
		next = append(next, newID) // newID > every existing id, stays sorted
		active = next
	}
	return linkage, nil
}

// QuasiDiagonalize returns the leaf order (a permutation of 0..n-1)
// obtained by unrolling the single-linkage dendrogram so that assets
// belonging to the same cluster are placed adjacently. This is HRP
// Stage 2: reordering rows/columns of the covariance matrix by this
// permutation makes large values sit near the diagonal, so the recursive
// bisection splits along genuine cluster boundaries.
//
// The order is the depth-first leaf sequence of the tree whose root is
// the last-created cluster (id 2n-2). At each internal node the left
// child (LinkageStep.A) is emitted before the right child
// (LinkageStep.B), matching the convention used by the RubberDuck twin
// and by SciPy's dendrogram leaf ordering when no distance-based leaf
// rotation is applied.
//
// Source: Lopez de Prado (2016), Stage 2 (Matrix Seriation /
// Quasi-Diagonalization).
//
// n must equal the number of leaves the linkage was built from and
// linkage must have exactly n-1 steps; a mismatched (linkage, n) pair
// returns ErrLinkageMismatch. n == 0 returns an empty slice; n == 1
// returns []int{0}.
func QuasiDiagonalize(linkage []LinkageStep, n int) ([]int, error) {
	if n < 0 {
		return nil, ErrLinkageMismatch
	}
	if n == 0 {
		return []int{}, nil
	}
	if n == 1 {
		if len(linkage) != 0 {
			return nil, ErrLinkageMismatch
		}
		return []int{0}, nil
	}
	if len(linkage) != n-1 {
		return nil, ErrLinkageMismatch
	}

	order := make([]int, 0, n)
	stack := []int{2*n - 2} // root cluster id
	for len(stack) > 0 {
		node := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if node < n {
			order = append(order, node) // leaf
			continue
		}
		idx := node - n
		if idx < 0 || idx >= len(linkage) {
			return nil, ErrLinkageMismatch
		}
		// Push right then left so the left child is popped (emitted) first.
		stack = append(stack, linkage[idx].B, linkage[idx].A)
	}
	if len(order) != n {
		return nil, ErrLinkageMismatch
	}
	return order, nil
}

// RecursiveBisection performs HRP Stage 3: it walks the quasi-diagonal
// leaf order top-down, at each split allocating capital between the two
// halves in inverse proportion to their cluster variance, then returns
// the final weight vector aligned to the ORIGINAL asset indices (not the
// leaf order).
//
// At a contiguous segment [start,end) of the leaf order the midpoint
// mid = (start+end)/2 splits it into a left and right sub-cluster. Each
// sub-cluster's variance is computed under its own inverse-variance
// portfolio (the diagonal-only IVP weights w_i proportional to
// 1/sigma_i^2, normalized, then variance = w^T Cov w). The split factor
//
//	alpha = (1/leftVar) / (1/leftVar + 1/rightVar)  =  1 - leftVar/(leftVar+rightVar)
//
// scales every leaf allocation in the left half by alpha and every leaf
// in the right half by (1-alpha); recursion proceeds into each half. A
// segment of length 1 is a base case. The resulting allocations already
// multiply to a sum of 1 by construction, but a final renormalization is
// applied to absorb the variance floor (below) and any accumulated
// rounding, so the returned weights sum to exactly 1 (to within one ulp
// times n).
//
// A variance floor of 1e-10 (matching the RubberDuck twin) guards the
// 1/variance reciprocals against a zero-variance (constant-return) asset
// or degenerate cluster; without it a genuinely riskless leg would take
// infinite inverse-variance weight. This floor is a documented deviation
// from a pure-real algorithm and the only place the output is not scale-
// exact; callers with near-zero variances should be aware the floor caps
// the maximum tilt toward a near-riskless asset.
//
// Source: Lopez de Prado (2016), Stage 3 (Recursive Bisection),
// getRecBipart / getClusterVar / getIVP.
//
// cov must be square with dimension at least max(order)+1. order must be
// a permutation of a subset of valid covariance indices with no
// duplicates and every entry in [0, len(cov)). Violations return
// ErrIndexOutOfRange (out-of-bounds or duplicate index) or ErrNotSquare
// (ragged covariance). An empty order returns an empty weight slice.
// The returned weight slice has length len(cov); assets not present in
// order receive weight 0.
func RecursiveBisection(cov [][]float64, order []int) ([]float64, error) {
	m := len(cov)
	if m == 0 {
		if len(order) == 0 {
			return []float64{}, nil
		}
		return nil, ErrIndexOutOfRange
	}
	for i := range cov {
		if len(cov[i]) != m {
			return nil, ErrNotSquare
		}
	}
	seen := make([]bool, m)
	for _, idx := range order {
		if idx < 0 || idx >= m {
			return nil, ErrIndexOutOfRange
		}
		if seen[idx] {
			return nil, ErrIndexOutOfRange
		}
		seen[idx] = true
	}

	n := len(order)
	weights := make([]float64, m)
	if n == 0 {
		return weights, nil
	}

	alloc := make([]float64, n)
	for i := range alloc {
		alloc[i] = 1
	}
	recBisect(cov, order, alloc, 0, n)

	total := 0.0
	for i := 0; i < n; i++ {
		w := alloc[i]
		if w < 0 {
			w = 0
		}
		weights[order[i]] = w
		total += w
	}
	if total > 0 {
		for _, idx := range order {
			weights[idx] /= total
		}
	}
	return weights, nil
}

const varianceFloor = 1e-10

func recBisect(cov [][]float64, order []int, alloc []float64, start, end int) {
	if end-start <= 1 {
		return
	}
	mid := (start + end) / 2
	leftVar := clusterVariance(cov, order[start:mid])
	rightVar := clusterVariance(cov, order[mid:end])
	inv := 1/math.Max(leftVar, varianceFloor) + 1/math.Max(rightVar, varianceFloor)
	leftWeight := (1 / math.Max(leftVar, varianceFloor)) / inv
	rightWeight := 1 - leftWeight
	for i := start; i < mid; i++ {
		alloc[i] *= leftWeight
	}
	for i := mid; i < end; i++ {
		alloc[i] *= rightWeight
	}
	recBisect(cov, order, alloc, start, mid)
	recBisect(cov, order, alloc, mid, end)
}

// clusterVariance returns the variance of the inverse-variance portfolio
// over the assets named by idx: w_i proportional to 1/cov[i][i]
// (floored), normalized to sum 1, then variance = w^T Cov w. Floored at
// varianceFloor so downstream reciprocals stay finite.
func clusterVariance(cov [][]float64, idx []int) float64 {
	count := len(idx)
	if count == 0 {
		return varianceFloor
	}
	w := make([]float64, count)
	totalInv := 0.0
	for i := 0; i < count; i++ {
		v := cov[idx[i]][idx[i]]
		iv := 1 / math.Max(v, varianceFloor)
		w[i] = iv
		totalInv += iv
	}
	if totalInv > 0 {
		for i := 0; i < count; i++ {
			w[i] /= totalInv
		}
	}
	cv := 0.0
	for i := 0; i < count; i++ {
		for j := 0; j < count; j++ {
			cv += w[i] * cov[idx[i]][idx[j]] * w[j]
		}
	}
	return math.Max(cv, varianceFloor)
}

// HRPWeights runs the full Hierarchical Risk Parity pipeline of Lopez de
// Prado (2016) end-to-end and returns the portfolio weight vector aligned
// to the original asset order:
//
//	Stage 1  d = CorrelationDistance(corr)
//	Stage 2a linkage = SingleLinkage(d)           (deterministic tie-break)
//	Stage 2b order   = QuasiDiagonalize(linkage)
//	Stage 3  w       = RecursiveBisection(cov, order)
//
// cov is the asset covariance matrix (used for the recursive-bisection
// variances); corr is the correlation matrix HRP clusters on. Passing a
// separately-cleaned correlation matrix (e.g. from random-matrix-theory
// denoising or shrinkage) is supported and encouraged — it is why corr
// is a distinct argument rather than being derived from cov here.
//
// Both matrices must be square and of the same dimension n. n == 0
// returns ErrEmptyMatrix; n == 1 returns []float64{1} (a single asset
// takes the whole book). The returned weights are non-negative and sum
// to 1.
//
// Source: Marcos Lopez de Prado, "Building Diversified Portfolios that
// Outperform Out-of-Sample", J. Portfolio Management 42(4), 2016.
func HRPWeights(cov, corr [][]float64) ([]float64, error) {
	n := len(cov)
	if n == 0 {
		return nil, ErrEmptyMatrix
	}
	if len(corr) != n {
		return nil, ErrDimensionMismatch
	}
	for i := 0; i < n; i++ {
		if len(cov[i]) != n || len(corr[i]) != n {
			return nil, ErrNotSquare
		}
	}
	if n == 1 {
		return []float64{1}, nil
	}

	dist, err := CorrelationDistance(corr)
	if err != nil {
		return nil, err
	}
	linkage, err := SingleLinkage(dist)
	if err != nil {
		return nil, err
	}
	order, err := QuasiDiagonalize(linkage, n)
	if err != nil {
		return nil, err
	}
	return RecursiveBisection(cov, order)
}
