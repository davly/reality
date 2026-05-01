package persistent

import (
	"math"
	"sort"
)

// Simplex is a sorted slice of vertex indices.  By convention the
// indices are in strictly-increasing order and the dimension of the
// simplex is len(s) - 1 (a vertex is a 0-simplex, an edge is a
// 1-simplex, a triangle is a 2-simplex).  The boundary of a k-simplex
// in F_2 coefficients is the symmetric difference of its (k+1) faces;
// see barcode.go.
type Simplex []int

// Dim returns the simplicial dimension (len(s) - 1).
func (s Simplex) Dim() int {
	return len(s) - 1
}

// Equal reports whether two simplices have identical vertex sets.
// Both slices must already be in canonical (sorted-ascending) form,
// which is the invariant maintained by VietorisRipsComplex.
func (s Simplex) Equal(other Simplex) bool {
	if len(s) != len(other) {
		return false
	}
	for i := range s {
		if s[i] != other[i] {
			return false
		}
	}
	return true
}

// Filtration is a Vietoris-Rips simplicial filtration: a list of
// simplices in non-decreasing order of birth time (the "filtration
// time"), each simplex paired with its scalar entry parameter.  The
// filtration is the canonical input to ComputeBarcode in barcode.go.
//
// Invariants:
//
//   - len(Simplices) == len(Times).
//   - Times is non-decreasing (sort.Float64sAreSorted).
//   - For every simplex s of dimension k >= 1, every face of s
//     (obtained by deleting one vertex) appears earlier in Simplices
//     with a Time at most Times[i].  This is the "filtration order"
//     condition that the matrix-reduction algorithm depends on for
//     correctness.
type Filtration struct {
	Simplices []Simplex
	Times     []float64
}

// Len returns the number of simplices in the filtration.
func (f Filtration) Len() int {
	return len(f.Simplices)
}

// VietorisRipsComplex builds the Vietoris-Rips simplicial filtration
// of a point cloud in R^d under the Euclidean metric.  A simplex
// {v_0, ..., v_k} is included with filtration time equal to the
// diameter of its vertex set (max pairwise distance) iff that diameter
// does not exceed maxRadius.
//
// In Phase-A scope (maxDim in {0, 1}), the filtration consists of:
//
//   - n vertices (0-simplices), each born at time 0.
//   - All edges (1-simplices) {i, j} with |p_i - p_j| <= maxRadius,
//     each born at time |p_i - p_j|.
//   - All triangles (2-simplices) {i, j, k} with the diameter of
//     the triple <= maxRadius, each born at the triangle's diameter.
//     Triangles are needed even at maxDim=1 because they are what
//     *kill* H_1 classes (a loop dies when its filling triangle
//     enters), and the matrix-reduction algorithm requires the full
//     (maxDim+1)-skeleton.
//
// The output is sorted by filtration time, with stable secondary sort
// by dimension (ascending) and lexicographic vertex order, so that
// the matrix-reduction step in barcode.go produces a deterministic
// barcode regardless of point-cloud iteration order.
//
// Complexity: O(n^3) time + O(n^3) space in the worst case (full
// 2-skeleton on n points).  For Phase-A consumers (Tether import-
// graph, Insights service blast-radius, RubberDuck 30-asset crash
// detection), n is bounded by the C# reference implementation's
// MaxAssets = 50, so the O(n^3) factor is at most 125,000 simplices.
//
// Returns ErrEmptyPoints, ErrInvalidMaxDim, ErrInconsistentDim, or
// ErrInvalidMaxRadius if the input is malformed.
func VietorisRipsComplex(points [][]float64, maxRadius float64, maxDim int) (Filtration, error) {
	if err := validateVRInput(points, maxRadius, maxDim); err != nil {
		return Filtration{}, err
	}

	n := len(points)

	// 0-simplices (vertices): all born at time 0.
	type entry struct {
		simplex Simplex
		time    float64
	}
	entries := make([]entry, 0, n)
	for i := 0; i < n; i++ {
		entries = append(entries, entry{Simplex{i}, 0})
	}

	// 1-simplices (edges): birth time = pairwise Euclidean distance.
	// Cache the distance matrix because triangles need it.
	dist := pairwiseDistanceMatrix(points)
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			d := dist[i][j]
			if d <= maxRadius {
				entries = append(entries, entry{Simplex{i, j}, d})
			}
		}
	}

	// 2-simplices (triangles): birth time = max of three pairwise
	// distances (the simplicial diameter).  Triangles fill loops in
	// H_1, so they are required even at maxDim = 1.
	if maxDim >= 1 {
		for i := 0; i < n; i++ {
			for j := i + 1; j < n; j++ {
				for k := j + 1; k < n; k++ {
					d := math.Max(dist[i][j], math.Max(dist[i][k], dist[j][k]))
					if d <= maxRadius {
						entries = append(entries, entry{Simplex{i, j, k}, d})
					}
				}
			}
		}
	}

	// Sort by (time, dim, lex(vertices)).  Stable filtration order is
	// what the matrix-reduction step relies on; ties on time are
	// broken by dimension first (a face must appear before any
	// coface) and then lexicographically.
	sort.SliceStable(entries, func(a, b int) bool {
		ea, eb := entries[a], entries[b]
		if ea.time != eb.time {
			return ea.time < eb.time
		}
		if ea.simplex.Dim() != eb.simplex.Dim() {
			return ea.simplex.Dim() < eb.simplex.Dim()
		}
		// Lex order on sorted vertex indices.
		for i := 0; i < len(ea.simplex) && i < len(eb.simplex); i++ {
			if ea.simplex[i] != eb.simplex[i] {
				return ea.simplex[i] < eb.simplex[i]
			}
		}
		return false
	})

	out := Filtration{
		Simplices: make([]Simplex, len(entries)),
		Times:     make([]float64, len(entries)),
	}
	for i, e := range entries {
		out.Simplices[i] = e.simplex
		out.Times[i] = e.time
	}
	return out, nil
}

// validateVRInput rejects malformed point clouds, max-radius, and
// max-dimension values.  Returning a typed error here keeps the hot
// path in VietorisRipsComplex allocation-free.
func validateVRInput(points [][]float64, maxRadius float64, maxDim int) error {
	if len(points) == 0 {
		return ErrEmptyPoints
	}
	if maxDim < 0 || maxDim > 1 {
		return ErrInvalidMaxDim
	}
	if math.IsNaN(maxRadius) || math.IsInf(maxRadius, 0) || maxRadius < 0 {
		return ErrInvalidMaxRadius
	}
	d := len(points[0])
	if d == 0 {
		return ErrInconsistentDim
	}
	for _, p := range points {
		if len(p) != d {
			return ErrInconsistentDim
		}
	}
	return nil
}

// pairwiseDistanceMatrix returns the full n*n Euclidean-distance
// matrix.  The matrix is symmetric with a zero diagonal; we fill
// only the upper triangle and mirror, which keeps the complexity at
// O(n^2 * d) and avoids redundant sqrt calls.
func pairwiseDistanceMatrix(points [][]float64) [][]float64 {
	n := len(points)
	d := make([][]float64, n)
	for i := range d {
		d[i] = make([]float64, n)
	}
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			s := 0.0
			for k := range points[i] {
				diff := points[i][k] - points[j][k]
				s += diff * diff
			}
			r := math.Sqrt(s)
			d[i][j] = r
			d[j][i] = r
		}
	}
	return d
}
