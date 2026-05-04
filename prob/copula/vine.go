package copula

import (
	"errors"
	"fmt"
	"math"
)

// Vine copula substrate — pair-copula construction (PCC) over a regular
// vine tree topology. Per S60 evening-review agent a241 (S60 carry-
// forward #2). This file ships the substrate primitives + minimal
// D-vine API; full higher-dim sequential fitting (Aas-Czado 2009 Algorithm
// 5) is the natural follow-up consumer in `flagships/folio` once the
// Solvency II 12-pot aggregation lands.
//
// Topology classes:
//   - **D-vine** — sequential dependencies along a path. Tree T_1 is the
//     path 1-2-3-...-n; tree T_k contains conditional dependencies given
//     the k-1 adjacent elements. Suited to ordered indices (e.g. EIOPA
//     module grouping: market → life → health → non-life).
//   - **C-vine** — one central node per tree. Tree T_1 has a root
//     connected to every other node; T_k cascades around k different
//     conditional roots. Suited to one dominant driver (e.g. equity
//     index as central node, all sectors conditioning on it).
//
// This file ships the D-vine builder; C-vine ships when a consumer
// concretely needs it (per PROMOTION_GATE Step 10 PM-051 — substrate
// follows demand-side pull, not anticipation).
//
// References:
//   - Bedford, T., Cooke, R.M. (2002). "Vines: a new graphical model for
//     dependent random variables." Annals of Statistics 30: 1031-1068.
//   - Aas, K., Czado, C., Frigessi, A., Bakken, H. (2009). "Pair-copula
//     constructions of multiple dependence." Insurance: Mathematics and
//     Economics 44: 182-198.
//   - Joe, H. (1996). "Families of m-variate distributions with given
//     margins and m(m-1)/2 bivariate dependence parameters." In Distri-
//     butions with Fixed Marginals and Related Topics, IMS LNS 28.

// ErrVineInvalidDimension is returned by D-vine / C-vine constructors
// when n < 2 (a vine over one variable is degenerate).
var ErrVineInvalidDimension = errors.New(
	"copula: vine dimension n must be >= 2")

// ErrVineEdgeMismatch is returned by NewDVine when the supplied edges
// have inconsistent length for the dimension.
var ErrVineEdgeMismatch = errors.New(
	"copula: D-vine tree must have n-1 edges in T_1, n-2 in T_2, etc.")

// ErrVineEdgeInvalid is returned by NewDVine when an edge has invalid
// theta / family combination.
var ErrVineEdgeInvalid = errors.New(
	"copula: vine edge has invalid theta for its family")

// VineEdge is a single bivariate copula edge in the vine. For tree T_k
// (k >= 1):
//
//   - In T_1, the edge couples raw observations (u_i, u_{i+1}).
//   - In T_k for k > 1, the edge couples *pseudo-observations* obtained
//     by applying h-functions of T_{k-1} to the original observations.
//     Indexing in the conditioning set follows the D-vine convention:
//     edge e in T_k couples (u_e | conditioning_set) with
//     (u_{e+k} | conditioning_set).
//
// Family + Theta together determine the bivariate copula CDF + h-function.
type VineEdge struct {
	Family ArchimedeanFamily
	Theta  float64
}

// Validate checks that Theta is admissible for Family.
func (e VineEdge) Validate() error {
	switch e.Family {
	case FamilyClayton:
		if !(e.Theta > 0) {
			return fmt.Errorf("%w: Clayton theta must be > 0, got %v",
				ErrVineEdgeInvalid, e.Theta)
		}
	case FamilyGumbel:
		if e.Theta < 1 {
			return fmt.Errorf("%w: Gumbel theta must be >= 1, got %v",
				ErrVineEdgeInvalid, e.Theta)
		}
	default:
		return fmt.Errorf("%w: unknown family %v", ErrVineEdgeInvalid, e.Family)
	}
	return nil
}

// DVine is a regular D-vine over n variables with n-1 trees. Tree T_k
// contains n-k edges. Total edge count: n(n-1)/2.
//
// Indexing: Trees[k-1][e] is the e-th edge of tree T_k (k = 1..n-1,
// e = 0..n-k-1). The D-vine path order is the variable index order
// supplied at construction.
type DVine struct {
	dim   int
	Trees [][]VineEdge
}

// NewDVine constructs a D-vine over `dim` variables from a triangular
// `trees` slice where trees[k][e] is the e-th edge of tree T_{k+1}.
// Validates the shape (n-1, n-2, ..., 1 edges per tree) and each edge's
// theta admissibility.
//
// For dim=3: trees must be [[T1[0], T1[1]], [T2[0]]] (3 edges total).
// For dim=4: trees must be [[T1[0..2]], [T2[0..1]], [T3[0]]] (6 edges).
// Generally: tree T_k has dim - k edges; total dim*(dim-1)/2.
func NewDVine(dim int, trees [][]VineEdge) (*DVine, error) {
	if dim < 2 {
		return nil, ErrVineInvalidDimension
	}
	if len(trees) != dim-1 {
		return nil, fmt.Errorf("%w: expected %d trees, got %d",
			ErrVineEdgeMismatch, dim-1, len(trees))
	}
	for k := 0; k < dim-1; k++ {
		expected := dim - 1 - k
		if len(trees[k]) != expected {
			return nil, fmt.Errorf("%w: tree T_%d expected %d edges, got %d",
				ErrVineEdgeMismatch, k+1, expected, len(trees[k]))
		}
		for e, edge := range trees[k] {
			if err := edge.Validate(); err != nil {
				return nil, fmt.Errorf("T_%d edge %d: %w", k+1, e, err)
			}
		}
	}
	return &DVine{dim: dim, Trees: trees}, nil
}

// Dim returns the number of variables in the vine.
func (v *DVine) Dim() int { return v.dim }

// EdgeCount returns the total number of bivariate copulas in the vine
// (= dim * (dim-1) / 2).
func (v *DVine) EdgeCount() int {
	return v.dim * (v.dim - 1) / 2
}

// HFunctionPass applies one tree of h-function transforms to the
// supplied row of pseudo-observations, returning the next tree's
// pseudo-observations.
//
// For tree T_k with input u (length dim - k + 1) and tree-T_k edges
// e_0..e_{n-k-1}, the output row has length dim - k:
//
//	out[i] = h_{e_i}(u[i] | u[i+1])
//
// where h_{e_i} is the h-function for edge e_i. This matches Aas-Czado
// 2009 Algorithm 4 step "compute v_{k+1, i}".
//
// Returns an error if any edge's h-function fails (invalid theta — but
// validation in NewDVine ensures this is unreachable in practice).
func (v *DVine) HFunctionPass(treeIdx int, u []float64) ([]float64, error) {
	if treeIdx < 0 || treeIdx >= v.dim-1 {
		return nil, fmt.Errorf("treeIdx %d out of range [0, %d)", treeIdx, v.dim-1)
	}
	expected := v.dim - treeIdx
	if len(u) != expected {
		return nil, fmt.Errorf("input length %d != expected %d for tree T_%d",
			len(u), expected, treeIdx+1)
	}
	tree := v.Trees[treeIdx]
	out := make([]float64, len(tree))
	for i, edge := range tree {
		hfn, err := HFnForFamily(edge.Family, edge.Theta)
		if err != nil {
			return nil, fmt.Errorf("tree T_%d edge %d: %w", treeIdx+1, i, err)
		}
		out[i] = hfn(u[i], u[i+1])
	}
	return out, nil
}

// LogPDF evaluates the D-vine's log joint density at a row of uniform
// pseudo-observations u (length dim). Sums log-densities of every
// bivariate copula across all trees per the Aas-Czado 2009 vine
// likelihood decomposition (§2.5).
//
// Algorithm:
//   For tree T_1 (raw observations), edge e couples u[e] with u[e+1]:
//     contribution = log c_e( u[e], u[e+1]; θ_e )
//   For tree T_k (k > 1, pseudo-observations), edge e couples
//   pseudo[e] with pseudo[e+1] where pseudo is the previous tree's
//   h-function output:
//     contribution = log c_e( pseudo[e], pseudo[e+1]; θ_e )
//   Sum all contributions across all trees.
//
// Returns -∞ when any input falls on the unit-hypercube boundary
// (the bivariate copula densities are degenerate there).
func (v *DVine) LogPDF(u []float64) (float64, error) {
	if len(u) != v.dim {
		return 0, fmt.Errorf("LogPDF input length %d != dim %d", len(u), v.dim)
	}
	for i, ui := range u {
		if ui <= 0 || ui >= 1 {
			// Boundary value — copula PDFs are degenerate there.
			return math.Inf(-1), fmt.Errorf("u[%d] = %v on hypercube boundary", i, ui)
		}
	}

	// Accumulate log-density edge by edge. Track current "row" of
	// pseudo-observations as we ascend the trees.
	logL := 0.0
	current := make([]float64, len(u))
	copy(current, u)

	for k := 0; k < v.dim-1; k++ {
		tree := v.Trees[k]
		// Sum log c for each edge in the current tree.
		next := make([]float64, len(tree)) // next-tree pseudo-obs row
		for i, edge := range tree {
			logPdf, err := LogPDFFnForFamily(edge.Family, edge.Theta)
			if err != nil {
				return 0, fmt.Errorf("tree T_%d edge %d log-pdf: %w", k+1, i, err)
			}
			ui := current[i]
			vi := current[i+1]
			contrib := logPdf(ui, vi)
			if math.IsInf(contrib, -1) {
				return math.Inf(-1), nil
			}
			logL += contrib

			// Compute next-tree pseudo-observation via the same edge's
			// h-function (only if there is a next tree to feed).
			if k+1 < v.dim-1 {
				hfn, err := HFnForFamily(edge.Family, edge.Theta)
				if err != nil {
					return 0, fmt.Errorf("tree T_%d edge %d h-fn: %w", k+1, i, err)
				}
				next[i] = hfn(ui, vi)
			}
		}
		current = next
	}

	return logL, nil
}
