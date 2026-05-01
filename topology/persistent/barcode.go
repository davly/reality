package persistent

import (
	"math"
	"sort"
)

// Bar is a single persistence interval.  Birth and Death are scalar
// filtration parameters; Death == math.Inf(+1) signals an essential
// class that never dies (the surviving connected component, an
// uncovered loop, etc.).  Persistence is Death - Birth (or +Inf for
// essential classes).
type Bar struct {
	Dim   int
	Birth float64
	Death float64
}

// Persistence returns the lifetime Death - Birth, or +Inf for
// essential classes.
func (b Bar) Persistence() float64 {
	if math.IsInf(b.Death, 1) {
		return math.Inf(1)
	}
	return b.Death - b.Birth
}

// IsEssential reports whether the bar is essential (never dies).
func (b Bar) IsEssential() bool {
	return math.IsInf(b.Death, 1)
}

// ComputeBarcode returns the persistence barcode of the supplied VR
// filtration up to dimension maxDim (must be 0 or 1 in v1).  Uses the
// standard column-reduction algorithm of Edelsbrunner-Letscher-
// Zomorodian 2000 over F_2: the boundary matrix D of the filtered
// simplicial complex is reduced from left to right, columns are added
// (modulo 2) by their lowest-row-index, and a column with empty
// reduced form births a class while a column with non-empty reduced
// form kills the class indexed by its pivot row.
//
// The algorithm:
//
//  1. Index simplices in filtration order; column j of D lists the
//     boundary faces of simplex j (their indices, F_2 coefficients).
//  2. For each column j left to right: while the lowest non-zero row
//     of column j has a pivot column j' < j, add column j' to
//     column j (XOR of index sets).  If column j becomes empty, j
//     births a new class; otherwise the class born at the lowest
//     pivot row is killed at j.
//  3. After reduction, classes whose birth was never killed are
//     essential — they receive Death = +Inf.
//
// Complexity: O(m^3) worst case in the number of simplices m, but in
// practice much faster on Vietoris-Rips filtrations because boundary
// columns are sparse.  For Phase-A consumer scale (n <= 50, m <= ~21k
// in the worst case at maxDim = 1) this is under a millisecond.
//
// Returns ErrInvalidMaxDim if maxDim is outside {0, 1}.
func ComputeBarcode(filtration Filtration, maxDim int) ([]Bar, error) {
	if maxDim < 0 || maxDim > 1 {
		return nil, ErrInvalidMaxDim
	}
	if filtration.Len() == 0 {
		return []Bar{}, nil
	}

	m := filtration.Len()

	// Build sparse boundary columns.  Each column is a sorted []int
	// of row indices (F_2 coefficient = 1 implicitly).  We index
	// every simplex by its position in filtration order so faces
	// can be mapped back to their column id.
	indexBy := make(map[string]int, m)
	for i, s := range filtration.Simplices {
		indexBy[simplexKey(s)] = i
	}

	cols := make([][]int, m)
	for j, s := range filtration.Simplices {
		if s.Dim() == 0 {
			continue // 0-simplices have empty boundary
		}
		cols[j] = boundaryColumn(s, indexBy)
	}

	// Matrix reduction: track which row indices are pivots for which
	// columns.  pivotCol[r] = j means column j has lowest = r.
	pivotCol := make(map[int]int, m)
	for j := 0; j < m; j++ {
		col := cols[j]
		for len(col) > 0 {
			low := col[len(col)-1]
			if jp, ok := pivotCol[low]; ok && jp < j {
				col = symDiff(col, cols[jp])
			} else {
				pivotCol[low] = j
				break
			}
		}
		cols[j] = col
	}

	// Read off bars.  A column j is a "creator" (births a class) iff
	// after reduction its column is empty.  A column j with a non-
	// empty reduced column kills the class born at its lowest row.
	bars := make([]Bar, 0)
	killed := make(map[int]bool, m)

	for j := 0; j < m; j++ {
		col := cols[j]
		if len(col) > 0 {
			birthIdx := col[len(col)-1]
			deathIdx := j
			birthDim := filtration.Simplices[birthIdx].Dim()
			if birthDim > maxDim {
				continue
			}
			// Persistence > 0 only if filtration time strictly
			// advances; equal-time birth/death contributes a
			// zero-persistence bar that we keep for consumers
			// that want the full diagram (it is still a valid
			// element of the persistence module).
			bars = append(bars, Bar{
				Dim:   birthDim,
				Birth: filtration.Times[birthIdx],
				Death: filtration.Times[deathIdx],
			})
			killed[birthIdx] = true
		}
	}

	// Essential classes: simplices whose column was empty after
	// reduction AND that were never killed by a higher-index column.
	for j := 0; j < m; j++ {
		if len(cols[j]) > 0 || killed[j] {
			continue
		}
		dim := filtration.Simplices[j].Dim()
		if dim > maxDim {
			continue
		}
		bars = append(bars, Bar{
			Dim:   dim,
			Birth: filtration.Times[j],
			Death: math.Inf(1),
		})
	}

	// Stable canonical order: dim asc, birth asc, death asc.  This
	// makes ComputeBarcode deterministic across runs and pin-friendly
	// for cross-substrate parity tests.
	sort.SliceStable(bars, func(a, b int) bool {
		if bars[a].Dim != bars[b].Dim {
			return bars[a].Dim < bars[b].Dim
		}
		if bars[a].Birth != bars[b].Birth {
			return bars[a].Birth < bars[b].Birth
		}
		return bars[a].Death < bars[b].Death
	})

	return bars, nil
}

// simplexKey returns a canonical string key for a (sorted) simplex.
// Used as a map key in boundaryColumn lookup.  We avoid fmt.Sprintf
// to keep the inner loop allocation-cheap; the manual encoding is
// uniquely decodable because integers are separated by ','.
func simplexKey(s Simplex) string {
	if len(s) == 0 {
		return ""
	}
	// Compute exact buffer size to avoid grow churn.
	bufLen := len(s) // commas + final char count rough estimate
	for _, v := range s {
		if v < 10 {
			bufLen += 1
		} else if v < 100 {
			bufLen += 2
		} else if v < 1000 {
			bufLen += 3
		} else {
			bufLen += 5 // safe upper for typical Phase-A scale
		}
	}
	buf := make([]byte, 0, bufLen)
	for i, v := range s {
		if i > 0 {
			buf = append(buf, ',')
		}
		buf = appendInt(buf, v)
	}
	return string(buf)
}

func appendInt(b []byte, v int) []byte {
	if v == 0 {
		return append(b, '0')
	}
	if v < 0 {
		b = append(b, '-')
		v = -v
	}
	// Push digits in reverse, then reverse them in place.
	start := len(b)
	for v > 0 {
		b = append(b, byte('0'+v%10))
		v /= 10
	}
	// Reverse [start:].
	for i, j := start, len(b)-1; i < j; i, j = i+1, j-1 {
		b[i], b[j] = b[j], b[i]
	}
	return b
}

// boundaryColumn returns the F_2 boundary of a simplex as a sorted
// slice of column indices.  Each face is the simplex with one
// vertex removed; its index is looked up in indexBy.
func boundaryColumn(s Simplex, indexBy map[string]int) []int {
	if len(s) <= 1 {
		return nil
	}
	rows := make([]int, 0, len(s))
	face := make(Simplex, len(s)-1)
	for skip := 0; skip < len(s); skip++ {
		// Build face = s without s[skip].
		k := 0
		for i := 0; i < len(s); i++ {
			if i == skip {
				continue
			}
			face[k] = s[i]
			k++
		}
		idx, ok := indexBy[simplexKey(face)]
		if !ok {
			// Face not in filtration (clipped by maxRadius or
			// maxDim).  Drop the simplex's contribution at this
			// face — equivalent to F_2 coefficient 0.
			continue
		}
		rows = append(rows, idx)
	}
	sort.Ints(rows)
	return rows
}

// symDiff returns the F_2 symmetric difference of two sorted []int.
// Both inputs must already be sorted ascending; the output is also
// sorted ascending and contains each element of (a XOR b).  This is
// the F_2 column-add operation in the matrix-reduction algorithm.
func symDiff(a, b []int) []int {
	out := make([]int, 0, len(a)+len(b))
	i, j := 0, 0
	for i < len(a) && j < len(b) {
		switch {
		case a[i] < b[j]:
			out = append(out, a[i])
			i++
		case a[i] > b[j]:
			out = append(out, b[j])
			j++
		default:
			// equal: cancels in F_2.
			i++
			j++
		}
	}
	for i < len(a) {
		out = append(out, a[i])
		i++
	}
	for j < len(b) {
		out = append(out, b[j])
		j++
	}
	return out
}
