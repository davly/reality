package sequence

// ---------------------------------------------------------------------------
// Token-level shortest edit script (Myers O(ND) diff)
// ---------------------------------------------------------------------------
//
// DiffTokens computes the shortest edit script (SES) that transforms token
// slice a into token slice b, using Myers' O(ND) greedy difference algorithm
// with the linear-space divide-and-conquer refinement (the "middle snake"
// recursion). The result is an ordered list of edit operations — Equal,
// Insert, Delete — such that:
//
//   - filtering out Insert ops and reading the tokens reproduces a exactly;
//   - filtering out Delete ops and reading the tokens reproduces b exactly;
//   - the number of Insert+Delete ops equals the true edit distance D (the SES
//     is minimal — it maximises the count of Equal ops, i.e. the LCS).
//
// Unlike a full Wagner-Fischer / LCS table (O(n·m) time AND O(n·m) memory),
// this runs in O((n+m)·D) time and O(n+m) space, where D is the edit distance.
// For workloads where most tokens are shared (e.g. an original vs. its
// lightly-corrected rewrite) D is small, so both time and memory stay far
// below the quadratic table an equivalent LCS diff would allocate.
//
// Reference: Eugene W. Myers, "An O(ND) Difference Algorithm and Its
// Variations", Algorithmica 1 (1986), 251-266 — §2-3 (greedy O(ND) SES) and
// §4b (linear-space middle-snake refinement). The middle-snake construction is
// the same one used by git's diff engine and GNU diff's Hirschberg-style split.

// EditOpKind identifies the kind of a single token-level edit operation.
type EditOpKind string

const (
	// OpEqual marks a token present, in the same relative position, in both
	// sequences (part of the longest common subsequence).
	OpEqual EditOpKind = "equal"
	// OpInsert marks a token that appears only in b (added).
	OpInsert EditOpKind = "insert"
	// OpDelete marks a token that appears only in a (removed).
	OpDelete EditOpKind = "delete"
)

// EditOp is one operation in a shortest edit script: a kind plus the token it
// applies to. For OpEqual and OpDelete the token is drawn from a; for OpInsert
// it is drawn from b.
type EditOp struct {
	Op    EditOpKind
	Token string
}

// DiffTokens returns the shortest edit script transforming a into b as an
// ordered slice of EditOp. It never returns nil for non-empty input; for two
// empty inputs it returns an empty (non-nil) slice.
//
// Determinism: the algorithm is fully deterministic. Where several minimal
// scripts exist, Myers' greedy rule breaks ties consistently (it prefers
// advancing deletions over insertions at equal cost), so the same inputs
// always yield byte-identical output — a requirement for golden-file testing.
//
// Time complexity:  O((n+m)·D), D = edit distance (Insert+Delete count).
// Space complexity: O(n+m) — two integer frontier arrays, reused across the
//
//	recursion; the recursion depth is O(log(n+m)) because each middle-snake
//	split more than halves the larger remaining problem.
//
// Precision: exact (integer arithmetic only); no floating point, no tolerance.
func DiffTokens(a, b []string) []EditOp {
	ops := make([]EditOp, 0, len(a)+len(b))
	// Frontier arrays are allocated once and shared across the whole
	// recursion to keep total working memory at O(n+m).
	max := len(a) + len(b)
	vf := make([]int, 2*max+1)
	vb := make([]int, 2*max+1)
	diffRec(a, b, vf, vb, &ops)
	return ops
}

// diffRec appends the SES of a→b to *ops using the linear-space middle-snake
// recursion. vf and vb are scratch frontier arrays (length 2*(cap)+1) shared
// across every call; each call only touches the index band it needs.
func diffRec(a, b []string, vf, vb []int, ops *[]EditOp) {
	n, m := len(a), len(b)

	// Trim a shared prefix and suffix cheaply before the (more expensive)
	// middle-snake search. This is a correctness-neutral optimisation that
	// also handles the common "mostly equal" case in linear time.
	prefix := 0
	for prefix < n && prefix < m && a[prefix] == b[prefix] {
		prefix++
	}
	suffix := 0
	for suffix < n-prefix && suffix < m-prefix && a[n-1-suffix] == b[m-1-suffix] {
		suffix++
	}

	// Emit the shared prefix.
	for i := 0; i < prefix; i++ {
		*ops = append(*ops, EditOp{Op: OpEqual, Token: a[i]})
	}

	// The unmatched middle.
	am := a[prefix : n-suffix]
	bm := b[prefix : m-suffix]

	switch {
	case len(am) == 0:
		// Everything left in b is an insertion.
		for _, t := range bm {
			*ops = append(*ops, EditOp{Op: OpInsert, Token: t})
		}
	case len(bm) == 0:
		// Everything left in a is a deletion.
		for _, t := range am {
			*ops = append(*ops, EditOp{Op: OpDelete, Token: t})
		}
	default:
		// Both sides non-empty: find the middle snake and recurse on the two
		// halves it induces.
		x, y, u, v, d := middleSnake(am, bm, vf, vb)
		if d > 1 {
			diffRec(am[:x], bm[:y], vf, vb, ops)
			// The snake am[x:u] == bm[y:v] is a run of equal tokens.
			for i := x; i < u; i++ {
				*ops = append(*ops, EditOp{Op: OpEqual, Token: am[i]})
			}
			diffRec(am[u:], bm[v:], vf, vb, ops)
		} else {
			// d <= 1: at most one indel remains after a shared prefix. Emit it
			// directly (the middle-snake recursion is unnecessary and the base
			// case is unambiguous here).
			emitSmall(am, bm, ops)
		}
	}

	// Emit the shared suffix.
	for i := n - suffix; i < n; i++ {
		*ops = append(*ops, EditOp{Op: OpEqual, Token: a[i]})
	}
}

// emitSmall handles the base case where the edit distance between a and b is
// at most 1 (so |len(a)-len(b)| <= 1 and at most a single insert or delete
// separates them, around a shared prefix). It produces the unique minimal
// script directly.
func emitSmall(a, b []string, ops *[]EditOp) {
	n, m := len(a), len(b)
	// Shared prefix.
	p := 0
	for p < n && p < m && a[p] == b[p] {
		p++
	}
	for i := 0; i < p; i++ {
		*ops = append(*ops, EditOp{Op: OpEqual, Token: a[i]})
	}
	switch {
	case n == m:
		// d==0 after prefix means the sequences are identical; prefix covered
		// them. (n==m with d<=1 cannot be a single substitution — that is d==2.)
	case n > m:
		// Exactly one deletion: a[p] is removed, then a[p+1:] == b[p:].
		*ops = append(*ops, EditOp{Op: OpDelete, Token: a[p]})
		for i := p + 1; i < n; i++ {
			*ops = append(*ops, EditOp{Op: OpEqual, Token: a[i]})
		}
	default:
		// Exactly one insertion: b[p] is added, then a[p:] == b[p+1:].
		*ops = append(*ops, EditOp{Op: OpInsert, Token: b[p]})
		for i := p; i < n; i++ {
			*ops = append(*ops, EditOp{Op: OpEqual, Token: a[i]})
		}
	}
}

// middleSnake finds the middle snake of an optimal edit path between a and b,
// per Myers §4b. It returns the snake's start (x,y) and end (u,v) in a/b
// coordinates together with the total SES length d for a→b. The snake
// a[x:u] == b[y:v] is a (possibly empty) run of matched tokens that lies on an
// optimal path, splitting the problem into a[:x]/b[:y] and a[u:]/b[v:].
//
// Both frontier arrays are indexed by diagonal k with an offset so that
// negative k are representable. The forward search advances from (0,0); the
// backward search advances from (n,m) in reflected coordinates.
func middleSnake(a, b []string, vf, vb []int) (x, y, u, v, d int) {
	n, m := len(a), len(b)
	delta := n - m
	odd := delta&1 != 0
	off := len(vf) / 2 // index offset so vf[off+k] is valid for k in [-off, off]

	// Initialise the two frontiers (both frontiers seed diagonal k=1 at 0 so the
	// k=0 recurrence at d=0 reads a valid 0).
	vf[off+1] = 0
	vb[off+1] = 0

	// hmax bounds the half-distance we must search: ceil((n+m)/2).
	hmax := (n + m + 1) / 2
	for dd := 0; dd <= hmax; dd++ {
		// ---- forward pass: extend furthest-reaching D-paths from (0,0) ----
		for k := -dd; k <= dd; k += 2 {
			var px int
			if k == -dd || (k != dd && vf[off+k-1] < vf[off+k+1]) {
				px = vf[off+k+1] // step down (insertion): x unchanged
			} else {
				px = vf[off+k-1] + 1 // step right (deletion): x+1
			}
			py := px - k
			xs, ys := px, py
			// Follow the diagonal snake of equal tokens.
			for px < n && py < m && a[px] == b[py] {
				px++
				py++
			}
			vf[off+k] = px
			// Overlap test: only meaningful when delta is odd, and only on the
			// backward diagonals already explored (band around delta).
			if odd && k >= delta-(dd-1) && k <= delta+(dd-1) {
				// Backward reach on the same diagonal k is stored (reflected)
				// at vb[off + (k-delta)] as a distance from the end; its
				// absolute x is n - vb[off + (k-delta)].
				if vf[off+k]+vb[off+(delta-k)] >= n {
					// Paths overlap: middle snake is the forward snake we just
					// followed, from (xs,ys) to (px,py). Total length 2*dd-1.
					return xs, ys, px, py, 2*dd - 1
				}
			}
		}
		// ---- backward pass: extend furthest-reaching D-paths from (n,m) ----
		for k := -dd; k <= dd; k += 2 {
			var px int
			if k == -dd || (k != dd && vb[off+k-1] < vb[off+k+1]) {
				px = vb[off+k+1]
			} else {
				px = vb[off+k-1] + 1
			}
			py := px - k
			xs, ys := px, py
			// Reflected diagonal: compare from the ends inward.
			for px < n && py < m && a[n-1-px] == b[m-1-py] {
				px++
				py++
			}
			vb[off+k] = px
			// Inverse (forward) diagonal for this backward k is delta-k; the
			// overlap test applies when delta is even.
			if !odd && (delta-k) >= -dd && (delta-k) <= dd {
				if vb[off+k]+vf[off+(delta-k)] >= n {
					// Convert the backward (reflected) snake to forward
					// coordinates: reflected (xs,ys) maps to (n-px, m-py) start
					// and (n-xs, m-ys) end.
					return n - px, m - py, n - xs, m - ys, 2 * dd
				}
			}
		}
	}
	// Unreachable for finite input (a middle snake always exists), but keep the
	// compiler happy and fail safe: treat everything as changed.
	return 0, 0, n, m, n + m
}
