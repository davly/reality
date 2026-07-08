package combinatorics

// derangement.go — constrained-derangement (constrained-assignment) solver
// with deterministic, verifiable seeding.
//
// PROMOTION PROVENANCE: promoted from giftdrawcraft
// (src/api/internal/ai/gateway/giftdraw_draw.go on branch
// feat/giftdrawcraft-producer, commit 57c023cb); pattern-setter for
// craft-solver promotion — next candidates: fixturecraft round-robin
// scheduler (fixture_schedule.go), puzzlecraft seeded RNG. Crafts swap
// local copies for this package at merge-train rebase, post
// reality-v0.11.0 tag. Domain-free maths only: no request/response types
// from any craft travel here.
//
// THE PROBLEM: assign each of n agents exactly one of n slots (a
// permutation) such that
//   - no agent is assigned to itself (no fixed points — a derangement),
//   - every blocked edge is avoided (the caller decides what is blocked;
//     CanonicalizeExclusions + BuildBlocked implement the common symmetric
//     pair-exclusion case),
//   - the result is exact: ok=false is a PROOF that no valid assignment
//     exists, never a sampling timeout.
//
// DETERMINISM CONTRACT: the RNG is seeded from a SHA-256 hash of a
// canonicalized payload (SeedFromCanonical), so the same
// (n, canonical constraints, seed) always yields the IDENTICAL assignment,
// while any payload change yields an independent draw. The hex seed string
// is short enough to surface to users as a reproducibility receipt and
// useless for reversing the payload.
//
// THE ALGORITHM (two phases, both consuming the same seeded rng in order):
//
//  1. BOUNDED REJECTION SAMPLING — DerangementSampleAttempts seeded
//     Fisher–Yates permutations, first valid one wins. When constraints are
//     loose (the common case) a valid derangement is found in a handful of
//     draws and every valid derangement is equally likely.
//
//  2. BACKTRACKING AUGMENTING-PATH MATCHING — when sampling fails (dense
//     constraints), a perfect matching is constructed on the bipartite
//     agents→slots graph (blocked edges removed) via Kuhn's algorithm with
//     rng-shuffled adjacency: a depth-first BACKTRACKING search over
//     alternating paths that re-assigns earlier agents when a later agent
//     is stuck. Polynomial (O(V·E)) and EXACT: it finds a valid assignment
//     whenever one exists, and the shuffled candidate order preserves the
//     property that the constructed assignment still respects the seed
//     rather than collapsing to a lexicographic one.
//
//  3. EXACT IMPOSSIBILITY — a perfect matching exists iff a valid
//     assignment exists, so when phase 2 cannot match every agent the
//     constraints PROVABLY admit no valid constrained derangement
//     (Berge/König: augmenting-path failure from a free vertex with all
//     alternatives exhausted is a proof).

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math/rand"
	"sort"
)

// DerangementSampleAttempts bounds the rejection-sampling phase of
// ConstrainedDerangement. 64 attempts makes the sampling path
// overwhelmingly likely for feasible loose constraints (a plain
// derangement is hit with p ≈ 1-1/e ≈ 0.63 per try) while keeping the
// worst case trivially cheap before the exact solver takes over.
const DerangementSampleAttempts = 64

// Constraint is one symmetric exclusion pair in canonical [low, high]
// order: neither index may be assigned to the other.
type Constraint [2]int

// CanonicalizeExclusions normalizes symmetric exclusion pairs for BOTH the
// solver and the seed hash: each pair ordered [low, high], the list sorted
// lexicographically, exact duplicates removed. Symmetric duplicates
// ([0,1] and [1,0]) therefore collapse to ONE pair, so an input that
// differs only in pair order or orientation produces the IDENTICAL
// canonical form — hence the identical seed and the identical assignment.
//
// Normalization only: the caller validates arity (each pair must have at
// least two elements; only the first two are read), index range, and
// self-pairs before calling.
func CanonicalizeExclusions(exclusions [][]int) []Constraint {
	canonical := make([]Constraint, 0, len(exclusions))
	seen := make(map[Constraint]bool, len(exclusions))
	for _, pair := range exclusions {
		a, b := pair[0], pair[1]
		if a > b {
			a, b = b, a
		}
		key := Constraint{a, b}
		if seen[key] {
			continue
		}
		seen[key] = true
		canonical = append(canonical, key)
	}
	sort.Slice(canonical, func(i, j int) bool {
		if canonical[i][0] != canonical[j][0] {
			return canonical[i][0] < canonical[j][0]
		}
		return canonical[i][1] < canonical[j][1]
	})
	return canonical
}

// SeedFromCanonical derives a deterministic RNG seed and a short
// verifiable seed string from a canonicalized payload: SHA-256 over the
// payload's JSON encoding, first 8 bytes big-endian as the rand source,
// the same 8 bytes hex-encoded as the seed string (16 hex chars — enough
// to verify reproducibility, useless for reversing the payload).
//
// Determinism is the CALLER's contract to keep: pass a struct (field order
// is fixed by declaration; map keys are sorted by encoding/json) whose
// constraint fields have already been canonicalized (CanonicalizeExclusions),
// so that semantically identical payloads marshal byte-identically. An
// unmarshalable payload (channel, func, cyclic value) returns an error.
func SeedFromCanonical(payload any) (int64, string, error) {
	b, err := json.Marshal(payload)
	if err != nil {
		return 0, "", fmt.Errorf("combinatorics: seed payload not canonicalizable: %w", err)
	}
	h := sha256.Sum256(b)
	return int64(binary.BigEndian.Uint64(h[:8])), fmt.Sprintf("%x", h[:8]), nil
}

// BuildBlocked builds the blocked-edge matrix for ConstrainedDerangement:
// blocked[i][j] is true when agent i may NOT be assigned slot j. The
// diagonal is blocked by construction (no fixed points — the derangement
// property) and every canonical exclusion pair blocks BOTH directions.
func BuildBlocked(n int, canonical []Constraint) [][]bool {
	blocked := make([][]bool, n)
	for i := range blocked {
		blocked[i] = make([]bool, n)
		blocked[i][i] = true // no fixed points, ever
	}
	for _, pair := range canonical {
		blocked[pair[0]][pair[1]] = true
		blocked[pair[1]][pair[0]] = true
	}
	return blocked
}

// IsValidAssignment reports whether perm (perm[i] = the slot assigned to
// agent i) is a valid constrained derangement: a permutation avoiding
// every blocked edge. Used by the sampling phase and exported as the
// independent verifier.
func IsValidAssignment(perm []int, blocked [][]bool) bool {
	seen := make([]bool, len(perm))
	for agent, slot := range perm {
		if slot < 0 || slot >= len(perm) || seen[slot] || blocked[agent][slot] {
			return false
		}
		seen[slot] = true
	}
	return true
}

// ConstrainedDerangement produces the assignment: perm[i] = the slot for
// agent i, or ok=false when the constraints PROVABLY admit no valid
// constrained derangement. Deterministic for a given (n, blocked, seeded
// rng): both phases consume the same rng in the same order, so the same
// (n, canonical constraints, seed) always yields the same assignment.
// Pure function — no I/O, no globals.
//
// blocked must be an n×n matrix (BuildBlocked); blocked[i][i] should be
// true for the derangement property (BuildBlocked enforces it).
func ConstrainedDerangement(n int, blocked [][]bool, rng *rand.Rand) ([]int, bool) {
	// Phase 1 — bounded rejection sampling (the uniform-ish path).
	perm := make([]int, n)
	for attempt := 0; attempt < DerangementSampleAttempts; attempt++ {
		for i := range perm {
			perm[i] = i
		}
		rng.Shuffle(n, func(i, j int) { perm[i], perm[j] = perm[j], perm[i] })
		if IsValidAssignment(perm, blocked) {
			return perm, true
		}
	}

	// Phase 2 — backtracking augmenting-path matching (exact). Adjacency
	// lists are rng-shuffled so the constructed assignment still varies
	// with the seed rather than collapsing to a lexicographic assignment.
	adj := make([][]int, n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if !blocked[i][j] {
				adj[i] = append(adj[i], j)
			}
		}
		rng.Shuffle(len(adj[i]), func(a, b int) { adj[i][a], adj[i][b] = adj[i][b], adj[i][a] })
	}

	matchedAgent := make([]int, n) // slot -> agent (-1 = free)
	for i := range matchedAgent {
		matchedAgent[i] = -1
	}

	var augment func(agent int, visited []bool) bool
	augment = func(agent int, visited []bool) bool {
		for _, slot := range adj[agent] {
			if visited[slot] {
				continue
			}
			visited[slot] = true
			// Take a free slot, or BACKTRACK: re-route the agent currently
			// holding this slot onto another one.
			if matchedAgent[slot] == -1 || augment(matchedAgent[slot], visited) {
				matchedAgent[slot] = agent
				return true
			}
		}
		return false
	}

	for agent := 0; agent < n; agent++ {
		if !augment(agent, make([]bool, n)) {
			// EXACT impossibility: this agent cannot be matched even with
			// full re-routing, so no perfect matching — hence no valid
			// assignment — exists.
			return nil, false
		}
	}

	for slot, agent := range matchedAgent {
		perm[agent] = slot
	}
	return perm, true
}
