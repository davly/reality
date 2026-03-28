package gametheory

// ---------------------------------------------------------------------------
// Stable Matching
//
// The Gale-Shapley algorithm (deferred acceptance) for finding stable
// matchings between two equal-sized sets of agents. The algorithm is
// proposer-optimal: the resulting matching is the best possible stable
// matching for the proposing side.
// ---------------------------------------------------------------------------

// GaleShapley computes a stable matching using the Gale-Shapley deferred
// acceptance algorithm. Both proposerPrefs and receiverPrefs are preference
// lists: proposerPrefs[i] is an ordered slice of receiver indices that
// proposer i prefers (most preferred first), and receiverPrefs[j] is an
// ordered slice of proposer indices that receiver j prefers.
//
// Returns matching where matching[proposer] = receiver. The matching is
// proposer-optimal: among all stable matchings, each proposer gets the
// best partner possible. Equivalently, each receiver gets the worst
// partner possible among all stable matchings.
//
// Time complexity: O(n^2) where n = len(proposerPrefs).
// Space complexity: O(n^2) for the rank lookup table.
//
// Definition: A matching is stable if there is no blocking pair (i, j) such
// that i prefers j to their current partner AND j prefers i to their current
// partner.
//
// Preconditions:
//   - len(proposerPrefs) == len(receiverPrefs) (equal-sized sets)
//   - Each preference list is a permutation of [0, n-1]
//
// Reference: Gale, D. & Shapley, L.S. (1962) "College Admissions and the
// Stability of Marriage", American Mathematical Monthly 69(1):9-15.
func GaleShapley(proposerPrefs, receiverPrefs [][]int) []int {
	n := len(proposerPrefs)
	if n == 0 {
		return nil
	}

	// Build rank table for receivers: rank[j][i] = position of proposer i
	// in receiver j's preference list (lower is better).
	rank := make([][]int, n)
	for j := 0; j < n; j++ {
		rank[j] = make([]int, n)
		for pos, proposer := range receiverPrefs[j] {
			rank[j][proposer] = pos
		}
	}

	// matching[proposer] = receiver (-1 = unmatched)
	matching := make([]int, n)
	for i := range matching {
		matching[i] = -1
	}

	// receiverPartner[receiver] = proposer (-1 = unmatched)
	receiverPartner := make([]int, n)
	for j := range receiverPartner {
		receiverPartner[j] = -1
	}

	// nextProposal[proposer] = index into proposer's preference list
	// (next receiver to propose to)
	nextProposal := make([]int, n)

	// Free proposers queue. Initially all proposers are free.
	free := make([]int, n)
	for i := 0; i < n; i++ {
		free[i] = i
	}

	for len(free) > 0 {
		// Pick a free proposer.
		proposer := free[0]
		free = free[1:]

		if nextProposal[proposer] >= n {
			// Exhausted all proposals (shouldn't happen with valid input).
			continue
		}

		// Propose to the next receiver on the list.
		receiver := proposerPrefs[proposer][nextProposal[proposer]]
		nextProposal[proposer]++

		currentPartner := receiverPartner[receiver]
		if currentPartner == -1 {
			// Receiver is free — accept.
			matching[proposer] = receiver
			receiverPartner[receiver] = proposer
		} else if rank[receiver][proposer] < rank[receiver][currentPartner] {
			// Receiver prefers new proposer — switch.
			matching[proposer] = receiver
			receiverPartner[receiver] = proposer
			matching[currentPartner] = -1
			free = append(free, currentPartner)
		} else {
			// Receiver prefers current partner — reject.
			free = append(free, proposer)
		}
	}

	return matching
}

// IsStableMatching checks whether the given matching is stable with respect
// to the given preference lists. A matching is stable if there are no
// blocking pairs.
//
// A blocking pair (i, j) exists when:
//   - proposer i is not matched to receiver j, AND
//   - proposer i prefers receiver j to their current match, AND
//   - receiver j prefers proposer i to their current match
//
// matching[proposer] = receiver.
// Returns true if the matching is stable, false if a blocking pair exists.
//
// Time complexity: O(n^2) for checking all potential blocking pairs.
// Reference: Gale & Shapley (1962)
func IsStableMatching(matching []int, proposerPrefs, receiverPrefs [][]int) bool {
	n := len(matching)
	if n == 0 {
		return true
	}

	// Build rank tables for both sides.
	proposerRank := make([][]int, n) // proposerRank[i][j] = rank of receiver j for proposer i
	for i := 0; i < n; i++ {
		proposerRank[i] = make([]int, n)
		for pos, receiver := range proposerPrefs[i] {
			proposerRank[i][receiver] = pos
		}
	}

	receiverRank := make([][]int, n) // receiverRank[j][i] = rank of proposer i for receiver j
	for j := 0; j < n; j++ {
		receiverRank[j] = make([]int, n)
		for pos, proposer := range receiverPrefs[j] {
			receiverRank[j][proposer] = pos
		}
	}

	// Build inverse matching: receiverPartner[receiver] = proposer.
	receiverPartner := make([]int, n)
	for i := 0; i < n; i++ {
		receiverPartner[matching[i]] = i
	}

	// Check all pairs for blocking.
	for i := 0; i < n; i++ {
		currentReceiver := matching[i]
		for _, j := range proposerPrefs[i] {
			if j == currentReceiver {
				// We've reached proposer i's current match — no blocking pair
				// with any lower-ranked receiver.
				break
			}
			// Proposer i prefers receiver j to currentReceiver.
			// Check if receiver j prefers proposer i to their current partner.
			currentProposerOfJ := receiverPartner[j]
			if receiverRank[j][i] < receiverRank[j][currentProposerOfJ] {
				return false // blocking pair found
			}
		}
	}

	return true
}
