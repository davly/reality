package combinatorics

// ---------------------------------------------------------------------------
// Generation Functions
// ---------------------------------------------------------------------------

// GeneratePermutations returns all permutations of the given slice using
// Heap's algorithm. The input slice is not modified; all permutations are
// returned as newly-allocated slices.
//
// The number of results is n! where n = len(items). For n > 10 this will
// produce over 3.6 million results — caller beware.
//
// Algorithm: Heap's algorithm generates each permutation from the previous
// one by a single swap, visiting all n! permutations exactly once.
// Time complexity: O(n * n!) — n! permutations, each copied in O(n).
// Reference: Heap, B.R. (1963) "Permutations by Interchanges"
func GeneratePermutations(items []int) [][]int {
	n := len(items)
	if n == 0 {
		return nil
	}

	// Working copy to permute in place.
	work := make([]int, n)
	copy(work, items)

	var result [][]int
	snapshot := func() {
		perm := make([]int, n)
		copy(perm, work)
		result = append(result, perm)
	}

	// Heap's algorithm (iterative).
	c := make([]int, n)
	snapshot()

	i := 0
	for i < n {
		if c[i] < i {
			if i%2 == 0 {
				work[0], work[i] = work[i], work[0]
			} else {
				work[c[i]], work[i] = work[i], work[c[i]]
			}
			snapshot()
			c[i]++
			i = 0
		} else {
			c[i] = 0
			i++
		}
	}

	return result
}

// GenerateCombinations returns all k-element combinations of {0, 1, ..., n-1}
// in lexicographic order. Returns nil if k < 0, k > n, or n < 0.
//
// The number of results is C(n, k). Each combination is a sorted slice of
// k distinct integers from [0, n).
//
// Algorithm: iterative generation of combinations in lexicographic order.
// Time complexity: O(k * C(n,k)) — C(n,k) combinations, each built in O(k).
// Reference: Knuth, TAOCP vol. 4A, Algorithm T
func GenerateCombinations(n, k int) [][]int {
	if k < 0 || k > n || n < 0 {
		return nil
	}
	if k == 0 {
		return [][]int{{}}
	}

	var result [][]int
	combo := make([]int, k)
	for i := range combo {
		combo[i] = i
	}

	for {
		// Snapshot current combination.
		c := make([]int, k)
		copy(c, combo)
		result = append(result, c)

		// Find rightmost element that can be incremented.
		i := k - 1
		for i >= 0 && combo[i] == n-k+i {
			i--
		}
		if i < 0 {
			break
		}

		// Increment and reset all elements to the right.
		combo[i]++
		for j := i + 1; j < k; j++ {
			combo[j] = combo[j-1] + 1
		}
	}

	return result
}

// NextPermutation rearranges perm into the next lexicographically greater
// permutation, returning true. If perm is already the last permutation
// (fully descending), it returns false without modifying perm.
//
// Algorithm: standard "next permutation" algorithm.
// 1. Find largest i such that perm[i] < perm[i+1]. If none, this is the
//    last permutation.
// 2. Find largest j > i such that perm[j] > perm[i].
// 3. Swap perm[i] and perm[j].
// 4. Reverse perm[i+1:].
//
// Time complexity: O(n) amortised over all permutations.
// Reference: Knuth, TAOCP vol. 4A, Algorithm L; also Dijkstra (1976)
func NextPermutation(perm []int) bool {
	n := len(perm)
	if n <= 1 {
		return false
	}

	// Step 1: find largest i with perm[i] < perm[i+1].
	i := n - 2
	for i >= 0 && perm[i] >= perm[i+1] {
		i--
	}
	if i < 0 {
		return false
	}

	// Step 2: find largest j > i with perm[j] > perm[i].
	j := n - 1
	for perm[j] <= perm[i] {
		j--
	}

	// Step 3: swap.
	perm[i], perm[j] = perm[j], perm[i]

	// Step 4: reverse perm[i+1:].
	lo, hi := i+1, n-1
	for lo < hi {
		perm[lo], perm[hi] = perm[hi], perm[lo]
		lo++
		hi--
	}

	return true
}

// RandomSubset returns a random k-element subset of {0, 1, ..., n-1} using
// a partial Fisher-Yates shuffle. The result is not sorted.
// Returns nil if k < 0, k > n, or n < 0.
//
// The rng parameter must implement Intn(n int) int, returning a uniform
// random integer in [0, n). This allows the caller to supply any PRNG
// (e.g. math/rand.Rand).
//
// Algorithm: partial Fisher-Yates shuffle — shuffle only the first k
// elements, then return them.
// Time complexity: O(k) swaps + O(n) for the initial array.
// Reference: Fisher, Yates (1938); Knuth, TAOCP vol. 2
func RandomSubset(n, k int, rng interface{ Intn(int) int }) []int {
	if k < 0 || k > n || n < 0 {
		return nil
	}
	if k == 0 {
		return []int{}
	}

	// Build [0, 1, ..., n-1].
	pool := make([]int, n)
	for i := range pool {
		pool[i] = i
	}

	// Partial Fisher-Yates: shuffle first k positions.
	for i := 0; i < k; i++ {
		j := i + rng.Intn(n-i)
		pool[i], pool[j] = pool[j], pool[i]
	}

	result := make([]int, k)
	copy(result, pool[:k])
	return result
}
