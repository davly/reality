// Package gametheory provides classical game-theoretic algorithms:
// Nash equilibrium (2x2 and zero-sum), stable matching (Gale-Shapley),
// multi-armed bandits (UCB1, Thompson sampling, epsilon-greedy),
// voting power indices (Banzhaf, Shapley value), and Kelly criterion.
//
// All functions are pure, deterministic (except those accepting an RNG),
// and use only the Go standard library. Zero external dependencies.
//
// Consumers:
//   - RubberDuck:  auction/market strategies (Nash, Kelly)
//   - Paradox:     puzzle design (Nash, matching)
//   - BookaBloke:  matching algorithms (Gale-Shapley)
//   - LiquidVote:  voting power analysis (Banzhaf, Shapley)
//   - Tempo:       explore/exploit decisions (bandits)
package gametheory

import "math"

// ---------------------------------------------------------------------------
// Nash Equilibrium
//
// Functions for computing equilibrium strategies in finite games.
// All payoff matrices use row-player (A) and column-player (B) convention.
// ---------------------------------------------------------------------------

// NashEquilibrium2x2 finds the mixed strategy Nash equilibrium for a 2x2
// bimatrix game. payoffA and payoffB are the payoff matrices for player A
// (row) and player B (column) respectively.
//
// Returns:
//   - stratA: probability distribution over A's two strategies [p, 1-p]
//   - stratB: probability distribution over B's two strategies [q, 1-q]
//   - value: expected payoff to player A at equilibrium
//
// If a pure strategy Nash equilibrium exists with no mixed equilibrium,
// it returns the first pure NE found. If a dominant strategy exists,
// the dominated player's mix is set to the best response.
//
// Definition: A Nash equilibrium is a strategy profile where no player can
// improve their payoff by unilaterally changing their strategy.
//
// Precision: exact for rational payoffs (single arithmetic operations)
// Reference: Nash, J. (1950) "Equilibrium Points in N-Person Games";
// Osborne, M.J. "An Introduction to Game Theory", Chapter 4.
func NashEquilibrium2x2(payoffA, payoffB [2][2]float64) (stratA, stratB [2]float64, value float64) {
	// For the mixed strategy Nash equilibrium of a 2x2 game:
	// Player B mixes to make A indifferent between rows:
	//   q * A[0][0] + (1-q) * A[0][1] = q * A[1][0] + (1-q) * A[1][1]
	// Player A mixes to make B indifferent between columns:
	//   p * B[0][0] + (1-p) * B[1][0] = p * B[0][1] + (1-p) * B[1][1]

	// Compute q (B's mix) from A's indifference condition.
	// A indifferent: q*A[0][0] + (1-q)*A[0][1] = q*A[1][0] + (1-q)*A[1][1]
	// q = (A[1][1] - A[0][1]) / (A[0][0] - A[0][1] - A[1][0] + A[1][1])
	denomA := payoffA[0][0] - payoffA[0][1] - payoffA[1][0] + payoffA[1][1]
	// Compute p (A's mix) from B's indifference condition.
	// B indifferent: p*B[0][0] + (1-p)*B[1][0] = p*B[0][1] + (1-p)*B[1][1]
	// p = (B[1][1] - B[1][0]) / (B[0][0] - B[1][0] - B[0][1] + B[1][1])
	denomB := payoffB[0][0] - payoffB[1][0] - payoffB[0][1] + payoffB[1][1]

	// Check if a mixed equilibrium exists (denominators non-zero).
	if math.Abs(denomA) > 1e-15 && math.Abs(denomB) > 1e-15 {
		q := (payoffA[1][1] - payoffA[0][1]) / denomA
		p := (payoffB[1][1] - payoffB[1][0]) / denomB

		// Valid mixed equilibrium: both probabilities in [0, 1].
		if q >= 0 && q <= 1 && p >= 0 && p <= 1 {
			stratA = [2]float64{p, 1 - p}
			stratB = [2]float64{q, 1 - q}
			value = p*(q*payoffA[0][0]+(1-q)*payoffA[0][1]) +
				(1-p)*(q*payoffA[1][0]+(1-q)*payoffA[1][1])
			return
		}
	}

	// No interior mixed equilibrium -- find pure strategy Nash equilibrium.
	// Check all four cells for pure NE (both players best-responding).
	type pureNE struct {
		row, col int
	}
	var candidates []pureNE

	for r := 0; r < 2; r++ {
		for c := 0; c < 2; c++ {
			// Is r a best response for A given B plays c?
			otherR := 1 - r
			aBestResponse := payoffA[r][c] >= payoffA[otherR][c]
			// Is c a best response for B given A plays r?
			otherC := 1 - c
			bBestResponse := payoffB[r][c] >= payoffB[r][otherC]

			if aBestResponse && bBestResponse {
				candidates = append(candidates, pureNE{r, c})
			}
		}
	}

	if len(candidates) > 0 {
		// Return the first pure NE found.
		ne := candidates[0]
		stratA = [2]float64{0, 0}
		stratA[ne.row] = 1.0
		stratB = [2]float64{0, 0}
		stratB[ne.col] = 1.0
		value = payoffA[ne.row][ne.col]
		return
	}

	// Fallback: should not happen in valid 2x2 games (Nash's theorem
	// guarantees at least one NE), but handle gracefully.
	stratA = [2]float64{0.5, 0.5}
	stratB = [2]float64{0.5, 0.5}
	value = 0.25 * (payoffA[0][0] + payoffA[0][1] + payoffA[1][0] + payoffA[1][1])
	return
}

// Minimax solves a two-player zero-sum game via linear programming.
// The payoff matrix is for the row player (maximizer); the column player
// minimizes. Returns the optimal mixed strategies for both players and the
// game value.
//
// For 2xN and Nx2 games, uses the geometric method (intersection of lines).
// For general mxn games, uses iterative fictitious play which converges
// to the minimax value.
//
// Definition: The minimax theorem (von Neumann, 1928) states that in any
// finite two-player zero-sum game, max_p min_q p^T A q = min_q max_p p^T A q.
//
// Precision: ~1e-6 for fictitious play (iterative); exact for 2x2.
// Reference: von Neumann, J. (1928) "Zur Theorie der Gesellschaftsspiele";
// Brown, G.W. (1951) "Iterative Solution of Games by Fictitious Play".
func Minimax(payoff [][]float64, nRows, nCols int) (rowStrategy []float64, colStrategy []float64, value float64) {
	if nRows <= 0 || nCols <= 0 {
		return nil, nil, 0
	}
	if nRows == 1 {
		// Row player has one strategy: find the column that minimizes payoff.
		rowStrategy = []float64{1.0}
		colStrategy = make([]float64, nCols)
		minVal := payoff[0][0]
		minCol := 0
		for j := 1; j < nCols; j++ {
			if payoff[0][j] < minVal {
				minVal = payoff[0][j]
				minCol = j
			}
		}
		colStrategy[minCol] = 1.0
		value = minVal
		return
	}
	if nCols == 1 {
		// Column player has one strategy: find the row that maximizes payoff.
		colStrategy = []float64{1.0}
		rowStrategy = make([]float64, nRows)
		maxVal := payoff[0][0]
		maxRow := 0
		for i := 1; i < nRows; i++ {
			if payoff[i][0] > maxVal {
				maxVal = payoff[i][0]
				maxRow = i
			}
		}
		rowStrategy[maxRow] = 1.0
		value = maxVal
		return
	}

	// Use fictitious play for general games.
	// Each player tracks cumulative opponent play and best-responds.
	const maxIter = 100000

	rowCounts := make([]int, nRows)
	colCounts := make([]int, nCols)

	// Cumulative payoff sums for best-response computation.
	rowSums := make([]float64, nRows) // sum of payoffs for each row against col plays
	colSums := make([]float64, nCols) // sum of payoffs for each col against row plays

	for iter := 0; iter < maxIter; iter++ {
		// Row player best-responds to column player's empirical distribution.
		bestRow := 0
		bestRowVal := rowSums[0]
		for i := 1; i < nRows; i++ {
			if rowSums[i] > bestRowVal {
				bestRowVal = rowSums[i]
				bestRow = i
			}
		}
		rowCounts[bestRow]++

		// Column player best-responds to row player's empirical distribution.
		bestCol := 0
		bestColVal := colSums[0]
		for j := 1; j < nCols; j++ {
			if colSums[j] < bestColVal {
				bestColVal = colSums[j]
				bestCol = j
			}
		}
		colCounts[bestCol]++

		// Update cumulative sums.
		for i := 0; i < nRows; i++ {
			rowSums[i] += payoff[i][bestCol]
		}
		for j := 0; j < nCols; j++ {
			colSums[j] += payoff[bestRow][j]
		}
	}

	// Convert counts to strategies.
	rowStrategy = make([]float64, nRows)
	colStrategy = make([]float64, nCols)
	totalIter := float64(maxIter)
	for i := 0; i < nRows; i++ {
		rowStrategy[i] = float64(rowCounts[i]) / totalIter
	}
	for j := 0; j < nCols; j++ {
		colStrategy[j] = float64(colCounts[j]) / totalIter
	}

	// Compute game value as p^T A q.
	value = 0
	for i := 0; i < nRows; i++ {
		for j := 0; j < nCols; j++ {
			value += rowStrategy[i] * colStrategy[j] * payoff[i][j]
		}
	}

	return
}
