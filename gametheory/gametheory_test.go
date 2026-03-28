package gametheory

import (
	"math"
	"math/rand"
	"testing"

	"github.com/davly/reality/testutil"
)

// ---------------------------------------------------------------------------
// NashEquilibrium2x2
// ---------------------------------------------------------------------------

func TestNash2x2_MatchingPennies(t *testing.T) {
	// Zero-sum: each player mixes 50/50.
	payA := [2][2]float64{{1, -1}, {-1, 1}}
	payB := [2][2]float64{{-1, 1}, {1, -1}}
	stratA, stratB, val := NashEquilibrium2x2(payA, payB)

	assertClose(t, "stratA[0]", stratA[0], 0.5, 1e-12)
	assertClose(t, "stratA[1]", stratA[1], 0.5, 1e-12)
	assertClose(t, "stratB[0]", stratB[0], 0.5, 1e-12)
	assertClose(t, "stratB[1]", stratB[1], 0.5, 1e-12)
	assertClose(t, "value", val, 0.0, 1e-12)
}

func TestNash2x2_PrisonersDilemma(t *testing.T) {
	// Dominant strategy: both defect (row 1, col 1).
	payA := [2][2]float64{{3, 0}, {5, 1}}
	payB := [2][2]float64{{3, 5}, {0, 1}}
	stratA, stratB, val := NashEquilibrium2x2(payA, payB)

	assertClose(t, "stratA[1]", stratA[1], 1.0, 1e-12) // defect
	assertClose(t, "stratB[1]", stratB[1], 1.0, 1e-12) // defect
	assertClose(t, "value", val, 1.0, 1e-12)
}

func TestNash2x2_BattleOfSexes(t *testing.T) {
	// Mixed equilibrium: A plays (3/5, 2/5), B plays (2/5, 3/5).
	payA := [2][2]float64{{3, 0}, {0, 2}}
	payB := [2][2]float64{{2, 0}, {0, 3}}
	stratA, stratB, val := NashEquilibrium2x2(payA, payB)

	assertClose(t, "stratA[0]", stratA[0], 0.6, 1e-12)
	assertClose(t, "stratB[0]", stratB[0], 0.4, 1e-12)
	assertClose(t, "value", val, 1.2, 1e-12)
}

func TestNash2x2_DominantStrategy(t *testing.T) {
	// A: row 0 dominates. B: col 1 dominates.
	payA := [2][2]float64{{4, 3}, {1, 2}}
	payB := [2][2]float64{{1, 2}, {3, 4}}
	stratA, stratB, val := NashEquilibrium2x2(payA, payB)

	assertClose(t, "stratA[0]", stratA[0], 1.0, 1e-12)
	assertClose(t, "stratB[1]", stratB[1], 1.0, 1e-12)
	assertClose(t, "value", val, 3.0, 1e-12)
}

func TestNash2x2_CoordinationGame(t *testing.T) {
	payA := [2][2]float64{{2, 0}, {0, 1}}
	payB := [2][2]float64{{1, 0}, {0, 2}}
	stratA, stratB, val := NashEquilibrium2x2(payA, payB)

	assertClose(t, "stratA[0]", stratA[0], 2.0/3.0, 1e-10)
	assertClose(t, "stratB[0]", stratB[0], 1.0/3.0, 1e-10)
	assertClose(t, "value", val, 2.0/3.0, 1e-10)
}

func TestNash2x2_EqualPayoffs(t *testing.T) {
	payA := [2][2]float64{{5, 5}, {5, 5}}
	payB := [2][2]float64{{5, 5}, {5, 5}}
	_, _, val := NashEquilibrium2x2(payA, payB)

	assertClose(t, "value", val, 5.0, 1e-10)
}

func TestNash2x2_HawkDove(t *testing.T) {
	payA := [2][2]float64{{0, 3}, {1, 2}}
	payB := [2][2]float64{{0, 1}, {3, 2}}
	stratA, stratB, val := NashEquilibrium2x2(payA, payB)

	assertClose(t, "stratA[0]", stratA[0], 0.5, 1e-12)
	assertClose(t, "stratB[0]", stratB[0], 0.5, 1e-12)
	assertClose(t, "value", val, 1.5, 1e-12)
}

// Golden-file tests for Nash2x2.
func TestNash2x2_Golden(t *testing.T) {
	gf := testutil.LoadGolden(t, "../testdata/gametheory/nash_2x2.json")

	if gf.Function != "GameTheory.NashEquilibrium2x2" {
		t.Fatalf("golden file function = %q, want GameTheory.NashEquilibrium2x2", gf.Function)
	}

	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			payA := extractPayoff2x2(t, tc, "payoffA")
			payB := extractPayoff2x2(t, tc, "payoffB")

			stratA, stratB, val := NashEquilibrium2x2(payA, payB)

			// Expected format: [stratA[0], stratA[1], stratB[0], stratB[1], value]
			expected, ok := toFloat64Slice(tc.Expected)
			if !ok || len(expected) != 5 {
				t.Fatalf("expected must be a 5-element float64 array")
			}

			assertClose(t, "stratA[0]", stratA[0], expected[0], tc.Tolerance)
			assertClose(t, "stratA[1]", stratA[1], expected[1], tc.Tolerance)
			assertClose(t, "stratB[0]", stratB[0], expected[2], tc.Tolerance)
			assertClose(t, "stratB[1]", stratB[1], expected[3], tc.Tolerance)
			assertClose(t, "value", val, expected[4], tc.Tolerance)
		})
	}
}

// ---------------------------------------------------------------------------
// Minimax
// ---------------------------------------------------------------------------

func TestMinimax_RockPaperScissors(t *testing.T) {
	// Zero-sum symmetric game: each strategy played with probability 1/3.
	payoff := [][]float64{
		{0, -1, 1},
		{1, 0, -1},
		{-1, 1, 0},
	}
	rowStrat, colStrat, val := Minimax(payoff, 3, 3)

	// Game value should be 0 (symmetric).
	assertClose(t, "value", val, 0.0, 0.05)

	// Each strategy should be close to 1/3.
	for i, s := range rowStrat {
		if math.Abs(s-1.0/3.0) > 0.05 {
			t.Errorf("rowStrat[%d] = %v, want ~0.333", i, s)
		}
	}
	for i, s := range colStrat {
		if math.Abs(s-1.0/3.0) > 0.05 {
			t.Errorf("colStrat[%d] = %v, want ~0.333", i, s)
		}
	}
}

func TestMinimax_SingleRow(t *testing.T) {
	payoff := [][]float64{{3, 1, 4, 1, 5}}
	row, col, val := Minimax(payoff, 1, 5)

	if len(row) != 1 || row[0] != 1.0 {
		t.Errorf("row strategy should be [1.0], got %v", row)
	}
	assertClose(t, "value", val, 1.0, 1e-12)
	if col[1] != 1.0 {
		t.Errorf("column should pick index 1 (min), got %v", col)
	}
}

func TestMinimax_SingleCol(t *testing.T) {
	payoff := [][]float64{{3}, {7}, {2}}
	row, col, val := Minimax(payoff, 3, 1)

	if len(col) != 1 || col[0] != 1.0 {
		t.Errorf("col strategy should be [1.0], got %v", col)
	}
	assertClose(t, "value", val, 7.0, 1e-12)
	if row[1] != 1.0 {
		t.Errorf("row should pick index 1 (max), got %v", row)
	}
}

func TestMinimax_2x2_SaddlePoint(t *testing.T) {
	// Game with saddle point at (1, 0): row max of col min = col min of row max = 3.
	payoff := [][]float64{
		{1, 4},
		{3, 2},
	}
	_, _, val := Minimax(payoff, 2, 2)
	// Value should be between 2 and 3 for the mixed strategy.
	if val < 1.5 || val > 3.5 {
		t.Errorf("minimax value = %v, want ~2.5 range", val)
	}
}

func TestMinimax_Empty(t *testing.T) {
	row, col, _ := Minimax(nil, 0, 0)
	if row != nil || col != nil {
		t.Errorf("empty game should return nil strategies")
	}
}

// ---------------------------------------------------------------------------
// GaleShapley
// ---------------------------------------------------------------------------

func TestGaleShapley_Trivial1x1(t *testing.T) {
	proposerPrefs := [][]int{{0}}
	receiverPrefs := [][]int{{0}}
	matching := GaleShapley(proposerPrefs, receiverPrefs)

	if len(matching) != 1 || matching[0] != 0 {
		t.Errorf("1x1 matching = %v, want [0]", matching)
	}
}

func TestGaleShapley_2x2_Aligned(t *testing.T) {
	proposerPrefs := [][]int{{0, 1}, {1, 0}}
	receiverPrefs := [][]int{{0, 1}, {1, 0}}
	matching := GaleShapley(proposerPrefs, receiverPrefs)

	if matching[0] != 0 || matching[1] != 1 {
		t.Errorf("2x2 aligned matching = %v, want [0, 1]", matching)
	}
}

func TestGaleShapley_2x2_Competing(t *testing.T) {
	// Both proposers prefer receiver 0. Receiver 0 prefers proposer 0.
	proposerPrefs := [][]int{{0, 1}, {0, 1}}
	receiverPrefs := [][]int{{0, 1}, {0, 1}}
	matching := GaleShapley(proposerPrefs, receiverPrefs)

	// Proposer 0 gets receiver 0 (proposer-optimal).
	if matching[0] != 0 || matching[1] != 1 {
		t.Errorf("2x2 competing matching = %v, want [0, 1]", matching)
	}
}

func TestGaleShapley_3x3_Classic(t *testing.T) {
	proposerPrefs := [][]int{{1, 0, 2}, {0, 1, 2}, {0, 1, 2}}
	receiverPrefs := [][]int{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}}
	matching := GaleShapley(proposerPrefs, receiverPrefs)

	if matching[0] != 1 || matching[1] != 0 || matching[2] != 2 {
		t.Errorf("3x3 classic matching = %v, want [1, 0, 2]", matching)
	}
}

func TestGaleShapley_ProposerOptimal(t *testing.T) {
	// Verify that the result is proposer-optimal: no proposer can do
	// better in any other stable matching.
	proposerPrefs := [][]int{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}}
	receiverPrefs := [][]int{{2, 1, 0}, {2, 1, 0}, {2, 1, 0}}
	matching := GaleShapley(proposerPrefs, receiverPrefs)

	// Receiver 0 prefers proposer 2 most, so proposer 2 gets receiver 0.
	if matching[2] != 0 {
		t.Errorf("proposer 2 should get receiver 0, got %d", matching[2])
	}
}

func TestGaleShapley_StabilityVerification(t *testing.T) {
	proposerPrefs := [][]int{{1, 0, 2}, {0, 1, 2}, {0, 1, 2}}
	receiverPrefs := [][]int{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}}
	matching := GaleShapley(proposerPrefs, receiverPrefs)

	if !IsStableMatching(matching, proposerPrefs, receiverPrefs) {
		t.Errorf("Gale-Shapley result should be stable, but IsStableMatching returned false")
	}
}

func TestGaleShapley_Empty(t *testing.T) {
	matching := GaleShapley(nil, nil)
	if matching != nil {
		t.Errorf("empty matching should be nil, got %v", matching)
	}
}

func TestGaleShapley_4x4_ReversePrefs(t *testing.T) {
	proposerPrefs := [][]int{{3, 2, 1, 0}, {3, 2, 1, 0}, {3, 2, 1, 0}, {3, 2, 1, 0}}
	receiverPrefs := [][]int{{0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}}
	matching := GaleShapley(proposerPrefs, receiverPrefs)

	// Proposer-optimal: proposer 0 gets receiver 3 (most preferred).
	if matching[0] != 3 {
		t.Errorf("proposer 0 should get receiver 3, got %d", matching[0])
	}
	// Verify stability.
	if !IsStableMatching(matching, proposerPrefs, receiverPrefs) {
		t.Errorf("4x4 reverse matching should be stable")
	}
}

// Golden-file tests for GaleShapley.
func TestGaleShapley_Golden(t *testing.T) {
	gf := testutil.LoadGolden(t, "../testdata/gametheory/gale_shapley.json")

	if gf.Function != "GameTheory.GaleShapley" {
		t.Fatalf("golden file function = %q, want GameTheory.GaleShapley", gf.Function)
	}

	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			n := extractInt(t, tc, "n")
			proposerPrefs := extractIntMatrix(t, tc, "proposerPrefs")
			receiverPrefs := extractIntMatrix(t, tc, "receiverPrefs")

			matching := GaleShapley(proposerPrefs, receiverPrefs)

			expected := extractIntSlice(t, tc)
			if len(matching) != n {
				t.Fatalf("matching length = %d, want %d", len(matching), n)
			}
			for i := 0; i < n; i++ {
				if matching[i] != expected[i] {
					t.Errorf("matching[%d] = %d, want %d", i, matching[i], expected[i])
				}
			}

			// Also verify stability.
			if !IsStableMatching(matching, proposerPrefs, receiverPrefs) {
				t.Errorf("golden file matching should be stable")
			}
		})
	}
}

// ---------------------------------------------------------------------------
// IsStableMatching
// ---------------------------------------------------------------------------

func TestIsStableMatching_Stable(t *testing.T) {
	proposerPrefs := [][]int{{0, 1}, {1, 0}}
	receiverPrefs := [][]int{{0, 1}, {1, 0}}
	matching := []int{0, 1}

	if !IsStableMatching(matching, proposerPrefs, receiverPrefs) {
		t.Error("expected stable matching")
	}
}

func TestIsStableMatching_Unstable(t *testing.T) {
	// Swap assignments to create a blocking pair.
	proposerPrefs := [][]int{{0, 1}, {0, 1}}
	receiverPrefs := [][]int{{0, 1}, {0, 1}}
	matching := []int{1, 0} // proposer 0 gets receiver 1, but prefers 0

	if IsStableMatching(matching, proposerPrefs, receiverPrefs) {
		t.Error("expected unstable matching (blocking pair exists)")
	}
}

func TestIsStableMatching_Empty(t *testing.T) {
	if !IsStableMatching(nil, nil, nil) {
		t.Error("empty matching should be considered stable")
	}
}

// ---------------------------------------------------------------------------
// UCB1
// ---------------------------------------------------------------------------

func TestUCB1_UnexploredArm(t *testing.T) {
	counts := []int{10, 0, 5}
	rewards := []float64{8.0, 0.0, 4.0}
	got := UCB1(counts, rewards, 15)

	if got != 1 {
		t.Errorf("UCB1 should select unexplored arm 1, got %d", got)
	}
}

func TestUCB1_AllUnexplored(t *testing.T) {
	counts := []int{0, 0, 0}
	rewards := []float64{0, 0, 0}
	got := UCB1(counts, rewards, 0)

	if got != 0 {
		t.Errorf("UCB1 should select first unexplored arm 0, got %d", got)
	}
}

func TestUCB1_ExploitsHighReward(t *testing.T) {
	// All arms well-explored. Arm 0 has much higher average reward.
	counts := []int{1000, 1000, 1000}
	rewards := []float64{900.0, 100.0, 100.0}
	got := UCB1(counts, rewards, 3000)

	if got != 0 {
		t.Errorf("UCB1 should exploit arm 0, got %d", got)
	}
}

func TestUCB1_ExplorationBoost(t *testing.T) {
	// Arm 2 has low count — exploration bonus should make it attractive.
	counts := []int{100, 100, 1}
	rewards := []float64{50.0, 50.0, 0.4}
	got := UCB1(counts, rewards, 201)

	if got != 2 {
		t.Errorf("UCB1 should explore arm 2 (low count), got %d", got)
	}
}

func TestUCB1_Empty(t *testing.T) {
	if UCB1(nil, nil, 0) != -1 {
		t.Error("UCB1 with no arms should return -1")
	}
}

// ---------------------------------------------------------------------------
// ThompsonSampling
// ---------------------------------------------------------------------------

func TestThompsonSampling_BestArm(t *testing.T) {
	// Arm 0 has overwhelming evidence of being best.
	successes := []int{1000, 10, 10}
	failures := []int{10, 1000, 1000}
	rng := rand.New(rand.NewSource(42))

	// Run many times — arm 0 should be selected most often.
	counts := make([]int, 3)
	for i := 0; i < 1000; i++ {
		arm := ThompsonSampling(successes, failures, rng)
		counts[arm]++
	}

	if counts[0] < 900 {
		t.Errorf("arm 0 should be selected ~1000 times, got %d", counts[0])
	}
}

func TestThompsonSampling_Uniform(t *testing.T) {
	// All arms have same prior — should be roughly uniform.
	successes := []int{1, 1, 1}
	failures := []int{1, 1, 1}
	rng := rand.New(rand.NewSource(42))

	counts := make([]int, 3)
	for i := 0; i < 3000; i++ {
		arm := ThompsonSampling(successes, failures, rng)
		counts[arm]++
	}

	// Each arm should be selected roughly 1000 times (within wide margin).
	for i, c := range counts {
		if c < 500 || c > 1500 {
			t.Errorf("arm %d selected %d times, expected ~1000", i, c)
		}
	}
}

func TestThompsonSampling_Empty(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	if ThompsonSampling(nil, nil, rng) != -1 {
		t.Error("Thompson with no arms should return -1")
	}
}

// ---------------------------------------------------------------------------
// EpsilonGreedy
// ---------------------------------------------------------------------------

func TestEpsilonGreedy_Exploit(t *testing.T) {
	rewards := []float64{10.0, 5.0, 1.0}
	counts := []int{10, 10, 10}
	// epsilon = 0 means always exploit.
	rng := rand.New(rand.NewSource(42))
	got := EpsilonGreedy(rewards, counts, 0, rng)

	if got != 0 {
		t.Errorf("EpsilonGreedy(eps=0) should exploit arm 0, got %d", got)
	}
}

func TestEpsilonGreedy_Explore(t *testing.T) {
	rewards := []float64{10.0, 5.0, 1.0}
	counts := []int{10, 10, 10}
	rng := rand.New(rand.NewSource(42))

	// epsilon = 1 means always explore.
	exploreCounts := make([]int, 3)
	for i := 0; i < 3000; i++ {
		arm := EpsilonGreedy(rewards, counts, 1.0, rng)
		exploreCounts[arm]++
	}

	// All arms should be explored roughly equally.
	for i, c := range exploreCounts {
		if c < 500 || c > 1500 {
			t.Errorf("arm %d explored %d times, expected ~1000", i, c)
		}
	}
}

func TestEpsilonGreedy_Empty(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	if EpsilonGreedy(nil, nil, 0.1, rng) != -1 {
		t.Error("EpsilonGreedy with no arms should return -1")
	}
}

func TestEpsilonGreedy_ZeroCounts(t *testing.T) {
	rewards := []float64{0, 0, 0}
	counts := []int{0, 0, 0}
	rng := rand.New(rand.NewSource(42))
	got := EpsilonGreedy(rewards, counts, 0, rng)

	// All averages are 0, should pick arm 0 (first max).
	if got != 0 {
		t.Errorf("EpsilonGreedy with zero counts should pick arm 0, got %d", got)
	}
}

// ---------------------------------------------------------------------------
// BanzhafIndex
// ---------------------------------------------------------------------------

func TestBanzhaf_3_2_1_Quota4(t *testing.T) {
	// Classic example: [3, 2, 1] with quota 4.
	// Winning coalitions: {3,2}, {3,1}, {3,2,1}
	// Swings: voter 0 (w=3): critical in {3,2}, {3,1}, {3,2,1} = 3
	//         voter 1 (w=2): critical in {3,2} = 1
	//         voter 2 (w=1): critical in {3,1} = 1
	// Total swings = 5. Banzhaf: [3/5, 1/5, 1/5]
	idx := BanzhafIndex([]float64{3, 2, 1}, 4)

	assertClose(t, "voter 0", idx[0], 3.0/5.0, 1e-12)
	assertClose(t, "voter 1", idx[1], 1.0/5.0, 1e-12)
	assertClose(t, "voter 2", idx[2], 1.0/5.0, 1e-12)
}

func TestBanzhaf_EqualWeights(t *testing.T) {
	// Equal weights: all voters have equal power.
	idx := BanzhafIndex([]float64{1, 1, 1}, 2)

	for i, v := range idx {
		assertClose(t, "voter", v, 1.0/3.0, 1e-12)
		_ = i
	}
}

func TestBanzhaf_Dictator(t *testing.T) {
	// One voter has enough weight alone — they are the dictator.
	idx := BanzhafIndex([]float64{10, 1, 1}, 10)

	assertClose(t, "dictator", idx[0], 1.0, 1e-12)
	assertClose(t, "dummy 1", idx[1], 0.0, 1e-12)
	assertClose(t, "dummy 2", idx[2], 0.0, 1e-12)
}

func TestBanzhaf_UNSecurityCouncil(t *testing.T) {
	// Simplified UN Security Council: 5 permanent members (veto power)
	// need all 5 + at least 4 of 10 non-permanent members.
	// We model with 15 voters, but use a simpler approximation.
	// Full UN model is complex — test the 5-voter veto model.
	//
	// 5 voters with veto power: quota requires all 5.
	// Each veto member has weight 1, quota = 5.
	// Only the grand coalition wins. Every member is critical.
	idx := BanzhafIndex([]float64{1, 1, 1, 1, 1}, 5)

	for i, v := range idx {
		assertClose(t, "veto member", v, 0.2, 1e-12)
		_ = i
	}
}

func TestBanzhaf_DummyVoter(t *testing.T) {
	// Voter with weight 0 is never critical.
	idx := BanzhafIndex([]float64{5, 3, 0}, 5)

	// Voter 2 (w=0) is never critical.
	assertClose(t, "dummy", idx[2], 0.0, 1e-12)
}

func TestBanzhaf_Empty(t *testing.T) {
	if BanzhafIndex(nil, 5) != nil {
		t.Error("BanzhafIndex with no voters should return nil")
	}
}

func TestBanzhaf_SumsToOne(t *testing.T) {
	idx := BanzhafIndex([]float64{4, 3, 2, 1}, 6)

	sum := 0.0
	for _, v := range idx {
		sum += v
	}
	assertClose(t, "sum", sum, 1.0, 1e-12)
}

// ---------------------------------------------------------------------------
// ShapleyValue
// ---------------------------------------------------------------------------

func TestShapley_SymmetricGame(t *testing.T) {
	// Symmetric game: v(S) = |S|. All players have equal Shapley value.
	n := 3
	charFunc := func(coalition []bool) float64 {
		count := 0
		for _, in := range coalition {
			if in {
				count++
			}
		}
		return float64(count)
	}

	values := ShapleyValue(n, charFunc)
	for i, v := range values {
		assertClose(t, "player", v, 1.0, 1e-12)
		_ = i
	}
}

func TestShapley_Dictator(t *testing.T) {
	// Dictator game: v(S) = 1 if player 0 is in S, else 0.
	n := 3
	charFunc := func(coalition []bool) float64 {
		if coalition[0] {
			return 1
		}
		return 0
	}

	values := ShapleyValue(n, charFunc)
	assertClose(t, "dictator", values[0], 1.0, 1e-12)
	assertClose(t, "dummy 1", values[1], 0.0, 1e-12)
	assertClose(t, "dummy 2", values[2], 0.0, 1e-12)
}

func TestShapley_Unanimity(t *testing.T) {
	// Unanimity game: v(S) = 1 only if S = N (grand coalition).
	// Shapley value: 1/n for each player.
	n := 4
	charFunc := func(coalition []bool) float64 {
		for _, in := range coalition {
			if !in {
				return 0
			}
		}
		return 1
	}

	values := ShapleyValue(n, charFunc)
	for i, v := range values {
		assertClose(t, "player", v, 0.25, 1e-12)
		_ = i
	}
}

func TestShapley_SumsToGrandCoalition(t *testing.T) {
	// Efficiency axiom: sum of Shapley values = v(N).
	n := 4
	charFunc := func(coalition []bool) float64 {
		count := 0
		for _, in := range coalition {
			if in {
				count++
			}
		}
		return float64(count * count) // superadditive
	}

	values := ShapleyValue(n, charFunc)
	sum := 0.0
	for _, v := range values {
		sum += v
	}

	// v(grand coalition) = 4^2 = 16
	assertClose(t, "sum", sum, 16.0, 1e-10)
}

func TestShapley_WeightedVoting(t *testing.T) {
	// [3, 2, 1] with quota 4. Shapley-Shubik index.
	values := ShapleyValueWeightedVoting([]float64{3, 2, 1}, 4)

	// Expected: voter 0 = 4/6, voter 1 = 1/6, voter 2 = 1/6
	assertClose(t, "voter 0", values[0], 4.0/6.0, 1e-12)
	assertClose(t, "voter 1", values[1], 1.0/6.0, 1e-12)
	assertClose(t, "voter 2", values[2], 1.0/6.0, 1e-12)
}

func TestShapley_Empty(t *testing.T) {
	if ShapleyValue(0, nil) != nil {
		t.Error("ShapleyValue(0) should return nil")
	}
}

// ---------------------------------------------------------------------------
// KellyFraction
// ---------------------------------------------------------------------------

func TestKelly_FairCoin(t *testing.T) {
	// Fair coin, even odds: f* = (0.5*1 - 0.5)/1 = 0.
	got := KellyFraction(0.5, 1.0)
	assertClose(t, "fair coin", got, 0.0, 1e-12)
}

func TestKelly_BiasedCoin(t *testing.T) {
	// p=0.6, b=1: f* = (0.6*1 - 0.4)/1 = 0.2
	got := KellyFraction(0.6, 1.0)
	assertClose(t, "biased coin", got, 0.2, 1e-12)
}

func TestKelly_HighOdds(t *testing.T) {
	// p=0.3, b=3: f* = (0.3*3 - 0.7)/3 = (0.9 - 0.7)/3 = 0.2/3
	got := KellyFraction(0.3, 3.0)
	assertClose(t, "high odds", got, 0.2/3.0, 1e-12)
}

func TestKelly_NegativeEdge(t *testing.T) {
	// p=0.3, b=1: f* = (0.3 - 0.7)/1 = -0.4 (don't bet, or bet against).
	got := KellyFraction(0.3, 1.0)
	assertClose(t, "negative edge", got, -0.4, 1e-12)
}

func TestKelly_Certainty(t *testing.T) {
	// p=1.0: degenerate, returns 0.
	got := KellyFraction(1.0, 2.0)
	assertClose(t, "certainty", got, 0.0, 1e-12)
}

func TestKelly_ZeroProb(t *testing.T) {
	got := KellyFraction(0.0, 2.0)
	assertClose(t, "zero prob", got, 0.0, 1e-12)
}

func TestKelly_ZeroOdds(t *testing.T) {
	got := KellyFraction(0.5, 0.0)
	assertClose(t, "zero odds", got, 0.0, 1e-12)
}

func TestKelly_SureBet(t *testing.T) {
	// p=0.9, b=10: f* = (0.9*10 - 0.1)/10 = 8.9/10 = 0.89
	got := KellyFraction(0.9, 10.0)
	assertClose(t, "sure bet", got, 0.89, 1e-12)
}

// ---------------------------------------------------------------------------
// KellyFractionMultiple
// ---------------------------------------------------------------------------

func TestKellyMultiple_Independent(t *testing.T) {
	probs := []float64{0.6, 0.55}
	odds := []float64{1.0, 1.0}
	fractions := KellyFractionMultiple(probs, odds)

	// Individual Kelly: 0.2 and 0.1, total = 0.3 < 1, no scaling needed.
	assertClose(t, "bet 0", fractions[0], 0.2, 1e-12)
	assertClose(t, "bet 1", fractions[1], 0.1, 1e-12)
}

func TestKellyMultiple_NeedsScaling(t *testing.T) {
	// Many positive bets that total > 1.
	probs := []float64{0.9, 0.9, 0.9}
	odds := []float64{10.0, 10.0, 10.0}
	fractions := KellyFractionMultiple(probs, odds)

	// Individual Kelly: 0.89 each, total = 2.67. Scaled to sum <= 1.
	total := 0.0
	for _, f := range fractions {
		total += f
	}
	if total > 1.0+1e-12 {
		t.Errorf("total allocation = %v, should be <= 1.0", total)
	}
}

func TestKellyMultiple_MismatchLength(t *testing.T) {
	if KellyFractionMultiple([]float64{0.5}, []float64{1.0, 2.0}) != nil {
		t.Error("mismatched lengths should return nil")
	}
}

func TestKellyMultiple_Empty(t *testing.T) {
	if KellyFractionMultiple(nil, nil) != nil {
		t.Error("empty input should return nil")
	}
}

// ---------------------------------------------------------------------------
// KellyGrowthRate
// ---------------------------------------------------------------------------

func TestKellyGrowthRate_OptimalFraction(t *testing.T) {
	// At the optimal Kelly fraction, growth rate should be maximized.
	prob := 0.6
	odds := 1.0
	optF := KellyFraction(prob, odds) // 0.2

	optGrowth := KellyGrowthRate(prob, odds, optF)
	halfGrowth := KellyGrowthRate(prob, odds, optF/2)
	doubleGrowth := KellyGrowthRate(prob, odds, optF*2)

	if optGrowth <= halfGrowth {
		t.Errorf("optimal growth (%v) should exceed half-Kelly growth (%v)", optGrowth, halfGrowth)
	}
	if optGrowth <= doubleGrowth {
		t.Errorf("optimal growth (%v) should exceed double-Kelly growth (%v)", optGrowth, doubleGrowth)
	}
}

func TestKellyGrowthRate_ZeroBet(t *testing.T) {
	got := KellyGrowthRate(0.6, 1.0, 0.0)
	assertClose(t, "zero bet growth", got, 0.0, 1e-12)
}

func TestKellyGrowthRate_InvalidProb(t *testing.T) {
	if !math.IsNaN(KellyGrowthRate(0.0, 1.0, 0.5)) {
		t.Error("invalid prob should return NaN")
	}
}

func TestKellyGrowthRate_Ruin(t *testing.T) {
	// Bet everything (f=1) on a non-certain outcome: guaranteed ruin on loss.
	got := KellyGrowthRate(0.5, 1.0, 1.0)
	if !math.IsNaN(got) && !math.IsInf(got, -1) {
		t.Errorf("full bet with p<1 should be -Inf or NaN, got %v", got)
	}
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func assertClose(t *testing.T, name string, got, want, tol float64) {
	t.Helper()
	if math.Abs(got-want) > tol {
		t.Errorf("%s = %v, want %v (diff %v > tol %v)",
			name, got, want, math.Abs(got-want), tol)
	}
}

// extractPayoff2x2 pulls a [2][2]float64 from a golden file test case.
func extractPayoff2x2(t *testing.T, tc testutil.TestCase, key string) [2][2]float64 {
	t.Helper()
	val, exists := tc.Inputs[key]
	if !exists {
		t.Fatalf("missing input %q", key)
	}
	rows, ok := val.([]any)
	if !ok || len(rows) != 2 {
		t.Fatalf("input %q must be a 2x2 array", key)
	}
	var result [2][2]float64
	for i, row := range rows {
		cols, ok := row.([]any)
		if !ok || len(cols) != 2 {
			t.Fatalf("input %q row %d must have 2 elements", key, i)
		}
		for j, v := range cols {
			f, ok := v.(float64)
			if !ok {
				t.Fatalf("input %q[%d][%d] not a float64", key, i, j)
			}
			result[i][j] = f
		}
	}
	return result
}

// extractInt pulls an integer from a golden file test case.
func extractInt(t *testing.T, tc testutil.TestCase, key string) int {
	t.Helper()
	val, exists := tc.Inputs[key]
	if !exists {
		t.Fatalf("missing input %q", key)
	}
	f, ok := val.(float64)
	if !ok {
		t.Fatalf("input %q not a number", key)
	}
	return int(f)
}

// extractIntMatrix pulls a [][]int from a golden file test case.
func extractIntMatrix(t *testing.T, tc testutil.TestCase, key string) [][]int {
	t.Helper()
	val, exists := tc.Inputs[key]
	if !exists {
		t.Fatalf("missing input %q", key)
	}
	rows, ok := val.([]any)
	if !ok {
		t.Fatalf("input %q must be a 2D array", key)
	}
	result := make([][]int, len(rows))
	for i, row := range rows {
		cols, ok := row.([]any)
		if !ok {
			t.Fatalf("input %q row %d must be an array", key, i)
		}
		result[i] = make([]int, len(cols))
		for j, v := range cols {
			f, ok := v.(float64)
			if !ok {
				t.Fatalf("input %q[%d][%d] not a number", key, i, j)
			}
			result[i][j] = int(f)
		}
	}
	return result
}

// extractIntSlice pulls a []int from the expected field of a golden file test case.
func extractIntSlice(t *testing.T, tc testutil.TestCase) []int {
	t.Helper()
	arr, ok := tc.Expected.([]any)
	if !ok {
		t.Fatalf("expected must be an int array")
	}
	result := make([]int, len(arr))
	for i, v := range arr {
		f, ok := v.(float64)
		if !ok {
			t.Fatalf("expected[%d] not a number", i)
		}
		result[i] = int(f)
	}
	return result
}

// toFloat64Slice converts a JSON-decoded value to []float64.
func toFloat64Slice(v any) ([]float64, bool) {
	arr, ok := v.([]any)
	if !ok {
		return nil, false
	}
	result := make([]float64, len(arr))
	for i, elem := range arr {
		f, ok := elem.(float64)
		if !ok {
			return nil, false
		}
		result[i] = f
	}
	return result, true
}
