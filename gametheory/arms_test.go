package gametheory

import (
	"math"
	"math/rand"
	"testing"
)

// ---------------------------------------------------------------------------
// THE DOMINANCE TEST: sum-contract vs mean-misuse.
//
// Construction: the misuse (passing per-arm MEANS where cumulative SUMS are
// expected) makes UCB1's internal average mean/Count instead of mean, which
// collapses the exploitation term toward zero and leaves selection driven
// almost purely by the exploration bonus (smaller Count wins). So we pick
// arms where correct semantics and pure-exploration disagree:
//
//	armA: Count=100, RewardSum=90  (true mean 0.9 — clearly best)
//	armB: Count=50,  RewardSum=5   (true mean 0.1)
//
// Correct (sum contract), totalPulls=150, ln(150)≈5.0106:
//	score(A) = 0.9 + sqrt(2·ln150/100) ≈ 0.9 + 0.3166 = 1.2166
//	score(B) = 0.1 + sqrt(2·ln150/50)  ≈ 0.1 + 0.4477 = 0.5477  → picks A
//
// Misuse (means fed as rewards): rewards=[0.9, 0.1]
//	score(A) = 0.9/100 + 0.3166 = 0.3256
//	score(B) = 0.1/50  + 0.4477 = 0.4497                        → picks B
//
// If a future change ever regresses toward mean-passing, these tests fail
// loudly.
// ---------------------------------------------------------------------------

var dominanceArms = []Arm{
	{Count: 100, RewardSum: 90}, // arm 0: true mean 0.9
	{Count: 50, RewardSum: 5},   // arm 1: true mean 0.1
}

func TestUCB1FromArms_SumContractDominance(t *testing.T) {
	got := UCB1FromArms(dominanceArms)
	if got != 0 {
		t.Fatalf("UCB1FromArms(sum contract) = %d, want 0 (arm with true mean 0.9 must beat true mean 0.1 at these counts)", got)
	}
}

func TestUCB1_MeanMisuseSelectsDifferently(t *testing.T) {
	counts := []int{dominanceArms[0].Count, dominanceArms[1].Count}
	totalPulls := counts[0] + counts[1]

	// The misuse: feed per-arm MEANS into the rewards parameter that the
	// contract (bandit.go: "total cumulative reward") requires to be SUMS.
	means := []float64{
		dominanceArms[0].RewardSum / float64(dominanceArms[0].Count), // 0.9
		dominanceArms[1].RewardSum / float64(dominanceArms[1].Count), // 0.1
	}
	misuseGot := UCB1(counts, means, totalPulls)
	correctGot := UCB1FromArms(dominanceArms)

	if misuseGot == correctGot {
		t.Fatalf("mean-misuse selected the same arm (%d) as the sum contract; the dominance construction must make them disagree", misuseGot)
	}
	if misuseGot != 1 {
		t.Fatalf("mean-misuse UCB1 = %d, want 1 (exploitation collapses, exploration bonus of the low-count arm dominates)", misuseGot)
	}

	// Document the collapse of the exploitation term with assertions: under
	// misuse, the internal average for arm 0 becomes mean/Count = 0.009,
	// two orders of magnitude below its true mean of 0.9.
	misuseAvgA := means[0] / float64(counts[0])
	trueMeanA := dominanceArms[0].RewardSum / float64(dominanceArms[0].Count)
	if misuseAvgA >= 0.01 {
		t.Fatalf("expected misuse to collapse arm 0's exploitation term below 0.01, got %g", misuseAvgA)
	}
	if trueMeanA != 0.9 {
		t.Fatalf("true mean of arm 0 = %g, want 0.9", trueMeanA)
	}

	// And verify the score arithmetic the comment block above claims.
	logTotal := math.Log(float64(totalPulls))
	correctScoreA := trueMeanA + math.Sqrt(2*logTotal/float64(counts[0]))
	correctScoreB := dominanceArms[1].RewardSum/float64(counts[1]) + math.Sqrt(2*logTotal/float64(counts[1]))
	if !(correctScoreA > correctScoreB) {
		t.Fatalf("sum-contract scores: A=%g must exceed B=%g", correctScoreA, correctScoreB)
	}
	misuseScoreA := misuseAvgA + math.Sqrt(2*logTotal/float64(counts[0]))
	misuseScoreB := means[1]/float64(counts[1]) + math.Sqrt(2*logTotal/float64(counts[1]))
	if !(misuseScoreB > misuseScoreA) {
		t.Fatalf("misuse scores: B=%g must exceed A=%g", misuseScoreB, misuseScoreA)
	}
}

// ---------------------------------------------------------------------------
// UCB1FromArms edge-case parity with UCB1.
// ---------------------------------------------------------------------------

func TestUCB1FromArms_Empty(t *testing.T) {
	if got := UCB1FromArms(nil); got != -1 {
		t.Fatalf("UCB1FromArms(nil) = %d, want -1", got)
	}
	if got := UCB1FromArms([]Arm{}); got != -1 {
		t.Fatalf("UCB1FromArms(empty) = %d, want -1", got)
	}
}

func TestUCB1FromArms_UnexploredArmFirst(t *testing.T) {
	arms := []Arm{
		{Count: 10, RewardSum: 10},
		{Count: 0, RewardSum: 0},
		{Count: 0, RewardSum: 0},
	}
	if got := UCB1FromArms(arms); got != 1 {
		t.Fatalf("UCB1FromArms with unexplored arms = %d, want 1 (lowest-index zero-count arm)", got)
	}
}

func TestUCB1FromArms_MatchesRawForm(t *testing.T) {
	arms := []Arm{
		{Count: 7, RewardSum: 3.5},
		{Count: 12, RewardSum: 9.1},
		{Count: 4, RewardSum: 2.2},
	}
	counts := []int{7, 12, 4}
	rewards := []float64{3.5, 9.1, 2.2}
	want := UCB1(counts, rewards, 23)
	if got := UCB1FromArms(arms); got != want {
		t.Fatalf("UCB1FromArms = %d, want %d (must delegate exactly to UCB1)", got, want)
	}
}

// ---------------------------------------------------------------------------
// EpsilonGreedyFromArms.
// ---------------------------------------------------------------------------

// scriptedRNG replays a fixed sequence of Float64 values.
type scriptedRNG struct {
	vals []float64
	i    int
}

func (r *scriptedRNG) Float64() float64 {
	v := r.vals[r.i%len(r.vals)]
	r.i++
	return v
}

func TestEpsilonGreedyFromArms_Empty(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	if got := EpsilonGreedyFromArms(nil, 0.1, rng); got != -1 {
		t.Fatalf("EpsilonGreedyFromArms(nil) = %d, want -1", got)
	}
}

func TestEpsilonGreedyFromArms_ExploitsHighestMean(t *testing.T) {
	arms := []Arm{
		{Count: 10, RewardSum: 1}, // mean 0.1
		{Count: 10, RewardSum: 9}, // mean 0.9
		{Count: 10, RewardSum: 5}, // mean 0.5
	}
	// epsilon = 0 → always exploit; first rng draw (0.99) is the explore
	// check, which fails, so the highest-average arm must be returned.
	rng := &scriptedRNG{vals: []float64{0.99}}
	if got := EpsilonGreedyFromArms(arms, 0, rng); got != 1 {
		t.Fatalf("EpsilonGreedyFromArms exploit = %d, want 1 (highest mean)", got)
	}
}

func TestEpsilonGreedyFromArms_ExplorePath(t *testing.T) {
	arms := []Arm{
		{Count: 10, RewardSum: 9},
		{Count: 10, RewardSum: 1},
		{Count: 10, RewardSum: 1},
	}
	// First draw 0.0 < epsilon=1.0 → explore; second draw 0.7 → arm
	// int(0.7*3) = 2, even though arm 0 has the best mean.
	rng := &scriptedRNG{vals: []float64{0.0, 0.7}}
	if got := EpsilonGreedyFromArms(arms, 1.0, rng); got != 2 {
		t.Fatalf("EpsilonGreedyFromArms explore = %d, want 2", got)
	}
}

func TestEpsilonGreedyFromArms_ZeroCountTreatedAsZeroAvg(t *testing.T) {
	arms := []Arm{
		{Count: 10, RewardSum: -5}, // mean -0.5
		{Count: 0, RewardSum: 0},   // unexplored → avg treated as 0
	}
	rng := &scriptedRNG{vals: []float64{0.99}}
	if got := EpsilonGreedyFromArms(arms, 0, rng); got != 1 {
		t.Fatalf("EpsilonGreedyFromArms = %d, want 1 (zero-count avg 0 beats mean -0.5)", got)
	}
}

func TestEpsilonGreedyFromArms_MatchesRawForm(t *testing.T) {
	arms := []Arm{
		{Count: 5, RewardSum: 2},
		{Count: 8, RewardSum: 7},
	}
	rngA := rand.New(rand.NewSource(7))
	rngB := rand.New(rand.NewSource(7))
	want := EpsilonGreedy([]float64{2, 7}, []int{5, 8}, 0.3, rngA)
	if got := EpsilonGreedyFromArms(arms, 0.3, rngB); got != want {
		t.Fatalf("EpsilonGreedyFromArms = %d, want %d (must delegate exactly to EpsilonGreedy)", got, want)
	}
}

// ---------------------------------------------------------------------------
// ThompsonFromArmsBernoulli.
// ---------------------------------------------------------------------------

func TestThompsonFromArmsBernoulli_Empty(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	if got := ThompsonFromArmsBernoulli(nil, rng); got != -1 {
		t.Fatalf("ThompsonFromArmsBernoulli(nil) = %d, want -1", got)
	}
}

func TestThompsonFromArmsBernoulli_MappingMatchesThompsonSampling(t *testing.T) {
	// With identical seeds, the wrapper must be indistinguishable from
	// calling ThompsonSampling with successes=round(RewardSum) and
	// failures=Count-successes.
	arms := []Arm{
		{Count: 20, RewardSum: 13}, // 13 successes, 7 failures
		{Count: 15, RewardSum: 4},  // 4 successes, 11 failures
		{Count: 9, RewardSum: 9},   // 9 successes, 0 failures
	}
	for seed := int64(1); seed <= 25; seed++ {
		rngA := rand.New(rand.NewSource(seed))
		rngB := rand.New(rand.NewSource(seed))
		want := ThompsonSampling([]int{13, 4, 9}, []int{7, 11, 0}, rngA)
		got := ThompsonFromArmsBernoulli(arms, rngB)
		if got != want {
			t.Fatalf("seed %d: ThompsonFromArmsBernoulli = %d, want %d (mapping must match documented Bernoulli unpacking)", seed, got, want)
		}
	}
}

func TestThompsonFromArmsBernoulli_PrefersDominantArm(t *testing.T) {
	arms := []Arm{
		{Count: 1000, RewardSum: 990}, // ~99% success
		{Count: 1000, RewardSum: 10},  // ~1% success
	}
	rng := rand.New(rand.NewSource(42))
	wins := 0
	const trials = 100
	for i := 0; i < trials; i++ {
		if ThompsonFromArmsBernoulli(arms, rng) == 0 {
			wins++
		}
	}
	if wins < 95 {
		t.Fatalf("dominant arm selected %d/%d times, want >= 95", wins, trials)
	}
}

func TestThompsonFromArmsBernoulli_ClampsOutOfContractSums(t *testing.T) {
	// RewardSum above Count clamps to Count (all successes); negative
	// RewardSum clamps to 0 (all failures). Verified by seed-for-seed
	// equivalence with the explicitly clamped raw call — and no panic from
	// negative Beta parameters.
	arms := []Arm{
		{Count: 5, RewardSum: 7.6},  // clamps to successes=5, failures=0
		{Count: 5, RewardSum: -2.0}, // clamps to successes=0, failures=5
	}
	for seed := int64(1); seed <= 25; seed++ {
		rngA := rand.New(rand.NewSource(seed))
		rngB := rand.New(rand.NewSource(seed))
		want := ThompsonSampling([]int{5, 0}, []int{0, 5}, rngA)
		got := ThompsonFromArmsBernoulli(arms, rngB)
		if got != want {
			t.Fatalf("seed %d: clamped ThompsonFromArmsBernoulli = %d, want %d", seed, got, want)
		}
	}
}

func TestThompsonFromArmsBernoulli_RoundsFloatDrift(t *testing.T) {
	// 12.9999999 rounds to 13, not truncates to 12.
	arms := []Arm{
		{Count: 20, RewardSum: 12.9999999},
		{Count: 15, RewardSum: 4.0000001},
	}
	for seed := int64(1); seed <= 25; seed++ {
		rngA := rand.New(rand.NewSource(seed))
		rngB := rand.New(rand.NewSource(seed))
		want := ThompsonSampling([]int{13, 4}, []int{7, 11}, rngA)
		got := ThompsonFromArmsBernoulli(arms, rngB)
		if got != want {
			t.Fatalf("seed %d: rounded ThompsonFromArmsBernoulli = %d, want %d", seed, got, want)
		}
	}
}
