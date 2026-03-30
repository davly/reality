package prob

import (
	"math"
	"testing"

	"github.com/davly/reality/testutil"
)

// ═══════════════════════════════════════════════════════════════════════════
// MarkovSteadyState unit tests
// ═══════════════════════════════════════════════════════════════════════════

func TestMarkovSteadyState_Nil(t *testing.T) {
	if got := MarkovSteadyState(nil, 0); got != nil {
		t.Errorf("expected nil for n=0, got %v", got)
	}
}

func TestMarkovSteadyState_BadSize(t *testing.T) {
	if got := MarkovSteadyState([]float64{1, 2, 3}, 2); got != nil {
		t.Errorf("expected nil for mismatched matrix size, got %v", got)
	}
}

func TestMarkovSteadyState_NegativeN(t *testing.T) {
	if got := MarkovSteadyState([]float64{1}, -1); got != nil {
		t.Errorf("expected nil for negative n, got %v", got)
	}
}

func TestMarkovSteadyState_SingleState(t *testing.T) {
	result := MarkovSteadyState([]float64{1.0}, 1)
	if len(result) != 1 || math.Abs(result[0]-1.0) > 1e-10 {
		t.Errorf("single state: got %v, want [1.0]", result)
	}
}

func TestMarkovSteadyState_SymmetricUniform(t *testing.T) {
	// Symmetric matrix → uniform distribution.
	m := []float64{0.5, 0.5, 0.5, 0.5}
	result := MarkovSteadyState(m, 2)
	for i, v := range result {
		if math.Abs(v-0.5) > 1e-10 {
			t.Errorf("state %d: got %v, want 0.5", i, v)
		}
	}
}

func TestMarkovSteadyState_SumsToOne(t *testing.T) {
	m := []float64{0.7, 0.3, 0.4, 0.6}
	result := MarkovSteadyState(m, 2)
	sum := 0.0
	for _, v := range result {
		sum += v
	}
	if math.Abs(sum-1.0) > 1e-10 {
		t.Errorf("sum = %v, want 1.0", sum)
	}
}

func TestMarkovSteadyState_WeatherModel(t *testing.T) {
	// Classic weather Markov chain: sunny/rainy.
	// P(sunny|sunny)=0.7, P(rainy|sunny)=0.3
	// P(sunny|rainy)=0.4, P(rainy|rainy)=0.6
	// Steady state: pi_sunny = 4/7, pi_rainy = 3/7
	m := []float64{0.7, 0.3, 0.4, 0.6}
	result := MarkovSteadyState(m, 2)
	if math.Abs(result[0]-4.0/7.0) > 1e-6 {
		t.Errorf("sunny: got %v, want %v", result[0], 4.0/7.0)
	}
	if math.Abs(result[1]-3.0/7.0) > 1e-6 {
		t.Errorf("rainy: got %v, want %v", result[1], 3.0/7.0)
	}
}

func TestMarkovSteadyState_Cycle(t *testing.T) {
	// Pure cycle: 0→1→2→0, uniform steady state.
	m := []float64{0, 1, 0, 0, 0, 1, 1, 0, 0}
	result := MarkovSteadyState(m, 3)
	for i, v := range result {
		if math.Abs(v-1.0/3.0) > 1e-6 {
			t.Errorf("state %d: got %v, want 1/3", i, v)
		}
	}
}

func TestMarkovSteadyState_AbsorbingState(t *testing.T) {
	// State 1 is absorbing.
	m := []float64{0, 1, 0, 1}
	result := MarkovSteadyState(m, 2)
	if math.Abs(result[0]) > 1e-10 {
		t.Errorf("non-absorbing state: got %v, want 0", result[0])
	}
	if math.Abs(result[1]-1.0) > 1e-10 {
		t.Errorf("absorbing state: got %v, want 1", result[1])
	}
}

func TestMarkovSteadyState_DoublyStochastic(t *testing.T) {
	// Doubly stochastic → uniform steady state.
	m := []float64{0.2, 0.3, 0.5, 0.5, 0.2, 0.3, 0.3, 0.5, 0.2}
	result := MarkovSteadyState(m, 3)
	for i, v := range result {
		if math.Abs(v-1.0/3.0) > 1e-6 {
			t.Errorf("state %d: got %v, want 1/3", i, v)
		}
	}
}

func TestMarkovSteadyState_NonNegative(t *testing.T) {
	m := []float64{0.1, 0.6, 0.3, 0.4, 0.2, 0.4, 0.3, 0.3, 0.4}
	result := MarkovSteadyState(m, 3)
	for i, v := range result {
		if v < -1e-15 {
			t.Errorf("state %d is negative: %v", i, v)
		}
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Golden-file: MarkovSteadyState
// ═══════════════════════════════════════════════════════════════════════════

func TestGolden_MarkovSteadyState(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/prob/markov_steady_state.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			matrix := testutil.InputFloat64Slice(t, tc, "matrix")
			n := testutil.InputInt(t, tc, "n")

			result := MarkovSteadyState(matrix, n)

			testutil.AssertFloat64Slice(t, tc, result)
		})
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// MarkovSimulate unit tests
// ═══════════════════════════════════════════════════════════════════════════

func TestMarkovSimulate_Nil(t *testing.T) {
	if got := MarkovSimulate(nil, 0, 0, 10); got != nil {
		t.Errorf("expected nil for n=0, got %v", got)
	}
}

func TestMarkovSimulate_BadSize(t *testing.T) {
	if got := MarkovSimulate([]float64{1, 2, 3}, 2, 0, 10); got != nil {
		t.Errorf("expected nil for mismatched matrix, got %v", got)
	}
}

func TestMarkovSimulate_InvalidInitialState(t *testing.T) {
	if got := MarkovSimulate([]float64{1}, 1, 1, 10); got != nil {
		t.Errorf("expected nil for out of range initial state, got %v", got)
	}
}

func TestMarkovSimulate_NegativeInitial(t *testing.T) {
	if got := MarkovSimulate([]float64{1}, 1, -1, 10); got != nil {
		t.Errorf("expected nil for negative initial state, got %v", got)
	}
}

func TestMarkovSimulate_NegativeSteps(t *testing.T) {
	if got := MarkovSimulate([]float64{1}, 1, 0, -1); got != nil {
		t.Errorf("expected nil for negative steps, got %v", got)
	}
}

func TestMarkovSimulate_ZeroSteps(t *testing.T) {
	result := MarkovSimulate([]float64{1}, 1, 0, 0)
	if len(result) != 1 || result[0] != 0 {
		t.Errorf("zero steps: got %v, want [0]", result)
	}
}

func TestMarkovSimulate_Length(t *testing.T) {
	m := []float64{0.5, 0.5, 0.5, 0.5}
	result := MarkovSimulate(m, 2, 0, 20)
	if len(result) != 21 {
		t.Errorf("length: got %d, want 21", len(result))
	}
}

func TestMarkovSimulate_StartsCorrectly(t *testing.T) {
	m := []float64{0.5, 0.5, 0.5, 0.5}
	for start := 0; start < 2; start++ {
		result := MarkovSimulate(m, 2, start, 5)
		if result[0] != start {
			t.Errorf("initial state: got %d, want %d", result[0], start)
		}
	}
}

func TestMarkovSimulate_DeterministicCycle(t *testing.T) {
	// 0→1→2→0 deterministically.
	m := []float64{0, 1, 0, 0, 0, 1, 1, 0, 0}
	result := MarkovSimulate(m, 3, 0, 6)
	expected := []int{0, 1, 2, 0, 1, 2, 0}
	for i, v := range expected {
		if result[i] != v {
			t.Errorf("step %d: got %d, want %d", i, result[i], v)
		}
	}
}

func TestMarkovSimulate_AbsorbingTraps(t *testing.T) {
	// State 1 is absorbing.
	m := []float64{0, 1, 0, 1}
	result := MarkovSimulate(m, 2, 0, 10)
	// After step 1, should be in state 1 forever.
	for i := 1; i <= 10; i++ {
		if result[i] != 1 {
			t.Errorf("step %d: got %d, want 1", i, result[i])
		}
	}
}

func TestMarkovSimulate_AllValidStates(t *testing.T) {
	m := []float64{0.5, 0.5, 0.5, 0.5}
	result := MarkovSimulate(m, 2, 0, 100)
	for i, v := range result {
		if v < 0 || v >= 2 {
			t.Errorf("step %d: state %d out of range [0, 2)", i, v)
		}
	}
}

func TestMarkovSimulate_Reproducible(t *testing.T) {
	m := []float64{0.5, 0.5, 0.5, 0.5}
	r1 := MarkovSimulate(m, 2, 0, 20)
	r2 := MarkovSimulate(m, 2, 0, 20)
	for i := range r1 {
		if r1[i] != r2[i] {
			t.Errorf("step %d: not reproducible: %d vs %d", i, r1[i], r2[i])
		}
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Golden-file: MarkovSimulate
// ═══════════════════════════════════════════════════════════════════════════

func TestGolden_MarkovSimulate(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/prob/markov_simulate.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			matrix := testutil.InputFloat64Slice(t, tc, "matrix")
			n := testutil.InputInt(t, tc, "n")
			initialState := testutil.InputInt(t, tc, "initial_state")
			steps := testutil.InputInt(t, tc, "steps")

			result := MarkovSimulate(matrix, n, initialState, steps)

			expected, ok := tc.Expected.([]any)
			if !ok {
				t.Fatalf("expected is not an array: %T", tc.Expected)
			}
			if len(result) != len(expected) {
				t.Fatalf("length mismatch: got %d, want %d", len(result), len(expected))
			}
			for i, elem := range expected {
				want := int(elem.(float64))
				if result[i] != want {
					t.Errorf("step %d: got %d, want %d", i, result[i], want)
				}
			}
		})
	}
}
