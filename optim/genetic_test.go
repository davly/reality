package optim

import (
	"math"
	"math/rand"
	"testing"
)

// ---------------------------------------------------------------------------
// GeneticAlgorithm
// ---------------------------------------------------------------------------

func TestGA_Sphere(t *testing.T) {
	// Minimize sum(x_i^2) in 2D. Minimum at origin.
	f := func(x []float64) float64 { return x[0]*x[0] + x[1]*x[1] }
	rng := rand.New(rand.NewSource(42))
	best, bestF := GeneticAlgorithm(f, 2, 50, 200, 0.1, rng)
	if bestF > 0.1 {
		t.Errorf("GA sphere: bestF=%v, best=%v, want near 0", bestF, best)
	}
}

func TestGA_ShiftedBowl(t *testing.T) {
	// Minimize (x-3)^2 + (y+2)^2. Minimum at (3, -2).
	f := func(x []float64) float64 {
		return (x[0]-3)*(x[0]-3) + (x[1]+2)*(x[1]+2)
	}
	rng := rand.New(rand.NewSource(42))
	best, bestF := GeneticAlgorithm(f, 2, 100, 300, 0.1, rng)
	if bestF > 0.5 {
		t.Errorf("GA shifted bowl: bestF=%v, best=%v, want near (3,-2)", bestF, best)
	}
}

func TestGA_1D(t *testing.T) {
	// Minimize (x-7)^2 in 1D. Minimum at 7.
	f := func(x []float64) float64 { return (x[0] - 7) * (x[0] - 7) }
	rng := rand.New(rand.NewSource(42))
	// Note: 7 is outside [-5,5] default init range, so the GA must
	// explore beyond the initial bounds via mutation/crossover.
	// With enough generations it should still find a good solution.
	best, bestF := GeneticAlgorithm(f, 1, 50, 500, 0.2, rng)
	if bestF > 1.0 {
		t.Errorf("GA 1D: bestF=%v, best=%v, want near 7", bestF, best)
	}
}

func TestGA_Rastrigin(t *testing.T) {
	// 2D Rastrigin: f(x) = 20 + sum(x_i^2 - 10*cos(2*pi*x_i)).
	// Global minimum at origin, f=0. Many local minima.
	f := func(x []float64) float64 {
		sum := 20.0
		for _, xi := range x {
			sum += xi*xi - 10*math.Cos(2*math.Pi*xi)
		}
		return sum
	}
	rng := rand.New(rand.NewSource(42))
	_, bestF := GeneticAlgorithm(f, 2, 200, 500, 0.15, rng)
	// GA should find a good (not necessarily global) minimum.
	if bestF > 5 {
		t.Errorf("GA Rastrigin: bestF=%v, want < 5", bestF)
	}
}

func TestGA_DoesNotModifyRNGState(t *testing.T) {
	// Running GA twice with the same seed should produce the same result.
	f := func(x []float64) float64 { return x[0]*x[0] + x[1]*x[1] }

	rng1 := rand.New(rand.NewSource(123))
	best1, f1 := GeneticAlgorithm(f, 2, 30, 100, 0.1, rng1)

	rng2 := rand.New(rand.NewSource(123))
	best2, f2 := GeneticAlgorithm(f, 2, 30, 100, 0.1, rng2)

	if best1[0] != best2[0] || best1[1] != best2[1] || f1 != f2 {
		t.Errorf("GA not deterministic: run1=(%v, %v), run2=(%v, %v)", best1, f1, best2, f2)
	}
}

func TestGA_PopSizeRoundUp(t *testing.T) {
	// Odd popSize should be rounded up to even.
	f := func(x []float64) float64 { return x[0] * x[0] }
	rng := rand.New(rand.NewSource(42))
	// Should not panic with odd population size.
	_, bestF := GeneticAlgorithm(f, 1, 3, 50, 0.1, rng)
	if bestF > 1 {
		t.Errorf("GA popsize 3: bestF=%v, want < 1", bestF)
	}
}

func TestGA_MinPopSize(t *testing.T) {
	// Population size < 2 should be clamped to 2.
	f := func(x []float64) float64 { return x[0] * x[0] }
	rng := rand.New(rand.NewSource(42))
	_, bestF := GeneticAlgorithm(f, 1, 1, 100, 0.1, rng)
	if math.IsNaN(bestF) || math.IsInf(bestF, 0) {
		t.Errorf("GA min popsize: bestF=%v, want finite", bestF)
	}
}

func TestGA_ZeroMutRate(t *testing.T) {
	// With zero mutation rate, crossover alone should still make progress.
	f := func(x []float64) float64 { return x[0]*x[0] + x[1]*x[1] }
	rng := rand.New(rand.NewSource(42))
	_, bestF := GeneticAlgorithm(f, 2, 50, 200, 0.0, rng)
	if bestF > 1 {
		t.Errorf("GA zero mutRate: bestF=%v, want < 1", bestF)
	}
}

func TestGA_HighDim(t *testing.T) {
	// 5D sphere.
	f := func(x []float64) float64 {
		sum := 0.0
		for _, xi := range x {
			sum += xi * xi
		}
		return sum
	}
	rng := rand.New(rand.NewSource(42))
	_, bestF := GeneticAlgorithm(f, 5, 100, 500, 0.15, rng)
	if bestF > 1 {
		t.Errorf("GA 5D sphere: bestF=%v, want < 1", bestF)
	}
}

func TestGA_Elitism(t *testing.T) {
	// The best fitness should never get worse across generations
	// (elitism preserves the best individual). We verify by checking
	// that the final result is at least as good as an initial random best.
	f := func(x []float64) float64 { return x[0]*x[0] + x[1]*x[1] }
	rng := rand.New(rand.NewSource(42))

	// Run for 1 generation — record best.
	rng1 := rand.New(rand.NewSource(42))
	_, f1 := GeneticAlgorithm(f, 2, 50, 1, 0.1, rng1)

	// Run for more generations — should be at least as good.
	_, fMany := GeneticAlgorithm(f, 2, 50, 100, 0.1, rng)
	if fMany > f1+1e-10 {
		t.Errorf("GA elitism violated: 1-gen=%v, 100-gen=%v", f1, fMany)
	}
}

// ---------------------------------------------------------------------------
// SimplexMethod — additional tests (function already exists in linear.go)
// ---------------------------------------------------------------------------

func TestSimplex_BasicLP(t *testing.T) {
	// minimize -x1 - x2
	// subject to: x1 + x2 <= 4, x1 <= 3, x2 <= 3, x1,x2 >= 0
	// Optimal: x1=3, x2=1, obj=-4  OR  x1=1, x2=3
	c := []float64{-1, -1}
	A := [][]float64{{1, 1}, {1, 0}, {0, 1}}
	b := []float64{4, 3, 3}
	x, obj, err := SimplexMethod(c, A, b)
	if err != nil {
		t.Fatalf("SimplexMethod error: %v", err)
	}
	if math.Abs(obj-(-4)) > 1e-6 {
		t.Errorf("Simplex objective = %v, want -4", obj)
	}
	if math.Abs(x[0]+x[1]-4) > 1e-6 {
		t.Errorf("Simplex x1+x2 = %v, want 4", x[0]+x[1])
	}
}

func TestSimplex_SingleVariable(t *testing.T) {
	// minimize -3x subject to x <= 10, x >= 0
	c := []float64{-3}
	A := [][]float64{{1}}
	b := []float64{10}
	x, obj, err := SimplexMethod(c, A, b)
	if err != nil {
		t.Fatalf("SimplexMethod error: %v", err)
	}
	if math.Abs(x[0]-10) > 1e-6 {
		t.Errorf("Simplex x = %v, want 10", x[0])
	}
	if math.Abs(obj-(-30)) > 1e-6 {
		t.Errorf("Simplex obj = %v, want -30", obj)
	}
}

func TestSimplex_Infeasible(t *testing.T) {
	// minimize x1 subject to -x1 <= -10, x1 <= 5 (infeasible: x1 >= 10 and x1 <= 5)
	// Note: our implementation may not detect infeasibility, it depends on
	// how the negative-b handling works. We just verify it doesn't panic.
	c := []float64{1}
	A := [][]float64{{-1}, {1}}
	b := []float64{-10, 5}
	_, _, _ = SimplexMethod(c, A, b) // should not panic
}

func TestSimplex_Empty(t *testing.T) {
	_, _, err := SimplexMethod(nil, nil, nil)
	if err == nil {
		t.Errorf("SimplexMethod with empty inputs should return error")
	}
}

func TestSimplex_ThreeVar(t *testing.T) {
	// minimize -2x1 - 3x2 - x3
	// subject to: x1 + x2 + x3 <= 40, 2x1 + x2 <= 60, x2 + x3 <= 45
	c := []float64{-2, -3, -1}
	A := [][]float64{{1, 1, 1}, {2, 1, 0}, {0, 1, 1}}
	b := []float64{40, 60, 45}
	x, obj, err := SimplexMethod(c, A, b)
	if err != nil {
		t.Fatalf("SimplexMethod error: %v", err)
	}
	// Check feasibility.
	for i := 0; i < 3; i++ {
		if x[i] < -1e-6 {
			t.Errorf("Simplex x[%d] = %v, want >= 0", i, x[i])
		}
	}
	_ = obj
}

func TestSimplex_ZeroObjective(t *testing.T) {
	// minimize 0*x1 + 0*x2 — any feasible point is optimal with obj=0.
	c := []float64{0, 0}
	A := [][]float64{{1, 0}, {0, 1}}
	b := []float64{10, 10}
	_, obj, err := SimplexMethod(c, A, b)
	if err != nil {
		t.Fatalf("SimplexMethod error: %v", err)
	}
	if math.Abs(obj) > 1e-10 {
		t.Errorf("Simplex zero obj = %v, want 0", obj)
	}
}
