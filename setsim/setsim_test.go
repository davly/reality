package setsim

import (
	"math"
	"testing"
)

const eps = 1e-12

func approx(got, want float64) bool {
	return math.Abs(got-want) <= eps
}

// TestSetJaccard_Golden pins hand-computed Jaccard values.
//
// Hand computations (A∩B / A∪B):
//
//	{1,2,3} vs {2,3,4}: ∩={2,3}=2, ∪={1,2,3,4}=4 → 2/4 = 0.5
//	{1,2}   vs {3,4}   : ∩=0,       ∪=4           → 0/4 = 0.0   (disjoint)
//	{1,2,3} vs {1,2,3} : ∩=3,       ∪=3           → 3/3 = 1.0   (identical)
//	{1,2}   vs {1,2,3,4}:∩=2,       ∪=4           → 2/4 = 0.5   (subset)
//	{}      vs {}      : ∪=0                       → 0.0         (convention)
//	{1,2}   vs {}      : ∩=0,       ∪=2           → 0/2 = 0.0   (one empty)
//	{1,1,2,3} vs {2,3,3,4}: dedups to {1,2,3} vs {2,3,4} → 0.5  (dedup)
func TestSetJaccard_Golden(t *testing.T) {
	cases := []struct {
		name string
		a, b []int
		want float64
	}{
		{"overlap_half", []int{1, 2, 3}, []int{2, 3, 4}, 0.5},
		{"disjoint", []int{1, 2}, []int{3, 4}, 0.0},
		{"identical", []int{1, 2, 3}, []int{1, 2, 3}, 1.0},
		{"subset", []int{1, 2}, []int{1, 2, 3, 4}, 0.5},
		{"empty_empty", []int{}, []int{}, 0.0},
		{"nil_nil", nil, nil, 0.0},
		{"one_empty", []int{1, 2}, []int{}, 0.0},
		{"dedup_to_half", []int{1, 1, 2, 3}, []int{2, 3, 3, 4}, 0.5},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := SetJaccard(tc.a, tc.b); !approx(got, tc.want) {
				t.Errorf("SetJaccard(%v, %v) = %v, want %v", tc.a, tc.b, got, tc.want)
			}
			// Jaccard is symmetric.
			if got := SetJaccard(tc.b, tc.a); !approx(got, tc.want) {
				t.Errorf("SetJaccard(%v, %v) [swapped] = %v, want %v (symmetry)", tc.b, tc.a, got, tc.want)
			}
		})
	}
}

// TestSetDice_Golden pins hand-computed Dice values (2|A∩B|/(|A|+|B|)).
//
//	{1,2,3} vs {2,3,4} : 2·2/(3+3) = 4/6 = 0.666666...
//	{1,2}   vs {3,4}   : 0
//	{1,2,3} vs {1,2,3} : 2·3/(3+3) = 6/6 = 1.0
//	{1,2}   vs {1,2,3,4}: 2·2/(2+4) = 4/6 = 0.666666...
//	{}      vs {}      : 0 (convention; denom=0)
//	{1,2}   vs {}      : 2·0/(2+0) = 0
func TestSetDice_Golden(t *testing.T) {
	const twoThirds = 2.0 / 3.0
	cases := []struct {
		name string
		a, b []int
		want float64
	}{
		{"overlap", []int{1, 2, 3}, []int{2, 3, 4}, twoThirds},
		{"disjoint", []int{1, 2}, []int{3, 4}, 0.0},
		{"identical", []int{1, 2, 3}, []int{1, 2, 3}, 1.0},
		{"subset", []int{1, 2}, []int{1, 2, 3, 4}, twoThirds},
		{"empty_empty", nil, nil, 0.0},
		{"one_empty", []int{1, 2}, nil, 0.0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := SetDice(tc.a, tc.b); !approx(got, tc.want) {
				t.Errorf("SetDice(%v, %v) = %v, want %v", tc.a, tc.b, got, tc.want)
			}
		})
	}
}

// TestSetOverlapCoefficient_Golden pins hand-computed overlap values
// (|A∩B|/min(|A|,|B|)). The defining property vs Jaccard/Dice: a subset
// scores 1.0.
//
//	{1,2,3} vs {2,3,4} : 2/min(3,3) = 2/3 = 0.666666...
//	{1,2}   vs {1,2,3,4}: 2/min(2,4) = 2/2 = 1.0   (subset → 1)
//	{1,2}   vs {3,4}   : 0/min(2,2) = 0
//	{1,2,3} vs {1,2,3} : 3/3 = 1.0
//	{}      vs {}      : 0 (convention; min=0)
//	{1,2}   vs {}      : 0 (min=0)
func TestSetOverlapCoefficient_Golden(t *testing.T) {
	const twoThirds = 2.0 / 3.0
	cases := []struct {
		name string
		a, b []int
		want float64
	}{
		{"overlap", []int{1, 2, 3}, []int{2, 3, 4}, twoThirds},
		{"subset_is_one", []int{1, 2}, []int{1, 2, 3, 4}, 1.0},
		{"disjoint", []int{1, 2}, []int{3, 4}, 0.0},
		{"identical", []int{1, 2, 3}, []int{1, 2, 3}, 1.0},
		{"empty_empty", nil, nil, 0.0},
		{"one_empty", []int{1, 2}, nil, 0.0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := SetOverlapCoefficient(tc.a, tc.b); !approx(got, tc.want) {
				t.Errorf("SetOverlapCoefficient(%v, %v) = %v, want %v", tc.a, tc.b, got, tc.want)
			}
		})
	}
}

// TestSetOverlapCounts_Golden pins the raw (intersection, union) primitive.
//
//	{1,2,3} vs {2,3,4} : (2, 4)
//	{1,2}   vs {3,4}   : (0, 4)
//	{1,2,3} vs {1,2,3} : (3, 3)
//	{1,1,2} vs {2,2,3} : dedups {1,2} vs {2,3} → (1, 3)
//	{}      vs {}      : (0, 0)
func TestSetOverlapCounts_Golden(t *testing.T) {
	cases := []struct {
		name           string
		a, b           []int
		wantI, wantU   int
	}{
		{"overlap", []int{1, 2, 3}, []int{2, 3, 4}, 2, 4},
		{"disjoint", []int{1, 2}, []int{3, 4}, 0, 4},
		{"identical", []int{1, 2, 3}, []int{1, 2, 3}, 3, 3},
		{"dedup", []int{1, 1, 2}, []int{2, 2, 3}, 1, 3},
		{"empty_empty", nil, nil, 0, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			gi, gu := SetOverlapCounts(tc.a, tc.b)
			if gi != tc.wantI || gu != tc.wantU {
				t.Errorf("SetOverlapCounts(%v, %v) = (%d, %d), want (%d, %d)", tc.a, tc.b, gi, gu, tc.wantI, tc.wantU)
			}
		})
	}
}

// TestMapKeyJaccard_Golden pins map-key Jaccard. Values are deliberately
// different to prove they are ignored.
//
//	keys{a,b,c} vs keys{b,c,d}: ∩={b,c}=2, ∪={a,b,c,d}=4 → 0.5
//	keys{a,b}   vs keys{a,b}  : 1.0  (different values, same keys)
//	keys{a}     vs keys{b}    : 0.0  (disjoint keys)
//	{} vs {}                  : 0.0  (convention)
func TestMapKeyJaccard_Golden(t *testing.T) {
	cases := []struct {
		name string
		a, b map[string]int
		want float64
	}{
		{
			"overlap_half",
			map[string]int{"a": 1, "b": 2, "c": 3},
			map[string]int{"b": 9, "c": 9, "d": 9},
			0.5,
		},
		{
			"same_keys_diff_values",
			map[string]int{"a": 1, "b": 2},
			map[string]int{"a": 100, "b": 200},
			1.0,
		},
		{
			"disjoint",
			map[string]int{"a": 1},
			map[string]int{"b": 1},
			0.0,
		},
		{
			"empty_empty",
			map[string]int{},
			map[string]int{},
			0.0,
		},
		{
			"nil_nil",
			nil,
			nil,
			0.0,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := MapKeyJaccard(tc.a, tc.b); !approx(got, tc.want) {
				t.Errorf("MapKeyJaccard(%v, %v) = %v, want %v", tc.a, tc.b, got, tc.want)
			}
		})
	}
}

// TestRelations cross-checks the algebraic relation dice = 2J/(1+J), which
// holds for any non-empty pair. This is an independent check on both formulas:
// if either SetJaccard or SetDice drifted, the identity would break.
func TestRelations(t *testing.T) {
	pairs := [][2][]int{
		{{1, 2, 3}, {2, 3, 4}},
		{{1, 2, 3, 4, 5}, {4, 5, 6}},
		{{1, 2}, {1, 2, 3, 4}},
		{{7, 8, 9}, {7, 8, 9}},
	}
	for _, p := range pairs {
		j := SetJaccard(p[0], p[1])
		d := SetDice(p[0], p[1])
		wantD := 2 * j / (1 + j)
		if !approx(d, wantD) {
			t.Errorf("dice/jaccard identity broke for %v,%v: dice=%v, 2J/(1+J)=%v", p[0], p[1], d, wantD)
		}
	}
}

// TestStringElements confirms the generic primitive works on string sets,
// matching the original reinvented token/tag overlap use case.
func TestStringElements(t *testing.T) {
	a := []string{"the", "quick", "brown", "fox"}
	b := []string{"the", "lazy", "brown", "dog"}
	// ∩ = {the, brown} = 2 ; ∪ = {the,quick,brown,fox,lazy,dog} = 6 → 2/6 = 1/3
	want := 1.0 / 3.0
	if got := SetJaccard(a, b); !approx(got, want) {
		t.Errorf("SetJaccard(strings) = %v, want %v", got, want)
	}
}
