package fairness

import (
	"math"
	"testing"
)

// ---------------------------------------------------------------------------
// SelectionRate
// ---------------------------------------------------------------------------

func TestSelectionRate(t *testing.T) {
	tests := []struct {
		name     string
		selected int
		total    int
		want     float64
	}{
		{"30 of 100", 30, 100, 0.30},
		{"50 of 100", 50, 100, 0.50},
		{"all selected", 7, 7, 1.0},
		{"none selected", 0, 40, 0.0},
		{"zero total", 5, 0, 0.0},
		{"negative total", 5, -1, 0.0},
		{"over-count clamps to 1", 12, 10, 1.0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := SelectionRate(tt.selected, tt.total)
			if math.Abs(got-tt.want) > 1e-15 {
				t.Errorf("SelectionRate(%d, %d) = %v, want %v",
					tt.selected, tt.total, got, tt.want)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// AdverseImpactRatio
// ---------------------------------------------------------------------------

func TestAdverseImpactRatio(t *testing.T) {
	tests := []struct {
		name  string
		rateA float64
		rateB float64
		want  float64
	}{
		// Discriminating golden: 0.30 / 0.50 = 0.60 (a wrong divisor or a
		// flipped numerator/denominator would not produce 0.60).
		{"fail case 0.30 vs 0.50", 0.30, 0.50, 0.60},
		// Order independence: same inputs swapped give the same ratio.
		{"fail case swapped", 0.50, 0.30, 0.60},
		// Passing golden: 0.45 / 0.50 = 0.90.
		{"pass case 0.45 vs 0.50", 0.45, 0.50, 0.90},
		// Exactly at the four-fifths boundary: 0.40 / 0.50 = 0.80.
		{"boundary 0.40 vs 0.50", 0.40, 0.50, 0.80},
		{"perfect parity", 0.50, 0.50, 1.0},
		{"both zero -> 0", 0.0, 0.0, 0.0},
		{"one zero rate -> 0", 0.0, 0.40, 0.0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := AdverseImpactRatio(tt.rateA, tt.rateB)
			if math.Abs(got-tt.want) > 1e-15 {
				t.Errorf("AdverseImpactRatio(%v, %v) = %v, want %v",
					tt.rateA, tt.rateB, got, tt.want)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// PassesFourFifths
// ---------------------------------------------------------------------------

func TestPassesFourFifths(t *testing.T) {
	tests := []struct {
		name string
		air  float64
		want bool
	}{
		{"below threshold fails", 0.60, false},
		{"just below threshold fails", 0.7999, false},
		{"exactly at threshold passes", 0.80, true},
		{"above threshold passes", 0.90, true},
		{"perfect parity passes", 1.0, true},
		{"zero fails", 0.0, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := PassesFourFifths(tt.air)
			if got != tt.want {
				t.Errorf("PassesFourFifths(%v) = %v, want %v", tt.air, got, tt.want)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// WilsonScoreInterval
// ---------------------------------------------------------------------------

func TestWilsonScoreInterval(t *testing.T) {
	// wantLow/wantHigh of NaN means "do not assert this bound exactly" (still
	// checked for finite + in-range).
	tests := []struct {
		name     string
		selected int
		total    int
		z        float64
		wantLow  float64
		wantHigh float64
		tol      float64
	}{
		// Hand-computed golden point: selected=50, total=100, z=1.96.
		//   pHat   = 0.5
		//   z^2    = 3.8416
		//   denom  = 1 + 3.8416/100        = 1.038416
		//   centre = (0.5 + 3.8416/200)/denom = 0.519208/1.038416
		//   margin = 1.96*sqrt((0.25 + 3.8416/400)/100)/denom
		//          = 1.96*sqrt(0.00259604)/1.038416
		//   low    = 0.4038298286, high = 0.5961701714
		{"p=0.5 n=100 golden", 50, 100, 1.96, 0.4038298286, 0.5961701714, 1e-9},
		// p=0.45 n=100: low=0.3561437511, high=0.5475557297
		{"p=0.45 n=100 golden", 45, 100, 1.96, 0.3561437511, 0.5475557297, 1e-9},
		// p=0.30 n=100: low=0.2189475387, high=0.3958503843
		{"p=0.30 n=100 golden", 30, 100, 1.96, 0.2189475387, 0.3958503843, 1e-9},
		// Near the upper boundary, the interval stays inside [0,1].
		// 99/100: low=0.9455124752, high=0.9982326134
		{"p=0.99 n=100 stays below 1", 99, 100, 1.96, 0.9455124752, 0.9982326134, 1e-9},
		// Zero selections: lower bound hits exactly 0, upper stays < 1.
		// 0/30: low=0.0, high=0.1135170914
		{"p=0.0 n=30 lower hits 0", 0, 30, 1.96, 0.0, 0.1135170914, 1e-9},
		// z defaults to 1.96 when <= 0 -> same as the p=0.5 golden.
		{"z<=0 defaults to 1.96", 50, 100, 0.0, 0.4038298286, 0.5961701714, 1e-9},
		// Degenerate total -> (0,0).
		{"total=0 -> (0,0)", 5, 0, 1.96, 0.0, 0.0, 1e-15},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			low, high := WilsonScoreInterval(tt.selected, tt.total, tt.z)
			if math.IsNaN(low) || math.IsNaN(high) {
				t.Fatalf("got NaN bound: low=%v high=%v", low, high)
			}
			if low < 0 || low > 1 || high < 0 || high > 1 {
				t.Errorf("bounds escaped [0,1]: low=%v high=%v", low, high)
			}
			if !math.IsNaN(tt.wantLow) && math.Abs(low-tt.wantLow) > tt.tol {
				t.Errorf("low = %v, want ~%v (tol %v)", low, tt.wantLow, tt.tol)
			}
			if !math.IsNaN(tt.wantHigh) && math.Abs(high-tt.wantHigh) > tt.tol {
				t.Errorf("high = %v, want ~%v (tol %v)", high, tt.wantHigh, tt.tol)
			}
			if low > high {
				t.Errorf("low (%v) > high (%v)", low, high)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// AdverseImpact (end-to-end)
// ---------------------------------------------------------------------------

func TestAdverseImpactFailCase(t *testing.T) {
	// groupA 30/100 = 0.30, groupB 50/100 = 0.50 => AIR = 0.60 < 0.80 FAIL.
	rep := AdverseImpact([]GroupCount{
		{Label: "groupA", Selected: 30, Total: 100},
		{Label: "groupB", Selected: 50, Total: 100},
	}, 1.96)

	if !rep.Applicable {
		t.Fatalf("expected applicable report")
	}
	if math.Abs(rep.AIR-0.60) > 1e-12 {
		t.Errorf("AIR = %v, want 0.60", rep.AIR)
	}
	if rep.Pass {
		t.Errorf("expected four-fifths FAIL, got Pass=true (AIR=%v)", rep.AIR)
	}
	if rep.MinLabel != "groupA" || rep.MaxLabel != "groupB" {
		t.Errorf("min/max labels = %q/%q, want groupA/groupB", rep.MinLabel, rep.MaxLabel)
	}
	if len(rep.Groups) != 2 {
		t.Fatalf("expected 2 group rows, got %d", len(rep.Groups))
	}
	// Groups are sorted by label; groupA first.
	if rep.Groups[0].Label != "groupA" {
		t.Errorf("groups not sorted by label: first = %q", rep.Groups[0].Label)
	}
	if math.Abs(rep.Groups[0].SelectionRate-0.30) > 1e-15 {
		t.Errorf("groupA rate = %v, want 0.30", rep.Groups[0].SelectionRate)
	}
	if math.Abs(rep.Groups[1].SelectionRate-0.50) > 1e-15 {
		t.Errorf("groupB rate = %v, want 0.50", rep.Groups[1].SelectionRate)
	}
	// Wilson CI on groupA (30/100) should bracket its hand-computed bounds.
	if math.Abs(rep.Groups[0].CILow-0.2189475387) > 1e-9 ||
		math.Abs(rep.Groups[0].CIHigh-0.3958503843) > 1e-9 {
		t.Errorf("groupA CI = [%v, %v], want [0.2189475387, 0.3958503843]",
			rep.Groups[0].CILow, rep.Groups[0].CIHigh)
	}
}

func TestAdverseImpactPassCase(t *testing.T) {
	// groupA 45/100 = 0.45, groupB 50/100 = 0.50 => AIR = 0.90 >= 0.80 PASS.
	rep := AdverseImpact([]GroupCount{
		{Label: "groupA", Selected: 45, Total: 100},
		{Label: "groupB", Selected: 50, Total: 100},
	}, 1.96)

	if !rep.Applicable {
		t.Fatalf("expected applicable report")
	}
	if math.Abs(rep.AIR-0.90) > 1e-12 {
		t.Errorf("AIR = %v, want 0.90", rep.AIR)
	}
	if !rep.Pass {
		t.Errorf("expected four-fifths PASS, got Pass=false (AIR=%v)", rep.AIR)
	}
}

func TestAdverseImpactBoundary(t *testing.T) {
	// Exactly at 0.80 must PASS (>= comparison).
	// 40/100 = 0.40 vs 50/100 = 0.50 => AIR = 0.80.
	rep := AdverseImpact([]GroupCount{
		{Label: "a", Selected: 40, Total: 100},
		{Label: "b", Selected: 50, Total: 100},
	}, 1.96)
	if math.Abs(rep.AIR-0.80) > 1e-12 {
		t.Errorf("AIR = %v, want 0.80", rep.AIR)
	}
	if !rep.Pass {
		t.Errorf("AIR exactly 0.80 must PASS")
	}
}

func TestAdverseImpactThreeGroups(t *testing.T) {
	// AIR is the worst pair: min rate / max rate over ALL eligible groups.
	// a=20/100=0.20, b=50/100=0.50, c=40/100=0.40 => AIR = 0.20/0.50 = 0.40.
	rep := AdverseImpact([]GroupCount{
		{Label: "b", Selected: 50, Total: 100},
		{Label: "c", Selected: 40, Total: 100},
		{Label: "a", Selected: 20, Total: 100},
	}, 1.96)
	if math.Abs(rep.AIR-0.40) > 1e-12 {
		t.Errorf("AIR = %v, want 0.40", rep.AIR)
	}
	if rep.MinLabel != "a" || rep.MaxLabel != "b" {
		t.Errorf("min/max = %q/%q, want a/b", rep.MinLabel, rep.MaxLabel)
	}
	if rep.Pass {
		t.Errorf("expected FAIL for AIR 0.40")
	}
}

func TestAdverseImpactNotApplicable(t *testing.T) {
	// Single group -> rule not applicable.
	rep := AdverseImpact([]GroupCount{
		{Label: "only", Selected: 30, Total: 100},
	}, 1.96)
	if rep.Applicable {
		t.Errorf("single group should not be applicable")
	}
	if rep.Pass {
		t.Errorf("not-applicable report must not Pass")
	}
	if rep.AIR != 0 {
		t.Errorf("AIR = %v, want 0 when not applicable", rep.AIR)
	}

	// Empty input.
	rep = AdverseImpact(nil, 1.96)
	if rep.Applicable || rep.Pass || len(rep.Groups) != 0 {
		t.Errorf("empty input: applicable=%v pass=%v groups=%d",
			rep.Applicable, rep.Pass, len(rep.Groups))
	}

	// Zero-total groups are excluded from AIR; with only one eligible group
	// the rule is not applicable.
	rep = AdverseImpact([]GroupCount{
		{Label: "a", Selected: 5, Total: 10},
		{Label: "b", Selected: 0, Total: 0},
	}, 1.96)
	if rep.Applicable {
		t.Errorf("one eligible group should not be applicable")
	}
	if len(rep.Groups) != 2 {
		t.Errorf("zero-total group should still appear in Groups: got %d rows", len(rep.Groups))
	}
}

func TestAdverseImpactAllZeroSelections(t *testing.T) {
	// Two eligible groups but nobody selected -> maxRate 0 -> not applicable.
	rep := AdverseImpact([]GroupCount{
		{Label: "a", Selected: 0, Total: 50},
		{Label: "b", Selected: 0, Total: 50},
	}, 1.96)
	if rep.Applicable {
		t.Errorf("all-zero selection should not be applicable (maxRate=0)")
	}
	if rep.AIR != 0 {
		t.Errorf("AIR = %v, want 0", rep.AIR)
	}
}

func TestAdverseImpactInputNotMutated(t *testing.T) {
	in := []GroupCount{
		{Label: "z", Selected: 5, Total: 10},
		{Label: "a", Selected: 3, Total: 10},
	}
	backup := make([]GroupCount, len(in))
	copy(backup, in)
	AdverseImpact(in, 1.96)
	for i := range in {
		if in[i] != backup[i] {
			t.Errorf("AdverseImpact mutated input[%d]: %+v -> %+v", i, backup[i], in[i])
		}
	}
}
