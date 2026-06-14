package evidence

import (
	"math"
	"testing"
)

// ---------------------------------------------------------------------------
// Summarize
// ---------------------------------------------------------------------------

func TestSummarize(t *testing.T) {
	tests := []struct {
		name      string
		samples   []int
		threshold int
		want      Summary
	}{
		// Hand-computed weakest-link case (mirrors the argus teacher test):
		// {3000,120,800} -> count 3, total 3920, min 120, 120<500 underpowered.
		{
			"weakest link below threshold",
			[]int{3000, 120, 800}, 500,
			Summary{Count: 3, Total: 3920, Min: 120, Underpowered: true},
		},
		// Real argus DB case: matched samples 2400 & 950, default threshold 500.
		// total 3350, min 950 >= 500 -> not underpowered.
		{
			"two samples default threshold not underpowered",
			[]int{2400, 950}, 0, // <=0 -> DefaultMinSampleSize (500)
			Summary{Count: 2, Total: 3350, Min: 950, Underpowered: false},
		},
		// Same samples, stricter threshold 1000: weakest link 950 < 1000.
		{
			"stricter threshold flags weakest link",
			[]int{2400, 950}, 1000,
			Summary{Count: 2, Total: 3350, Min: 950, Underpowered: true},
		},
		// Empty: zero-valued and NOT underpowered (nothing to under-power).
		{
			"empty is zero and not underpowered",
			nil, 500,
			Summary{Count: 0, Total: 0, Min: 0, Underpowered: false},
		},
		// Boundary: min exactly at threshold is NOT underpowered (strict <).
		{
			"min equals threshold not underpowered",
			[]int{500, 900}, 500,
			Summary{Count: 2, Total: 1400, Min: 500, Underpowered: false},
		},
		// Boundary: one below the threshold IS underpowered.
		{
			"min one below threshold underpowered",
			[]int{499, 900}, 500,
			Summary{Count: 2, Total: 1399, Min: 499, Underpowered: true},
		},
		// Single sample.
		{
			"single strong sample",
			[]int{2400}, 500,
			Summary{Count: 1, Total: 2400, Min: 2400, Underpowered: false},
		},
		// Min is first element (ordering independence).
		{
			"min is first element",
			[]int{120, 3000, 800}, 500,
			Summary{Count: 3, Total: 3920, Min: 120, Underpowered: true},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Summarize(tt.samples, tt.threshold)
			if got != tt.want {
				t.Errorf("Summarize(%v, %d) = %+v, want %+v",
					tt.samples, tt.threshold, got, tt.want)
			}
		})
	}
}

func TestSummarizeDefaultThreshold(t *testing.T) {
	// A non-positive threshold must resolve to DefaultMinSampleSize (500).
	// min 400 < 500 -> underpowered under the default.
	if !Summarize([]int{400}, 0).Underpowered {
		t.Error("threshold 0 must default to 500; min 400 < 500 -> underpowered")
	}
	if !Summarize([]int{400}, -10).Underpowered {
		t.Error("negative threshold must default to 500; min 400 < 500 -> underpowered")
	}
	// min 600 >= 500 -> not underpowered under the default.
	if Summarize([]int{600}, 0).Underpowered {
		t.Error("min 600 >= default 500 must not be underpowered")
	}
}

func TestSummarizeDoesNotMutateInput(t *testing.T) {
	in := []int{3000, 120, 800}
	backup := []int{3000, 120, 800}
	Summarize(in, 500)
	for i := range in {
		if in[i] != backup[i] {
			t.Errorf("Summarize mutated input[%d]: %d -> %d", i, backup[i], in[i])
		}
	}
}

// ---------------------------------------------------------------------------
// SampleBackingFactor
// ---------------------------------------------------------------------------

func TestSampleBackingFactor(t *testing.T) {
	tests := []struct {
		name           string
		n              int
		halfSaturation float64
		want           float64
	}{
		// At n == k the factor is exactly 0.5.
		{"at half-saturation", 500, 500, 0.5},
		// n=0 -> 0.
		{"zero sample", 0, 500, 0.0},
		// Negative sample -> 0.
		{"negative sample", -10, 500, 0.0},
		// 1500/(1500+500) = 0.75.
		{"three times k", 1500, 500, 0.75},
		// 100/(100+100) = 0.5 with a different k.
		{"custom k at half", 100, 100, 0.5},
		// 4500/(4500+500) = 0.9.
		{"nine times k", 4500, 500, 0.9},
		// Non-positive halfSaturation defaults to 500: 500/(500+500)=0.5.
		{"default k when zero", 500, 0, 0.5},
		{"default k when negative", 500, -1, 0.5},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := SampleBackingFactor(tt.n, tt.halfSaturation)
			if math.Abs(got-tt.want) > 1e-12 {
				t.Errorf("SampleBackingFactor(%d, %v) = %v, want %v",
					tt.n, tt.halfSaturation, got, tt.want)
			}
		})
	}
}

func TestSampleBackingFactorMonotoneConcave(t *testing.T) {
	// Strictly increasing and bounded in [0,1); concave (diminishing returns).
	// Concavity is tested over EQUALLY-spaced n (step 500): the marginal gain per
	// fixed-size step must strictly decrease. (Over geometrically-spaced n the
	// gain can rise — that does not contradict concavity.)
	prev := SampleBackingFactor(0, 500) // 0
	prevGain := math.Inf(1)
	for n := 500; n <= 10000; n += 500 {
		f := SampleBackingFactor(n, 500)
		if f < 0 || f >= 1 {
			t.Errorf("factor out of [0,1): n=%d -> %v", n, f)
		}
		if f <= prev {
			t.Errorf("not strictly increasing at n=%d: %v <= %v", n, f, prev)
		}
		gain := f - prev
		if gain >= prevGain {
			t.Errorf("not concave at n=%d: gain %v >= prev gain %v", n, gain, prevGain)
		}
		prevGain = gain
		prev = f
	}
}

// ---------------------------------------------------------------------------
// Score
// ---------------------------------------------------------------------------

func TestScore(t *testing.T) {
	tests := []struct {
		name       string
		n          int
		effect     float64
		tierWeight float64
		k          float64
		want       float64
	}{
		// 0.75 backing * 1.0 * 1.0 = 0.75.
		{"strong all dimensions", 1500, 1.0, 1.0, 500, 0.75},
		// 0.5 backing * 0.5 effect * 1.0 = 0.25.
		{"moderate sample weak effect", 500, 0.5, 1.0, 500, 0.25},
		// 0.75 backing * 0.8 effect * 0.5 tier = 0.30.
		{"tier discount", 1500, 0.8, 0.5, 500, 0.30},
		// n=0 -> backing 0 -> 0 regardless of effect/tier.
		{"no sample caps to zero", 0, 1.0, 1.0, 500, 0.0},
		// effect > 1 clamps to 1: 0.5 backing * 1 * 1 = 0.5.
		{"effect over one clamps", 500, 2.0, 1.0, 500, 0.5},
		// negative effect clamps to 0 -> score 0.
		{"negative effect clamps to zero", 1500, -0.3, 1.0, 500, 0.0},
		// negative tier weight clamps to 0 -> score 0.
		{"negative tier clamps to zero", 1500, 1.0, -1.0, 500, 0.0},
		// tier > 1 clamps to 1: 0.9 backing * 1 * 1 = 0.9.
		{"tier over one clamps", 4500, 1.0, 5.0, 500, 0.9},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Score(tt.n, tt.effect, tt.tierWeight, tt.k)
			if math.Abs(got-tt.want) > 1e-12 {
				t.Errorf("Score(%d, %v, %v, %v) = %v, want %v",
					tt.n, tt.effect, tt.tierWeight, tt.k, got, tt.want)
			}
		})
	}
}

func TestScoreNaNInputsAreZeroed(t *testing.T) {
	// A NaN effect or tier weight must not propagate; clamp01 maps NaN -> 0.
	if got := Score(1500, math.NaN(), 1.0, 500); got != 0 {
		t.Errorf("Score with NaN effect = %v, want 0", got)
	}
	if got := Score(1500, 1.0, math.NaN(), 500); got != 0 {
		t.Errorf("Score with NaN tier = %v, want 0", got)
	}
}

// ---------------------------------------------------------------------------
// GradeScore / Grade.String
// ---------------------------------------------------------------------------

func TestGradeScore(t *testing.T) {
	tests := []struct {
		name  string
		score float64
		want  Grade
	}{
		{"zero is none", 0.0, GradeNone},
		{"just below weak", 0.2499, GradeNone},
		{"weak lower edge inclusive", 0.25, GradeWeak},
		{"just below moderate", 0.4999, GradeWeak},
		{"moderate lower edge inclusive", 0.50, GradeModerate},
		{"just below strong", 0.7499, GradeModerate},
		{"strong lower edge inclusive", 0.75, GradeStrong},
		{"one is strong", 1.0, GradeStrong},
		// Out-of-range / NaN clamp first.
		{"above one is strong", 1.5, GradeStrong},
		{"negative is none", -0.4, GradeNone},
		{"NaN is none", math.NaN(), GradeNone},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := GradeScore(tt.score)
			if got != tt.want {
				t.Errorf("GradeScore(%v) = %v, want %v", tt.score, got, tt.want)
			}
		})
	}
}

func TestGradeString(t *testing.T) {
	tests := []struct {
		g    Grade
		want string
	}{
		{GradeNone, "None"},
		{GradeWeak, "Weak"},
		{GradeModerate, "Moderate"},
		{GradeStrong, "Strong"},
		{Grade(99), "Unknown"},
	}
	for _, tt := range tests {
		if got := tt.g.String(); got != tt.want {
			t.Errorf("Grade(%d).String() = %q, want %q", int(tt.g), got, tt.want)
		}
	}
}

// ---------------------------------------------------------------------------
// End-to-end: Summarize -> Score -> Grade composes coherently.
// ---------------------------------------------------------------------------

func TestEndToEndComposition(t *testing.T) {
	// A strong, well-powered conclusion: large weakest-link sample, full effect,
	// top tier -> Strong grade.
	s := Summarize([]int{4500, 6000}, 500)
	if s.Underpowered {
		t.Fatalf("expected not underpowered, got %+v", s)
	}
	score := Score(s.Min, 1.0, 1.0, 500) // weakest link 4500 -> 0.9
	if math.Abs(score-0.9) > 1e-12 {
		t.Errorf("composed score = %v, want 0.9", score)
	}
	if g := GradeScore(score); g != GradeStrong {
		t.Errorf("composed grade = %v, want Strong", g)
	}

	// A thin conclusion: weakest link 120 is underpowered and scores Weak/None.
	s2 := Summarize([]int{3000, 120}, 500)
	if !s2.Underpowered {
		t.Fatalf("expected underpowered, got %+v", s2)
	}
	score2 := Score(s2.Min, 1.0, 1.0, 500) // 120/(120+500) ~= 0.1935
	if g := GradeScore(score2); g != GradeNone {
		t.Errorf("thin-evidence grade = %v (score %v), want None", g, score2)
	}
}
