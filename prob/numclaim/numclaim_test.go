package numclaim

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

// ---- golden-file loading ----------------------------------------------------

type goldenOptions struct {
	MaxRoundDP   int     `json:"maxRoundDP"`
	PercentScale bool    `json:"percentScale"`
	Tolerance    float64 `json:"tolerance"`
}

func (g goldenOptions) opts() Options {
	return Options{MaxRoundDP: g.MaxRoundDP, PercentScale: g.PercentScale, Tolerance: g.Tolerance}
}

func loadJSON(t *testing.T, path string, v any) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	if err := json.Unmarshal(data, v); err != nil {
		t.Fatalf("parse %s: %v", path, err)
	}
}

// TestNumericEquivalent_Golden validates the boolean classifier AND the
// transform/roundDP labelling against the language-neutral contract in
// testdata/numeric_equivalent.json.
func TestNumericEquivalent_Golden(t *testing.T) {
	var gf struct {
		Options goldenOptions `json:"options"`
		Cases   []struct {
			Description string    `json:"description"`
			Claim       float64   `json:"claim"`
			Truth       float64   `json:"truth"`
			Matched     bool      `json:"matched"`
			Transform   Transform `json:"transform"`
			RoundDP     int       `json:"roundDP"`
		} `json:"cases"`
	}
	loadJSON(t, "testdata/numeric_equivalent.json", &gf)
	if len(gf.Cases) == 0 {
		t.Fatal("no cases loaded")
	}
	opts := gf.Options.opts()

	for _, c := range gf.Cases {
		if got := NumericEquivalent(c.Claim, c.Truth, opts); got != c.Matched {
			t.Errorf("[%s] NumericEquivalent(%v,%v)=%v, want %v",
				c.Description, c.Claim, c.Truth, got, c.Matched)
		}
		m := classify(c.Claim, c.Truth, opts)
		if m.matched != c.Matched {
			t.Errorf("[%s] classify matched=%v, want %v", c.Description, m.matched, c.Matched)
		}
		if m.transform != c.Transform {
			t.Errorf("[%s] transform=%q, want %q", c.Description, m.transform, c.Transform)
		}
		if m.roundDP != c.RoundDP {
			t.Errorf("[%s] roundDP=%d, want %d", c.Description, m.roundDP, c.RoundDP)
		}
	}
}

// TestClaimConsistency_Golden validates the per-claim verdicts (including
// nearest-miss reporting) against testdata/claim_consistency.json.
func TestClaimConsistency_Golden(t *testing.T) {
	var gf struct {
		Options goldenOptions `json:"options"`
		Cases   []struct {
			Description string    `json:"description"`
			Claims      []float64 `json:"claims"`
			Truths      []float64 `json:"truths"`
			Verdicts    []struct {
				ClaimIndex int       `json:"claimIndex"`
				Claim      float64   `json:"claim"`
				Matched    bool      `json:"matched"`
				Transform  Transform `json:"transform"`
				RoundDP    int       `json:"roundDP"`
				TruthIndex int       `json:"truthIndex"`
				Truth      float64   `json:"truth"`
			} `json:"verdicts"`
		} `json:"cases"`
	}
	loadJSON(t, "testdata/claim_consistency.json", &gf)
	if len(gf.Cases) == 0 {
		t.Fatal("no cases loaded")
	}
	opts := gf.Options.opts()

	for _, c := range gf.Cases {
		got := ClaimConsistency(c.Claims, c.Truths, opts)
		if len(got) != len(c.Verdicts) {
			t.Fatalf("[%s] got %d verdicts, want %d", c.Description, len(got), len(c.Verdicts))
		}
		for i, want := range c.Verdicts {
			g := got[i]
			if g.ClaimIndex != want.ClaimIndex || g.Claim != want.Claim {
				t.Errorf("[%s] verdict %d: claim (%d,%v), want (%d,%v)",
					c.Description, i, g.ClaimIndex, g.Claim, want.ClaimIndex, want.Claim)
			}
			if g.Matched != want.Matched {
				t.Errorf("[%s] verdict %d: matched=%v, want %v", c.Description, i, g.Matched, want.Matched)
			}
			if g.Transform != want.Transform {
				t.Errorf("[%s] verdict %d: transform=%q, want %q", c.Description, i, g.Transform, want.Transform)
			}
			if g.RoundDP != want.RoundDP {
				t.Errorf("[%s] verdict %d: roundDP=%d, want %d", c.Description, i, g.RoundDP, want.RoundDP)
			}
			if g.TruthIndex != want.TruthIndex {
				t.Errorf("[%s] verdict %d: truthIndex=%d, want %d", c.Description, i, g.TruthIndex, want.TruthIndex)
			}
			if math.Abs(g.Truth-want.Truth) > 1e-12 {
				t.Errorf("[%s] verdict %d: truth=%v, want %v", c.Description, i, g.Truth, want.Truth)
			}
		}
	}
}

// ---- option-sensitivity unit tests -----------------------------------------

func TestMaxRoundDP_ControlsRoundingReach(t *testing.T) {
	// 0.4% (=0.4) vs 0.00366: percent-scaled claim is 0.004, which equals the
	// truth only after rounding to 3 decimal places.
	claim, truth := 0.4, 0.00366
	if NumericEquivalent(claim, truth, DefaultOptions()) {
		t.Error("with MaxRoundDP=2, 0.4% should NOT match 0.00366")
	}
	deep := Options{MaxRoundDP: 3, PercentScale: true, Tolerance: 1e-9}
	if !NumericEquivalent(claim, truth, deep) {
		t.Error("with MaxRoundDP=3, 0.4% should match 0.00366 (0.004 rounds 0.00366)")
	}
	m := classify(claim, truth, deep)
	if m.transform != TransformPercentToFraction || m.roundDP != 3 {
		t.Errorf("transform=%q roundDP=%d, want percent_to_fraction/3", m.transform, m.roundDP)
	}
}

func TestNegativeMaxRoundDP_DisablesRounding(t *testing.T) {
	off := Options{MaxRoundDP: -1, PercentScale: true, Tolerance: 1e-9}
	if NumericEquivalent(53, 52.6, off) {
		t.Error("with rounding disabled, 53 should NOT match 52.6")
	}
	if !NumericEquivalent(53, 53.0, off) {
		t.Error("exact match must still hold with rounding disabled")
	}
}

func TestPercentScaleOff(t *testing.T) {
	off := Options{MaxRoundDP: 2, PercentScale: false, Tolerance: 1e-9}
	if NumericEquivalent(45, 0.45, off) {
		t.Error("with PercentScale off, 45 should NOT match 0.45")
	}
}

func TestNegativeToleranceClampedToZero(t *testing.T) {
	// A negative tolerance must not admit spurious matches; exact still matches.
	opts := Options{MaxRoundDP: 2, PercentScale: true, Tolerance: -1}
	if NumericEquivalent(52.61, 52.6, opts) {
		t.Error("52.61 must not match 52.6 with clamped-zero tolerance")
	}
	if !NumericEquivalent(52.6, 52.6, opts) {
		t.Error("exact equality must hold with clamped-zero tolerance")
	}
}

// ---- edge cases -------------------------------------------------------------

func TestNonFiniteNeverMatches(t *testing.T) {
	opts := DefaultOptions()
	for _, bad := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		if NumericEquivalent(bad, 1.0, opts) {
			t.Errorf("claim %v must not match", bad)
		}
		if NumericEquivalent(1.0, bad, opts) {
			t.Errorf("truth %v must not match", bad)
		}
	}
}

func TestClaimConsistency_EmptyTruths(t *testing.T) {
	got := ClaimConsistency([]float64{42.0}, nil, DefaultOptions())
	if len(got) != 1 {
		t.Fatalf("got %d verdicts", len(got))
	}
	v := got[0]
	if v.Matched {
		t.Error("no truths => must be unmatched")
	}
	if v.TruthIndex != -1 {
		t.Errorf("truthIndex=%d, want -1 for empty pool", v.TruthIndex)
	}
	if !math.IsNaN(v.Truth) {
		t.Errorf("truth=%v, want NaN for empty pool", v.Truth)
	}
}

func TestClaimConsistency_NoClaims(t *testing.T) {
	got := ClaimConsistency(nil, []float64{1, 2, 3}, DefaultOptions())
	if len(got) != 0 {
		t.Fatalf("got %d verdicts, want 0", len(got))
	}
}

func TestRelError_Advisory(t *testing.T) {
	// nearest-miss relError for the £520 vs pool {52.6, 188.8}: closest is
	// 188.8 at |520-188.8|/520 = 331.2/520.
	got := ClaimConsistency([]float64{520}, []float64{52.6, 188.8}, DefaultOptions())
	v := got[0]
	if v.Matched {
		t.Fatal("520 should be unmatched")
	}
	if v.TruthIndex != 1 {
		t.Fatalf("nearest truthIndex=%d, want 1", v.TruthIndex)
	}
	want := 331.2 / 520.0
	if math.Abs(v.RelError-want) > 1e-9 {
		t.Errorf("relError=%v, want %v", v.RelError, want)
	}
	// exact match has ~0 relError.
	exact := ClaimConsistency([]float64{52.6}, []float64{52.6}, DefaultOptions())
	if exact[0].RelError != 0 {
		t.Errorf("exact-match relError=%v, want 0", exact[0].RelError)
	}
}

func TestFirstMatchingTruthWins(t *testing.T) {
	// Two equally-valid truths; the lower index must be reported.
	got := ClaimConsistency([]float64{45}, []float64{0.45, 0.45}, DefaultOptions())
	if got[0].TruthIndex != 0 {
		t.Errorf("truthIndex=%d, want 0 (first match wins)", got[0].TruthIndex)
	}
}
