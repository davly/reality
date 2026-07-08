package forge

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/davly/reality/forge/session40"
)

// parityFixture mirrors the schema of testdata/decide_parity.json. Non-finite
// dominance/confidence are encoded as the string tokens "+Inf"/"-Inf"/"NaN" so
// the fixture stays JSON-portable across substrates; json.Number lets us
// distinguish a numeric value from a token without losing precision.
type parityFixture struct {
	Schema      string `json:"schema"`
	Description string `json:"description"`
	Thresholds  struct {
		ConvergedDominance  float64 `json:"converged_dominance"`
		ConvergedConfidence float64 `json:"converged_confidence"`
		EscapeThreshold     float64 `json:"escape_threshold"`
		MinObservations     int     `json:"min_observations"`
	} `json:"thresholds"`
	Cases []struct {
		Name       string      `json:"name"`
		Dominance  interface{} `json:"dominance"`
		Confidence interface{} `json:"confidence"`
		Total      int         `json:"total"`
		Verdict    string      `json:"verdict"`
	} `json:"cases"`
}

// decodeFloat maps a JSON value (a float64 from encoding/json, or one of the
// non-finite string tokens) to a Go float64.
func decodeFloat(t *testing.T, v interface{}) float64 {
	t.Helper()
	switch x := v.(type) {
	case float64:
		return x
	case string:
		switch x {
		case "+Inf":
			return math.Inf(1)
		case "-Inf":
			return math.Inf(-1)
		case "NaN":
			return math.NaN()
		default:
			t.Fatalf("unknown non-finite token %q", x)
		}
	default:
		t.Fatalf("unexpected JSON type %T for float field", v)
	}
	return 0
}

func loadFixture(t *testing.T) parityFixture {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", "decide_parity.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var fx parityFixture
	if err := json.Unmarshal(raw, &fx); err != nil {
		t.Fatalf("unmarshal fixture: %v", err)
	}
	return fx
}

// TestDecideParityFixture runs the language-neutral golden KAT: every case's
// verdict must be byte-identical to the fixture's expected verdict string.
func TestDecideParityFixture(t *testing.T) {
	fx := loadFixture(t)
	if fx.Schema != "forge.decide.parity.v1" {
		t.Fatalf("unexpected schema %q", fx.Schema)
	}
	if len(fx.Cases) == 0 {
		t.Fatal("fixture has no cases")
	}

	// Fixture thresholds must match the live consts (single source of truth).
	if fx.Thresholds.ConvergedDominance != ConvergedDominance {
		t.Errorf("fixture converged_dominance %v != const %v", fx.Thresholds.ConvergedDominance, ConvergedDominance)
	}
	if fx.Thresholds.ConvergedConfidence != ConvergedConfidence {
		t.Errorf("fixture converged_confidence %v != const %v", fx.Thresholds.ConvergedConfidence, ConvergedConfidence)
	}
	if fx.Thresholds.EscapeThreshold != EscapeThreshold {
		t.Errorf("fixture escape_threshold %v != const %v", fx.Thresholds.EscapeThreshold, EscapeThreshold)
	}
	if fx.Thresholds.MinObservations != MinObservations {
		t.Errorf("fixture min_observations %v != const %v", fx.Thresholds.MinObservations, MinObservations)
	}

	for _, c := range fx.Cases {
		dom := decodeFloat(t, c.Dominance)
		conf := decodeFloat(t, c.Confidence)
		got := Decide(dom, conf, c.Total).String()
		if got != c.Verdict {
			t.Errorf("case %q: Decide(%v, %v, %d) = %q, want %q",
				c.Name, c.Dominance, c.Confidence, c.Total, got, c.Verdict)
		}
	}
}

// TestDecide_BoundaryInclusivity nails the >=/< edges of every threshold.
func TestDecide_BoundaryInclusivity(t *testing.T) {
	cases := []struct {
		name       string
		dominance  float64
		confidence float64
		total      int
		want       Verdict
	}{
		// MinObservations is inclusive: total==3 is enough.
		{"min_obs_exact", 0.99, 0.99, 3, VerdictConverged},
		{"min_obs_minus_one", 0.99, 0.99, 2, VerdictUncertain},

		// Converged floors are inclusive (>=).
		{"converged_dom_exact", 0.70, 0.65, 5, VerdictConverged},
		{"converged_conf_exact", 0.80, 0.65, 5, VerdictConverged},
		{"converged_dom_below", 0.6999999, 0.99, 5, VerdictUncertain},
		{"converged_conf_below", 0.99, 0.6499999, 5, VerdictUncertain},

		// EscapeThreshold is exclusive (< escapes; == does NOT escape).
		{"escape_threshold_exact_not_escape", 0.60, 0.90, 5, VerdictUncertain},
		{"escape_just_below", 0.5999999, 0.90, 5, VerdictEscape},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := Decide(c.dominance, c.confidence, c.total); got != c.want {
				t.Errorf("Decide(%v, %v, %d) = %v, want %v", c.dominance, c.confidence, c.total, got, c.want)
			}
		})
	}
}

// TestDecide_NonFiniteFailsClosed is the paired regression test (R145) for the
// fail-closed behavior choice (OD-2). Without the non-finite guard,
// Decide(+Inf, +Inf, 5) would return VerdictConverged because +Inf satisfies
// both >= floors. The canonical fails CLOSED -> VerdictUncertain.
func TestDecide_NonFiniteFailsClosed(t *testing.T) {
	posInf := math.Inf(1)
	negInf := math.Inf(-1)
	nan := math.NaN()

	cases := []struct {
		name       string
		dominance  float64
		confidence float64
	}{
		{"pos_inf_both", posInf, posInf},
		{"pos_inf_dominance", posInf, 0.99},
		{"pos_inf_confidence", 0.99, posInf},
		{"neg_inf_dominance", negInf, 0.99},
		{"neg_inf_confidence", 0.99, negInf},
		{"nan_dominance", nan, 0.99},
		{"nan_confidence", 0.99, nan},
		{"nan_both", nan, nan},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := Decide(c.dominance, c.confidence, 5); got != VerdictUncertain {
				t.Errorf("Decide(%v, %v, 5) = %v, want VerdictUncertain (fail-closed)", c.dominance, c.confidence, got)
			}
		})
	}
}

// TestConstParityWithSession40 locks the forge thresholds to the basis-point
// constants baked into session40 (single source of truth, mirrors R61 const
// parity discipline): 0.65 <-> 6500 bps and min-obs 3.
func TestConstParityWithSession40(t *testing.T) {
	if got := uint16(math.Round(ConvergedConfidence * 10000)); got != session40.CanonicalVerdictConvergedFloor {
		t.Errorf("ConvergedConfidence*10000 = %d, want session40.CanonicalVerdictConvergedFloor = %d",
			got, session40.CanonicalVerdictConvergedFloor)
	}
	if MinObservations != session40.CanonicalMinObservations {
		t.Errorf("MinObservations = %d, want session40.CanonicalMinObservations = %d",
			MinObservations, session40.CanonicalMinObservations)
	}
}

// TestVerdictString round-trips the three canonical names.
func TestVerdictString(t *testing.T) {
	cases := map[Verdict]string{
		VerdictUncertain: "uncertain",
		VerdictConverged: "converged",
		VerdictEscape:    "escape",
	}
	for v, want := range cases {
		if got := v.String(); got != want {
			t.Errorf("Verdict(%d).String() = %q, want %q", int(v), got, want)
		}
	}
	if got := Verdict(99).String(); got != "unknown" {
		t.Errorf("Verdict(99).String() = %q, want \"unknown\"", got)
	}
}
