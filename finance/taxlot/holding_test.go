package taxlot

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

type holdingGolden struct {
	Function string `json:"function"`
	Source   string `json:"source"`
	Cases    []struct {
		Description     string `json:"description"`
		AcquisitionDate string `json:"acquisition_date"`
		DisposalDate    string `json:"disposal_date"`
		Expected        string `json:"expected"`
	} `json:"cases"`
}

// TestGoldenHoldingPeriod validates the IRC §1222 short/long-term boundary
// against IRS Pub 544's published worked example and leap-year boundaries.
func TestGoldenHoldingPeriod(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("testdata", "holding_period.json"))
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g holdingGolden
	if err := json.Unmarshal(data, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	if len(g.Cases) == 0 {
		t.Fatal("golden file has no cases")
	}
	for _, c := range g.Cases {
		acq := parseTestDate(t, c.AcquisitionDate)
		disp := parseTestDate(t, c.DisposalDate)
		got := Classify(acq, disp).String()
		if got != c.Expected {
			t.Errorf("%s\n  Classify(%s, %s) = %s, want %s",
				c.Description, c.AcquisitionDate, c.DisposalDate, got, c.Expected)
		}
		// IsLongTerm must agree with Classify.
		if (got == "long-term") != IsLongTerm(acq, disp) {
			t.Errorf("%s: IsLongTerm disagrees with Classify", c.Description)
		}
	}
}

func TestClassifyNegativeHoldingIsShortTerm(t *testing.T) {
	// A disposal date before acquisition cannot be long-term.
	if IsLongTerm(D(2023, 5, 1), D(2023, 1, 1)) {
		t.Error("negative holding period must not be long-term")
	}
}
