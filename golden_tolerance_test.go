package reality_test

// Golden tolerance ratchet.
//
// An absolute tolerance at least as large as every nonzero expected value
// accepts 0 and twice the expected values alike, so the case checks nothing.
// This test reads every golden file in the module and fails on any such case
// unless it is listed in vacuousGoldenTolerances. A listed case that is no longer
// vacuous also fails, so the list can only shrink. To fix a listed case, give
// it an expected value checked against an independent oracle and
// "tolerance_kind": "rel" (see testutil.TestCase).

import (
	"encoding/json"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

// vacuousGoldenTolerances maps "<file>#<case description>" to why the case is
// listed. Remove an entry in the change that fixes it.
var vacuousGoldenTolerances = map[string]string{}

type goldenToleranceCase struct {
	Description   string   `json:"description"`
	Expected      any      `json:"expected"`
	Tolerance     *float64 `json:"tolerance"`
	ToleranceKind string   `json:"tolerance_kind"`
}

// numericLeaves collects every finite, nonzero number in a decoded JSON value
// (a scalar, an array, or nested arrays).
func numericLeaves(v any, out *[]float64) {
	switch x := v.(type) {
	case float64:
		if x != 0 && !math.IsInf(x, 0) && !math.IsNaN(x) {
			*out = append(*out, math.Abs(x))
		}
	case []any:
		for _, e := range x {
			numericLeaves(e, out)
		}
	}
}

func TestGoldenTolerancesAreNotVacuous(t *testing.T) {
	root, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	skip := map[string]bool{"testdata/determinism": true, "testdata/stress": true}

	vacuous := map[string]string{}
	scanned := 0
	err = filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel(root, path)
		rel = filepath.ToSlash(rel)
		if d.IsDir() {
			if d.Name() == ".git" || skip[rel] {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(rel, ".json") || !strings.Contains("/"+rel, "/testdata/") {
			return nil
		}
		data, err := os.ReadFile(path) // #nosec G304 -- golden files inside this module
		if err != nil {
			return err
		}
		var gf struct {
			Cases []goldenToleranceCase `json:"cases"`
		}
		if json.Unmarshal(data, &gf) != nil {
			return nil // not a golden file
		}
		for _, c := range gf.Cases {
			if c.Tolerance == nil {
				continue
			}
			scanned++
			if c.ToleranceKind != "" && c.ToleranceKind != "abs" {
				continue
			}
			var leaves []float64
			numericLeaves(c.Expected, &leaves)
			if len(leaves) == 0 {
				continue
			}
			largest := leaves[0]
			for _, v := range leaves[1:] {
				largest = math.Max(largest, v)
			}
			if *c.Tolerance >= largest {
				vacuous[rel+"#"+c.Description] = ""
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if scanned == 0 {
		t.Fatal("found no golden cases: the scan is broken (this module has hundreds)")
	}

	var unlisted, stale []string
	for key := range vacuous {
		if _, ok := vacuousGoldenTolerances[key]; !ok {
			unlisted = append(unlisted, key)
		}
	}
	for key := range vacuousGoldenTolerances {
		if _, ok := vacuous[key]; !ok {
			stale = append(stale, key)
		}
	}
	sort.Strings(unlisted)
	sort.Strings(stale)
	if len(unlisted) > 0 {
		t.Errorf("%d golden case(s) have an absolute tolerance at least as large as every nonzero expected value, so they accept 0 and twice the values alike. Use \"tolerance_kind\": \"rel\" with an oracle-checked expected value:\n  %s",
			len(unlisted), strings.Join(unlisted, "\n  "))
	}
	if len(stale) > 0 {
		t.Errorf("%d listed case(s) are no longer vacuous; remove them from vacuousGoldenTolerances:\n  %s",
			len(stale), strings.Join(stale, "\n  "))
	}
	t.Logf("scanned %d golden cases; %d vacuous, %d of them listed", scanned, len(vacuous), len(vacuous)-len(unlisted))
}
