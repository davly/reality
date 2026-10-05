// Package testutil provides golden-file test infrastructure for the Reality
// library. Every function in Reality is validated against golden-file test
// vectors stored as JSON. The vectors are language-neutral, so a port to
// another language can validate against the same files.
//
// Golden files live in testdata/ directories relative to each package.
// The canonical format is:
//
//	{
//	  "function": "PackageName.FunctionName",
//	  "cases": [
//	    {
//	      "description": "human-readable description",
//	      "inputs": {"x": 1.0, "y": 2.0},
//	      "expected": 3.0,
//	      "tolerance": 1e-15
//	    }
//	  ]
//	}
//
// expected may be a single float64 or an array of float64 for vector-valued
// functions. tolerance is per-case, not global — some functions need 1e-15,
// others need 1e-6 for iterative algorithms. An optional "tolerance_kind"
// says how tolerance is applied: "abs" (the default), "rel" or "ulp"; see
// TestCase.
package testutil

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// TestCase holds a single golden-file test vector.
type TestCase struct {
	// Description is a human-readable label for this test vector.
	Description string `json:"description"`

	// Inputs maps parameter names to their values. Values may be float64,
	// []float64, string, bool, or nested maps depending on the function.
	Inputs map[string]any `json:"inputs"`

	// Expected holds the expected result. For scalar functions this is a
	// float64. For vector-valued functions this is a []float64 (decoded as
	// []any from JSON, then converted).
	Expected any `json:"expected"`

	// Tolerance bounds the difference between the computed result and the
	// expected value, in the unit ToleranceKind selects. Per-case, not global.
	Tolerance float64 `json:"tolerance"`

	// ToleranceKind says how Tolerance is applied:
	//
	//	"" or "abs"  |got - expected| <= tolerance
	//	"rel"        |got - expected| <= tolerance * |expected|; an expected
	//	             value of exactly 0 falls back to the absolute check
	//	"ulp"        |got - expected| <= tolerance units in the last place
	//	             of expected
	//
	// Prefer "rel" for values that span many magnitudes: an absolute
	// tolerance at least as large as |expected| accepts 0 and 2*expected
	// alike, so it checks nothing.
	ToleranceKind string `json:"tolerance_kind,omitempty"`
}

// GoldenFile represents a complete set of test vectors for a single function.
type GoldenFile struct {
	// Function identifies the function under test in "Package.Function" form.
	Function string `json:"function"`

	// Cases is the ordered list of test vectors.
	Cases []TestCase `json:"cases"`
}

// LoadGolden reads and parses a golden-file JSON from the given path.
// The path is resolved relative to the directory of the calling test file,
// making it safe to use from any package's test suite.
//
// If the file cannot be read or parsed, the test is failed immediately
// via t.Fatal.
func LoadGolden(t *testing.T, path string) GoldenFile {
	t.Helper()

	// Resolve path relative to the caller's source file directory.
	_, callerFile, _, ok := runtime.Caller(1)
	if !ok {
		t.Fatal("testutil.LoadGolden: unable to determine caller file path")
	}
	callerDir := filepath.Dir(callerFile)
	if !filepath.IsAbs(callerDir) {
		// Built with -trimpath: the compile-time path is module-relative
		// (for example github.com/davly/reality/prob). go test runs each test
		// binary in its package directory, which is the caller's directory.
		wd, err := os.Getwd()
		if err != nil {
			t.Fatalf("testutil.LoadGolden: resolving %s under -trimpath: %v", path, err)
		}
		callerDir = wd
	}
	absPath := filepath.Join(callerDir, path)

	data, err := os.ReadFile(absPath) // #nosec G304 -- test helper: reads golden files from the calling test's own testdata
	if err != nil {
		t.Fatalf("testutil.LoadGolden: failed to read %s: %v", absPath, err)
	}

	var gf GoldenFile
	if err := json.Unmarshal(data, &gf); err != nil {
		t.Fatalf("testutil.LoadGolden: failed to parse %s: %v", absPath, err)
	}

	if len(gf.Cases) == 0 {
		t.Fatalf("testutil.LoadGolden: %s contains no test cases", absPath)
	}

	return gf
}

// AssertFloat64 checks that a single float64 result matches the expected
// value within the test case's tolerance, applied as the case's
// ToleranceKind says (absolute by default).
//
// Special values are handled: if both got and expected are NaN, the assertion
// passes. If both are the same infinity, the assertion passes.
func AssertFloat64(t *testing.T, tc TestCase, got float64) {
	t.Helper()

	expected, ok := toFloat64(tc.Expected)
	if !ok {
		t.Fatalf("[%s] expected value is not a float64: %v (type %T)",
			tc.Description, tc.Expected, tc.Expected)
	}

	// Handle special float64 values.
	if math.IsNaN(expected) && math.IsNaN(got) {
		return
	}
	if math.IsInf(expected, 1) && math.IsInf(got, 1) {
		return
	}
	if math.IsInf(expected, -1) && math.IsInf(got, -1) {
		return
	}

	measured, ok, err := Within(tc, got, expected)
	if err != nil {
		t.Fatalf("[%s] %v", tc.Description, err)
	}
	if !ok {
		t.Errorf("[%s] got %v, expected %v (%s error %v exceeds tolerance %v)",
			tc.Description, got, expected, kindName(tc), measured, tc.Tolerance)
	}
}

// AssertFloat64Slice checks that a []float64 result matches the expected
// slice element-wise, each within the test case's tolerance.
func AssertFloat64Slice(t *testing.T, tc TestCase, got []float64) {
	t.Helper()

	expected, ok := toFloat64Slice(tc.Expected)
	if !ok {
		t.Fatalf("[%s] expected value is not a float64 slice: %v (type %T)",
			tc.Description, tc.Expected, tc.Expected)
	}

	if len(got) != len(expected) {
		t.Fatalf("[%s] length mismatch: got %d elements, expected %d",
			tc.Description, len(got), len(expected))
	}

	for i := range expected {
		if math.IsNaN(expected[i]) && math.IsNaN(got[i]) {
			continue
		}
		if math.IsInf(expected[i], 1) && math.IsInf(got[i], 1) {
			continue
		}
		if math.IsInf(expected[i], -1) && math.IsInf(got[i], -1) {
			continue
		}
		measured, ok, err := Within(tc, got[i], expected[i])
		if err != nil {
			t.Fatalf("[%s] %v", tc.Description, err)
		}
		if !ok {
			t.Errorf("[%s] element %d: got %v, expected %v (%s error %v exceeds tolerance %v)",
				tc.Description, i, got[i], expected[i], kindName(tc), measured, tc.Tolerance)
		}
	}
}

// Within reports whether got matches expected under tc's tolerance, and the
// error measured in the unit tc.ToleranceKind selects (an absolute
// difference, a relative difference, or a count of units in the last place).
// It returns an error for an unknown ToleranceKind.
func Within(tc TestCase, got, expected float64) (measured float64, ok bool, err error) {
	diff := math.Abs(got - expected)
	switch tc.ToleranceKind {
	case "", "abs":
		return diff, diff <= tc.Tolerance, nil
	case "rel":
		if expected == 0 {
			return diff, diff <= tc.Tolerance, nil
		}
		rel := diff / math.Abs(expected)
		return rel, rel <= tc.Tolerance, nil
	case "ulp":
		ulps := diff / ulp(expected)
		return ulps, ulps <= tc.Tolerance, nil
	}
	return 0, false, fmt.Errorf("unknown tolerance_kind %q (want \"abs\", \"rel\" or \"ulp\")", tc.ToleranceKind)
}

// ulp returns the spacing between |x| and the next larger float64.
func ulp(x float64) float64 {
	x = math.Abs(x)
	return math.Nextafter(x, math.Inf(1)) - x
}

func kindName(tc TestCase) string {
	if tc.ToleranceKind == "" {
		return "abs"
	}
	return tc.ToleranceKind
}

// InputFloat64 extracts a named float64 input from a test case's Inputs map.
// Fails the test if the key is missing or the value is not a float64.
func InputFloat64(t *testing.T, tc TestCase, key string) float64 {
	t.Helper()

	val, exists := tc.Inputs[key]
	if !exists {
		t.Fatalf("[%s] missing input key %q", tc.Description, key)
	}

	f, ok := toFloat64(val)
	if !ok {
		t.Fatalf("[%s] input %q is not a float64: %v (type %T)",
			tc.Description, key, val, val)
	}

	return f
}

// InputFloat64Slice extracts a named []float64 input from a test case's
// Inputs map. Fails the test if the key is missing or cannot be converted.
func InputFloat64Slice(t *testing.T, tc TestCase, key string) []float64 {
	t.Helper()

	val, exists := tc.Inputs[key]
	if !exists {
		t.Fatalf("[%s] missing input key %q", tc.Description, key)
	}

	s, ok := toFloat64Slice(val)
	if !ok {
		t.Fatalf("[%s] input %q is not a float64 slice: %v (type %T)",
			tc.Description, key, val, val)
	}

	return s
}

// InputInt extracts a named integer input from a test case's Inputs map.
// JSON numbers are decoded as float64, so this converts to int after checking
// that the value is an integer (no fractional part). Fails the test if the
// key is missing, the value is not numeric, or has a fractional part.
func InputInt(t *testing.T, tc TestCase, key string) int {
	t.Helper()

	val, exists := tc.Inputs[key]
	if !exists {
		t.Fatalf("[%s] missing input key %q", tc.Description, key)
	}

	f, ok := toFloat64(val)
	if !ok {
		t.Fatalf("[%s] input %q is not numeric: %v (type %T)",
			tc.Description, key, val, val)
	}

	i := int(f)
	if float64(i) != f {
		t.Fatalf("[%s] input %q is not an integer: %v",
			tc.Description, key, f)
	}

	return i
}

// toFloat64 converts a JSON-decoded value to float64.
// JSON numbers decode as float64, so this handles the common case.
func toFloat64(v any) (float64, bool) {
	switch val := v.(type) {
	case float64:
		return val, true
	case int:
		return float64(val), true
	case int64:
		return float64(val), true
	case json.Number:
		f, err := val.Float64()
		return f, err == nil
	default:
		return 0, false
	}
}

// toFloat64Slice converts a JSON-decoded value to []float64.
// JSON arrays decode as []any, so each element is converted individually.
func toFloat64Slice(v any) ([]float64, bool) {
	arr, ok := v.([]any)
	if !ok {
		return nil, false
	}

	result := make([]float64, len(arr))
	for i, elem := range arr {
		f, ok := toFloat64(elem)
		if !ok {
			return nil, false
		}
		result[i] = f
	}
	return result, true
}
