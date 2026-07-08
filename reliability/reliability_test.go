package reliability

import (
	"math"
	"testing"

	"github.com/davly/reality/graph"
	"github.com/davly/reality/testutil"
)

// ═══════════════════════════════════════════════════════════════════════════
// Golden-file tests. Reference values are produced by an INDEPENDENT Python
// implementation of the same textbook RBD formulas (Rausand & Hoyland 2004;
// Trivedi 2001; Birnbaum 1969); the Go implementation must reproduce them.
// ═══════════════════════════════════════════════════════════════════════════

func TestGolden_SeriesAvailability(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/reliability/series_availability.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			testutil.AssertFloat64(t, tc, SeriesAvailability(inputFloatSlice(t, tc, "a")))
		})
	}
}

func TestGolden_ParallelAvailability(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/reliability/parallel_availability.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			testutil.AssertFloat64(t, tc, ParallelAvailability(inputFloatSlice(t, tc, "a")))
		})
	}
}

func TestGolden_KofN(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/reliability/kofn.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			k := testutil.InputInt(t, tc, "k")
			n := testutil.InputInt(t, tc, "n")
			a := testutil.InputFloat64(t, tc, "a")
			testutil.AssertFloat64(t, tc, KofN(k, n, a))
		})
	}
}

func TestGolden_AvailabilityFromMTBF(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/reliability/availability_from_mtbf.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			mtbf := testutil.InputFloat64(t, tc, "mtbf")
			mttr := testutil.InputFloat64(t, tc, "mttr")
			testutil.AssertFloat64(t, tc, AvailabilityFromMTBF(mtbf, mttr))
		})
	}
}

func TestGolden_SystemAvailability(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/reliability/system_availability.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			edges := inputEdges(t, tc)
			avail := inputAvail(t, tc)
			target := inputString(t, tc, "target")
			testutil.AssertFloat64(t, tc, SystemAvailability(edges, avail, target))
		})
	}
}

func TestGolden_BirnbaumImportance(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/reliability/birnbaum_importance.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			edges := inputEdges(t, tc)
			avail := inputAvail(t, tc)
			target := inputString(t, tc, "target")
			component := inputString(t, tc, "component")
			testutil.AssertFloat64(t, tc, BirnbaumImportance(edges, avail, target, component))
		})
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Behavioural / invariant unit tests
// ═══════════════════════════════════════════════════════════════════════════

func TestSeries_EmptyIsIdentity(t *testing.T) {
	if got := SeriesAvailability(nil); got != 1.0 {
		t.Errorf("empty series = %v, want 1.0", got)
	}
}

func TestParallel_EmptyIsZero(t *testing.T) {
	if got := ParallelAvailability(nil); got != 0.0 {
		t.Errorf("empty parallel = %v, want 0.0", got)
	}
}

// KofN must agree with the closed forms at its two boundaries: k==n is series
// (a^n) and k==1 is parallel (1-(1-a)^n).
func TestKofN_ReducesToSeriesAndParallel(t *testing.T) {
	for _, a := range []float64{0.1, 0.5, 0.9, 0.99} {
		for _, n := range []int{1, 2, 3, 5} {
			series := math.Pow(a, float64(n))
			if got := KofN(n, n, a); math.Abs(got-series) > 1e-12 {
				t.Errorf("KofN(%d,%d,%g)=%v, want series %v", n, n, a, got, series)
			}
			parallel := 1 - math.Pow(1-a, float64(n))
			if got := KofN(1, n, a); math.Abs(got-parallel) > 1e-12 {
				t.Errorf("KofN(1,%d,%g)=%v, want parallel %v", n, a, got, parallel)
			}
		}
	}
}

// KofN rows must sum consistently: KofN(k) - KofN(k+1) = P(exactly k up), all
// non-negative and monotone non-increasing in k.
func TestKofN_MonotoneInK(t *testing.T) {
	a, n := 0.8, 6
	prev := 2.0
	for k := 0; k <= n+1; k++ {
		v := KofN(k, n, a)
		if v < 0 || v > 1 {
			t.Errorf("KofN(%d,%d,%g)=%v out of [0,1]", k, n, a, v)
		}
		if v > prev+1e-15 {
			t.Errorf("KofN not monotone: KofN(%d)=%v > KofN(%d)=%v", k, v, k-1, prev)
		}
		prev = v
	}
}

func TestBinom_KnownValues(t *testing.T) {
	cases := []struct {
		n, k int
		want float64
	}{{0, 0, 1}, {5, 0, 1}, {5, 5, 1}, {5, 2, 10}, {6, 3, 20}, {10, 4, 210}, {5, 6, 0}, {5, -1, 0}}
	for _, c := range cases {
		if got := binom(c.n, c.k); got != c.want {
			t.Errorf("binom(%d,%d)=%v, want %v", c.n, c.k, got, c.want)
		}
	}
}

// A diamond dependency must count the shared node exactly once: the naive
// recursion would square db's availability. Verify SystemAvailability == the
// product over the unique closure set.
func TestSystemAvailability_DiamondCountsSharedOnce(t *testing.T) {
	edges := []graph.Edge{{"app", "A"}, {"app", "B"}, {"A", "db"}, {"B", "db"}}
	avail := map[string]float64{"app": 0.99, "A": 0.995, "B": 0.995, "db": 0.98}
	want := 0.99 * 0.995 * 0.995 * 0.98 // db once
	got := SystemAvailability(edges, avail, "app")
	if math.Abs(got-want) > 1e-12 {
		t.Errorf("diamond system availability = %v, want %v (shared db once)", got, want)
	}
	naiveDoubleCount := 0.99 * 0.995 * 0.995 * 0.98 * 0.98
	if math.Abs(got-naiveDoubleCount) < 1e-9 {
		t.Errorf("system availability appears to double-count shared dependency")
	}
}

// SystemAvailability of a service must equal a_service * product(dep availabilities)
// and must never exceed the availability of any single required dependency
// (the "you can never beat your weakest series link" invariant).
func TestSystemAvailability_CappedByWeakestLink(t *testing.T) {
	edges := []graph.Edge{{"svc", "db"}, {"db", "disk"}}
	avail := map[string]float64{"svc": 0.999, "db": 0.99, "disk": 0.9995}
	sys := SystemAvailability(edges, avail, "svc")
	for _, a := range avail {
		if sys > a+1e-12 {
			t.Errorf("system availability %v exceeds a component availability %v", sys, a)
		}
	}
}

// Birnbaum importance of a series component must equal A_sys / a_component, and
// the lowest-availability dependency must have the highest importance.
func TestBirnbaum_SeriesClosedForm(t *testing.T) {
	edges := []graph.Edge{{"t", "x"}, {"t", "y"}, {"t", "z"}}
	avail := map[string]float64{"t": 1.0, "x": 0.9, "y": 0.8, "z": 0.95}
	sys := SystemAvailability(edges, avail, "t")
	for _, comp := range []string{"x", "y", "z"} {
		want := sys / avail[comp]
		got := BirnbaumImportance(edges, avail, "t", comp)
		if math.Abs(got-want) > 1e-12 {
			t.Errorf("I_B(%s)=%v, want A_sys/a=%v", comp, got, want)
		}
	}
	// y has the lowest availability -> highest Birnbaum importance.
	imps := BirnbaumImportances(edges, avail, "t")
	if imps["y"] <= imps["x"] || imps["y"] <= imps["z"] {
		t.Errorf("expected y (min availability) to have max importance; got %v", imps)
	}
}

func TestBirnbaum_OutsideClosureIsZero(t *testing.T) {
	edges := []graph.Edge{{"t", "x"}}
	avail := map[string]float64{"t": 0.99, "x": 0.9, "w": 0.5}
	if got := BirnbaumImportance(edges, avail, "t", "w"); got != 0.0 {
		t.Errorf("I_B of node outside closure = %v, want 0", got)
	}
}

func TestLimitingDependency_PicksMinAvailability(t *testing.T) {
	edges := []graph.Edge{{"svc", "db"}, {"svc", "cache"}, {"db", "disk"}}
	avail := map[string]float64{"svc": 0.9999, "db": 0.997, "cache": 0.9995, "disk": 0.9998}
	name, a := LimitingDependency(edges, avail, "svc")
	if name != "db" || math.Abs(a-0.997) > 1e-12 {
		t.Errorf("LimitingDependency = (%q,%v), want (db,0.997)", name, a)
	}
}

func TestLimitingDependency_NoDeps(t *testing.T) {
	edges := []graph.Edge{{"a", "b"}}
	name, a := LimitingDependency(edges, map[string]float64{"a": 0.9, "b": 0.8}, "b")
	if name != "" || !math.IsNaN(a) {
		t.Errorf("LimitingDependency with no deps = (%q,%v), want (\"\",NaN)", name, a)
	}
}

func TestLimitingDependency_DeterministicTieBreak(t *testing.T) {
	edges := []graph.Edge{{"svc", "z"}, {"svc", "a"}}
	avail := map[string]float64{"svc": 0.99, "z": 0.95, "a": 0.95}
	for i := 0; i < 10; i++ {
		name, _ := LimitingDependency(edges, avail, "svc")
		if name != "a" {
			t.Errorf("tie should break to lexicographically smallest 'a', got %q", name)
		}
	}
}

func TestAvailabilityFromMTBF(t *testing.T) {
	if got := AvailabilityFromMTBF(999, 1); math.Abs(got-0.999) > 1e-12 {
		t.Errorf("A(999,1)=%v, want 0.999", got)
	}
	if got := AvailabilityFromMTBF(0, 0); !math.IsNaN(got) {
		t.Errorf("A(0,0)=%v, want NaN", got)
	}
	if got := AvailabilityFromMTBF(100, 0); got != 1.0 {
		t.Errorf("A(100,0)=%v, want 1.0", got)
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Golden-file input helpers
// ═══════════════════════════════════════════════════════════════════════════

func inputFloatSlice(t *testing.T, tc testutil.TestCase, key string) []float64 {
	t.Helper()
	val, ok := tc.Inputs[key]
	if !ok {
		t.Fatalf("[%s] missing input %q", tc.Description, key)
	}
	arr, ok := val.([]any)
	if !ok {
		t.Fatalf("[%s] input %q is not an array: %T", tc.Description, key, val)
	}
	out := make([]float64, len(arr))
	for i, e := range arr {
		f, ok := e.(float64)
		if !ok {
			t.Fatalf("[%s] input %q[%d] is not a number: %T", tc.Description, key, i, e)
		}
		out[i] = f
	}
	return out
}

func inputEdges(t *testing.T, tc testutil.TestCase) []graph.Edge {
	t.Helper()
	val, ok := tc.Inputs["edges"]
	if !ok {
		t.Fatalf("[%s] missing input 'edges'", tc.Description)
	}
	arr, ok := val.([]any)
	if !ok {
		t.Fatalf("[%s] edges is not an array: %T", tc.Description, val)
	}
	edges := make([]graph.Edge, len(arr))
	for i, elem := range arr {
		pair, ok := elem.([]any)
		if !ok || len(pair) != 2 {
			t.Fatalf("[%s] edge[%d] is not a 2-element array", tc.Description, i)
		}
		src, ok1 := pair[0].(string)
		dst, ok2 := pair[1].(string)
		if !ok1 || !ok2 {
			t.Fatalf("[%s] edge[%d] elements are not strings", tc.Description, i)
		}
		edges[i] = graph.Edge{src, dst}
	}
	return edges
}

func inputAvail(t *testing.T, tc testutil.TestCase) map[string]float64 {
	t.Helper()
	val, ok := tc.Inputs["avail"]
	if !ok {
		t.Fatalf("[%s] missing input 'avail'", tc.Description)
	}
	m, ok := val.(map[string]any)
	if !ok {
		t.Fatalf("[%s] avail is not an object: %T", tc.Description, val)
	}
	out := make(map[string]float64, len(m))
	for k, v := range m {
		f, ok := v.(float64)
		if !ok {
			t.Fatalf("[%s] avail[%q] is not a number: %T", tc.Description, k, v)
		}
		out[k] = f
	}
	return out
}

func inputString(t *testing.T, tc testutil.TestCase, key string) string {
	t.Helper()
	val, ok := tc.Inputs[key]
	if !ok {
		t.Fatalf("[%s] missing input %q", tc.Description, key)
	}
	s, ok := val.(string)
	if !ok {
		t.Fatalf("[%s] input %q is not a string: %T", tc.Description, key, val)
	}
	return s
}
