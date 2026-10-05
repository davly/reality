package testutil

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"testing"
)

// DistinctOutputs calls f `calls` times and returns how many distinct
// fingerprints it produced. f must return an exact fingerprint of its result:
// use FloatBits for floats, so that values which print alike but differ in
// the last bit are told apart.
func DistinctOutputs(calls int, f func() string) int {
	seen := map[string]struct{}{}
	for i := 0; i < calls; i++ {
		seen[f()] = struct{}{}
	}
	return len(seen)
}

// AssertDeterministic fails the test unless f returns the same fingerprint on
// every one of `calls` calls. Go randomises map iteration, so a function that
// accumulates or picks winners while ranging over a map can return different
// results for the same input; repeated calls in one process expose that.
func AssertDeterministic(t *testing.T, name string, calls int, f func() string) {
	t.Helper()
	if n := DistinctOutputs(calls, f); n != 1 {
		t.Errorf("%s: %d distinct outputs over %d identical calls; it must be deterministic", name, n, calls)
	}
}

// FloatBits fingerprints floats exactly (their IEEE-754 bit patterns).
func FloatBits(xs ...float64) string {
	var b strings.Builder
	for i, x := range xs {
		if i > 0 {
			b.WriteByte(',')
		}
		fmt.Fprintf(&b, "%016x", math.Float64bits(x))
	}
	return b.String()
}

// SortedMapBits fingerprints a map with float values exactly, in key order.
func SortedMapBits[K comparable](m map[K]float64) string {
	keys := make([]string, 0, len(m))
	byKey := make(map[string]float64, len(m))
	for k, v := range m {
		ks := fmt.Sprint(k)
		keys = append(keys, ks)
		byKey[ks] = v
	}
	sort.Strings(keys)
	var b strings.Builder
	for _, k := range keys {
		fmt.Fprintf(&b, "%s=%s;", k, FloatBits(byKey[k]))
	}
	return b.String()
}
