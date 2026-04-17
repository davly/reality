package testutil

import (
	"encoding/hex"
	"fmt"
	"hash/fnv"
	"math"
	"sort"
	"strconv"
	"testing"
)

// Session 38 — forge-primitive cross-language golden vectors.
//
// FNV-1a, SituationHash, and BasisPoints round-trip are cross-language
// *protocol* contracts rather than package-internal math, so reality hosts
// them as shared vectors under testdata/forge/. Every substrate in the
// ecosystem is expected to reproduce these byte-identical numbers.
//
// These tests reference the canonical Go stdlib hash/fnv.New64a() as the
// reference implementation — everyone else mirrors Go.

// parseHexU64 parses a 16-char hex string into a uint64. The expected values
// are stored as hex strings (not numbers) to preserve uint64 precision
// through JSON round-trip (JSON float64 has only ~15-16 digits).
func parseHexU64(t *testing.T, s string) uint64 {
	t.Helper()
	v, err := strconv.ParseUint(s, 16, 64)
	if err != nil {
		t.Fatalf("parseHexU64(%q): %v", s, err)
	}
	return v
}

// resolveInputBytes extracts the input bytes. Prefers `input_hex` (for raw
// binary inputs that can't survive UTF-8 round-trip), falls back to `input`
// (for text inputs).
func resolveInputBytes(t *testing.T, inputs map[string]any) []byte {
	t.Helper()
	if raw, ok := inputs["input_hex"].(string); ok {
		bs, err := hex.DecodeString(raw)
		if err != nil {
			t.Fatalf("bad input_hex %q: %v", raw, err)
		}
		return bs
	}
	if s, ok := inputs["input"].(string); ok {
		return []byte(s)
	}
	t.Fatalf("no input / input_hex in test case")
	return nil
}

func TestGoldenFnv1a_Canonical12(t *testing.T) {
	gf := LoadGolden(t, "../testdata/forge/fnv1a.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			in := resolveInputBytes(t, tc.Inputs)
			h := fnv.New64a()
			h.Write(in)
			got := h.Sum64()

			hexWant, ok := tc.Expected.(string)
			if !ok {
				t.Fatalf("expected not a hex string: %v (type %T)", tc.Expected, tc.Expected)
			}
			want := parseHexU64(t, hexWant)
			if got != want {
				t.Errorf("FNV-1a(bytes=%x) = %016x, want %016x", in, got, want)
			}
		})
	}
}

func TestGoldenSituationHash(t *testing.T) {
	gf := LoadGolden(t, "../testdata/forge/situation_hash.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			raw, ok := tc.Inputs["dimensions"].(map[string]any)
			if !ok {
				t.Fatalf("dimensions not a map: %v", tc.Inputs["dimensions"])
			}
			dims := make(map[string]string, len(raw))
			for k, v := range raw {
				s, _ := v.(string)
				dims[k] = s
			}
			got := situationHash(dims)

			hexWant, ok := tc.Expected.(string)
			if !ok {
				t.Fatalf("expected not a hex string: %v", tc.Expected)
			}
			want := parseHexU64(t, hexWant)
			if got != want {
				t.Errorf("SituationHash(%v) = %016x, want %016x", dims, got, want)
			}
		})
	}
}

func TestGoldenBasisPointsRoundTrip(t *testing.T) {
	gf := LoadGolden(t, "../testdata/forge/basispoints_roundtrip.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			p := InputFloat64(t, tc, "f64")
			got := bpsFromF64(p)
			AssertFloat64(t, tc, float64(got))
		})
	}
}

// situationHash replicates the canonical SituationHash algorithm so this
// test doesn't import any other package (testutil is upstream of prob).
func situationHash(in map[string]string) uint64 {
	keys := make([]string, 0, len(in))
	for k := range in {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	h := fnv.New64a()
	for _, k := range keys {
		h.Write([]byte(k))
		h.Write([]byte{0x00})
		h.Write([]byte(in[k]))
		h.Write([]byte{0x00})
	}
	return h.Sum64()
}

// bpsFromF64 replicates the canonical round-half-up conversion (mirrors
// aicore/innovations.BasisPointsFromF64 but without the dependency).
func bpsFromF64(p float64) uint16 {
	if math.IsNaN(p) || p <= 0 {
		return 0
	}
	if p >= 1 {
		return 10000
	}
	bp := math.Floor(p*10000.0 + 0.5)
	if bp > 10000 {
		return 10000
	}
	return uint16(bp)
}

// Sanity: forces a failure if anyone "fixes" FNV by using a different
// algorithm.
func TestFnv1a_EmptyInput_IsOffsetBasis(t *testing.T) {
	h := fnv.New64a()
	got := h.Sum64()
	if got != 14695981039346656037 {
		t.Errorf("FNV-1a empty = %d, canonical offset basis is 14695981039346656037", got)
	}
	_ = fmt.Sprintf // retain import (used by other tests when debugging)
}
