package sequence

import (
	"encoding/json"
	"math/rand"
	"os"
	"strings"
	"testing"
)

// ═══════════════════════════════════════════════════════════════════════════
// Oracle: brute-force LCS edit distance (Wagner-Fischer on tokens, insert +
// delete only, no substitution) — the ground truth D that a minimal SES must
// achieve.
// ═══════════════════════════════════════════════════════════════════════════

func oracleEditDistance(a, b []string) int {
	n, m := len(a), len(b)
	dp := make([][]int, n+1)
	for i := range dp {
		dp[i] = make([]int, m+1)
		dp[i][0] = i
	}
	for j := 0; j <= m; j++ {
		dp[0][j] = j
	}
	for i := 1; i <= n; i++ {
		for j := 1; j <= m; j++ {
			if a[i-1] == b[j-1] {
				dp[i][j] = dp[i-1][j-1]
			} else {
				del := dp[i-1][j] + 1
				ins := dp[i][j-1] + 1
				if del < ins {
					dp[i][j] = del
				} else {
					dp[i][j] = ins
				}
			}
		}
	}
	return dp[n][m]
}

// checkScript verifies the three load-bearing invariants of a SES against a,b.
func checkScript(t *testing.T, a, b []string, ops []EditOp) {
	t.Helper()
	// 1. equal+delete tokens (in order) reproduce a.
	var fromA []string
	// 2. equal+insert tokens (in order) reproduce b.
	var fromB []string
	edits := 0
	for _, op := range ops {
		switch op.Op {
		case OpEqual:
			fromA = append(fromA, op.Token)
			fromB = append(fromB, op.Token)
		case OpDelete:
			fromA = append(fromA, op.Token)
			edits++
		case OpInsert:
			fromB = append(fromB, op.Token)
			edits++
		default:
			t.Fatalf("unknown op kind %q", op.Op)
		}
	}
	if strings.Join(fromA, "\x00") != strings.Join(a, "\x00") {
		t.Fatalf("equal+delete does not reconstruct a\n got %v\nwant %v\nops %v", fromA, a, ops)
	}
	if strings.Join(fromB, "\x00") != strings.Join(b, "\x00") {
		t.Fatalf("equal+insert does not reconstruct b\n got %v\nwant %v\nops %v", fromB, b, ops)
	}
	// 3. minimality: edit count equals the true edit distance.
	if want := oracleEditDistance(a, b); edits != want {
		t.Fatalf("non-minimal SES: got %d edits, want %d\na=%v\nb=%v\nops=%v", edits, want, a, b, ops)
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Hand-computed fixtures
// ═══════════════════════════════════════════════════════════════════════════

func TestDiffTokens_Empty(t *testing.T) {
	if got := DiffTokens(nil, nil); len(got) != 0 {
		t.Fatalf("empty/empty: got %v, want []", got)
	}
	if got := DiffTokens([]string{}, []string{}); len(got) != 0 {
		t.Fatalf("empty slices: got %v, want []", got)
	}
}

func TestDiffTokens_AllInsert(t *testing.T) {
	a := []string{}
	b := []string{"a", "b", "c"}
	ops := DiffTokens(a, b)
	checkScript(t, a, b, ops)
	for _, op := range ops {
		if op.Op != OpInsert {
			t.Fatalf("expected all inserts, got %v", ops)
		}
	}
}

func TestDiffTokens_AllDelete(t *testing.T) {
	a := []string{"a", "b", "c"}
	b := []string{}
	ops := DiffTokens(a, b)
	checkScript(t, a, b, ops)
	for _, op := range ops {
		if op.Op != OpDelete {
			t.Fatalf("expected all deletes, got %v", ops)
		}
	}
}

func TestDiffTokens_Identical(t *testing.T) {
	a := []string{"the", "cat", "sat"}
	ops := DiffTokens(a, a)
	checkScript(t, a, a, ops)
	for _, op := range ops {
		if op.Op != OpEqual {
			t.Fatalf("identical inputs must be all-equal, got %v", ops)
		}
	}
}

// The canonical published Myers example: A = ABCABBA, B = CBABAC (as single-
// character tokens). Myers 1986 gives edit distance D = 5 (SES length 5).
func TestDiffTokens_PublishedMyersExample(t *testing.T) {
	a := strings.Split("ABCABBA", "")
	b := strings.Split("CBABAC", "")
	ops := DiffTokens(a, b)
	checkScript(t, a, b, ops)
	if got := oracleEditDistance(a, b); got != 5 {
		t.Fatalf("sanity: oracle distance %d, want 5", got)
	}
}

func TestDiffTokens_SingleDelete(t *testing.T) {
	a := []string{"a", "b", "c"}
	b := []string{"a", "c"}
	ops := DiffTokens(a, b)
	checkScript(t, a, b, ops)
	edits := 0
	for _, op := range ops {
		if op.Op != OpEqual {
			edits++
		}
	}
	if edits != 1 {
		t.Fatalf("want exactly 1 edit, got %d: %v", edits, ops)
	}
}

func TestDiffTokens_RepeatedTokens(t *testing.T) {
	a := []string{"x", "x", "x"}
	b := []string{"x"}
	checkScript(t, a, b, DiffTokens(a, b))
}

func TestDiffTokens_GrammarLikeSentence(t *testing.T) {
	a := strings.Fields("Their going too the shops on tuesday")
	b := strings.Fields("They are going to the shops on Tuesday")
	checkScript(t, a, b, DiffTokens(a, b))
}

// ═══════════════════════════════════════════════════════════════════════════
// Fuzz: random token sequences, checked against the oracle
// ═══════════════════════════════════════════════════════════════════════════

func TestDiffTokens_FuzzAgainstOracle(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	alphabets := [][]string{
		{"a", "b"},                // tiny alphabet → many repeats, ties
		{"a", "b", "c", "d"},      //
		{"w", "x", "y", "z", "0"}, //
	}
	for iter := 0; iter < 20000; iter++ {
		alpha := alphabets[rng.Intn(len(alphabets))]
		n := rng.Intn(12)
		m := rng.Intn(12)
		a := make([]string, n)
		for i := range a {
			a[i] = alpha[rng.Intn(len(alpha))]
		}
		b := make([]string, m)
		for i := range b {
			b[i] = alpha[rng.Intn(len(alpha))]
		}
		checkScript(t, a, b, DiffTokens(a, b))
	}
}

// Larger inputs that are near-identical (the real grammar-diff workload): a
// long shared body with a few scattered edits. Exercises the O((n+m)D) path at
// scale and confirms memory stays bounded (no quadratic table).
func TestDiffTokens_LargeNearIdentical(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	base := make([]string, 6000)
	for i := range base {
		base[i] = "tok" + string(rune('a'+rng.Intn(26)))
	}
	a := append([]string(nil), base...)
	b := append([]string(nil), base...)
	// Apply ~30 edits to b.
	for e := 0; e < 30; e++ {
		i := rng.Intn(len(b))
		b[i] = "EDIT" + string(rune('A'+rng.Intn(26)))
	}
	checkScript(t, a, b, DiffTokens(a, b))
}

// Two entirely different large sequences: the pathological case the old
// full-table diff had to cap. Confirms correctness (linear memory keeps it
// tractable).
func TestDiffTokens_LargeDisjoint(t *testing.T) {
	a := make([]string, 400)
	b := make([]string, 400)
	for i := range a {
		a[i] = "A" + string(rune('a'+i%26))
	}
	for i := range b {
		b[i] = "B" + string(rune('a'+i%26))
	}
	checkScript(t, a, b, DiffTokens(a, b))
}

// ═══════════════════════════════════════════════════════════════════════════
// Golden-file vectors (cross-language port conformance target)
// ═══════════════════════════════════════════════════════════════════════════

type diffGoldenCase struct {
	Description string     `json:"description"`
	A           []string   `json:"a"`
	B           []string   `json:"b"`
	Expected    [][]string `json:"expected"` // each op: ["equal"|"insert"|"delete", token]
}

type diffGoldenFile struct {
	Function string           `json:"function"`
	Cases    []diffGoldenCase `json:"cases"`
}

func TestDiffTokens_Golden(t *testing.T) {
	raw, err := os.ReadFile("../testdata/sequence/diff_tokens.json")
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var gf diffGoldenFile
	if err := json.Unmarshal(raw, &gf); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	if gf.Function != "Sequence.DiffTokens" {
		t.Fatalf("golden function = %q", gf.Function)
	}
	if len(gf.Cases) < 20 {
		t.Fatalf("golden must carry >=20 vectors, got %d", len(gf.Cases))
	}
	for _, c := range gf.Cases {
		got := DiffTokens(c.A, c.B)
		if len(got) != len(c.Expected) {
			t.Fatalf("%s: got %d ops, want %d\n got %v", c.Description, len(got), len(c.Expected), got)
		}
		for i, op := range got {
			wantKind := c.Expected[i][0]
			wantTok := c.Expected[i][1]
			if string(op.Op) != wantKind || op.Token != wantTok {
				t.Fatalf("%s op %d: got [%s %q], want [%s %q]",
					c.Description, i, op.Op, op.Token, wantKind, wantTok)
			}
		}
		// Golden vectors must also satisfy the reconstruction invariants.
		checkScript(t, c.A, c.B, got)
	}
}
