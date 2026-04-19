package canonical

import (
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestR85IsCanonicalSourceDeclared asserts the R85 marker is true.
func TestR85IsCanonicalSourceDeclared(t *testing.T) {
	if !IsCanonicalSource {
		t.Fatalf("R85: IsCanonicalSource must be true for reality (foundation-tier canonical source)")
	}
}

// TestR85CanonicalPrimitivesNonEmpty asserts the primitive list is populated.
// A canonical source with zero declared primitives fails R85 clause 2.
func TestR85CanonicalPrimitivesNonEmpty(t *testing.T) {
	prims := CanonicalPrimitives()
	if len(prims) == 0 {
		t.Fatalf("R85: CanonicalPrimitives() must list at least one primitive")
	}
	for i, p := range prims {
		if strings.TrimSpace(p) == "" {
			t.Fatalf("R85: CanonicalPrimitives()[%d] is blank", i)
		}
	}
}

// TestR85ZeroDivergences walks the repo tree and fails if any source file
// contains a CanonicalDivergence{ struct-literal usage. A flagship declaring
// IsCanonicalSource = true MUST hold zero R74 divergence entries — a source
// cannot diverge from itself (R85 clause 1).
//
// Implementation note: this is a simple textual grep. False positives (e.g.
// matches inside strings or comments) are acceptable because they would still
// indicate the R74 registry pattern is present, which violates R85.
func TestR85ZeroDivergences(t *testing.T) {
	root := findRepoRoot(t)

	var hits []string
	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			name := d.Name()
			// Skip VCS, vendor, and build output directories.
			if name == ".git" || name == "vendor" || name == "node_modules" {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(path, ".go") {
			return nil
		}
		// Skip this test file — it legitimately mentions the token.
		if strings.HasSuffix(path, "canonical_test.go") {
			return nil
		}
		data, readErr := os.ReadFile(path)
		if readErr != nil {
			return readErr
		}
		if strings.Contains(string(data), "CanonicalDivergence{") {
			hits = append(hits, path)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("R85: walk failed: %v", err)
	}
	if len(hits) > 0 {
		t.Fatalf("R85: expected zero CanonicalDivergence{ struct usages in reality (canonical source) but found %d:\n  %s",
			len(hits), strings.Join(hits, "\n  "))
	}
}

// findRepoRoot locates the nearest ancestor containing go.mod.
func findRepoRoot(t *testing.T) string {
	t.Helper()
	wd, err := os.Getwd()
	if err != nil {
		t.Fatalf("R85: getwd: %v", err)
	}
	dir := wd
	for i := 0; i < 10; i++ {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	t.Fatalf("R85: could not locate go.mod from %s", wd)
	return ""
}
