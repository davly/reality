package canonical

import (
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
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

// TestR85ZeroDivergences walks the repo tree and fails if any production
// source file (not *_test.go) contains a CanonicalDivergence{} struct
// literal whose Primitive field matches one of this flagship's
// CanonicalPrimitives(). Per the Standard's R85 clause: "a canonical
// source cannot diverge from itself" — interpreted scope-aware: the
// flagship is canonical for some primitives, and registering R74
// divergences for THOSE primitives is the violation. Registering R74
// divergences for OTHER primitives is legitimate domain documentation,
// not a self-divergence.
//
// Test fixtures (*_test.go) are excluded — they construct
// CanonicalDivergence values for testing the registry's API, not as
// production registrations.
//
// Implementation: textual grep + regex-extract of the Primitive field.
// False positives (matches inside strings or comments that look like
// registrations) are conservatively flagged, then filtered by the
// canonical-primitive check.
func TestR85ZeroDivergences(t *testing.T) {
	root := findRepoRoot(t)
	canonical := make(map[string]struct{}, len(CanonicalPrimitives()))
	for _, p := range CanonicalPrimitives() {
		canonical[p] = struct{}{}
	}

	primitiveRe := regexp.MustCompile(`CanonicalDivergence\{[^}]*?Primitive:\s*([A-Za-z][A-Za-z0-9_]*|"[^"]*")`)

	type violation struct {
		path      string
		primitive string
	}
	var hits []violation
	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			name := d.Name()
			if name == ".git" || name == "vendor" || name == "node_modules" {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(path, ".go") {
			return nil
		}
		if strings.HasSuffix(path, "_test.go") {
			return nil
		}
		data, readErr := os.ReadFile(path)
		if readErr != nil {
			return readErr
		}
		matches := primitiveRe.FindAllStringSubmatch(string(data), -1)
		for _, m := range matches {
			prim := m[1]
			if strings.HasPrefix(prim, `"`) && strings.HasSuffix(prim, `"`) {
				prim = strings.Trim(prim, `"`)
			} else {
				prim = resolveRealityPrimitiveIdent(prim)
			}
			if _, isCanonical := canonical[prim]; isCanonical {
				hits = append(hits, violation{path: path, primitive: prim})
			}
		}
		return nil
	})
	if err != nil {
		t.Fatalf("R85: walk failed: %v", err)
	}
	if len(hits) > 0 {
		var b strings.Builder
		for _, h := range hits {
			b.WriteString("  ")
			b.WriteString(h.path)
			b.WriteString(" -> primitive=")
			b.WriteString(h.primitive)
			b.WriteString("\n")
		}
		t.Fatalf("R85: expected zero CanonicalDivergence{} entries for primitives in CanonicalPrimitives()=%v, but found %d:\n%s",
			CanonicalPrimitives(), len(hits), b.String())
	}
}

// resolveRealityPrimitiveIdent maps reality's session40 primitive
// identifiers to their literal string values.
func resolveRealityPrimitiveIdent(ident string) string {
	switch ident {
	case "PrimitiveConduitTimeout":
		return "conduit_timeout"
	case "PrimitiveBridgeTimeout":
		return "bridge_timeout"
	case "PrimitiveFnvHash":
		return "fnv_hash"
	case "PrimitiveJeffreysPrior":
		return "jeffreys_prior"
	case "PrimitiveEscapeThreshold":
		return "escape_threshold"
	case "PrimitiveVerdictScheme":
		return "verdict_scheme"
	}
	return ident
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
