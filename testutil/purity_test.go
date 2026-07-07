package testutil

import (
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// moduleRoot returns the reality module root (this file lives at
// <root>/testutil/purity_test.go). runtime.Caller embeds the path at compile
// time, so it resolves correctly under any checkout, including CI.
func moduleRoot(t *testing.T) string {
	t.Helper()
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	return filepath.Dir(filepath.Dir(file))
}

// allowedImport reports whether an import path is permitted under reality's
// Tier-0 zero-external-dependency invariant: it must be either Go stdlib (no dot
// in the path — stdlib import paths like "math/rand" never contain a dot) or a
// sub-package of github.com/davly/reality.
func allowedImport(path string) bool {
	if !strings.Contains(path, ".") {
		return true // stdlib
	}
	return strings.HasPrefix(path, "github.com/davly/reality")
}

// TestAllowedImport_Predicate proves the gate is non-trivial: it actually
// rejects external imports (stdlib and reality sub-packages pass; anything else
// with a dotted path fails).
func TestAllowedImport_Predicate(t *testing.T) {
	cases := []struct {
		path string
		want bool
	}{
		{"fmt", true},
		{"math/rand", true},
		{"encoding/json", true},
		{"net/http", true}, // stdlib (allowed by the zero-DEP gate; contained by the impure allowlist)
		{"github.com/davly/reality/linalg", true},
		{"github.com/davly/reality/prob/copula", true},
		{"golang.org/x/crypto/sha3", false},
		{"github.com/pkg/errors", false},
		{"gonum.org/v1/gonum/mat", false},
	}
	for _, c := range cases {
		if got := allowedImport(c.path); got != c.want {
			t.Errorf("allowedImport(%q)=%v want %v", c.path, got, c.want)
		}
	}
}

// TestZeroExternalDependencies enforces the Tier-0 invariant — the load-bearing
// reason reality can credibly claim "universal truth" — as a build gate rather
// than a convention (ARCHITECTURE.md §3.1 previously said it was "checked
// informally by grepping during review"). It asserts:
//  1. go.mod declares no `require` clauses (no external module dependency), and
//  2. every import in every .go file is stdlib or github.com/davly/reality/*.
//
// One careless `go get` in a future change is otherwise invisible until a
// downstream consumer discovers reality is no longer a pure importable leaf.
func TestZeroExternalDependencies(t *testing.T) {
	root := moduleRoot(t)

	// (1) go.mod: no require clauses.
	gomod, err := os.ReadFile(filepath.Join(root, "go.mod"))
	if err != nil {
		t.Fatalf("read go.mod: %v", err)
	}
	for _, line := range strings.Split(string(gomod), "\n") {
		l := strings.TrimSpace(line)
		// Match a `require` directive (single-line `require x v` or a `require (`
		// block opener). The bare module/go directives never start with require.
		if strings.HasPrefix(l, "require") {
			t.Errorf("go.mod contains a require directive (%q); reality must have zero external dependencies", l)
		}
	}

	// (2) every import is stdlib or davly/reality.
	fset := token.NewFileSet()
	walkErr := filepath.WalkDir(root, func(p string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			name := d.Name()
			if p != root && (name == ".git" || name == ".github" || name == "reviews" || name == "docs") {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(p, ".go") {
			return nil
		}
		f, perr := parser.ParseFile(fset, p, nil, parser.ImportsOnly)
		if perr != nil {
			return nil // unparseable file — not our concern here
		}
		for _, imp := range f.Imports {
			path := strings.Trim(imp.Path.Value, `"`)
			if !allowedImport(path) {
				rel, _ := filepath.Rel(root, p)
				t.Errorf("%s imports %q — a non-stdlib, non-reality package, violating the Tier-0 zero-dependency invariant", rel, path)
			}
		}
		return nil
	})
	if walkErr != nil {
		t.Fatalf("walk module: %v", walkErr)
	}
}

// TestImpureStdlibImportsAllowlisted keeps the network/IO surface contained: only
// the deliberately-impure shims may import net/http or os/exec. This is the
// machine-checked form of ARCHITECTURE.md's prose "conduit is the one package
// that is not pure math" — a future package that silently adds an HTTP client
// (e.g. a "metrics" shim) trips this instead of quietly eroding the purity story.
func TestImpureStdlibImportsAllowlisted(t *testing.T) {
	root := moduleRoot(t)
	// Directory (package) -> permitted to import the impure-stdlib set below.
	allowedImpureDirs := map[string]bool{
		"conduit":             true, // documented fail-silent event-emit shim (net/http)
		"cmd/reality-compute": true, // executable entry point: Nexus MCP HTTP server (net/http)
	}
	impure := []string{"net/http", "os/exec", "net"}

	fset := token.NewFileSet()
	walkErr := filepath.WalkDir(root, func(p string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			name := d.Name()
			if p != root && (name == ".git" || name == ".github" || name == "reviews" || name == "docs") {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(p, ".go") || strings.HasSuffix(p, "_test.go") {
			return nil // tests may use net/httptest etc.
		}
		rel, _ := filepath.Rel(root, p)
		dir := filepath.ToSlash(filepath.Dir(rel))
		if allowedImpureDirs[dir] {
			return nil
		}
		f, perr := parser.ParseFile(fset, p, nil, parser.ImportsOnly)
		if perr != nil {
			return nil
		}
		for _, imp := range f.Imports {
			path := strings.Trim(imp.Path.Value, `"`)
			for _, bad := range impure {
				if path == bad {
					t.Errorf("%s imports %q outside the impure allowlist (%d dirs, e.g. conduit); keep network/IO out of the pure-math packages", rel, path, len(allowedImpureDirs))
				}
			}
		}
		return nil
	})
	if walkErr != nil {
		t.Fatalf("walk module: %v", walkErr)
	}
}
