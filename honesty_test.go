// Package reality_test holds repo-root "honesty" invariants: tests that turn
// admitted-but-untooled conventions in the documentation into TRUE-BY-CONSTRUCTION
// laws. A drift away from the documented contract becomes a failing build.
//
// This is the calibrated-honesty discipline applied to the substrate itself:
//
//   - TestImportPurity codifies the zero-dependency convention that
//     ARCHITECTURE.md:27 explicitly admits is "enforced by convention, not by
//     tooling." It promotes that grep into real tooling.
//
//   - TestNoUnbackedCrossLanguageClaim guards against the README claiming that
//     non-Go (Python/C++/C#) implementations exist and validate the golden files
//     when ZERO non-Go source files ship in this repository. The golden files are
//     a language-neutral CONTRACT; the README must not over-claim shipping
//     implementations that do not exist on disk.
//
// Pure Go stdlib only — no new dependency is introduced, preserving reality's
// Tier-0 zero-dependency law.
package reality_test

import (
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// modulePrefix is the module path declared in go.mod. Any import that starts
// with this prefix is an internal reality package and is always allowed.
const modulePrefix = "github.com/davly/reality/"

// netHTTPAllowlist pins the ONE documented external-shaped exception described in
// ARCHITECTURE.md:90 — reality is strictly zero-dependency, and net/http is
// strictly stdlib, but it is only meant to appear in the deliberate
// event-publishing shim conduit/emit.go. We pin exactly that {file, import}
// pair: a SECOND net/http importer anywhere else must fail the build. (net/http
// is stdlib, so it is not "external"; the allowlist is about *which file* may
// reach for the HTTP shim, keeping the rest of the substrate HTTP-free.)
//
// The key is the module-relative slash path of the file.
var netHTTPAllowlist = map[string]bool{
	"conduit/emit.go": true,
}

// TestImportPurity codifies ARCHITECTURE.md:27, which states:
//
//	"This is enforced by convention, not by tooling. The test is:
//	 grep -rh '^\s*"[a-z][a-z/]*"' --include="*.go" should only return stdlib
//	 paths and paths starting with github.com/davly/reality/."
//
// This test IS that tooling. It walks every non-_test .go file in the module,
// parses its import block, and fails if any import path is neither (a) a Go
// stdlib path nor (b) prefixed with github.com/davly/reality/. The single
// documented exception (net/http in conduit/emit.go) is pinned via an
// allowlist; any other reach for net/http, or any external module import
// anywhere, fails the build.
//
// It MUST be GREEN on the current tree (the documented grep passes today).
func TestImportPurity(t *testing.T) {
	root, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}

	fset := token.NewFileSet()

	walkErr := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			// Skip VCS metadata and vendor (there is no vendor today, but be safe).
			name := d.Name()
			if name == ".git" || name == "vendor" {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(path, ".go") {
			return nil
		}
		// Purity is a property of shipped (non-test) code. Test files may legitimately
		// import net/http/httptest etc.; the zero-dep guarantee is about what a CONSUMER
		// who imports reality pulls in, which is the non-test compilation unit.
		if strings.HasSuffix(path, "_test.go") {
			return nil
		}

		rel, relErr := filepath.Rel(root, path)
		if relErr != nil {
			return relErr
		}
		rel = filepath.ToSlash(rel)

		f, perr := parser.ParseFile(fset, path, nil, parser.ImportsOnly)
		if perr != nil {
			return perr
		}

		for _, imp := range f.Imports {
			// imp.Path.Value is the quoted import string, e.g. "\"math/rand\"".
			ip := strings.Trim(imp.Path.Value, `"`)

			switch {
			case isStdlib(ip):
				// net/http is stdlib, but we additionally constrain WHICH file may
				// import it: only the documented conduit/emit.go shim.
				if ip == "net/http" && !netHTTPAllowlist[rel] {
					t.Errorf("PURITY VIOLATION: %s imports net/http, but net/http is only "+
						"permitted in the documented conduit/emit.go event-publishing shim "+
						"(ARCHITECTURE.md:90). Add it to netHTTPAllowlist only if this is a "+
						"new, deliberately-documented exception.", rel)
				}
			case strings.HasPrefix(ip, modulePrefix):
				// Internal reality package — always allowed.
			default:
				t.Errorf("PURITY VIOLATION: %s imports %q, which is neither Go stdlib nor "+
					"an internal github.com/davly/reality/ package. reality is Tier-0 "+
					"zero-dependency (ARCHITECTURE.md:25-27); external imports are forbidden.",
					rel, ip)
			}
		}
		return nil
	})
	if walkErr != nil {
		t.Fatalf("walking module tree: %v", walkErr)
	}
}

// isStdlib reports whether an import path is part of the Go standard library.
//
// Heuristic: stdlib import paths never contain a dot in their FIRST path
// segment. Every third-party/module path begins with a domain (e.g.
// "github.com/...", "golang.org/x/..."), which contains a dot. Standard library
// paths ("math", "math/rand", "net/http", "encoding/json") do not. This is the
// same rule the Go toolchain uses to distinguish stdlib from module imports.
func isStdlib(importPath string) bool {
	if importPath == "" {
		return false
	}
	first := importPath
	if i := strings.IndexByte(importPath, '/'); i >= 0 {
		first = importPath[:i]
	}
	return !strings.Contains(first, ".")
}

// TestNoUnbackedCrossLanguageClaim enforces honest documentation about
// cross-language implementations.
//
// The golden JSON vectors in testdata/ are a language-neutral CONTRACT: the Go
// implementation validates against them, and the JSON format is designed so that
// independent Python/C++/C# implementations COULD validate against the same
// vectors. But that is only honest while no non-Go implementation actually ships
// here. The README must not state that Python/C++/C# implementations "are used"
// to validate the golden files when ZERO non-Go source files exist on disk —
// that is an author-independent, missing-file fact.
//
// Rule:
//   - If ANY non-Go implementation source (*.py, *.cpp, *.cc, *.cxx, *.cs)
//     exists in the repo, the over-claim is backed and the test passes.
//   - If NONE exist, the README must NOT assert that those languages have
//     implementations that USE/validate the golden files.
func TestNoUnbackedCrossLanguageClaim(t *testing.T) {
	root, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}

	nonGoExts := map[string]bool{
		".py":  true,
		".cpp": true,
		".cc":  true,
		".cxx": true,
		".cs":  true,
	}

	var nonGoImpls []string
	walkErr := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			if d.Name() == ".git" || d.Name() == "vendor" {
				return filepath.SkipDir
			}
			return nil
		}
		if nonGoExts[strings.ToLower(filepath.Ext(path))] {
			rel, _ := filepath.Rel(root, path)
			nonGoImpls = append(nonGoImpls, filepath.ToSlash(rel))
		}
		return nil
	})
	if walkErr != nil {
		t.Fatalf("walking module tree: %v", walkErr)
	}

	// If a real non-Go implementation ships, the cross-language claim is backed.
	if len(nonGoImpls) > 0 {
		return
	}

	readmePath := filepath.Join(root, "README.md")
	data, err := os.ReadFile(readmePath)
	if err != nil {
		t.Fatalf("reading README.md: %v", err)
	}
	readme := string(data)

	// Phrases that assert non-Go implementations EXIST and validate the golden
	// files. These are FALSE while no non-Go source ships. The honest framing
	// ("language-neutral contract", "could validate", "only the Go
	// implementation ships") deliberately does not match any of these.
	forbidden := []string{
		// The exact line 129 over-claim (and close variants).
		"used by Go, Python, C++, and C# implementations",
		"used by Go, Python, C++ and C# implementations",
		// "Python, C++, and C# validate against" — asserts shipping impls.
		"Python, C++, and C# validate",
		"Python, C++ and C# validate",
	}

	for _, phrase := range forbidden {
		if strings.Contains(readme, phrase) {
			t.Errorf("UNBACKED CROSS-LANGUAGE CLAIM: README.md contains %q, but there are "+
				"ZERO non-Go implementation source files in this repo (searched *.py *.cpp "+
				"*.cc *.cxx *.cs). The golden files are a language-neutral CONTRACT; the "+
				"README must not claim shipping Python/C++/C# implementations that do not "+
				"exist on disk. State that only the Go implementation ships here today.",
				phrase)
		}
	}
}
