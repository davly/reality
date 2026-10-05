package reality_test

// Determinism ratchet, static half.
//
// Go randomises map iteration order. A `range` over a map whose body
// accumulates floats, appends to a result, or picks a winner on ties can give
// different outputs for the same input (several functions here did). Naming
// that hazard in documentation does not stop the next instance; this test
// does. It type-checks every package in the module and inventories each
// `range` over a map-typed expression in non-test code. Every site must be
// classified in testdata/determinism/map_ranges.json:
//
//   - order-insensitive       the result cannot depend on iteration order
//   - sorted-after            the loop's output is sorted before it is used
//   - known-nondeterministic  a measured defect, fixed later (ratchet)
//   - unreviewed              present when this inventory was created and
//     not yet reviewed; review it, then reclassify
//
// A new, unclassified site fails the test, and so does a stale entry whose
// site no longer exists. Site keys are <package>.<function>#<k>, where k
// counts map ranges within that function in source order, so line-number
// churn does not invalidate them.
//
// To add new sites as "unreviewed" (then classify them by hand):
//
//	REALITY_UPDATE_MAP_RANGES=1 go test -run TestMapRangeInventory .

import (
	"encoding/json"
	"fmt"
	"go/ast"
	"go/build"
	"go/importer"
	"go/parser"
	"go/token"
	"go/types"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

const (
	moduleImportPath = "github.com/davly/reality"
	mapRangeLedger   = "testdata/determinism/map_ranges.json"
)

var mapRangeClasses = map[string]bool{
	"order-insensitive":      true,
	"sorted-after":           true,
	"known-nondeterministic": true,
	"unreviewed":             true,
}

type mapRangeEntry struct {
	Class string `json:"class"`
	Note  string `json:"note,omitempty"`
}

type mapRangeFile struct {
	Comment string                   `json:"_comment"`
	Sites   map[string]mapRangeEntry `json:"sites"`
}

// moduleImporter type-checks module packages from source and resolves the
// standard library through the default (export data) importer.
type moduleImporter struct {
	root  string
	fset  *token.FileSet
	std   types.Importer
	pkgs  map[string]*types.Package
	infos map[string]*types.Info
	files map[string][]*ast.File
}

func (m *moduleImporter) Import(path string) (*types.Package, error) {
	if path == moduleImportPath || strings.HasPrefix(path, moduleImportPath+"/") {
		return m.load(path)
	}
	return m.std.Import(path)
}

func (m *moduleImporter) load(path string) (*types.Package, error) {
	if p, ok := m.pkgs[path]; ok {
		return p, nil
	}
	rel := strings.TrimPrefix(strings.TrimPrefix(path, moduleImportPath), "/")
	dir := filepath.Join(m.root, filepath.FromSlash(rel))
	bp, err := build.Default.ImportDir(dir, 0)
	if err != nil {
		return nil, err
	}
	if len(bp.IgnoredGoFiles) > 0 {
		return nil, fmt.Errorf("%s has build-constrained files %v: the inventory would depend on the platform; extend the scanner first", path, bp.IgnoredGoFiles)
	}
	var files []*ast.File
	for _, name := range bp.GoFiles {
		f, err := parser.ParseFile(m.fset, filepath.Join(dir, name), nil, parser.SkipObjectResolution)
		if err != nil {
			return nil, err
		}
		files = append(files, f)
	}
	info := &types.Info{Types: map[ast.Expr]types.TypeAndValue{}}
	conf := types.Config{Importer: m}
	pkg, err := conf.Check(path, m.fset, files, info)
	if err != nil {
		return nil, fmt.Errorf("type-checking %s: %w", path, err)
	}
	m.pkgs[path], m.infos[path], m.files[path] = pkg, info, files
	return pkg, nil
}

type mapRangeSite struct {
	key string
	pos string
}

// scanMapRanges returns every range-over-map site in non-test code, keyed.
func scanMapRanges(t *testing.T, root string) []mapRangeSite {
	t.Helper()
	if build.Default.GOROOT == "" {
		// A test binary built with -trimpath carries no GOROOT, so go/build
		// cannot find the standard library. go test puts the go command on
		// PATH; ask it.
		out, err := exec.Command("go", "env", "GOROOT").Output()
		if err != nil {
			t.Fatalf("locating GOROOT for the type checker: %v", err)
		}
		build.Default.GOROOT = strings.TrimSpace(string(out))
	}
	fset := token.NewFileSet()
	m := &moduleImporter{
		root:  root,
		fset:  fset,
		std:   importer.ForCompiler(fset, "gc", nil),
		pkgs:  map[string]*types.Package{},
		infos: map[string]*types.Info{},
		files: map[string][]*ast.File{},
	}
	var paths []string
	walkErr := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() {
			return nil
		}
		name := d.Name()
		if path != root && (strings.HasPrefix(name, ".") || strings.HasPrefix(name, "_") || name == "testdata" || name == "vendor") {
			return filepath.SkipDir
		}
		bp, err := build.Default.ImportDir(path, 0)
		if err != nil || len(bp.GoFiles) == 0 {
			return nil // no non-test Go files here (or only tests)
		}
		rel, _ := filepath.Rel(root, path)
		ip := moduleImportPath
		if rel != "." {
			ip += "/" + filepath.ToSlash(rel)
		}
		paths = append(paths, ip)
		return nil
	})
	if walkErr != nil {
		t.Fatalf("walking module: %v", walkErr)
	}
	sort.Strings(paths)

	var sites []mapRangeSite
	for _, ip := range paths {
		if _, err := m.load(ip); err != nil {
			t.Fatalf("%v", err)
		}
		pkgLabel := strings.TrimPrefix(strings.TrimPrefix(ip, moduleImportPath), "/")
		if pkgLabel == "" {
			pkgLabel = "."
		}
		info := m.infos[ip]
		for _, f := range m.files[ip] {
			for _, decl := range f.Decls {
				fd, ok := decl.(*ast.FuncDecl)
				if !ok || fd.Body == nil {
					continue
				}
				fn := fd.Name.Name
				if fd.Recv != nil && len(fd.Recv.List) > 0 {
					fn = recvTypeName(fd.Recv.List[0].Type) + "." + fn
				}
				k := 0
				ast.Inspect(fd.Body, func(n ast.Node) bool {
					rs, ok := n.(*ast.RangeStmt)
					if !ok {
						return true
					}
					typ := info.TypeOf(rs.X)
					if typ == nil {
						return true
					}
					if _, isMap := typ.Underlying().(*types.Map); isMap {
						k++
						p := fset.Position(rs.For)
						rp, _ := filepath.Rel(root, p.Filename)
						sites = append(sites, mapRangeSite{
							key: fmt.Sprintf("%s.%s#%d", pkgLabel, fn, k),
							pos: fmt.Sprintf("%s:%d", filepath.ToSlash(rp), p.Line),
						})
					}
					return true
				})
			}
		}
	}
	return sites
}

func recvTypeName(e ast.Expr) string {
	switch v := e.(type) {
	case *ast.StarExpr:
		return recvTypeName(v.X)
	case *ast.IndexExpr:
		return recvTypeName(v.X)
	case *ast.IndexListExpr:
		return recvTypeName(v.X)
	case *ast.Ident:
		return v.Name
	}
	return "?"
}

// TestMapRangeInventory is the static determinism ratchet described above.
func TestMapRangeInventory(t *testing.T) {
	root, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	sites := scanMapRanges(t, root)
	if len(sites) == 0 {
		t.Fatal("scanner found zero map ranges: the scanner is broken (this module is known to have them)")
	}

	ledgerPath := filepath.Join(root, filepath.FromSlash(mapRangeLedger))
	var ledger mapRangeFile
	if data, err := os.ReadFile(ledgerPath); err == nil { // #nosec G304 -- fixed path inside this module
		if err := json.Unmarshal(data, &ledger); err != nil {
			t.Fatalf("parsing %s: %v", mapRangeLedger, err)
		}
	} else if os.Getenv("REALITY_UPDATE_MAP_RANGES") == "" {
		t.Fatalf("reading %s: %v", mapRangeLedger, err)
	}
	if ledger.Sites == nil {
		ledger.Sites = map[string]mapRangeEntry{}
	}

	found := map[string]string{}
	for _, s := range sites {
		if prev, dup := found[s.key]; dup {
			t.Fatalf("duplicate site key %s at %s and %s", s.key, prev, s.pos)
		}
		found[s.key] = s.pos
		if os.Getenv("REALITY_LIST_MAP_RANGES") != "" {
			t.Logf("%-45s %-14s %s", s.key, ledger.Sites[s.key].Class, s.pos)
		}
	}

	if os.Getenv("REALITY_UPDATE_MAP_RANGES") != "" {
		for key := range found {
			if _, ok := ledger.Sites[key]; !ok {
				ledger.Sites[key] = mapRangeEntry{Class: "unreviewed"}
			}
		}
		for key := range ledger.Sites {
			if _, ok := found[key]; !ok {
				delete(ledger.Sites, key)
			}
		}
		out, _ := json.MarshalIndent(ledger, "", "  ")
		if err := os.MkdirAll(filepath.Dir(ledgerPath), 0o750); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(ledgerPath, append(out, '\n'), 0o600); err != nil {
			t.Fatal(err)
		}
		t.Logf("wrote %s: %d sites", mapRangeLedger, len(ledger.Sites))
		return
	}

	var unclassified, stale, badClass []string
	for _, s := range sites {
		e, ok := ledger.Sites[s.key]
		if !ok {
			unclassified = append(unclassified, s.key+" ("+s.pos+")")
			continue
		}
		if !mapRangeClasses[e.Class] {
			badClass = append(badClass, fmt.Sprintf("%s: class %q", s.key, e.Class))
		}
	}
	for key := range ledger.Sites {
		if _, ok := found[key]; !ok {
			stale = append(stale, key)
		}
	}
	sort.Strings(unclassified)
	sort.Strings(stale)
	sort.Strings(badClass)
	if len(unclassified) > 0 {
		t.Errorf("%d range-over-map site(s) are not classified in %s. Go randomises map order; decide whether each result can depend on it, then add it:\n  %s",
			len(unclassified), mapRangeLedger, strings.Join(unclassified, "\n  "))
	}
	if len(stale) > 0 {
		t.Errorf("%d entr(y/ies) in %s no longer match a site (the loop moved or was removed); delete them:\n  %s",
			len(stale), mapRangeLedger, strings.Join(stale, "\n  "))
	}
	if len(badClass) > 0 {
		t.Errorf("unknown classes in %s:\n  %s", mapRangeLedger, strings.Join(badClass, "\n  "))
	}
}
