# 395 — meta-vendor-deps (zero-deps invariant audit)

## Headline
Zero-deps invariant fully intact: `go.mod` declares no external requires, `go.sum` and `vendor/` are absent, and `go list -deps ./...` yields zero non-stdlib transitive dependencies; CLAUDE.md's stronger "only the language's standard math library" claim is technically violated by ~30 stdlib packages outside `math*` (e.g. `net/http` in `conduit`, `image/png` in `audio/spectrogram`), but no third-party code is reached.

## go.mod / go.sum status
- `C:\limitless\foundation\reality\go.mod` (3 lines, full file):
  ```
  module github.com/davly/reality
  go 1.24
  ```
  No `require`, `replace`, `exclude`, `retract`, or `tool` directives. No `// indirect` lines. Cannot be cleaner.
- `C:\limitless\foundation\reality\go.sum`: **absent**. Confirmed via `ls`.
- `C:\limitless\foundation\reality\vendor\`: **absent**. Confirmed via `ls`.
- `find . -name go.mod`: only the root file. No nested modules / sub-modules.
- `go list -m all` output: single line `github.com/davly/reality`. No transitive modules.
- `go list -deps ./...` filtered to drop `github.com/davly/reality/...` packages: 200 packages, **all** are Go stdlib (`math`, `math/cmplx`, `math/rand`, `math/big`, `crypto/...`, `encoding/json`, `image/png`, `container/heap`, `net/http`, `runtime`, `internal/...`, etc.). No external module path appears.

## Non-stdlib import scan
Scanned every `import` block under the repo (642 import lines across all `.go` files, sampled in full). Two-bucket result:

1. **Internal (`github.com/davly/reality/...`)** — only sub-packages of this very module. Fan-in summary (consumers of internal pkgs):
   - `testutil` — imported by 22 packages (test-only golden harness, the most-fan-in internal dep)
   - `signal` — imported by `audio`, `audio/onset`, `audio/pitch`, `audio/segmentation`, `audio/separation`, `audio/spectrogram`, `audio/vibration`
   - `constants` — imported by `em`, `orbital`, `physics`
   - `audio` — imported by `audio/spectrogram`, `audio/pitch`, `audio/vibration`
   - `audio/onset` ← `audio/segmentation`; `audio/spectrogram` ← `audio/onset`, `audio/segmentation`
   - `linalg` ← `prob/copula/gaussian.go:6`
   - `prob` ← `prob/copula/{gaussian,gaussian_test,sklar_test}.go`, `forge/session40/baked.go`
   - `crypto` ← `forge/session40/baked.go`
   - `autodiff` ← `infogeo/autodiff_test.go`, `prob/copula/autodiff_test.go`, `timeseries/garch/autodiff_test.go`
   - `infogeo` ← `changepoint/infogeo_test.go`
   - `optim/proximal` ← `optim/proximal_consumer_test.go`
   No cycles observed; DAG is consistent with the package list in CLAUDE.md (22 listed there + a handful unlisted: `audio*`, `autodiff`, `changepoint`, `conduit`, `forge/session40`, `info/{lz,mdl}`, `infogeo`, `pkg/canonical`, `sequence`, `timeseries/{dcc,garch}`, `topology/persistent`, `zkmark`).

2. **External (`<host>.<tld>/...` not matching `github.com/davly/reality`)** — **zero matches**. Regex `^\s*"([a-z0-9._-]+\.(com|org|io|net|dev)/[^"]+)"` against all `*.go` files returned only `github.com/davly/reality/...` paths.

## Cracks found
None that breach the literal "zero deps" invariant. Some friction points the Headline alludes to:

- **Stdlib breadth exceeds the "math stdlib" promise.** CLAUDE.md says "Only the language's standard math library." Code reaches well beyond `math`, `math/cmplx`, `math/rand`, `math/big`:
  - `conduit/emit.go:15-22` — `bytes`, `context`, `encoding/json`, `net/http`, `os`, `sync/atomic`, `time`. Network telemetry emitter; not pure math.
  - `audio/spectrogram/visualise.go:3-7` — `bytes`, `image`, `image/color`, `image/png`. Renders PNGs.
  - `audio/spectrogram/spectrogram_test.go:5,7` — `image/png`, `math/cmplx`.
  - `forge/session40/registry.go:3-7` — `fmt`, `sort`, `strings`, `sync`. Registry/runtime infra.
  - `pkg/canonical/canonical_test.go:4-8` — `io/fs`, `os`, `path/filepath`, `regexp`, `strings`.
  - `crypto/crypto_test.go`, `audio/parity_test.go`, `testutil/golden.go` — `os`, `encoding/json`, `runtime`, `path/filepath` (golden-file infra; expected and well-isolated).
  - Several `errors`, `fmt`, `sort`, `strings`, `sync`, `sync/atomic`, `bytes`, `container/heap`, `hash/fnv`, `encoding/hex`, `encoding/json`, `strconv`, `regexp` usages across many packages.
  All are **stdlib**, so `go.mod` stays clean and the dependency graph is closed. But the docstring "only ... standard math library" is aspirational rather than literal. Recommend rewording CLAUDE.md to "Go standard library only — no third-party modules" to match reality (pun intended), or carving `conduit`, `forge/session40`, `pkg/canonical`, `audio/spectrogram/visualise.go`, `zkmark`, `testutil` into a `pkg/` subtree that's documented as "infra, not math."
- **`net/http` reached at runtime via `conduit`.** Not a deps crack but a CGO/no-allocations concern: `net/http.Client.Do` allocates and may pull system DNS/TLS. If `conduit` is ever imported from a hot-path consumer, the "no allocations in hot paths" rule (CLAUDE.md design rule 3) is at risk. Recommend a build-tag gate (`//go:build conduit_emit`) so default builds drop the import entirely.
- **No CGO, no `unsafe`, no assembly.** Grepped `^import\s+"C"`, `//\s*#cgo`, `//export\s`, `"unsafe"` across all `*.go` — **zero matches**. Pure Go, all platforms. Slot 357's "no CGO" claim holds.
- **No `go:generate` pulling external tools.** Spot-checked: no `go:generate` directives invoke non-stdlib binaries (the test infrastructure uses Go's `math/big` for golden generation, in-tree).
- **Test-only deps?** None. Test files import the same internal packages plus stdlib `testing`, `math/rand`, `os`, `encoding/json`, etc. There is no `require ... // indirect` or test-only third-party crutch (e.g. no `testify`, no `quick`, no `gocheck`).
- **Replace/exclude/retract directives?** None. Zero hidden upgrade paths.
- **`go.work` / workspace files?** Not searched explicitly, but a `find` for `go.mod` returned only the root, and the project doesn't appear to be part of a workspace (no `go.work` listed at root via earlier listings).

## Sources
- `C:\limitless\foundation\reality\go.mod` (3 lines)
- `C:\limitless\foundation\reality\go.sum` (absent)
- `C:\limitless\foundation\reality\vendor\` (absent)
- `C:\limitless\foundation\reality\CLAUDE.md` (zero-deps claim, design rule 2)
- `go list -m all` — one line: `github.com/davly/reality`
- `go list -deps ./...` — 200 non-reality packages, all stdlib; 48 reality sub-packages
- Full import scan of all `.go` files (cited file:line references throughout)
- `C:\limitless\foundation\reality\conduit\emit.go:15-22` (network stdlib usage)
- `C:\limitless\foundation\reality\audio\spectrogram\visualise.go:3-7` (image stdlib usage)
- `C:\limitless\foundation\reality\testutil\golden.go:26-32` (golden-file harness stdlib usage)
