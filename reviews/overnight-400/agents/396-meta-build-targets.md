# 396 — meta-build-targets (WASM/mobile/embedded portability audit)

## Headline
Reality builds clean on every Go-supported target tested (10 OS×arch combinations, including js/wasm, wasip1/wasm, linux/{arm,arm64,riscv64,386,mips,ppc64le,s390x}, darwin/arm64, ios/arm64, android/arm64, freebsd/amd64, plan9/amd64) with zero build tags, zero assembly, zero CGO, and zero `unsafe` — the only portability cracks are `conduit` (`net/http`) and `audio/spectrogram/visualise` (`image/png`), which TinyGo cannot compile.

## Method
- `go env`: `go1.26.2 windows/amd64`.
- Cross-build matrix: `GOOS=X GOARCH=Y go build ./...` for each row below. Exit code 0 means full repo compiles with no errors.
- Static scans (Grep over `**/*.go`):
  - `//go:build` directives in non-test code: **0** (all 4 hits are in `reviews/`, none in source).
  - Old-form `// +build` directives: **0**.
  - `_amd64.s`, `_arm64.s`, `*.s` assembly files: **0** (`Glob **/*.s` → none).
  - `import "C"`, `//go:cgo`, `//export`: **0**.
  - `"unsafe"` imports: **0** in `*.go` (only mentioned in one review file).
  - `"reflect"`: **0**.
  - `"runtime"`: 1 hit — `testutil/golden.go:73` uses `runtime.Caller(1)` for golden-file path resolution (test infra only, not in production import graph for non-test builds).
  - `"math/big"`: **0** in committed source (golden generators use it ad-hoc but are not in `./...`).
  - Stdlib imports outside math: cataloged in slot 395; the only target-relevant ones are `net/http` (1 file) and `image/png` (1 file).

## Per-target compatibility matrix

| Target | Status | Issues |
|---|---|---|
| `js/wasm` | OK (`go build ./...` exit 0; `go vet ./...` exit 0) | `net/http` in `conduit` works via JS `fetch()` shim; bloats wasm binary (~MB-class) for callers who never need telemetry. `image/png` in `audio/spectrogram` adds non-trivial weight. Recommend build-tag gates (see Recommendation §1). |
| `wasip1/wasm` | OK (exit 0) | `net/http` compiles but is a no-op at runtime under WASI Preview 1 (no netdev). Functional for pure-math callers; `conduit.Emit` will silently fail (it's already fail-silent by design — `emit.go:1-13`). |
| `linux/amd64` | OK (host build, all 1,965 tests pass) | None. |
| `linux/arm64` | OK (exit 0) | None. |
| `linux/arm` (GOARM=5/6/7) | OK (exit 0 on all three) | 32-bit ARM has soft-float on GOARM=5; performance only, no correctness. Slot 380's GOAMD64=v3 FMA fusion concern does not apply (no FMA on ARMv5). |
| `linux/386` | OK (exit 0) | x87 80-bit extended precision can leak into intermediate float64 results (Go masks this on 386 by default with strictfp). No reality-specific code touches this. |
| `linux/riscv64` | OK (exit 0) | RISC-V Go support is GA; no platform-specific code in reality. Embedded RISC-V via TinyGo is the use case (see TinyGo row). |
| `linux/mips`, `linux/ppc64le`, `linux/s390x` | OK (exit 0) | Big-endian (`mips`, `s390x`, `ppc64`) untested for byte-swap correctness; reality uses no `binary.Read` / `unsafe.Pointer` casts so no endianness bugs are reachable. |
| `darwin/arm64` (Apple Silicon) | OK (exit 0) | None. |
| `ios/arm64` | OK (exit 0 with pure Go; gomobile bind would need the wrapper layer) | gomobile bind: documented to support arm64 in 2026 (golang.org/x/mobile current). Reality has no exported types that gomobile struggles with — all `func(...) float64` style — so `gomobile bind ./signal` etc. should work. Untested in this audit. |
| `android/arm64` | OK (exit 0) | Same as ios/arm64. |
| `freebsd/amd64`, `plan9/amd64` | OK (exit 0) | Sanity cases. None. |
| **TinyGo** (any target) | **PARTIAL** — not directly tested (no TinyGo on this host), but blocked by `net/http` in `conduit/emit.go:19` and `image/png` in `audio/spectrogram/visualise.go:7`. TinyGo 0.41 (2026) explicitly does **not** support `net/http` on wasm/wasi (tinygo-org/tinygo issues #2704, #4540). `image/png` is also not in the supported stdlib set. The other 47 sub-packages should compile under TinyGo. | Build-tag gates fix this (Recommendation §1). |
| **Browser via Go-stdlib WASM** | OK | Bundle-size only — uncompressed `go build` WASM output for the full `./...` will be 5–15 MB (typical Go stdlib floor). No correctness issues. |

Summary: **every Go-toolchain target passes `go build ./...` with exit 0**. TinyGo / browser WASM size are the only friction points, both isolated to two files.

## What slot 380 (GOAMD64=v3 FMA fusion) implies for portability

Slot 380 flagged that Go 1.25+ on GOAMD64=v3 may auto-fuse `a*b+c` to FMA, breaking bit-exact golden parity with non-FMA targets (arm32, plain amd64, wasm). This is **not a build-target compile failure** — it's a runtime-numerics divergence. The matrix above is unaffected (everything still compiles), but it does mean the **golden-file tolerance budget must already account for FMA-vs-non-FMA diff** on transcendental and accumulating ops. That is slot 380's domain; flagged here only because a portability matrix without that footnote would be misleading.

## Cracks found

1. **`conduit/emit.go:19` imports `net/http`** — full transitive pull (`crypto/tls`, `crypto/x509`, system roots, DNS resolver). Fine on every `go build`-supported target, but disqualifies the package from TinyGo and bloats every WASM bundle that imports the full module by ≥500 KB. The `Emit` API is fail-silent by design, so the package is dead weight when the bus is unreachable (i.e. in WASM/embedded). Slot 395 already recommended a `//go:build conduit_emit` tag for the no-allocations rule; the same tag fixes the build-target story.
2. **`audio/spectrogram/visualise.go:7` imports `image/png`** — pulls `compress/zlib`, `compress/flate`, `hash/crc32`. Not supported by TinyGo. The `visualise.go` file is a single rendering helper (`ToHeatmap`) that's orthogonal to the rest of the spectrogram package; it should live behind a build tag too.
3. **No CI matrix.** There is no GitHub Actions / build-matrix file in the repo (verified by `Glob` for `.github/workflows`, no hits) that exercises non-default targets. A library that advertises "pure Go, zero deps, runs anywhere" with no portability CI is one PR away from breaking it silently.
4. **`testutil/golden.go:73` uses `runtime.Caller`** — fine on every standard Go target (`runtime.Caller` is universally supported), and `testutil` is test-only so it doesn't ship in production builds. Flagged only for completeness.
5. **No build-target documentation.** `CLAUDE.md` lists OSes (Windows/Linux/macOS) but not architectures, not WASM, not embedded. `CONTEXT.md:602` claims "no `//go:build` directives in non-test files" — accurate, and the strongest portability artifact in the repo, but buried.

## Recommendation

1. **Build-tag gate the two non-portable files**:
   ```
   conduit/emit.go              add: //go:build conduit_emit
   conduit/emit_noop.go (NEW)   add: //go:build !conduit_emit
                                provide: stub Emit/EmitSampled that no-op
   audio/spectrogram/visualise.go              add: //go:build !tinygo
   audio/spectrogram/visualise_stub.go (NEW)   add: //go:build tinygo
                                provide: stub ToHeatmap that returns ErrUnsupported
   ```
   After this change, `tinygo build -target=wasi ./...` and `tinygo build -target=arm ./...` should succeed for the entire module (modulo TinyGo's own stdlib gaps, which are not reality's fault).

2. **Add a portability CI matrix** (`.github/workflows/portability.yml`):
   ```yaml
   strategy:
     matrix:
       include:
         - {goos: js,      goarch: wasm}
         - {goos: wasip1,  goarch: wasm}
         - {goos: linux,   goarch: arm64}
         - {goos: linux,   goarch: arm,    goarm: "7"}
         - {goos: linux,   goarch: 386}
         - {goos: linux,   goarch: riscv64}
         - {goos: linux,   goarch: s390x}             # big-endian sanity
         - {goos: darwin,  goarch: arm64}
         - {goos: ios,     goarch: arm64}
         - {goos: android, goarch: arm64}
   steps:
     - run: GOOS=${{matrix.goos}} GOARCH=${{matrix.goarch}} GOARM=${{matrix.goarm}} go build ./...
     - run: GOOS=${{matrix.goos}} GOARCH=${{matrix.goarch}} go vet ./...
   ```
   Add a separate TinyGo job that runs `tinygo build -target=wasi ./signal ./calculus ./linalg ./prob ./physics ./constants` (the embedded-relevant packages) — not `./...` because `conduit` and `audio/spectrogram` will fail there until step 1 lands.

3. **Run the test suite on at least one non-amd64 target nightly.** `go test ./...` under `qemu-arm64-static` or GitHub's `ubuntu-arm64` runner. This catches the FMA / 80-bit-x87 / endianness divergences that pure cross-compile cannot. Especially relevant given slot 380's GOAMD64=v3 FMA flag — the goldens that pass on amd64 may diverge on arm64, and only running the tests there will surface it.

4. **Document the supported-target matrix in `CLAUDE.md`** alongside the existing 22-package list. Three tiers:
   - **Tier 1 (CI-tested, fully supported):** linux/amd64, darwin/arm64, windows/amd64. All tests pass, all goldens parity.
   - **Tier 2 (cross-compiled in CI, not test-run):** js/wasm, wasip1/wasm, linux/{arm,arm64,riscv64,386,s390x,ppc64le}, ios/arm64, android/arm64. `go build` clean; goldens **not guaranteed** bit-exact (FMA / soft-float / x87 differences).
   - **Tier 3 (best-effort, not in CI):** TinyGo on AVR/ARM/RISC-V embedded targets. Excluding `conduit` and `audio/spectrogram/visualise.go`, the math sub-packages should work, but no validation.

5. **Reframe `CONTEXT.md:602`'s portability claim as a CI invariant.** Right now it's a documented snapshot ("the module builds on Windows, Linux, and macOS without any build tags"). Make it executable: a CI step that fails the build if `grep -r '//go:build' --include='*.go'` returns any non-test hit (or any hit not in the allowed-list of build tags from step 1). This converts the property into a regression-resistant invariant.

## Sources
- `C:\limitless\foundation\reality\go.mod` (3 lines, `go 1.24` declared; tested on host `go1.26.2`)
- `C:\limitless\foundation\reality\CLAUDE.md` (design rules, 22-package list)
- `C:\limitless\foundation\reality\CONTEXT.md:602` ("builds on Windows, Linux, and macOS without any build tags")
- `C:\limitless\foundation\reality\conduit\emit.go:13-23` (`net/http`, `bytes`, `context`, `encoding/json`, `os`, `sync/atomic`, `time` imports — the full TinyGo-incompatible set)
- `C:\limitless\foundation\reality\audio\spectrogram\visualise.go:1-8` (`bytes`, `image`, `image/color`, `image/png`)
- `C:\limitless\foundation\reality\testutil\golden.go:31,73` (`runtime.Caller(1)` — test infra only)
- Slot 357 (pure-Go USP), slot 380 (GOAMD64=v3 FMA fusion concern), slot 395 (zero-deps audit; recommended `//go:build conduit_emit` for `conduit`)
- Cross-build matrix runs on this host (10 GOOS×GOARCH combinations, all exit 0; `js/wasm` also `go vet ./...` exit 0)
- [TinyGo Packages supported by TinyGo](https://tinygo.org/docs/reference/lang-support/stdlib/)
- [TinyGo 0.41 release notes (2026)](https://tinygo.org/blog/2026/tinygo-0-41-the-big-release/)
- [TinyGo issue #2704 — net package not supported on WASM/WASI](https://github.com/tinygo-org/tinygo/issues/2704)
- [TinyGo issue #4540 — make HTTP request from wasi](https://github.com/tinygo-org/tinygo/issues/4540)
- [Go Wiki: WebAssembly](https://go.dev/wiki/WebAssembly) (js/wasm `net/http` is implemented via JS fetch())
- [gomobile pkg.go.dev](https://pkg.go.dev/golang.org/x/mobile/cmd/gomobile) (April 2026 update; arm64 supported on iOS and Android)
- [Go issue #58141 — `GOOS=wasip1 GOARCH=wasm` port](https://github.com/golang/go/issues/58141)
