# Operations: reality

> Verified against `7e552be` (origin/master) on 2026-10-07 by the arch-doc sweep (run edc992).

Reality is mainly a library: it has no service of its own to operate except the optional `cmd/reality-compute`
binary. This file covers that binary, the environment variables the code reads, and the CI that guards the library.
What runs where in production is not observable from this repository.

## 1. Environment variables

Complete list: every `os.Getenv` in the module (`git grep -n Getenv -- '*.go'`).

### Read at run time

| Name | Default | Effect | Read at |
|---|---|---|---|
| `NEXUS_SERVICE_TOKEN` | unset | shared secret compared with the `X-Nexus-Service-Token` header; unset means every manifest GET and tool POST gets 401 (a wrong method gets 405 first), and a warning is logged at startup | `cmd/reality-compute/main.go:17`, `cmd/reality-compute/main.go:25-31` |
| `PORT` | `8090` | listen port of `reality-compute` (`:PORT`, all interfaces) | `cmd/reality-compute/main.go:19-21`, `cmd/reality-compute/main.go:46-50` |
| `CONDUIT_URL` | `http://localhost:8200/v1/events` | destination of `conduit.Emit`; read on every call | `conduit/emit.go:28`, `conduit/emit.go:80-83` |
| `REALITY_CONDUIT_SAMPLE` | 10000 | 1-in-N rate of `conduit.EmitSampled`; read once at package init; a non-integer or non-positive value keeps the default | `conduit/emit.go:33-45` |

### Read by tests only

| Name | Effect | Read at |
|---|---|---|
| `REALITY_UPDATE_MAP_RANGES` | rewrite `testdata/determinism/map_ranges.json` (new sites as `unreviewed`, stale ones dropped) | `determinism_test.go:253`, `determinism_test.go:271` |
| `REALITY_LIST_MAP_RANGES` | log every map-range site with its class | `determinism_test.go:266` |
| `REALITY_PRECISION_REPORT` | log every precision case with its error | `precision_test.go:231` |
| `REALITY_PRECISION_OUTCOMES` | write one JSON line per precision case to this path | `precision_test.go:353` |
| `REALITY_BUILD_TARGET` | label for those lines (default `GOOS/GOARCH`) | `precision_test.go:361-364` |
| `REALITY_PRECISION_JOIN_DIR` | directory of outcome files for the cross-build join; unset means `TestPrecisionBuildDependentJoin` skips | `precision_test.go:447-450` |

## 2. `reality-compute` startup

1. Read `NEXUS_SERVICE_TOKEN`; if empty, log a warning and keep starting, fail-closed (`cmd/reality-compute/main.go:25-31`).
2. Build the server: address `:` + `PORT` (default 8090), read timeout 10 s, read-header timeout 5 s, write timeout
   30 s, idle timeout 60 s, 1 MiB header cap (`cmd/reality-compute/main.go:45-59`).
3. Log the address and the tool name, then `ListenAndServe`; any error other than `http.ErrServerClosed` is fatal
   (`cmd/reality-compute/main.go:35-38`).

There is no graceful shutdown or signal handling, no TLS, no health or readiness route, no metrics endpoint and no
request logging. Logging is the two startup lines above through the standard `log` package. The server keeps no
state between requests.

## 3. Background work

None in the library. The only goroutine started by non-test code is the fire-and-forget POST in `conduit.Emit`
(`conduit/emit.go:85`), bounded by a 100 ms timeout. There are no schedulers or tickers.

## 4. Build and deploy artifacts in the repository

- No Dockerfile, compose file, Makefile, deploy manifest or release workflow is tracked (`git ls-files`). The library
  is consumed as the Go module `github.com/davly/reality`; the binary would be built with `go build
  ./cmd/reality-compute`.
- `.tool-versions` pins `golang 1.24`; `go.mod` declares `go 1.24`.
- `.github/dependabot.yml` opens weekly update PRs for Go modules and GitHub Actions.

## 5. CI

Two workflow files run on pushes and pull requests.

### `.github/workflows/ci.yml` (branches `main` and `master`, `.github/workflows/ci.yml:3-7`)

| Job | What it does | Lines |
|---|---|---|
| `test` | Go 1.24, 1.25 and 1.26 (fail-fast off): `go build ./...`, then `go test -race -coverprofile ... -count=1 ./...`; writes precision outcomes; fails under 80 % total coverage | `.github/workflows/ci.yml:18-74` |
| `fma` | default tests on `GOAMD64=v3` (ubuntu) and on arm64 (`ubuntu-24.04-arm`), Go 1.26, uploading precision outcomes | `.github/workflows/ci.yml:81-115` |
| `precision-join` | after `test` and `fma`: downloads every build's outcomes and runs `TestPrecisionBuildDependentJoin` | `.github/workflows/ci.yml:121-142` |
| `lint` | golangci-lint v2.14.0, action pinned by commit, config `.golangci.yml` | `.github/workflows/ci.yml:144-170` |
| `bench` | `go test -bench=. -benchmem -run='^$' ./...` on Go 1.24 (no threshold) | `.github/workflows/ci.yml:172-193` |
| `security` | gosec v2.29.0, govulncheck v1.8.0 and trivy (CRITICAL/HIGH, unfixed ignored), each failing the job | `.github/workflows/ci.yml:200-247` |

### `.github/workflows/test.yml` (branch `master`)

`go vet ./...`, a `gofmt -l` check that must print nothing, `go test ./... -race -count=1`, and
`go test ./... -trimpath -count=1` on Go 1.24 (`.github/workflows/test.yml:21-39`). The `-trimpath` leg exists
because golden files are found through compile-time source paths (`testutil/golden.go:96-104`).

### Test ratchets that act as gates

These run inside `go test ./...` (all at the repository root except the two `testutil` purity tests):

| Test | Fails when | Where |
|---|---|---|
| `TestImportPurity` | a non-test file imports a non-stdlib module, or `net/http` outside the allowlist | `honesty_test.go:69` |
| `TestMapRangeInventory` | a map-range site is unclassified, `unreviewed` or stale | `determinism_test.go:237` |
| `TestDeterministic_MapTouchingFunctions` | a map-touching function gives two outputs in 200 calls | `determinism_dynamic_test.go:120` |
| `TestGoldenTolerancesAreNotVacuous` | a golden case's absolute tolerance swallows its expected value | `golden_tolerance_test.go:50` |
| `TestPrecisionClaims` | an unlisted stress case misses its claim, or a listed one now meets it | `precision_test.go:200` |
| `TestAppliedPrecisionClaimsAreCovered` | an applied-package function states a precision with no stress case | `precision_applied_test.go:284` |
| `TestNoUnbackedCrossLanguageClaim` | the README claims non-Go implementations that are not in the tree | `honesty_test.go:179` |
| `TestZeroExternalDependencies` | `go.mod` has a `require` line, or any `.go` file (test files included) imports a non-stdlib, non-reality path | `testutil/purity_test.go:79` |
| `TestImpureStdlibImportsAllowlisted` | a non-test file outside `conduit` and `cmd/reality-compute` imports `net/http`, `net` or `os/exec` | `testutil/purity_test.go:135` |
| `forge/session40` `init()` | canonical constants drift (panics on import, so every test binary that imports it fails) | `forge/session40/baked.go:136-189` |

Whether these jobs currently pass on GitHub is not observable from the code.

