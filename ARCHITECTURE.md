# Reality: architecture

> Verified against `7e552be` (origin/master) on 2026-10-07 by the arch-doc sweep (run edc992).

Reality (`github.com/davly/reality`, `go.mod`) is a Go library of deterministic numerical functions: mathematics,
statistics, physics, signal processing and related science, built on the Go standard library alone. This document
describes the library as it exists in the code. It deliberately documents only the code in this repository; it does
not describe the private systems that consume it.

## Deep docs (start here)

| File | What it covers |
|---|---|
| `ARCHITECTURE.md` (this file) | what the library is, its layers, design rules, trust boundaries, known gaps |
| `docs/CODE_MAP.md` | every package: purpose, key files, internal imports, where to dig |
| `docs/API_SURFACE.md` | exported Go API per package, the `reality-compute` HTTP routes, the outbound event shim |
| `docs/DATA_MODEL.md` | persistent and in-memory state (there is almost none), golden and stress test data files |
| `docs/OPERATIONS.md` | environment variables, the `reality-compute` startup, CI jobs, test ratchets |
| `docs/IMPROVEMENTS.md` | ranked improvement proposals with receipts |
| `docs/ARCHITECTURE_CHANGELOG.md` | what changed between documentation sweeps |

## 1. What the system is

- One Go module, `github.com/davly/reality`, declaring `go 1.24` and **no `require` lines** (`go.mod`). The
  `.tool-versions` file pins `golang 1.24` for local tooling.
- **71 importable library packages** plus one command, `cmd/reality-compute` (counted as directories holding non-test
  `.go` files: 72 in total). 277 non-test Go files, about 49k lines, 962 exported `func` declarations (functions and
  methods, regex `^func (\([^)]*\) )?[A-Z]` over non-test files); 246 test files (names ending in _test.go, per `git ls-files '*_test.go'`), 7 of them at the module root.
- Shape: "numbers in, numbers out". Almost every exported symbol is a pure function over `float64`, slices and small
  structs, with its precision, valid range and source cited in its doc comment (for example
  `prob/distributions.go:78`, `constants/physics.go:70`).
- Licence: the `LICENSE` file is Apache 2.0 (`LICENSE`). Some source comments still say "MIT-licensed"
  (`forge/session40/baked.go:6`); see `docs/IMPROVEMENTS.md`.

## 2. Where it sits

- **Inbound:** Reality is imported as a Go module by other code. The README names one importer by name ("aicore
  imports reality", `README.md:182`); the repository itself cannot show who imports it, so no consumer list is given
  here.
- **Outbound:** the library packages make **no network calls** with one exception, the opt-in event shim
  `conduit.Emit` (`conduit/emit.go:66`), which POSTs a JSON event to `CONDUIT_URL` (default
  `http://localhost:8200/v1/events`, `conduit/emit.go:28`). No non-test package in this repository imports
  `conduit` (checked with `git grep` over non-test files), so it is reached only by a caller that imports it directly.
- **Served:** `cmd/reality-compute` is a small HTTP server that exposes one function,
  `prob/conformal.AdaptiveInterval`, as the tool `reality.conformal_interval` over a `/mcp/tools/` wire contract
  (`cmd/reality-compute/server.go:72`, `cmd/reality-compute/server.go:163`). Its comments describe it as a producer
  for a gateway that sends an `X-Nexus-Service-Token` header (`cmd/reality-compute/server.go:65`). The repository
  contains no Dockerfile or deploy manifest for it, so whether and where it runs is not observable from code.

## 3. Components

| Layer | Packages (directory) | Notes |
|---|---|---|
| Foundations | `constants`, `linalg`, `calculus`, `combinatorics`, `crypto`, `geometry`, `signal`, `graph`, `autodiff` | `constants` is imported by `em`, `orbital`, `physics` and `prob` (`em/em.go:15`, `orbital/orbital.go:15`, `physics/mechanics.go:13`, `prob/sharpe.go:6`) |
| Probability and statistics | `prob` and `prob/{agreement,conformal,copula,evt,hmm,numclaim,risk}`, `moments`, `changepoint`, `infogeo`, `info/{lz,mdl}`, `evidence`, `fairness`, `causal`, `setsim` | `prob` is the largest package (17 source files); `prob/copula` and `prob/risk` import `prob` (`prob/copula/betainc.go:3`, `prob/risk/var.go:7`) |
| Optimisation and decision | `optim` and `optim/{hrp,portfolio,proximal,transport}`, `gametheory`, `queue`, `retrymath`, `reliability`, `slo`, `spc`, `trust`, `finance/taxlot` | `prob/evt` and `spc` use `optim` (`prob/evt/mle.go:6`, `spc/arl.go:77`) |
| Time series | `timeseries`, `timeseries/{dcc,garch,statespace}`, `topology/persistent` | |
| Applied science | `physics`, `em`, `fluids`, `acoustics`, `orbital`, `chaos`, `control`, `color`, `compression`, `sequence` | |
| Audio | `audio` and `audio/{beat,cqt,idbench,onset,pitch,segmentation,separation,spectrogram,tempo,vibration}` | `audio/spectrogram` and `audio/vibration` use `signal` (`audio/spectrogram/stft.go:4`, `audio/vibration/fundamental.go:6`); `audio/idbench` is an evaluation harness, not a production path (`audio/idbench/idbench.go:1`) |
| Ecosystem markers | `forge`, `forge/session40`, `pkg/canonical`, `zkmark` | shared decision thresholds and canonical constants (section 5) |
| Infrastructure | `testutil` (golden-file loader and assertions), `conduit` (outbound event shim) | `testutil` imports `testing` and `os` (`testutil/golden.go:28-36`), so it is meant for test code |
| Command | `cmd/reality-compute` | the only inbound HTTP listener (`cmd/reality-compute/main.go:36`) |

The full package list with key files is in `docs/CODE_MAP.md`.

### Internal dependency graph (non-test imports)

Edges found with `git grep '"github.com/davly/reality'` over non-test files. The graph is shallow: no package
imports more than two others.

```
em, orbital, physics, prob        -> constants
prob/copula                       -> prob, linalg
prob/risk                         -> prob
prob/evt, spc                     -> optim
causal, reliability               -> graph
optim/portfolio                   -> gametheory, linalg
forge/session40                   -> crypto, prob
audio/spectrogram                 -> audio, signal
audio/vibration                   -> signal
audio/segmentation                -> audio/onset
audio/idbench                     -> audio
cmd/reality-compute               -> prob/conformal
```

## 4. Data flow

1. **Library call (the common path).** A caller passes numbers and receives numbers. There is no I/O, no shared state
   and no goroutine on the path, apart from the exceptions in section 6.
2. **Served call.** `POST /mcp/tools/reality.conformal_interval` → service-token check
   (`cmd/reality-compute/server.go:214`) → `X-User-Id` check (`cmd/reality-compute/server.go:221`) → body capped at
   5 MiB and 1,048,576 residuals (`cmd/reality-compute/server.go:77`, `cmd/reality-compute/server.go:83`) →
   `conformal.AdaptiveInterval`, `MarginalCoverageBounds` and `EffectiveSampleSize`
   (`cmd/reality-compute/server.go:291-315`) → JSON envelope `{content, is_error, error_message}`.
3. **Emitted event.** `conduit.Emit` fills defaults (`NewStatus` "OBSERVING", `ProjectID` "reality"), resolves
   `CONDUIT_URL` at call time, then posts from a goroutine with a 100 ms timeout and drops every error
   (`conduit/emit.go:66-105`). `EmitSampled` emits one call in `SampleRate` (`conduit/emit.go:114-125`).

## 5. Key design decisions (and the test that enforces each)

| Decision | Enforced by |
|---|---|
| **Zero external dependencies.** Non-test code may import only the standard library and this module's own packages; `net/http` is allowed only in `conduit/emit.go` and `cmd/reality-compute/{main,server}.go` | `TestImportPurity` (`honesty_test.go:69`, allowlist `honesty_test.go:44-53`); `testutil/purity_test.go:79` |
| **Golden files are the contract.** Test vectors are JSON (`function`, `cases[]` with `inputs`, `expected`, per-case `tolerance` and optional `tolerance_kind` abs/rel/ulp) | `testutil/golden.go:39-68`; loader `testutil/golden.go:85` |
| **No vacuous tolerances.** An absolute tolerance at least as large as every nonzero expected value is rejected; the exception list can only shrink | `TestGoldenTolerancesAreNotVacuous` (`golden_tolerance_test.go:50`) |
| **Precision claims are executed.** 797 stress cases (29 + 146 + 622 in `testdata/stress/precision_{cases,numerics,applied}.json`) generated from mpmath at 60 digits by `tools/stressgolden/*.py` check each function against the precision its doc comment states. Known misses are listed and the list can only shrink; a case tagged `build-dependent:` may differ between builds | `TestPrecisionClaims` (`precision_test.go:200`), `buildDependentPrefix` (`precision_test.go:95`), cross-build join `TestPrecisionBuildDependentJoin` (`precision_test.go:446`), coverage of applied claims `precision_applied_test.go:284` |
| **Deterministic per build.** Every `range` over a map in non-test code (47 sites) is classified `order-insensitive` or `sorted-after`; a new or stale site fails. Map-touching functions are called 200 times and must give one bit pattern | `TestMapRangeInventory` (`determinism_test.go:237`) with `testdata/determinism/map_ranges.json`; `determinism_dynamic_test.go:120`; helpers `testutil/determinism.go` |
| **Results may differ across builds in the last bits.** Fused multiply-add on arm64 and `GOAMD64=v3` rounds differently; CI runs both | `.github/workflows/ci.yml:81-115` |
| **No unbacked cross-language claim.** The README may not claim non-Go implementations; the Python generators do not count as one | `honesty_test.go:179`, `honesty_test.go:262` |
| **Canonical ecosystem constants fail fast.** `forge/session40` pins FNV-1a and Jeffreys constants and panics in `init()` if they drift from `crypto.FNV1a64` or `prob.JeffreysConfidence` | `forge/session40/baked.go:136-189` |
| **One shared three-way verdict.** `forge.Decide` returns Uncertain on non-finite input or fewer than 3 observations, Converged at dominance >= 0.70 and confidence >= 0.65, Escape below 0.60 | `forge/convergence.go:20-37`, `forge/convergence.go:71-90` |

## 6. Runtime state and concurrency (exceptions to "pure")

- `conduit.SampleRate` is read once from `REALITY_CONDUIT_SAMPLE` at package init (`conduit/emit.go:33-45`);
  `sampleCounter` is a process-wide atomic (`conduit/emit.go:62`); `Emit` starts a goroutine per event
  (`conduit/emit.go:85`).
- `forge/session40` keeps a process-wide divergence registry behind a `sync.RWMutex`; `init()` registers nothing
  (`forge/session40/registry.go:138-139`, `forge/session40/registry.go:237`).
- `optim.simplexMaxIter` is a package variable so tests can lower it (`optim/linear.go:18-23`).
- Everything else at package level is a read-only table or a derived constant (for example `combinatorics/counting.go:43`,
  `color/spectral.go:95`, `em/em.go:21`). Details: `docs/DATA_MODEL.md`.

## 7. Trust and security boundaries

- **`reality-compute` inbound:** the only check is a constant-time comparison of `X-Nexus-Service-Token` with the
  `NEXUS_SERVICE_TOKEN` environment value; an unset value rejects every manifest GET and tool POST with 401
  (`cmd/reality-compute/server.go:337-346`, `cmd/reality-compute/main.go:25-31`). A wrong HTTP method gets 405 before
  the token is checked (`cmd/reality-compute/server.go:169-175`, `cmd/reality-compute/server.go:204-210`). `X-User-Id` is required but only
  for attribution; the tool reads no per-user data (`cmd/reality-compute/server.go:219-227`). Unknown JSON fields are
  rejected (`cmd/reality-compute/server.go:272`). Server timeouts and a 1 MiB header cap are set in
  `cmd/reality-compute/main.go:49-59`. There is no TLS in the binary; transport security is outside this code.
- **`conduit` outbound:** the destination is environment configuration, sent unauthenticated over whatever scheme the
  URL names (`conduit/emit.go:80-99`).
- **Library:** functions validate their documented domain and return NaN, `false` or an error rather than panic in
  most cases; `forge/session40` deliberately panics at init on constant drift (`forge/session40/baked.go:147-189`).
- **Supply chain:** gosec, govulncheck and trivy gate CI with pinned versions (`.github/workflows/ci.yml:200-247`);
  golangci-lint v2.14.0 with the committed `.golangci.yml` (`.github/workflows/ci.yml:167-170`).

## 8. Known gaps

- `docs/STRUCTURE.md` is a generated snapshot from 2026-04-09 that lists files that were not in the tree at `7e552be` (an
  ARCHITECTURE.md of 252 lines, a CLAUDE.md of 63 lines and a CONTEXT.md; the first two names now hold the new files
  this sweep wrote, not the ones listed) and a `prob -> conduit` import edge that the code no
  longer has (`docs/STRUCTURE.md:8-10`, `docs/STRUCTURE.md:268`).
- The README points to a CONTEXT.md that is not in the repository (`README.md:9`, `README.md:19`); several code
  comments cite `ARCHITECTURE.md` line numbers and a docs/NEXUS_CAPABILITY_EXPOSURE.md from before those files were
  removed (`honesty_test.go:8`, `honesty_test.go:48`, `cmd/reality-compute/server.go:8`).
- A comment says a `+Inf` interval bound is sent as JSON `null`, but `jsonSafe` sends `math.MaxFloat64`
  (`cmd/reality-compute/server.go:250-252` against `cmd/reality-compute/server.go:372-382`).
- `optim` has no package doc comment (no `// Package optim` line in `optim/*.go`).

Ranked proposals: `docs/IMPROVEMENTS.md`.
