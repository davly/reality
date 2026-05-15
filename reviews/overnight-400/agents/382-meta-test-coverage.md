# 382 — meta-test-coverage (cross-cutting test audit)

## Headline
Reality has 97 test files / 636 golden vectors but **zero `FuzzN` functions, zero property-based generators**, and golden coverage is uneven (median ~7 vectors/function, far below the CLAUDE.md "min 20, target 30" mandate).

## Inventory (raw counts, repo-wide)

| Metric | Count | Notes |
|---|---|---|
| Packages (top-level) | 41 dirs | CLAUDE.md still says "22"; doc drift |
| `*_test.go` files | 97 | |
| `func Test*` + `Benchmark*` + `Example*` | ~2,400 occurrences across 97 files | grep `^func (Test\|Benchmark\|Example)\w+` |
| `func Fuzz*` | **0** | grep `^func Fuzz` returns no matches |
| `testing/quick` / `pgregory.net/rapid` / `gopter` imports | **0** | no property libs in go.mod or test imports |
| Golden JSON files | 80 | under `*/testdata/` and root `/testdata/` |
| Golden test cases (sum of `"description"` keys) | **636** | mean ≈8 cases/file, well below CLAUDE.md target=30 |
| Test files referencing NaN/Inf/Subnormal/MaxFloat | 24 / 97 (25%) | only ~¼ of test files exercise IEEE 754 specials |

## Coverage matrix (by package, sampled)

Vector counts derived from `grep -c '"description"'` per file in `*/testdata/`. "Edge" = file references `math.NaN()` / `math.Inf(` / subnormals.

| Package | Test files | Golden files | Σ vectors | Median/file | FuzzN | Edge cases |
|---|---|---|---|---|---|---|
| acoustics | 2 | 4 | 15 | 4 | 0 | yes (acoustics_edge_test.go) |
| calculus | 1 | 4 (root testdata) | 34 | 8.5 | 0 | partial |
| chaos | 1 | 1 | 10 | 10 | 0 | no |
| color | 1 | 2 | 20 | 10 | 0 | no |
| combinatorics | 1 | 1 | 10 | 10 | 0 | no |
| compression | 1 | 1 | 10 | 10 | 0 | no |
| constants | 2 | 1 (physics_constants) | 13 | 13 | 0 | no |
| control | 2 | 1 | 10 | 10 | 0 | yes (control_edge_test.go) |
| crypto | 2 | 2 | 20 | 10 | 0 | no |
| em | 1 | 10 | 43 | 4 | 0 | no |
| fluids | 2 | 5 | 19 | 4 | 0 | yes (fluids_edge_test.go) |
| gametheory | 1 | 2 | 18 | 9 | 0 | no |
| geometry | 1 | 2 | 22 | 11 | 0 | no |
| graph | 1 | 7 | 86 | 10 | 0 | no |
| linalg | 2 | 5 | 50 | 10 | 0 | no (numerics-heavy, gap) |
| optim | 2 | 1 | 8 | 8 | 0 | no |
| optim/proximal | 3 | 0 | 0 | — | 0 | yes (admm/fbs reference NaN) |
| optim/transport | 1 | 0 | 0 | — | 0 | yes |
| orbital | 1 | 8 | 39 | 4.5 | 0 | no |
| physics | 1 | 3 | 26 | 8 | 0 | no |
| prob | 9 | 10 | 114 | 10 | 0 | yes (distributions_test) |
| prob/copula | 8 | 0 | 0 | — | 0 | yes (5 files) |
| prob/conformal | 4 | 0 | 0 | — | 0 | yes (4 files) |
| queue | 1 | 2 | 18 | 9 | 0 | no |
| sequence | 4 | 1 | 10 | 10 | 0 | no |
| signal | 1 | 1 | 10 | 10 | 0 | no |
| testutil | 2 | 2 | 8 | 4 | 0 | yes (NaN/±Inf) |
| audio (8 sub-pkgs) | 11 | 1 (welford-parity) | 1 | 1 | 0 | yes (onset, mdl) |
| changepoint | 3 | 0 | 0 | — | 0 | yes |
| autodiff | 2 | 0 | 0 | — | 0 | no |
| infogeo | 4 | 0 | 0 | — | 0 | no |
| timeseries/{garch,dcc} | 5 | 0 | 0 | — | 0 | yes (garch) |
| topology/persistent | 1 | 0 | 0 | — | 0 | yes |
| info/{lz,mdl} | 2 | 0 | 0 | — | 0 | yes (mdl) |
| forge/session40 | 2 | 3 (basispoints/fnv1a/situation) | 32 | 11 | 0 | no |
| zkmark | 1 | 0 | 0 | — | 0 | no |
| pkg/canonical | 2 | 0 | 0 | — | 0 | no |
| conduit | 1 | 0 | 0 | — | 0 | no |

## Key findings

### F1 — `Fuzz*` adoption is **zero**
- 0 fuzz targets across 97 test files. Go 1.18 native fuzzing released March 2022; reality (Go module) ships none.
- Highest-value fuzz candidates already gated by parsers/decoders:
  - `crypto.MillerRabin`, `Mersenne` — bit-twiddle invariants
  - `compression.Huffman`, `compression.LZ77`, `info/lz76`, `info/mdl` — round-trip `decode(encode(x))==x`
  - `linalg` LU/QR/Cholesky — `A == L·U` reconstruction modulo eps
  - `signal.FFT/IFFT` — invertibility; Parseval
  - `geometry.QuaternionSlerp` — unit-norm preservation
  - `forge/session40` `basispoints_roundtrip` (already a roundtrip in golden form — trivial fuzz target)

### F2 — Property-based testing is **zero**
- No imports of `testing/quick` (stdlib), `pgregory.net/rapid`, or `leanovate/gopter`.
- "Properties" exist only as ad-hoc `t.Run` cases (e.g. `sequence_edge_test.go::TriangleInequality`). These are 1-shot, not generated.
- Strong fits for property tests:
  - `prob` distributions: ∫pdf=1, CDF monotone, F⁻¹(F(x))≈x
  - `linalg`: spectral inequalities, det(AB)=det(A)·det(B)
  - `gametheory.NashEq`: payoff bounds
  - `graph.Dijkstra` vs `BellmanFord` agreement on non-negative weights

### F3 — Golden-file vector counts are **3–4× under target**
- CLAUDE.md: "minimum 20 vectors per function, target 30."
- Observed median per golden file: **~7-10**.
- Files at or above target: `graph/pagerank.json` (21), `graph/bellman_ford.json` (20), `graph/kruskal_mst.json` (20). Three out of 80.
- Many golden files at **3–5 vectors**: `em/coulomb_force.json` (5), `acoustics/decibel_spl.json` (3), `prob/ema.json` (3), `fluids/darcy_weisbach.json` (3), `orbital/hohmann_transfer.json` (3), `orbital/hill_sphere.json` (3).

### F4 — IEEE 754 edge-case coverage is **25% of test files**
- 24/97 test files reference `math.NaN()/Inf()/MaxFloat64`. CLAUDE.md mandates this for *every* function.
- `linalg` has zero NaN/Inf references in its tests — a gap (matrix singularities, overflow in determinants).
- `signal/signal_test.go` references no IEEE specials (FFT of array containing NaN should propagate NaN).
- `prob/copula/*` have IEEE coverage; good. `prob/conformal/*` likewise.
- Subnormals (`math.SmallestNonzeroFloat64`): grep returned **zero** hits in any test file.

### F5 — Newer / "growth" packages have **no golden files**
Packages with `*_test.go` but no `testdata/` JSON corpus:
`autodiff`, `changepoint`, `infogeo`, `optim/proximal`, `optim/transport`, `prob/copula`, `prob/conformal`, `timeseries/garch`, `timeseries/dcc`, `topology/persistent`, `info/lz`, `info/mdl`, `pkg/canonical`, `conduit`, `zkmark`, `audio/{beat,cqt,onset,pitch,segmentation,separation,spectrogram,tempo,vibration}`.
- These are post-v0.10 additions. Implies golden-file mandate from CLAUDE.md was applied to v0.1–0.10 core but not enforced on growth packages.

### F6 — Stress / long-horizon tests
- No `-stress` build tag, no long-horizon iteration counts > 10000 in any test file (grepped `for.*<.*10000` — none in test loops).
- No benchmarks gated to multi-second runs.
- `optim/genetic_test.go` runs single GA solve, but generation count fixed and small.
- Slot 308 (KF 10k iters) reference is aspirational — no analogue currently exists.

### F7 — Doc drift
- CLAUDE.md says "Packages (22)" and "Tests: 1,965 (22 packages)". Actual: 41 top-level dirs, 97 test files, ~2,400 test funcs. Header is stale by ≥10 packages.

## Concrete recommendations

1. **Add `pgregory.net/rapid` as the single PBT dep.** Modern, generic-typed, auto-minimizes failing cases — strictly better than `testing/quick`. Single allowed test-only dep; CLAUDE.md "zero deps" applies to runtime, not test scaffolding (precedent: `testing` itself). Target: ≥1 `rapid.Check` per package (41 properties total).
2. **Land 30 `FuzzN` targets in one PR**, prioritized:
   - Round-trip pairs: `compression.{Huffman,RLE,Delta,LZ77}`, `info/lz76`, `info/mdl`, `forge.basispoints`, `forge.fnv1a`, `crypto.{Mersenne,LCG}`.
   - Numeric invariants: `signal.FFT`/`IFFT`, `linalg.{LU,QR,Cholesky}` reconstruct A.
   - Parsers (none in reality currently — skip).
3. **Backfill golden vectors to CLAUDE.md targets.** All files <20 vectors are out-of-spec. Regenerate via the same `math/big` 256-bit code path. Estimated ~60 files to extend; can be scripted (random in-domain inputs + Go reference impl). Add a CI check: `golden_lint.go` that fails when any `*.json` in `*/testdata/` has fewer than 20 cases.
4. **Mandatory IEEE 754 edge-case suite per function.** Add `*_ieee_test.go` template enforcing the 7 specials per CLAUDE.md: `+0, -0, +Inf, -Inf, NaN, MinSubnormal, MaxFloat64`. CI: AST-level lint that every exported `func` in non-test files has a corresponding `Test<Name>_IEEE` somewhere in the package, or an explicit `// reality:no-ieee` waiver comment with reason.
5. **Golden coverage in growth packages.** Highest-priority backfill list: `autodiff` (gradient pins), `prob/copula` (PDF/Kendall-tau), `prob/conformal` (split nonconformity scores), `timeseries/garch`, `optim/proximal/{admm,fbs}`. These have the largest API surface with zero JSON coverage today.
6. **Long-horizon stress harness.** Add `//go:build stress` files for: `chaos.Lorenz` 1e6 steps drift bounds, `optim.LBFGS` on Rosenbrock-1000 dim, KF/GARCH 10k-iter convergence (slot 308 alignment). Run nightly only.
7. **Update CLAUDE.md** quick-reference: "22 packages" → "41 packages", "Tests: 1,965" → real count post-PR. Add "Fuzz: N targets" and "PBT: N properties" lines.
8. **Coverage telemetry.** `go test -cover ./...` is not currently surfaced in any review artifact. Wire `go test -coverprofile` into `make test`, fail PRs that drop pkg-level coverage below floor (suggested: 85% statement on math packages, 70% on glue packages like `conduit`/`zkmark`).

## Quick wins (≤1 day each)
- Add `Fuzz_BasispointsRoundtrip`, `Fuzz_FNV1a`, `Fuzz_Levenshtein` (string monotonicity), `Fuzz_LZ76Roundtrip`. ~4 fuzz targets in <100 LOC.
- Generate-and-commit 20-vector golden files for the eight `em/*` files currently at 4–6 cases.
- Run `staticcheck`/`govulncheck` over `*_test.go` once — likely surfaces dead test code.

## Sources
- Go fuzzing tutorial: https://go.dev/doc/tutorial/fuzz (Go 1.18+ native `F.Add`, `F.Fuzz`; coverage-guided)
- Go fuzzing security guide: https://go.dev/doc/security/fuzz/
- "The State of Go Fuzzing — did we already reach the peak?" — https://0x434b.dev/the-state-of-go-fuzzing-did-we-already-reach-the-peak/ (adoption flat 2024–2026)
- pgregory.net/rapid (modern PBT, auto-minimization, generics-typed): https://pkg.go.dev/pgregory.net/rapid and https://github.com/flyingmutant/rapid
- DZone "Comprehensive Guide to Property-Based Testing in Go": https://dzone.com/articles/property-based-testing-guide-go
- Earthly blog "Property-Based Testing In Go": https://earthly.dev/blog/property-based-testing/
- JetBrains GoLand fuzz testing primer: https://blog.jetbrains.com/go/2022/12/14/understanding-fuzz-testing-in-go/
- Internal: `C:/limitless/foundation/reality/CLAUDE.md` (golden-file mandate, 20/30 vector targets, IEEE 754 edge-case rule)
- Internal: `C:/limitless/foundation/reality/testutil/golden_test.go` (NaN/±Inf assertion harness)
