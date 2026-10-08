# Architecture changelog: reality

Newest first. One entry per documentation sweep, recording what changed in the system and what the docs had got
wrong. Line-level history: `git log -p -- ARCHITECTURE.md docs/`. Frozen copies of earlier docs: the godfather repo's
`architecture/history/snapshots/`.

## 2026-10-07: arch-doc sweep (run edc992)

- **Code range covered:** initial architecture doc at `7e552be` (origin/master). An earlier ARCHITECTURE.md, CLAUDE.md
  and CONTEXT.md were removed from this public repository in `8b7c24c` (2026-08-13); this pack is written fresh from
  the code and does not restore them. For orientation, the 52 code commits from `8b7c24c` to `7e552be` are summarised
  below.
- **Architecture changes in this range:**
  - CI gained two build shapes and a join: an `fma` job runs the tests on `GOAMD64=v3` and on arm64, and a
    `precision-join` job compares precision outcomes across all builds (`.github/workflows/ci.yml:81-142`). Tool
    versions are pinned (golangci-lint v2.14.0 with a committed `.golangci.yml`, gosec v2.29.0, govulncheck v1.8.0;
    `.github/workflows/ci.yml:170`, `.github/workflows/ci.yml:226`, `.github/workflows/ci.yml:234`), the `test` matrix grew from Go
    1.24/1.25 to 1.24/1.25/1.26 with `fail-fast: false` (`.github/workflows/ci.yml:24-28`), lint and security moved to
    Go 1.26, and `test.yml` gained a `-trimpath` test leg (`.github/workflows/test.yml:39`).
  - New root-level test ratchets that gate every `go test ./...`: the map-range determinism inventory with its data
    file (`determinism_test.go:237`, `testdata/determinism/map_ranges.json`, helpers `testutil/determinism.go`), the
    dynamic determinism check (`determinism_dynamic_test.go:120`), the vacuous-tolerance check
    (`golden_tolerance_test.go:50`), and the precision ratchet over 797 mpmath stress cases
    (`precision_test.go:200`, `testdata/stress/precision_cases.json`, `testdata/stress/precision_numerics.json`,
    `testdata/stress/precision_applied.json`) with its Python generators (`tools/stressgolden/generate.py`,
    `tools/stressgolden/numerics.py`, `tools/stressgolden/applied.py`).
  - The golden-file contract gained an optional `tolerance_kind` (`abs`, `rel`, `ulp`) (`testutil/golden.go:61-67`),
    and golden files are now located correctly under `-trimpath` (`testutil/golden.go:96-104`).
  - Every range over a map in non-test code was made order-independent or sorted (commit `b21d655`: `graph`,
    `reliability`, `trust`).
  - `prob` special functions rewritten: regularized incomplete gamma and beta, with t-test p-values computed directly
    rather than as `1 - CDF` (`prob/mathutil.go`, commits `1ae893d`, `200a382`); `prob/copula` now shares the incomplete
    beta for its Student-t CDF (`prob/copula/betainc.go`), and `copula.StudentTQuantile` is deprecated in favour of
    `prob.StudentTQuantile` (`prob/copula/studentt.go:78`).
  - Behaviour fixes with interface effect: `evt.FitGEVMLE` and `evt.FitGPDMLE` now maximise the likelihood through
    `optim` instead of returning the closed-form fit (`prob/evt/mle.go:283`, `prob/evt/mle.go:333`); new
    `crypto.LCMChecked` beside the wrapping `crypto.LCM` (`crypto/prime.go:275`); `combinatorics.Factorial`,
    `BinomialCoeff` and `Permutations` are correctly rounded from an exact table (`combinatorics/counting.go:43`);
    Fisher exact and Benjamini-Hochberg tie handling, `NormalQuantile` refinement, `ExponentialQuantile` via `log1p`,
    `StefanBoltzmann` set to its exact value, `QuatToAxisAngle` and `TrueAnomalyFromMean` robustness (commits
    `228af86`, `a8feda0`, `4a0dc81`, `ca7bfa1`, `db44d2e`, `82cf494`, `f1628a8`).
  - `conduit.Emit` resolves `CONDUIT_URL` at call time instead of inside its goroutine, and `EmitSampled` treats a
    non-positive `SampleRate` as disabled instead of dividing by zero (`conduit/emit.go:80-83`,
    `conduit/emit.go:114-118`).
  - `audio.UpdateFingerprint` keeps its Welford update unfused so its cross-language parity golden holds on arm64
    (`audio/fingerprint.go:78`).
  - Mass hygiene, no architectural change: gofmt over 77 files, 36 lint fixes, 18 gosec findings resolved, quadratic
    sorts replaced by stdlib sorts, and an `actions/setup-go` bump (commits `e315311`, `c8cb880`, `b12f5ad`,
    `f177ab0`, `d48590a`).
  - Unchanged in the range: `go.mod` (still `go 1.24`, no `require` lines), the internal import graph, and the
    `reality-compute` routes.
- **Doc corrections:** no previous pack at `7e552be`. In the in-repo docs that remain, the following are stale and are
  listed in `docs/IMPROVEMENTS.md` item 4: `docs/STRUCTURE.md` (2026-04-09 tree, `prob -> conduit` edge that no
  longer exists, `docs/STRUCTURE.md:268`), `docs/COVERAGE.md` (49-package snapshot), the README's CONTEXT.md pointer
  and "70 packages" (`README.md:19`), and 15 code comments citing the removed ARCHITECTURE.md and CONTEXT.md. During
  this sweep the first draft's test-file count was corrected from 239 to 246 (it had left out the 7 root-level test
  files). The adversarial verifier then corrected: the licence-comment line (`forge/session40/baked.go:5` → `:6`),
  the zkmark private-path lines (`zkmark/README.md:71-72` → `:70-71`), the `Welford`/`WelfordVec` type lines
  (`moments/moments.go:112`/`262` → `:86`/`:255`; the old ones were the constructors), the colour-map range
  (`audio/spectrogram/colourmap.go:22-82` → `:22-99`), the `errors.go` list (6 → all 9 files), the 401 claim (a
  wrong HTTP method gets 405 before the token check, `cmd/reality-compute/server.go:169-175`), the STRUCTURE.md gap
  wording (two of its listed files exist again, rewritten), and added the two `testutil` purity ratchets to
  `docs/OPERATIONS.md` (`testutil/purity_test.go:79`, `testutil/purity_test.go:135`).
- **Re-verified unchanged:** not applicable (first pack). Spot re-checks at write time: package and file counts, the
  internal import graph, every `os.Getenv` site, the `reality-compute` routes and auth, the CI job list.
- **Open questions:**
  - This is a public repository, and `8b7c24c` removed ARCHITECTURE.md and CLAUDE.md from it as internal detail. This
    pack documents only code in the repository and names no private paths, but publishing it re-adds files of those
    names. Whether to land it on `master`, keep it on a branch, or move it to a private location is an operator
    decision.
  - Whether `cmd/reality-compute` is deployed, and where, is not observable from code (no Dockerfile or manifest).
  - Who imports the module besides the one importer the README names (`README.md:182`) is not observable from this
    repository.
  - How the 138 per-package golden files were generated: the README describes a `math/big` generator
    (`README.md:174`) that is not in the repository.
