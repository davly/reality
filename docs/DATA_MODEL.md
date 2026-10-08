# Data model: reality

> Verified against `7e552be` (origin/master) on 2026-10-07 by the arch-doc sweep (run edc992).

**No persistent state.** Reality has no database, no migrations, no cache and no file it writes at run time. The only
non-test file read in library code is the golden-file loader in `testutil`, which is test support
(`testutil/golden.go:106`). Checked with `git grep` for `os.ReadFile`, `os.Create`, `os.WriteFile`, `os.Open` and
`os.OpenFile` over non-test files: that one hit is the only one.

## 1. Process-wide in-memory state

Everything else at package level is a read-only lookup table or a derived constant.

| State | Where | Lifetime and notes |
|---|---|---|
| `conduit.SampleRate` (int, default 10000) | `conduit/emit.go:33` | set once at package init from `REALITY_CONDUIT_SAMPLE`; exported and assignable by importers |
| `conduit.sampleCounter` (atomic uint64) | `conduit/emit.go:62` | counts `EmitSampled` calls for the whole process; lost on exit, by design |
| R74 divergence registry (`[]CanonicalDivergence` behind `sync.RWMutex`) | `forge/session40/registry.go:138-139` | filled only by `Register`; `init()` registers nothing (`forge/session40/registry.go:237`), so it is empty unless an importer registers entries |
| `optim.simplexMaxIter` (10000) | `optim/linear.go:23` | package variable so tests can lower it |
| Read-only tables | `combinatorics/counting.go:43` (171 exact factorials), `color/spectral.go:95` (CIE observer), `audio/spectrogram/colourmap.go:22-99` (four 16-stop colour maps), `em/em.go:21`, `info/mdl/universal_int.go:16` | computed or written once, never mutated |

Nothing here should be durable: the counter and registry are diagnostic, not records.

## 2. Caller-owned state objects

Several packages export stateful values that the caller creates and keeps. They hold no global state and nothing
persists them; a caller that needs them across restarts must serialise their fields itself.

| Type | Where | What it accumulates |
|---|---|---|
| `moments.Welford`, `moments.WelfordVec` | `moments/moments.go:86`, `moments/moments.go:255` | count, mean and M2; `Merge` combines two (`moments/moments.go:229`) |
| `audio.Fingerprint`, `audio.DegradationTracker` | `audio/fingerprint.go:24`, `audio/degradation.go:28` | Welford mean and variance per feature; baseline and window statistics |
| `conformal.ACI`, `conformal.ACIStream` | `prob/conformal/aci.go:34`, `prob/conformal/aci.go:87` | adaptive conformal miscoverage level |
| `timeseries.EWMoments` | `timeseries/ewvar.go:104` | exponentially weighted mean and variance |
| `changepoint.Bocpd`, `changepoint.EDetector` | `changepoint/bocpd.go:85`, `changepoint/edetector.go:187` | run-length posterior; e-process values |
| `control.PIDController` | `control/pid.go:36` | integral and previous error |

## 3. Data files in the repository

All are test inputs, read by test files. 142 JSON files are tracked.

| Files | Count | Read by | Format |
|---|---|---|---|
| per-package golden files under `<package>/testdata/`, usually `<package>/testdata/<package>/*.json` (for example `prob/testdata/prob/`) | 112 | each package's tests through `testutil.LoadGolden` (`testutil/golden.go:85`) | `{"function", "cases":[{"description","inputs","expected","tolerance","tolerance_kind"?}]}` (`testutil/golden.go:39-77`) |
| shared golden files, `testdata/<package>/*.json` (calculus, chaos, constants, control, crypto, forge, gametheory, optim, queue, sequence, slo) | 26 | the matching package's tests | same format |
| `testdata/determinism/map_ranges.json` | 1 | `determinism_test.go:237` | `{"_comment", "sites": {"<pkg>.<func>#k": {"class", "note"}}}`; 47 sites, each `order-insensitive` or `sorted-after` |
| `testdata/stress/precision_cases.json`, `testdata/stress/precision_numerics.json`, `testdata/stress/precision_applied.json` | 3 (29, 146 and 622 cases) | `precision_test.go:200` with the evaluators in `precision_numerics_test.go` and `precision_applied_test.go` | `{"_comment", "generator", "cases":[...]}`; each case carries the claim it checks |

Golden tolerances are checked for vacuity across every golden file (`golden_tolerance_test.go:50`).

### Generators

- `tools/stressgolden/generate.py`, `tools/stressgolden/numerics.py` and `tools/stressgolden/applied.py` write the three
  stress files from mpmath at 60 significant digits, evaluated at the exact binary64 inputs
  (`tools/stressgolden/generate.py:2-10`). They are development tools: Go code never imports them and CI does not run
  them. The mpmath version is recorded in each output file's `generator` field.
- `REALITY_UPDATE_MAP_RANGES=1` makes `TestMapRangeInventory` rewrite `map_ranges.json`, adding new sites as
  `unreviewed` (which then fail) and dropping stale ones (`determinism_test.go:271-285`).
- How the per-package golden files were produced is not recorded in code. The README says they are generated from Go
  with `math/big` at 256-bit precision (`README.md:174`); no generator for them is in the repository.

## 4. Files written by tests

| File | Written when | Where |
|---|---|---|
| precision outcome lines (`.jsonl`) | `REALITY_PRECISION_OUTCOMES` names a path | `precision_test.go:351-372`; CI uploads them as artifacts (`.github/workflows/ci.yml:56`, `.github/workflows/ci.yml:98`) |
| `testdata/determinism/map_ranges.json` | `REALITY_UPDATE_MAP_RANGES` is set | `determinism_test.go:271-285` |
| `coverage.out` | CI coverage step | `.github/workflows/ci.yml:58` |
