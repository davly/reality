# Coverage — reality

*Static per-package test inventory snapshot.*

- **Generated:** 2026-05-20 from `go test -json ./...` against branch `ecosystem-uplift-2026-05-20-coverage-index`
- **Go version:** go1.26.2 windows/amd64
- **Reality tip at snapshot:** `1cc7465` (W1 reality lift — README + CLAUDE refreshed to 49-package / 2400-test inventory)
- **Status at snapshot:** All packages PASS. Zero failures, zero skips.

## What this is

The audit on 2026-05-19 (`reviews/CROSS_POLL_2026-05-19/per_project/reality.md` Weakness 3 / Recommendation 2) flagged that reality's 2,400+ top-level tests are spread across 49 packages with **no master coverage table**. That gap is what produced the `TestR85ZeroDivergences` strict-grep bug (fixed in `aa160ca`) — when no central index exists, per-package R-rule tests cannot be audited as a set.

This document is the static snapshot answering "which packages have how many tests, and which R-rule tests live where". It is **not** auto-regenerated on every commit; it is a once-per-coverage-pass artefact. The audit's longer-term recommendation is a CI-generated `COVERAGE.md` — that is a future step, not the goal here.

## How to re-generate

```bash
# From foundation/reality/ root
go test -json ./... > /tmp/reality_test.json
# Then summarise. The exact summarisation is intentionally script-free;
# the JSON is the source of truth. See ecosystem-uplift-2026-05-20-coverage-index
# branch impl-log for the one-shot Python helper used for the 2026-05-20 snapshot.
```

The JSON `Action: pass` events whose `Test` field has no `/` are top-level tests; those with `/` are subtests. Package-level pass events (no `Test` field) give per-package elapsed times.

## Aggregate

| Metric | Value |
|---|---|
| Importable packages | 49 |
| Top-level test functions | 2,400 |
| Subtest invocations | 922 |
| Total test invocations | 3,322 |
| Failures | 0 |
| Skips | 0 |
| Golden-file JSON fixtures | 80 |

The 2,400 / 3,322 figures match the README + CLAUDE claims exactly. Any future drift between this snapshot and those documents is itself a signal to re-run the coverage pass.

## Per-package inventory

| Package | Top-level | Subtests | Goldens | Elapsed (ms) | R-rule tests |
|---|---:|---:|---:|---:|---|
| `acoustics` | 62 | 15 | 4 | 817 | — |
| `audio` | 21 | 0 | 1 | 815 | Welford-parity (golden cross-substrate) |
| `audio/beat` | 10 | 0 | 0 | 749 | — |
| `audio/cqt` | 22 | 4 | 0 | 772 | — |
| `audio/onset` | 21 | 0 | 0 | 898 | 3-detector cross-validation (R132 saturation site) |
| `audio/pitch` | 20 | 0 | 0 | 748 | — |
| `audio/segmentation` | 21 | 0 | 0 | 783 | — |
| `audio/separation` | 20 | 0 | 0 | 741 | — |
| `audio/spectrogram` | 19 | 0 | 0 | 763 | — |
| `audio/tempo` | 15 | 0 | 0 | 758 | — |
| `audio/vibration` | 10 | 0 | 0 | 731 | — |
| `autodiff` | 40 | 0 | 0 | 768 | — (consumer-side R131/R132 attestations live in copula / garch / infogeo) |
| `calculus` | 45 | 34 | 4 | 799 | — |
| `changepoint` | 40 | 0 | 0 | 774 | infogeo fresh-start convergence witness (R132 first consumer) |
| `chaos` | 51 | 10 | 1 | 945 | — |
| `color` | 49 | 20 | 2 | 798 | — |
| `combinatorics` | 58 | 10 | 1 | 793 | — |
| `compression` | 57 | 10 | 1 | 790 | — |
| `conduit` | 8 | 0 | 0 | 2,292 | — (fire-and-forget; elapsed includes HTTP timeouts) |
| `constants` | 5 | 53 | 1 | 974 | — |
| `control` | 78 | 10 | 1 | 894 | — |
| `crypto` | 58 | 20 | 2 | 1,053 | — (FNV-1a cross-substrate golden) |
| `em` | 31 | 43 | 10 | 995 | — |
| `fluids` | 70 | 19 | 5 | 1,014 | — |
| `forge/session40` | 17 | 0 | 3 | 845 | — (bedrock-FNV cross-substrate golden) |
| `gametheory` | 66 | 18 | 2 | 1,044 | — |
| `geometry` | 91 | 22 | 2 | 1,073 | — |
| `graph` | 134 | 86 | 7 | 1,092 | — |
| `info/lz` | 21 | 0 | 0 | 818 | — |
| `info/mdl` | 33 | 0 | 0 | 804 | — |
| `infogeo` | 33 | 3 | 0 | 8,135 | autodiff KL gradient pin (R131 site) |
| `linalg` | 154 | 50 | 5 | 1,091 | — |
| `optim` | 73 | 8 | 1 | 1,939 | R123 validated-variant trap-catcher suite (7 tests); LASSO {FBS,FISTA,ADMM} orthogonal closed-form (R131 first consumer of `optim/proximal/`, deliberately run from parent `optim` package as cross-package consumer) |
| `optim/proximal` | 53 | 0 | 0 | 890 | — (R131 attestations live in `optim/proximal_consumer_test.go` in the parent `optim` package) |
| `optim/transport` | 30 | 0 | 0 | 860 | — |
| `orbital` | 27 | 39 | 8 | 1,039 | — |
| `physics` | 79 | 26 | 3 | 974 | — |
| `pkg/canonical` | 11 | 0 | 0 | 958 | R85 canonical-source declaration + zero-divergences + non-empty (3 tests; canonical R85 site) |
| `prob` | 192 | 316 | 10 | 1,131 | — (consumer-side R-rule attestations live in copula / conformal sub-packages) |
| `prob/conformal` | 47 | 0 | 0 | 1,086 | — (R124 canonical site; tests are conformal-coverage-property tests, not named R124) |
| `prob/copula` | 112 | 5 | 0 | 910 | Clayton log-PDF × autodiff gradient pin (R131 site); Solvency II Annex IV smoke (cross-domain regulatory) |
| `queue` | 63 | 18 | 2 | 1,024 | — |
| `sequence` | 141 | 41 | 1 | 1,058 | — |
| `signal` | 69 | 10 | 1 | 1,085 | — |
| `testutil` | 14 | 32 | 2 | 934 | — (golden-file harness self-tests) |
| `timeseries/dcc` | 35 | 0 | 0 | 798 | FilterSeries parity-with-single-step-loop |
| `timeseries/garch` | 43 | 0 | 0 | 1,250 | ForecastVariance closed-form (R131) + autodiff gradient parity (R132 mutually-attesting first consumer) |
| `topology/persistent` | 19 | 0 | 0 | 843 | — |
| `zkmark` | 12 | 0 | 0 | 750 | — (S61 NEW-1 Tranche 1; consumer-count audit pending) |
| **TOTAL** | **2,400** | **922** | **80** | — | — |

## R-rule test sites (auditable set)

The audit's specific concern: "per-package R-rule tests aren't centrally indexed". The current authoritative R-rule test sites are:

### R85 — OwnedPrimitives (canonical declaration of canonical-source)

- `pkg/canonical/canonical_test.go` — three R85-named tests: `TestR85IsCanonicalSourceDeclared`, `TestR85CanonicalPrimitivesNonEmpty`, `TestR85ZeroDivergences` (the last one had a strict-grep bug closed in `aa160ca`; current form is scope-aware per `feedback_mirror_problem_arc.md`)
- `pkg/canonical/canonical_expansion_test.go` — 8 property tests on the canonical-primitives set (dotted format, no duplicates, lowercase, determinism, etc.); the property tests are property-named, not R85-named, but they enforce the invariants R85 depends on

Reality is the **canonical reference implementation** for R85 — the `pkg/canonical` package itself defines what "canonical-primitive" means ecosystem-wide.

### R123 — Validated-variant convergence-trap catcher

- `optim/gradient_validated_test.go` — 7 tests across `TestR123ConvergenceTrap_*` covering vanilla-gradient-descent false-convergence + validated-variant catch + warm-start + nil-validate rejection + budget-exhaustion honest reporting

### R124 — ConformalBand (canonical site)

- `prob/conformal/split_test.go` + `adaptive_test.go` + `mondrian_test.go` + `nonconformity_test.go` — 47 top-level tests covering nominal coverage (`TestSplitInterval_AchievesNominalCoverage`), CQR (`TestCqrInterval_BasicShape`), marginal coverage bounds (`TestMarginalCoverageBounds_SandwichValues`), Mondrian conditional coverage, adaptive nonconformity, and cross-substrate FW-corpus precision (`TestCrossSubstratePrecision_FwCorpus_*`). The tests are property-named rather than R124-named because they test the coverage property the R-rule encodes.

### R131 — Closed-form-pinned-to-autodiff (canonical here)

Saturation sites (3/3 promotion 2026-05-06):
- `timeseries/garch/garch_test.go:TestForecastVariance_ClosedForm` — multi-step forecast pinned to closed-form variance recursion
- `optim/proximal_consumer_test.go:TestProximalLasso_{FBS,FISTA,ADMM}_OrthogonalClosedForm` — 3 solver variants vs `β* = soft(y, λ)` orthogonal-design closed form; LASSO is the first consumer of `optim/proximal`. Note: the tests live in the *parent* `optim` package, not in `optim/proximal/`, to exercise the cross-package consumer path
- `prob/copula/autodiff_test.go:TestClaytonLogPDF_AutodiffGradientMatchesAnalytic` — Clayton log-PDF gradient pin (autodiff against analytical closed form)
- `infogeo/autodiff_test.go:TestKL_AutodiffGradientMatchesQMinusP` — KL gradient matches q-p closed form

### R132 — Mutual-cross-validation-in-parity-test (canonical here)

Saturation sites (3/3 promotion 2026-05-06):
- `timeseries/garch/autodiff_test.go:TestNegLogLikGrad_AutodiffEquivalence` — GARCH analytic gradient × reverse-mode autodiff mutually-attesting parity (first consumer of both `garch` and `autodiff` for parity purposes)
- `audio/onset/cross_validation_test.go:TestThreeOnsetDetectors_AgreeOnPercussiveTrain` — 3-detector cross-validation: energy + spectral-flux + complex-domain
- `changepoint/infogeo_test.go:TestPosterior_FreshStartConvergence` — BOCPD fresh-start convergence witness against infogeo posterior
- `audio/parity_test.go:TestWelford_ParityWithGoldenVector` — golden-file as cross-substrate oracle, the original pattern this R-rule generalises

### Solvency II / regulatory smoke test

- `prob/copula/gaussian_test.go` — `TestSolvencyII_AnnexIV_SmokeTest` (EIOPA Reg 2015/35 Annex IV; load-bearing post-S55 relic-insurance pivot per `reference_revolutionary_trio.md`)

## Coverage gaps surfaced by this snapshot

1. **`audio/vibration` consumer attestation** — 10 top-level tests, 0 subtests, 0 goldens. Audit notes vibration is consumer-count 2/3 (commit `b0eb862`); this snapshot confirms the test surface but does not by itself close the consumer-count gap.

2. **`zkmark` consumer-count** — 12 top-level tests, no goldens, no R-rule tests yet. Audit Cross-Pollination "Could learn from others" notes the first downstream consumer post-`a53e693` is not yet clear. Pair with Nexus Drainer.deliver (Mirror-Mark on dispatch chokepoints) is the proposed next step.

3. **`autodiff` and `prob` self-tests vs consumer-side R-rule tests** — These two foundational packages carry no in-package R-rule test markers, but they are the substrate that R131 + R132 cross-validate against. The R-rule attestations correctly live at consumer-side (copula / garch / infogeo / proximal) per the saturation discipline. This is the right shape; calling it out here for completeness.

4. **`testdata/` layout has two shapes** — Some packages use `<pkg>/testdata/<pkg>/` (e.g. `acoustics/testdata/acoustics/`) and some use top-level `testdata/<pkg>/` (e.g. `testdata/calculus/`). Both work for Go's testdata convention; the inconsistency is cosmetic but worth noting in case future cross-language port-runners need to discover golden files by convention.

5. **`conduit` 2,292 ms elapsed** — high relative to siblings because tests exercise HTTP fire-and-forget paths with real timeouts. The architectural boundary is principled per audit Weakness 5, but the elapsed-time outlier is a useful signal that conduit is the only I/O surface in an otherwise zero-deps library.

## Not in scope

- **Cross-language parity-runner status** — Phase 3 (Python / C++ / C# from-scratch reimplements) is the audit's Recommendation 1, separate from this coverage snapshot. The 80 golden-file fixtures listed above are the inputs that work would consume.
- **NIST CODATA 2018 vs 2022 staleness** — audit Recommendation 3. The `constants` package has 5 top-level + 53 subtests + 1 golden as of this snapshot; staleness audit is a separate pass.
- **CI-regenerable form of this document** — the audit suggests this be auto-generated from `go test -json`. The current document is a static snapshot. Future work: add a `make coverage` target that regenerates this in-place.
