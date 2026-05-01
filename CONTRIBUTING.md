# Contributing to Reality

Reality is the foundation Go module that hosts the canonical Welford accumulator,
forge primitives, and parity contracts consumed across the Limitless ecosystem.

## Local development

Reality targets Go 1.24+ and has no external service dependencies for the test
suite.

```bash
# from the repo root
go test ./... -race -count=1
go vet ./...
gofmt -l .   # must produce empty output
```

CI runs the same three steps on every push and pull request to `master`. See
`.github/workflows/test.yml`.

## Canonical contracts

Reality is the substrate of record for several invariants that downstream
flagships consume verbatim:

- **Welford accumulator** — `prob/welford` is the single source. Cross-substrate
  ports (Kotlin in `mobile/shared` of cohort flagships, Swift, .NET, etc.) MUST
  match Reality's bit-equivalent output. See `prob/welford/parity_test.go`.
- **Forge primitives** — `forge/` carries the canonical FNV-1a hash, delve
  walker, and conduit timeout constants. Substrate-baked ports must round-trip
  through Reality's vectors.
- **R-pattern obligations** — `R74 CanonicalDivergence`, `R80a Welford parity`,
  `R97 Substrate-baked forge`, `R104 Mirror-Mark`, and `R115 SingleEnumOutcome`
  are all anchored on Reality. Adding or modifying these primitives requires
  updating the cross-substrate parity vectors.

## Commit style

- Subject in imperative present tense, scoped: `prob: add copula CDF parity test`
- Reference the relevant R-pattern when adding a canonical primitive: `R80a:`
- Bug fixes that close a parity drift MUST include a regression test that fails
  before the fix.

## Pull requests

PRs are gated on the `Reality tests` workflow passing. Coverage is not enforced
but new code is expected to ship with table-driven tests and parity vectors
when canonical.
