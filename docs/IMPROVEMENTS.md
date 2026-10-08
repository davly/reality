# Improvements: reality

> Verified against `7e552be` (origin/master) on 2026-10-07 by the arch-doc sweep (run edc992).

Ranked proposals, each with a receipt at `7e552be`. P0 = correctness or exposure risk, P1 = a stated contract the
code does not meet or a documentation trap, P2 = hygiene. This is the first improvements file for this repository,
so nothing is marked RESOLVED yet; the changelog (`docs/ARCHITECTURE_CHANGELOG.md`) records what the 2026-10 fix
commits already closed.

## P0

None found. The library has no persistent state, no secrets and one inbound listener whose auth fails closed
(`cmd/reality-compute/server.go:337-346`).

## P1

1. **Absolute private workspace paths in a public file.** `zkmark/README.md:70-71` cites two files by absolute path in
   a private workspace. Commit `8b7c24c` removed files from this repository for exactly this reason (its message names
   "leaked absolute workspace paths"), but this file was not among them. Proposal: replace the two lines with a
   plain-language description of the design origin.
2. **Internal-integration detail in comments of a public library.** 81 non-test `.go` files mention Nexus,
   flagship, RubberDuck or aicore (`git grep -l -i -E 'nexus|flagship|rubberduck|aicore' -- '*.go' ':!*_test.go'`),
   for example the gateway loader and token field names in `cmd/reality-compute/main.go:14-16` and
   `cmd/reality-compute/server.go:86-88`. The `8b7c24c` commit message lists de-branding the remaining text as an
   open operator decision. Proposal: decide it; if de-branding, the comments in `cmd/reality-compute/` and
   `forge/session40/baked.go` are the densest.
3. **Precision claims that the code does not meet are listed, not fixed.** The precision ratchet carries 52 known
   violations in the numerics corpus (`precision_numerics_test.go:214-276`) and 129 in the applied corpus
   (`precision_applied_test.go:355-485`), each with a measured error larger than the function's own doc comment
   promises (for example `precision_numerics_test.go:216`, Pearson at an offset of 1e12, relative error 3.1e-8). The
   ratchet stops them growing (`precision_test.go:200`) but each one is a doc comment that over-claims. Proposal:
   for each entry, either fix the algorithm or weaken the doc comment's stated precision, and delete the entry.
4. **Stale in-repo docs that contradict the code.**
   - `docs/STRUCTURE.md` is a 2026-04-09 file tree listing ARCHITECTURE.md, CLAUDE.md and CONTEXT.md
     (`docs/STRUCTURE.md:8-10`), none of which exist at `7e552be`, and an import edge `prob -> conduit` that the
     code no longer has (`docs/STRUCTURE.md:268`; the real edges are in `ARCHITECTURE.md` section 3).
   - `docs/COVERAGE.md` is a 2026-05-20 snapshot of a 49-package inventory (`docs/COVERAGE.md:5-7`); the module now
     has 71 library packages.
   - `README.md:9` and `README.md:19` point to a CONTEXT.md that is not in the repository, and `README.md:19` gives
     the package count as 70 "as of 2026-07-05".
   - 15 code comments cite line numbers or sections of the ARCHITECTURE.md and CONTEXT.md that `8b7c24c`
     removed (`git grep -n -E 'CONTEXT\.md|ARCHITECTURE\.md' -- '*.go'`), for example `honesty_test.go:55`
     ("codifies ARCHITECTURE.md:27"), `cmd/reality-compute/server.go:8` and `prob/hmm/doc.go:20`;
     `honesty_test.go:48` cites a docs/NEXUS_CAPABILITY_EXPOSURE.md removed in the same commit. The ARCHITECTURE.md
     written by this sweep is a different file, so those line references now point at unrelated text.
   Proposal: delete or regenerate `docs/STRUCTURE.md` and `docs/COVERAGE.md`, and repoint the README and the comments.
5. **The per-package golden files have no generator in the repository.** The README says they are generated from Go
   with `math/big` at 256-bit precision (`README.md:174`), but no such generator is tracked; only the stress-golden
   generators are (`tools/stressgolden/generate.py`). 138 of the 142 JSON goldens therefore cannot be regenerated or
   audited from this repository (`docs/DATA_MODEL.md` section 3). Proposal: commit the generator, or state in the
   README which goldens are hand-derived.

## P2

6. **The FMA guard covers one of four Welford updates.** `audio/fingerprint.go:78` rounds `delta*delta2` before the
   add so arm64 cannot fuse it (commit `081f031`); the same update is unguarded in `audio/degradation.go:62`,
   `moments/moments.go:131` and `moments/moments.go:293`. These are allowed to differ between builds in the last bits
   (`ARCHITECTURE.md` section 5), so this is only a defect if a cross-build or cross-language golden is ever added for
   them. Proposal: apply the same conversion, or record why they are exempt.
7. **A comment in `reality-compute` contradicts the code.** `cmd/reality-compute/server.go:250-252` says an infinite
   bound is sent as JSON `null`; `jsonSafe` sends `±math.MaxFloat64` (`cmd/reality-compute/server.go:372-382`). The
   same function maps a NaN bound to 0, which a client would read as a real bound; the input cannot carry NaN through
   JSON, so this is defensive only. Proposal: fix the comment; consider returning an error instead of 0 for NaN.
8. **`conduit` is network code with no caller in this repository.** No non-test file imports `conduit`
   (`git grep '"github.com/davly/reality/conduit"'` over non-test files returns nothing); `Emit` ignores the caller's
   `ctx` and drops every error (`conduit/emit.go:66-105`), and its payload must match a struct in another repository
   that cannot be checked from here (`conduit/emit.go:47-48`). Proposal: either move the shim to its consumer or add a
   contract test against a vendored copy of the receiving struct.
9. **`reality-compute` has no health route, graceful shutdown or deploy artifact.** Only `/mcp/tools/` is registered
   (`cmd/reality-compute/server.go:163`), `main` has no signal handling (`cmd/reality-compute/main.go:24-39`), and no
   Dockerfile or manifest is tracked. If it is deployed, the deploy shape lives outside this repository. Proposal:
   add a `GET /healthz` and a `Shutdown` on SIGTERM, or document that the binary is a reference only.
10. **Licence wording.** `LICENSE` is Apache 2.0 but `forge/session40/baked.go:6` calls Reality "MIT-licensed".
    Proposal: correct the comment.
11. **Two CI workflows overlap.** `.github/workflows/test.yml:34` runs `go test ./... -race -count=1` on Go 1.24,
    which the `test` job of `.github/workflows/ci.yml:58` already does on Go 1.24 to 1.26. `test.yml` adds `go vet`,
    `gofmt -l` and a `-trimpath` leg (`.github/workflows/test.yml:22-39`). Proposal: fold those three steps into
    `ci.yml` and delete `test.yml`, halving the race-test minutes.
12. **`optim` has no package doc comment** (no `// Package optim` line in any `optim/*.go`), so `go doc` shows nothing
    for the package that `prob/evt` and `spc` build on.
