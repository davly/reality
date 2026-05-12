# Reality — Security policy

**Status**: Tier-0 substrate. Foundation of foundations. A vulnerability here propagates to every consumer in the ecosystem (aicore → SDKs → flagships → apps).

Reality is mostly pure math with zero external dependencies, so the security surface is genuinely narrow. The exceptions — `crypto` (canonical ecosystem hashing / PRNGs) and `conduit` (fire-and-forget HTTP shim) — are listed below.

## Supported versions

| Version | Supported |
|---|---|
| v0.10.x (current) | yes |
| v0.9.x and earlier | no — upgrade required |

## Reporting a vulnerability

Email **david@vocala.co** with subject prefix `[SECURITY:reality]`. Please do **not** open public issues for vulnerabilities (this is the only public repository in the `davly/` organisation, so disclosure has higher blast radius than private repos).

Include:
- Affected package(s) (e.g. `crypto`, `conduit`, `prob`)
- Reproduction steps or proof-of-concept
- Suggested severity (low / medium / high / critical)
- Whether the vulnerability is already public (CVE id if available)

Acknowledgement: within 72 hours. Triaged severity + remediation plan: within 7 days for high/critical, 14 days for medium, 30 days for low.

## Security surfaces in reality

Reality's "no allocations in hot paths" and "deterministic output" properties also serve as security properties (no side channels via allocation timing, no platform-divergent output to fingerprint). These packages have first-order security exposure and are reviewed line-by-line on every change:

| Package | Threat model | Canonical guard |
|---|---|---|
| `crypto` | Hash collisions / weak PRNGs propagated through consumers. FNV-1a is **not** a cryptographic hash and must not be used for security purposes — see package doc. | `crypto/hash.go` matches `architecture/fnv1a_canonical_vectors.json` byte-for-byte; PRNGs require explicit seed. |
| `conduit` | Fire-and-forget HTTP emit; URL injection via `CONDUIT_URL` env. | 100 ms timeout, no response body read, no credentials transmitted. Endpoint override is local-trust (operator-controlled env). |
| `prob/conformal`, `prob/copula` | Numerical-pinning of regulator-grade guarantees (S55 L01 / L13). Adversarial inputs could degrade calibration if precision contract breaks. | Per-function golden vectors at `≤1e-12` cross-substrate; closed-form-pinned-to-autodiff (R131) where applicable. |

## What reality is **not**

- Reality does **not** implement cryptographic ciphers. The `crypto` package holds number-theory primitives (Miller-Rabin, ModPow, EEAD) and non-cryptographic hashes (FNV-1a, MurmurHash3). Security-critical encryption belongs in audited libraries (`crypto/aes`, `crypto/rsa`, etc.) — not here.
- Reality does **not** call out to the network on hot paths. `conduit/emit.go` is an opt-in shim that consumers may invoke; no other reality package imports `net/http`.
- Reality does **not** read environment variables on hot paths. The single exception is `conduit/emit.go` (`CONDUIT_URL`, `REALITY_CONDUIT_SAMPLE`), and even that is read once at boot.

## CI security gates

Per R142 scope-extension and OVERNIGHT_2026-05-12-13 brief safe-change list, every push and PR runs:

- **gosec** — static analysis for unsafe Go patterns (`exit-code: 1`)
- **govulncheck** — module-graph vulnerability scan (`exit-code: 1`). Reality has zero external dependencies, so this guards against regressions where one slips in.
- **trivy** — filesystem scan for CRITICAL/HIGH vulnerabilities (`exit-code: 1`)

See `.github/workflows/ci.yml` `security:` job.

## Known limitations

- The `conduit` package's `Emit()` is fire-and-forget by design (per `CONTEXT.md` §11 / `ARCHITECTURE.md` §2). Failures are silent — operators should monitor the Conduit bus side, not reality.
- `crypto.FNV1a32`/`FNV1a64` are **non-cryptographic** by design and remain so. Consumers using them for security purposes (e.g. authentication tokens) are misusing the API; the canonical ecosystem use case is situation-hashing in delve/walker code.

## Disclosure

Once a fix lands and downstream consumers (per `ARCHITECTURE.md` §1 dependency DAG: aicore → SDKs → flagships → apps) have updated to a patched release, we publish a coordinated advisory referencing the CVE if assigned.
