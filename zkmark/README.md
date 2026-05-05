# zkmark — Zero-Knowledge Mirror-Mark substrate

Tranche 1 of the regulator-grade audit augmentation per `S61_S65_PLAN.md` NEW-1.

## What this is

An honest-pending interface for ZK proofs over L43 Mirror-Mark statements. Tranche 1 ships the interface contract + an `HonestProver` that wraps the canonical mirror-mark digest as a `proof_pending=true` envelope. Tranche 2 (S62-S63, conditional on Marc dry-run encouragement) will land a real Halo2 implementation.

## Substrate-only ship (Tranche 1 boundary)

This package ships the **interface and stub-aware implementations** only. It deliberately does NOT depend on `nexus/internal/mirrormark` to keep the substrate boundary clean — instead it accepts a caller-injected `SignerFunc` / `VerifierFunc`. This means:

- `foundation/reality/zkmark/` has zero crypto dependencies
- Cross-substrate parity (R80b) translates structurally — a C# port at `Limitless.Nexus.Client/Forge/ZkMark/` can reuse the same `Func<...>` shape
- Wiring sites (CLI tools, services) inject the canonical mirror-mark Sign/Verify functions

## Wire status (as of S61 close)

| Site | Status | Notes |
|------|--------|-------|
| `foundation/reality/zkmark/` substrate | ✅ shipped | This package + tests |
| `nexus/cmd/lore-mark-verify --zk` flag | ⏳ S62 | Requires reality v0.11.0 release + nexus dep bump |
| C# port (`Limitless.Nexus.Client/Forge/ZkMark/`) | ⏳ S62 | Mirrors interface shape; reuses MirrorMarker.Sign |
| Marc handoff package `--zk` augmentation | ⏳ S62 | Pulls from above |

## API at a glance

```go
import "github.com/davly/reality/zkmark"
import "github.com/nexus/api/internal/mirrormark"

// Tranche 1: honest-pending prover
prover := zkmark.NewHonestProver(mirrormark.Sign)
proof, err := prover.Prove(payload, corpusSHA, key)
// proof.Algorithm == "honest-pending"
// proof.ProofPending == true
// proof.MarkChainStatement == real mirror-mark string (cold-verifiable)
// proof.ProofBytes == nil

// Tranche 1: matching verifier
verifier := zkmark.NewMarkVerifier(mirrormark.Verify)
ok, err := verifier.VerifyProof(proof, payload, key)
// ok == true if mirror-mark verifies

// Tranche 2 stub (returns ErrNotYetWired today)
halo2 := zkmark.NewHalo2Prover()
_, err = halo2.Prove(payload, corpusSHA, key)
// errors.Is(err, zkmark.ErrNotYetWired) == true
```

## Honesty-gate compliance (R128)

- Interface is real and shipped; **Halo2 backend is explicitly stubbed** and returns `ErrNotYetWired` so callers cannot accidentally treat unwired proofs as real
- `Proof.ProofPending` flag distinguishes "trust mark via cold-verifier" (Tranche 1) from "trust ProofBytes via Halo2" (Tranche 2)
- `Proof.Algorithm` constant lets parsers branch on prover identity without string-matching internal docs
- Forward-compatible: new algorithms add via constants without breaking existing parsers

## Cross-substrate parity contract (R80b)

When the C# port lands at `Limitless.Nexus.Client/Forge/ZkMark/`, the following must hold byte-for-byte:

- Same `Algorithm` constant strings ("honest-pending", "halo2")
- Same `Proof` field ordering when serialized to JSON (alphabetical: `algorithm`, `corpusSha`, `markChainStatement`, `proofBytes`, `proofPending`)
- Same `ErrNotYetWired` semantics (C#: `NotImplementedException` with matching message text)
- Same prover/verifier interface shapes adapted to .NET idioms (`IZkProver`, `IZkVerifier`)

Cross-substrate parity test (R80b): a future `nexus/zkmark/parity_test.go` will assert that the same `(payload, corpusSHA, key)` tuple produces byte-identical `Proof` JSON in Go and C#.

## See also

- `C:/LimitlessGodfather/S61_S65_PLAN.md` — sibling plan, NEW-1 section
- `C:/LimitlessGodfather/reviews/2026-05-05-aspirational/wave_01_strategic_visioning/agent_005_privacy_preserving_compute.md` — origin agent report
- `nexus/internal/mirrormark/` — canonical SignerFunc/VerifierFunc implementations to inject
- `governance/SUBSTRATE_GRANT.md` — IP framework
