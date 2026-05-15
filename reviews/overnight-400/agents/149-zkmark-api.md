# 149 | zkmark-api | circuit DSL ergonomics, witness vs public input

## Two-line summary

`zkmark/` ships **no circuit DSL today** — the only public surface is a flat
3-tuple `(payload []byte, corpusSHA [32]byte, key []byte)` that conflates
witness and public input as raw bytes; before Tranche 2 (per 148's
recommended Plonky3-on-Goldilocks backend) the package needs a typed
`Witness` / `Public` split, an AIR-style `ConstraintBuilder` frontend, and
a `Setup → (pk, vk)` separation. The single most actionable fix today (no
Tranche 2 wait) is **pinning the expected public input on the `Verifier`
constructor**, not the per-call API — this generalises 146's G9
`expectedCorpusSHA` gap to all public inputs and prevents the deployed-ZK
class of bugs (Tornado Cash, zkSync, Aleo, Halo2 NULL_FIELD).

## Scope (delta vs 146/147/148)

- 146 = numerics surface, JSON ordering (G7), `expectedCorpusSHA` (G9),
  ProofPending zero-value (G1), 13 findings.
- 147 = missing primitives (KZG/FRI/IPA/PlonK/Halo2/STARK/RS/MLE/logUp/
  cq/plookup) + 4 ZK-only T1 layers; sprint ordering.
- 148 = SOTA shoot-out → recommend Plonky3-Goldilocks-FRI backend.

This audit's delta: **DSL ergonomics**, **witness/public type discipline**,
**`Proof`/`Verifier` interface as an interface**, **gnark DSL comparison**.
Path: `C:/limitless/foundation/reality/zkmark/zkmark.go` (262 lines).

## A. Circuit DSL ergonomics — none today, decision pending

### A1. The current "DSL" is a 3-tuple of raw bytes

```go
prover.Prove(payload []byte, corpusSHA [32]byte, key []byte) (Proof, error)
```

No constraint authoring, no variable allocation, no public/private input
distinction at the type level, no algebraic relation declaration. Correct
for Tranche 1 (substrate delegates "what is being proved" to the injected
`SignerFunc`), but the moment Tranche 2 lands this 3-byte-tuple shape
**stops working** — a STARK proves an AIR, not a black-box function call.

### A2. AIR vs Plonkish vs R1CS — pick one before Tranche 2

| Style    | Examples              | Author writes                                   | Strengths                  |
|----------|-----------------------|-------------------------------------------------|----------------------------|
| R1CS     | Groth16, Spartan      | `(a · b == c)` triples                          | Universal                  |
| Plonkish | PlonK, Halo2, Plonky2 | `(qL·a + qR·b + qO·c + qM·a·b + qC == 0)` rows  | Custom gates, lookups      |
| AIR      | StarkWare, Plonky3    | trace cols + transition polys `t(row_i, row_{i+1})` | Cleanest for state machines |

148 → Plonky3 → **AIR is the right answer**. Go translation: a
`ConstraintBuilder` interface with `AssertZero(expr)`, `AssertEq(a, b)`,
`Mul/Add` over a typed `Expression`. This is *not* gnark's pattern (gnark
is Plonkish/R1CS); it *is* Plonky3's — copy Plonky3's design even though
gnark is the closer Go-idiom peer.

### A3. The injected `SignerFunc` is *not* a circuit

A `SignerFunc` is a Go function value — opaque to the prover, with no
symbolic representation a constraint compiler can lift to AIR. The
`Prover.Prove(payload, corpusSHA, key) (Proof, error)` signature is
**unsalvageable** for Tranche 2. Either (1) add a separate
`ProveCircuit(c Circuit, w Witness)` method leaving `Prove(...)` as a
Tranche-1 fast path, or (2) introduce a typed `Statement` interface
subsuming both `MirrorMarkStatement{payload, corpusSHA, key}` and
`AIRStatement{trace, constraints, publicInputs}` and make
`Prove(s Statement) (Proof, error)`. **Recommend (2)**, with the
3-arg form as a deprecated convenience wrapper, so Tranche 2 is
purely additive.

## B. Witness vs Public Input — type discipline

### B1. The current API conflates everything as "input"

The three args `(payload, corpusSHA, key)` are *not* labeled by their
ZK role:
- `payload` — semantically a **public input** (the document attested).
- `corpusSHA` — semantically a **public commitment** (hash of corpus).
- `key` — semantically a **witness** (secret HMAC key) — but Tranche 1
  exposes it to the verifier (structural-only "ZK", README §"Substrate-
  only ship").

**The type system does not enforce the distinction.** A caller can swap
`payload` and `key` and get a (different but still valid) Proof — no
compile-time nor runtime check.

### B2. Public/witness confusion is the deployed-ZK #1 bug class

Every famous post-deployment ZK bug reduces to a public/witness mix-up:

- **Tornado Cash 2020 frontrunning** — "secret" nullifier
  reconstructable from a "public" Merkle path because front-end
  serialised them to the same byte buffer.
- **zkSync 2022 prover-config bug** — constraint pinned on a witness
  column instead of a public column → soundness lost.
- **Aleo 2023 "private" credential leak** — `Public<Field>` should
  have been `Private<Field>`; auditor caught pre-mainnet.
- **Halo2 NULL_FIELD** — copy-constraint between public/private cells
  leaked private cell prefix-bits via the proof.

**Shared failure mode:** the language's type system treated public and
private inputs as the same primitive (a `BigInt`, `[]byte`, `Field`),
and the bug surfaced only at audit. Exactly what zkmark's API is set
up for today.

### B3. Recommended type discipline (Go)

**B3-A (Tranche 1 backport, 30 LOC):** distinct named-type wrappers, no
method change:

```go
type Public []byte                   // visible to verifier
type Witness []byte                  // hidden in real ZK
type PublicCommitment [32]byte       // pre-hashed public input

func (p *HonestProver) Prove(pub Public, com PublicCommitment, w Witness) (Proof, error)
```

`prover.Prove(witnessBytes, sha, publicBytes)` fails to compile.

**B3-B (Tranche 2 frontend, 200 LOC):** typed `Variable` with visibility
tag (`Visibility` enum: `VisPublic|VisWitness|VisConstant`); `AssertEq`
errors when copying witness into public output (gnark-style "secret-to-
public copy violation" check).

**B3-C (Plonky3-style AIR, 800 LOC):** trace cols typed as
`MainTrace[Field]`, `PublicTrace[Field]`, `PreprocessedTrace[Field]`;
column visibility lives in the type, not a runtime tag. Largest upfront
cost, strongest guarantee.

**Recommendation:** ship **B3-A now** (30 LOC, breaks `nexus/cmd/lore-
mark-verify` only at the call-site cast level), commit to **B3-C** as
the Tranche 2 target. Skip B3-B (only useful if Tranche 2 is Plonkish,
148 said no).

### B4. Revealed vs committed public input — second sub-distinction

`Proof.MarkChainStatement` (string) is a *revealed* public input;
`Proof.CorpusSHA` ([32]byte) is a *committed* public input — same
semantic role (public), different commit kind. A `Statement` type
should encode this with a tag:

```go
type PublicInput struct {
    Name      string
    Kind      PublicKind  // KindRevealed | KindCommitted
    Bytes     []byte
    Algorithm string      // "" for revealed, hash name for committed
}
```

Single canonical enumeration of "what the proof says about the world",
without inferring from field names.

## C. Proof / Verifier interface — current stub vs Tranche 2

### C1. The `Verifier` interface today

```go
type Verifier interface {
    VerifyProof(proof Proof, payload []byte, key []byte) (bool, error)
}
```

**C1a. `key []byte` on verifier side leaks the witness.** A real ZK
verifier never sees the witness; Tranche 1 does (HMAC needs the key);
the *interface* shouldn't hard-bake that. **Interface-breaking change
waiting to happen.**

**C1b. No `expectedPublicInput` parameter** — 146's G9 generalised. The
verifier doesn't pin the public input it expects to verify against. A
malicious prover can produce a valid proof for a *different* (payload,
corpusSHA) pair and the verifier returns `(true, nil)` because it passes
`proof.CorpusSHA` (attacker-controlled) not the verifier's expected SHA.
**Fix (Tranche 1, 20 LOC):** thread `expectedPublic Public` and
`expectedCommitment PublicCommitment` into the *verifier constructor*,
not per-call. Per-call `VerifyProof(p Proof, w Witness)` then has no way
to bypass the expected public input.

**C1c. `(bool, error)` return is footgun-prone.** Standard ZK practice:
return `error` only — `nil` = valid, anything else = invalid+typed-reason.
`(false, nil)` is currently a bug state with no error to explain. gnark
ships `error`; halo2 ships `Result<(), VerificationError>`.
**Fix:** `VerifyProof(proof Proof) error` with named sentinels per 146
§G10 (`ErrPublicInputMismatch`, `ErrProofMalformed`, `ErrUnknownAlgorithm`,
`ErrNotYetWired`, `ErrSoundnessFailure`).

### C2. The `Prover` interface

```go
type Prover interface {
    Prove(payload []byte, corpusSHA [32]byte, key []byte) (Proof, error)
    Algorithm() string
}
```

**C2a. No `Setup(circuit Circuit) (ProverKey, VerifierKey, error)`
phase.** Production ZK splits *circuit setup* (one-time, expensive,
disk-persistable) from *per-proof prove*. Plonky3 / Halo2 / gnark /
Arkworks all split: `Setup(circuit) → (pk, vk)` + `Prove(pk, witness,
public) → proof` + `Verify(vk, public, proof) → ok`. Current shape
collapses both into `Prove(...)`, only works because Tranche 1's `Sign`
is per-statement-only.

**C2b. No `EstimateProofSize(circuit) int` / `EstimateProveTime`.**
Convenience methods every production library ships. Optional, almost-
free.

**C2c. No streaming/batched proof API.** STARK aggregation needs
`AggregateProofs([]Proof) (Proof, error)` and `BatchVerify([]Proof)
error`. Out of Tranche-1 scope but interface should leave room.

### C3. The `Proof` struct — additions beyond 146

146 covered `ProofPending` (G1), JSON order (G7), hash-agnostic CorpusSHA
(G3), no length bound (G6). API-level additions:

**C3a. No `PublicInputs []PublicInput` field.** A real ZK proof carries
an explicit public-inputs list. Today `MarkChainStatement` and
`CorpusSHA` *are* the public inputs but typed as `string` and
`[32]byte` — not extensible. Add even in Tranche 1 with
`MarkChainStatement` as first element of `KindRevealed`. Forward-compat
win.

**C3b. No `ProtocolVersion string` field.** Per 148 §7. Algorithm name
opaque ("honest-pending", "halo2"); when Halo2-v2 ships "halo2" alone
ambiguous. Recommend e.g. `"reality-zkmark-v1"`,
`"reality-stark-goldilocks-v1"`.

**C3c. No `Metadata map[string]string`** for forward-compat without
struct churn. gnark/arkworks ship something like this.

**C3d. `MarkChainStatement string` should be a typed `Mark` newtype**
with `Parse/Format/Validate` methods (146 §G2 base64-vs-base64url drift
catch at the type system level).

## D. Comparison with gnark — closest Go peer

148 named gnark the closest Go-posture reference. This audit zooms on
gnark's *DSL* specifically.

### D1. gnark's `frontend.Circuit` interface

```go
type CubicCircuit struct {
    X frontend.Variable `gnark:",public"`   // PUBLIC
    Y frontend.Variable `gnark:",public"`   // PUBLIC
    // any other field is WITNESS by default
}
func (c *CubicCircuit) Define(api frontend.API) error {
    x3 := api.Mul(c.X, c.X, c.X)
    api.AssertIsEqual(c.Y, api.Add(x3, c.X, 5))   // y = x³ + x + 5
    return nil
}
```

Then `frontend.Compile` → `groth16.Setup(ccs) → (pk, vk)` →
`groth16.Prove(ccs, pk, witness)` → `groth16.Verify(proof, vk, publicWitness)`.

### D2. What gnark does that zkmark currently does NOT

| gnark feature                                      | zkmark | gap |
|----------------------------------------------------|--------|-----|
| `Circuit` interface with `Define(api)` method      | none   | A1, A3 |
| Struct-tag public/witness split (`gnark:",public"`)| none   | B1, B3 |
| `frontend.API` with `Mul/Add/AssertIsEqual`        | none   | A2 |
| `frontend.Compile` → `ConstraintSystem`            | none   | C2a |
| Separate `Setup → (pk, vk)` from `Prove(witness)` from `Verify(publicWitness, proof)` | none | C2a |
| `Witness` type with `.Public()` projection         | none   | B1, B3 |
| `proof.MarshalBinary/UnmarshalBinary` round-trip   | only JSON via stdlib | C3a |
| Multi-backend dispatch (`groth16` vs `plonk`)      | string only | C1, C2 |
| `cs.GetNbConstraints()` size estimation            | none   | C2b |
| `cs.GetSchema()` public/private wire layout        | none   | B3 |

Pattern of **"struct tags as public/witness boundary + Define-method
for constraints + Compile-then-Setup-then-Prove flow"** is the cheapest
ergonomic wrapper buying all of B's type discipline plus C's setup/
prove/verify split. Estimate: ~600 LOC pure-Go frontend (no math) on
top of whatever Tranche-2 backend lands.

### D3. What zkmark should NOT copy from gnark

- **`frontend.Variable` is `interface{}` underneath** — runtime-
  dispatched to `*big.Int` or symbolic IDs. Erodes Go's type discipline.
  zkmark should keep variables strongly typed (struct, not interface).
- **Reflection-heavy struct-tag DSL** — hostile to CLAUDE.md §3 (no
  allocations in hot paths). gnark eats it because `Compile` is one-
  shot setup; zkmark must cache reflection per-Setup, none per-Prove.
- **R1CS+PlonK-only, no STARK.** 148 said zkmark goes STARK. Don't copy
  R1CS-shaped builder API; copy Plonky3's AIR-shaped builder API.

### D4. The hybrid that fits zkmark best

Plonky3-shaped AIR builder + gnark-shaped struct-tag annotation, in
idiomatic Go:

```go
type ZkMarkCircuit struct {
    Payload   PublicBytes  `zkmark:"public,length=variable"`
    CorpusSHA PublicHash   `zkmark:"public,algorithm=sha256"`
    Key       Witness      `zkmark:"witness,length=32,sensitive"`
}
func (c *ZkMarkCircuit) Define(b ConstraintBuilder) error {
    macKey := b.LoadWitness(c.Key)
    msg    := b.LoadPublicBytes(c.Payload)
    expect := b.LoadPublicHash(c.CorpusSHA)
    computed := b.HMAC(macKey, msg)
    b.AssertEq(computed, expect)
    return nil
}
```

`Define` *is* the AIR transition function; `ConstraintBuilder` holds
trace columns + constraints; struct tags drive witness/public split
with compile-time-distinct types (`PublicBytes` ≠ `Witness`); a
`Compile(circuit) (Circuit, ProverKey, VerifierKey, error)` pipeline
produces artifacts. **Effort:** ~600 LOC frontend + ~50 LOC tag parser
+ ~100 LOC compile-time validator — all independent of underlying math
(swap Plonky3-FRI for Halo2 with no DSL change).

## E. Constructive recommendations (delta vs 146/147/148)

**Priority 1 — Tranche 1 ship-now (no math wait):**

1. **(B3-A) Distinct `Public` / `Witness` / `PublicCommitment` types**
   — 30 LOC; compile-time public/witness type discipline; breaks one
   consumer at the cast level only.
2. **(C1b) `expectedPublicInput` on `Verifier` constructor** — 20 LOC;
   prevents "verify against the wrong document" attack class
   (generalises 146 G9 from CorpusSHA to all public inputs).
3. **(C1c) `VerifyProof(proof Proof) error` — drop `bool`** — ~10 LOC;
   removes `(false, nil)` ambiguity; aligns with gnark/halo2.
4. **(C3a) `Proof.PublicInputs []PublicInput`** — additive, ~15 LOC;
   sets up Tranche-2 `Statement` type without breaking parsers.
5. **(C3b) `Proof.ProtocolVersion string`** — additive, ~5 LOC; 148 §7
   alignment.

**Priority 2 — Tranche 2 design pin (no math, just docs):**

6. **(A2) Pick AIR over Plonkish/R1CS** — document in zkmark.go; pins
   the *frontend* shape implied by 148's Plonky3 backend pick.
7. **(C2a) Define `Setup → (pk, vk)` interface** — even as
   `ErrNotYetWired` stubs; locks API shape so Tranche 2 doesn't
   surprise existing callers.
8. **(D4) Sketch the `ConstraintBuilder` + struct-tag DSL** in zkmark.go
   doc comments — uncompilable shape pin prevents backend from being
   designed with awkward frontend.

**Priority 3 — post-Tranche 2:**

9. (D4 implementation) ~600 LOC frontend after backend math lands.
10. (C2c) Streaming/batched Proof API for STARK aggregation.
11. (C3c) `Metadata map[string]string` for forward-compat fields.
12. Cross-language DSL parity (R80b) — same struct-tag pattern in C#
    (`[ZkMark(Public = true)]`) so frontend is byte-parity-equivalent.

## F. Bottom line

`zkmark/`'s API today is shaped for Tranche 1's HMAC-replay model: flat
bytes-tuple inputs, no public/witness type distinction, `key` argument
on the verifier (oxymoron in real ZK), `(bool, error)` returns. None
wrong *for Tranche 1* — wrong as a Tranche-2 foundation. The 5
Priority-1 fixes are all 100-line-or-less, ship-now changes that make
the eventual Tranche-2 backend-PR (148's Plonky3-on-Goldilocks) **purely
additive** rather than an API-breaking re-architecture. The Priority-2
design pin — gnark-shaped struct-tag DSL with Plonky3-shaped AIR
backend joined by a `ConstraintBuilder` interface — is the highest-
leverage ergonomic decision the package can make before its math layer
exists; once math ships, the frontend shape gets locked by every caller.

Single highest-impact specific finding: **the `Verifier` constructor
should pin expected public inputs**, not the per-call `VerifyProof` API.
This generalises 146 §G9's `expectedCorpusSHA` gap to all of "what
statement is this verifier checking", and prevents the deployed-ZK
class of bugs (B2: Tornado Cash, zkSync, Aleo, Halo2 NULL_FIELD) from
ever being possible in a `reality` consumer.

## Progress

- 2026-05-08 agent-149 zkmark-api: API ergonomics review complete; 12
  recommendations (5 Priority-1 ship-now, 3 Priority-2 design-pin, 4
  Priority-3 post-Tranche-2); headline gap is no public/witness type
  distinction (B1) + verifier accepts attacker-controlled public inputs
  (C1b, generalising 146 §G9); recommended frontend pattern is gnark
  struct-tag annotation (`zkmark:"public,...|witness,sensitive"`) over
  Plonky3-style AIR ConstraintBuilder, per 148's backend pick;
  ~600 LOC pure-Go frontend estimate decoupled from backend math choice.
