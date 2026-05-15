# 146 | zkmark-numerics | field arithmetic, Fiat-Shamir transcript discipline

## Two-line summary

The `zkmark` package is a Tranche-1 *interface stub*: no field arithmetic, no
Fiat-Shamir transcript, no polynomial commitment, no hash function — the
audit topics in the brief literally do not exist in the shipped code yet.
The honesty-gate is well-engineered (the stub *refuses* to lie via
`ErrNotYetWired`), but several latent forward-compatibility traps need to be
addressed before Tranche 2 (Halo2) lands so the regulator-facing envelope is
not silently widened in unsafe ways.

## Scope of audit

Path: `C:/limitless/foundation/reality/zkmark/`
Files:
- `C:/limitless/foundation/reality/zkmark/zkmark.go`           (262 lines)
- `C:/limitless/foundation/reality/zkmark/zkmark_test.go`      (212 lines)
- `C:/limitless/foundation/reality/zkmark/README.md`           (75 lines)

Topology:
- `Prover` / `Verifier` interfaces
- `HonestProver` (Tranche 1) — wraps a caller-injected `SignerFunc`
- `Halo2Prover` (Tranche 2 stub) — `Prove` returns `ErrNotYetWired`
- `MarkVerifier` (Tranche 1) — delegates to caller-injected `VerifierFunc`
- `Proof` struct: `MarkChainStatement, Algorithm, ProofPending, ProofBytes,
  CorpusSHA`

The package has **zero cryptographic content of its own**. The mirror-mark
digest (HMAC-style mark) and any future Halo2 circuit live in *other*
repositories (`nexus/internal/mirrormark`, plus a future Rust sidecar).
This is intentional ("substrate-only ship", README §"Substrate-only ship").

## Topic-by-topic findings

### A. Field arithmetic / big-integer modular reduction
**STATUS: not present.** The package contains no field elements, no big
integers, no `math/big`, no carry-propagating arithmetic. The brief's
"carry handling in big-integer modular reduction" topic is **vacuously
clean** because no such code exists. Tranche 2 (Halo2) will introduce field
arithmetic, but it will live in the Rust sidecar; per zkmark's stated zero-
dep policy, the Go side should **not** re-implement it. (Cross-language
golden-file vectors would be the right discipline if it ever migrates here
— see §G recommendations.)

### B. Constant-time discipline (per agents 056/058 themes)
**STATUS: secret-handling exists, no timing audit performed in code.**
`HonestProver.Prove` and `MarkVerifier.VerifyProof` accept a `key []byte`
and pass it to a caller-injected function. The package itself never
inspects, compares, or branches on `key` bytes — so leak surface is in the
**injected** Sign/Verify, not in zkmark.go.

Sub-finding B1 (latent): `MarkVerifier.VerifyProof` uses a Go `switch
proof.Algorithm` (line 251) with `case ""` and `default`. This is constant
in *which case fires* relative to the byte content of `proof.Algorithm`,
but Go's string-equality `==` is **not** constant-time — an attacker who
controls `proof.Algorithm` can in principle distinguish algorithm names by
timing. Severity: **negligible**, because the algorithm name is
non-secret and short-circuit on early-mismatch is the correct semantic; but
worth pinning in a doc comment so 056-style reviewers don't re-flag it.

Sub-finding B2: there is no `subtle.ConstantTimeCompare` in zkmark.go.
This is correct for now (no MAC / digest comparison happens in zkmark
itself; that lives in the injected `VerifierFunc`), but **must be enforced
in the Tranche 2 Halo2 verifier** when proof-byte equality / commitment
equality lands inside this package. Recommendation: add a comment near
`AlgorithmHalo2` reserving a `subtle.ConstantTimeCompare` requirement on
proof-byte equality at the Tranche 2 boundary.

### C. Fiat-Shamir transcript discipline
**STATUS: not present.** No transcript object, no `absorb`, no
`squeeze_challenge`, no domain-separation tag, no Merlin-style label
ordering. There is no Halo2 circuit yet, so the canonical FS hazards
(transcript ordering, label collision, missing field-element
absorption, challenge re-use) **cannot manifest yet**.

Forward-design recommendation (Tranche 2): the package must, before
landing real proofs:
1. Define a versioned transcript-domain-separation tag (e.g.
   `b"zkmark-halo2-v1"`).
2. Pin the exact ordering of `(public_inputs, commitments, challenges)`
   absorbed into the transcript and lock it via golden-file vectors,
   exactly as the rest of `reality` does for `prob` / `signal` / etc.
3. Reject empty-transcript proofs (a common foot-gun: an attacker submits
   an empty Halo2 transcript and the verifier accepts because every
   challenge derives from an empty hash that the prover can pre-image).

### D. Polynomial commitments (KZG / FRI / IPA)
**STATUS: not present.** README §"Wire status" pegs Halo2 (which uses
IPA) as Tranche 2 / pending. Note: Halo2 in its upstream form uses
**Inner Product Argument (IPA)** over the Pasta curves, not KZG —
README/comments should make this explicit so callers don't expect a
trusted setup. Halo2 is *transparent*; KZG is *not*. Misnaming this
across the C# port (R80b parity) is a real risk if not pinned now.

### E. Hash function selection (BLAKE / Poseidon / Merlin)
**STATUS: not selected.** Tranche 2 Halo2 will need a hash for
Fiat-Shamir; upstream Halo2 uses Blake2b for the transcript and
Poseidon as the in-circuit hash. The substrate must commit to one
**before** the C# port and Rust sidecar are implemented or the byte-
parity (R80b) goal cannot be reached. Recommend pinning the choices in
zkmark.go as constants with documented rationale, e.g.:

    const TranscriptHash = "blake2b-512"   // out-of-circuit Fiat-Shamir
    const InCircuitHash  = "poseidon-rate2-arity3"  // for membership

…well before Tranche 2 prover code lands.

### F. Soundness / collision-resistance / random-oracle model
**STATUS: not yet a meaningful question.** Today, zkmark's "soundness"
is *exactly* the soundness of the L43 cold-verifier mirror-mark digest
(an HMAC over `(payload, corpusSHA, key)` per the README and the fake-
verify shape in the test file, line 11-23). The zkmark layer adds **no
soundness gain** in Tranche 1 — and the README is honest about that
("structural ZK in Tranche 1, not cryptographic", line 152-153).

ROM dependency in Tranche 2: a Halo2 IPA proof reduces to ROM via
Fiat-Shamir on the transcript hash. This places the *eventual* trust
assumption on whichever hash is chosen in §E. Tranche 1 has no
ROM dependency.

### G. Other numerics-adjacent issues found

**G1. `Proof.ProofPending` zero-value trap (HIGH-severity forward-compat).**
zkmark.go lines 99-103 document, and the test file lines 198-211 pin,
that `Proof{}` zero-value has `ProofPending = false`, which is the
**unsafe** default ("trust ProofBytes"). The doc warns callers to
construct explicitly, but Go has no compile-time enforcement.
Recommendation: invert the field — call it `ProofTrusted` and default
to `false` (zero-value = "do not trust"), or wrap the struct in a
factory `NewProof(...)` that always sets `ProofPending = true` unless
overridden. The current shape is a footgun: `Proof{Algorithm:
AlgorithmHalo2, ProofBytes: attackerBytes}` constructed by hand by a
careless caller would be parsed as `ProofPending=false` → "real proof"
even before Tranche 2 lands. The `MarkVerifier.VerifyProof` function
**does** intercept `AlgorithmHalo2` with `ErrNotYetWired` (line 254),
but only the `MarkVerifier`. A future `Halo2Verifier` that
short-circuits on `ProofPending=false` would inherit the trap.

**G2. `MarkChainStatement` is a `string` of unspecified encoding
(MEDIUM).** Line 84-88 documents it as `"lore@v1:<base64url-of-corpus-
prefix-and-hmac>"` but does not enforce. A C# port (R80b) that emits
`base64` (with `+/=` padding) instead of `base64url` (with `-_`,
unpadded) would silently break byte parity. Recommend pinning the
encoding name in a constant, e.g. `MarkChainEncoding = "base64url-
nopad"`, and adding a parser that rejects malformed marks.

**G3. `CorpusSHA [32]byte` is hash-agnostic (LOW).** The field is
typed as 32 bytes but never names the hash algorithm. SHA-256? BLAKE2s-
256? BLAKE3-256? For collision-resistance arguments to mean anything,
the hash must be pinned. Recommend a `CorpusHashAlgorithm` constant
of value `"sha256"` plus a doc comment naming it on the field.

**G4. Algorithm whitelist closed-set vs open-set (LOW).** Line 251
switches over `Algorithm` and rejects unknown values (line 258-259).
This is **good** — it's the safer of the two design choices, because
an attacker who could inject a future-but-unimplemented algorithm
name cannot bypass verification. But the README's "Forward-compatible:
new algorithms add via constants without breaking existing parsers"
(README line 56) is mildly misleading: existing parsers built on this
package **will** reject the new algorithm until they're recompiled
against a newer zkmark.go. That's a feature, not a bug, but the README
sells it as transparent. Recommend a small doc fix.

**G5. `Halo2Prover.Prove` discards inputs without validation (LOW).**
Line 213 takes `payload, corpusSHA, key` and immediately returns
`ErrNotYetWired`. It does not check `len(key) > 0`, `len(payload) > 0`,
or `corpusSHA != [32]byte{}`. For honesty-gate parity, recommend a
log-and-error path that surfaces *why* a downstream caller's
inputs would have been rejected anyway, so callers can't pass garbage
in development and discover the constraints only in Tranche 2.
Marginal severity in Tranche 1 (function returns error before any
side effects), but a useful discipline-pin for Tranche 2 prep.

**G6. No length-bound on `ProofBytes` (MEDIUM, Tranche 2 latent).**
Field type is `[]byte`. A malicious `Proof` with a multi-GB
`ProofBytes` slice could DoS the future Halo2 verifier. Recommend a
documented `MaxProofBytes` constant (e.g. 64 KiB for a typical Halo2
IPA proof) and reject in `Halo2Verifier.VerifyProof` at the head of
that function once it lands.

**G7. JSON ordering pin is in the README only (LOW).** README line 63
asserts JSON field ordering `algorithm, corpusSha, markChainStatement,
proofBytes, proofPending`. Go's `encoding/json` *does* emit struct
fields in declaration order by default, but the **declaration order in
zkmark.go is `MarkChainStatement, Algorithm, ProofPending, ProofBytes,
CorpusSHA`** (lines 84-115) — i.e. **inconsistent** with the README's
declared canonical JSON order. Either:
- Reorder the struct fields to match `algorithm, corpusSha,
  markChainStatement, proofBytes, proofPending`; or
- Add a `MarshalJSON` that produces the declared order; or
- Update the README to admit the actual field order is different and
  pin it explicitly.
Whichever is chosen, lock it with a golden-file test (per `reality`'s
core discipline). This is the single most actionable parity bug in the
package today.

**G8. No `String()` method on `Proof` (negligible).** Easy
human-debug improvement; ensure it does **not** dump `ProofBytes`
verbatim (could leak commitments in logs). Lower priority than G1/G7.

**G9. `MarkVerifier.VerifyProof` does not check `proof.CorpusSHA`
matches the corpus the verifier was instantiated against (MEDIUM).**
Line 253 passes `proof.CorpusSHA` straight into the injected
`verify(...)`. A correct implementation must compare `proof.CorpusSHA`
to the *expected* corpus SHA (which the verifier doesn't have access
to in the current API surface — there's no `expectedCorpusSHA`
parameter on `VerifyProof`). This means a caller can be fooled into
verifying a proof against the *wrong* corpus, so long as the
mirror-mark itself was computed honestly over that wrong corpus.
Recommend either:
- Adding `expectedCorpusSHA [32]byte` to `Verifier.VerifyProof`, **or**
- Wrapping `MarkVerifier` in a higher-level type that pins the
  expected corpus at construction time.
This is the most consequential semantic gap I found — it's a "verify
against the wrong document" attack waiting to happen if any caller
trusts `proof.CorpusSHA` blindly.

**G10. `errors.New("…")` vs sentinel error pattern (style, LOW).**
Lines 176, 249, 257 use ad-hoc `errors.New`. Lines 40, 259 use
sentinels (`ErrNotYetWired`) and wrapped fmt errors. Recommend
defining `ErrNilSigner`, `ErrNilVerifier`, `ErrEmptyAlgorithm`,
`ErrUnknownAlgorithm` so callers can branch on `errors.Is` —
particularly important for the C# port to mirror exception types.

**G11. No fuzz tests, no property-based tests.** Per `reality`'s
discipline (1,965 tests, golden-file infra), the zkmark package's
test count is tiny (10 tests in zkmark_test.go). For a package
intended to interface with regulator-facing audit trails, the bar
should be higher even in Tranche 1. Recommend at minimum:
- Property test: `Prove ∘ Verify` round-trip over arbitrary
  `(payload, corpusSHA, key)` byte triples
- Property test: any single-bit flip of `payload` or `key` or
  `corpusSHA` causes verify to fail
- JSON round-trip golden vector pinning the field order from G7
- Cross-language golden vectors (Go, C#) once the C# port lands

**G12. Test helper `min(a,b int) int` (line 26-31) is shadowed by Go
1.21+ builtin (style, LOW).** Harmless but produces a `go vet` /
staticcheck warning on modern toolchains. Either delete it (use
builtin) or rename it.

**G13. `fakeSign` test stub uses `string(corpusSHA[:4])` and
`string(payload[:min(8, len(payload))])` (line 15) — UTF-8 hazard
(LOW).** Random `byte` slices are not valid UTF-8 in general, so the
resulting Go strings will contain replacement runes if ever printed.
Tests pass because comparison is `==` over the raw bytes, but this
will trap any future JSON-marshalling test. Recommend
`hex.EncodeToString` or `base64url` in the test stub.

## Constructive recommendations summary

Priority 1 (before Tranche 2 — prevent regulator-facing footguns):
1. **G7**: align struct field declaration order with README JSON canon,
   or add `MarshalJSON`, and lock with golden-file vector. This is the
   only finding that *currently* breaks a stated parity contract.
2. **G9**: add `expectedCorpusSHA` to verifier API or factor a corpus-
   pinning wrapper.
3. **G1**: invert `ProofPending` to `ProofTrusted` (default-deny) **or**
   add a `NewProof(...)` factory.
4. **D / E**: pin Halo2 = IPA + Blake2b transcript + Poseidon in-circuit
   as documented constants now, before C# port lands.

Priority 2 (improves auditability):
5. **G2**: pin `MarkChainEncoding = "base64url-nopad"` constant and
   parser.
6. **G3**: pin `CorpusHashAlgorithm = "sha256"` constant.
7. **G6**: pin `MaxProofBytes` constant (e.g. 64 KiB).
8. **G10**: replace `errors.New` with named sentinels for cross-port
   `errors.Is` discipline.
9. **G11**: add fuzz / round-trip / single-bit-flip property tests.

Priority 3 (style / hygiene):
10. **B1**, **B2**: reserve constant-time-comparison contract in the
    Halo2-verifier doc.
11. **G4**: README copy-edit.
12. **G5**: stub-input validation in `Halo2Prover.Prove`.
13. **G8**: add `String()` method that does *not* dump `ProofBytes`.
14. **G12**, **G13**: clean up test helpers.

## Bottom line

The audit topics in the brief (field arithmetic, FS transcript, KZG/FRI/IPA,
hash selection, ROM soundness) **do not yet exist as code** in
`zkmark/`. The package is exactly what its README says it is: a
substrate-only interface stub with an explicit not-yet-wired sentinel.
The honesty-gate posture is excellent. The most consequential issues I
found are *not* in the stubbed-out crypto primitives — they're in the
shipped envelope: the `ProofPending` zero-value trap (G1), the
struct-vs-README JSON ordering inconsistency (G7), and the missing
`expectedCorpusSHA` check at the verifier boundary (G9). All three are
quick fixes. None of them require waiting for Tranche 2.

## Progress

- 2026-05-08 agent-146 zkmark-numerics: audit complete, 13 findings
  (G1-G13), priority-1 items: G1 ProofPending zero-value trap, G7
  struct-vs-README JSON order mismatch, G9 missing expectedCorpusSHA
  check; field-arithmetic / Fiat-Shamir / commitment topics vacuous
  (Tranche 2 not yet implemented).
