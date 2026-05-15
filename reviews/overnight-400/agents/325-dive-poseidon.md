# 325 — dive-poseidon (Poseidon / Rescue / Griffin / Anemoi / MiMC algebraic-hash audit)

## Headline
Reality has zero algebraic (ZK-friendly) hashes; Poseidon-BN254 + a generic Sponge is the
~330-LOC day-1 PR that unblocks zkmark Fiat-Shamir, Plonk transcripts, and in-SNARK Merkle.

## Findings

### Surface check (zero algebraic-hash code in repo)

- `Grep` for `Poseidon|Rescue|Griffin|Anemoi|Vision|MiMC|Sponge|AlgebraicHash` across
  `*.go` returns only `audio/spectrogram/{doc,stft}.go` (string "spectrogram" matched).
  No algebraic-hash code anywhere. Confirms slot 322's general "zero hashes" finding
  for the *cryptographic* axis as well.
- `crypto/hash.go` (223 LOC) is non-cryptographic only: `FNV1a32`, `FNV1a64`,
  `MurmurHash3_32`, `ConsistentHash`, `SituationHashWithStructure`,
  `StructuralDescriptor`. Useful for hash-tables, useless for ZK.
- `crypto/{prime.go, modular.go, rng.go, crypto_test.go}` — number theory,
  Miller-Rabin, modexp, PRNGs. No prime-field-specific hash plumbing.
- `zkmark/zkmark.go` (261 LOC): `Grep` for `hash|fiatshamir|transcript|sponge` returns
  no matches. zkmark is currently a structural skeleton with no Fiat-Shamir layer at all.

Conclusion: the entire algebraic-hash family is greenfield. No existing API to
preserve, no migration cost.

### Why this matters specifically (vs slot 322's SHA-256 push)

SHA-256 inside an R1CS circuit is ~25k constraints per 512-bit block (per
[Ben-Sasson et al.] / circomlib reference). The dominant cost in any SNARK that
uses a Merkle tree (i.e. STARKs, FRI commitments, Plonk lookup tables, every
zk-rollup) is the in-circuit hash. Poseidon-BN254 with `t=3, R_F=8, R_P=57`:
~150 R1CS constraints per 2-to-1 compression — **~165× cheaper** than SHA-256
in-circuit. That is the reason every production ZK system since 2020 uses an
algebraic hash for Merkle and Fiat-Shamir, not SHA-256.

So the day-1 SHA-256 PR (slot 322) covers commit-to-public-data and external
interop. It does NOT cover in-circuit usage. zkmark needs both. They are
disjoint primitives.

### Primitive landscape (ranked by deployment maturity)

| Primitive | Year | Spec stable? | Production deployments | Notes |
|---|---|---|---|---|
| **MiMC / MiMC-Feistel** | 2016 | yes | Zcash Sapling pseudo, Iden3 legacy | x^3 (or x^5) round, ~73 rounds. Simple but slow per-bit. Largely superseded. |
| **Poseidon (1)** | 2019 | yes | StarkWare, Polygon zkEVM, Filecoin, Iden3, Aztec, dYdX, Loopring | Production ZK standard. Partial+full S-box. MDS matrix. |
| **Poseidon2** | 2023 | yes | Plonky3, Risc0, Horizen | Cheaper plain (non-circuit) eval, cleaner mixing matrix. Drop-in for new deployments. |
| **Rescue / Rescue-Prime** | 2020 | yes | StarkNet (parts), Aleo (Rescue-Prime Optimized) | Symmetric S-box (x^α and x^(1/α)). Slower native, simplest constraint. |
| **Griffin** | 2023 | yes | Research; some Plonk2/3 experiments | Beats Poseidon on some metrics; less battle-tested. |
| **Anemoi** | 2023 | yes | Research; some experimental rollups | Flystel S-box; very competitive constraint counts. |
| **Vision** | 2020 | yes | Niche (binary fields too) | Companion to Rescue; rarely deployed alone. |
| **Reinforced Concrete** | 2022 | yes | Research; aimed at hybrid use | Fast outside circuit (lookup-table), still cheap inside. Niche. |

Unambiguous current production standard: **Poseidon over the SNARK scalar
field** (BN254 or BLS12-381). Poseidon2 is the forward-looking choice for new
systems but the 2019 Poseidon parameter sets dominate existing test-vector
corpora.

### Field dependency (gating constraint)

These hashes are **not field-agnostic**. Each parameter set targets one prime
modulus:
- **BN254 scalar** (Ethereum/zkEVM/Polygon): `r ≈ 2.188e75`, x^5 invertible.
- **BLS12-381 scalar** (Filecoin/Zcash Halo2/Aleo): `r ≈ 5.244e76`, x^5 invertible.
- **Goldilocks** (Plonky2/Risc0): `2^64 - 2^32 + 1`, x^7 S-box.
- **Mersenne-31 / BabyBear** (Plonky3): 31-bit fields, x^3 or x^5 S-box.
- **Pallas/Vesta** (Halo2): different again.

Round counts (`R_F`, `R_P`) and round constants are **derived** from the field
and security parameter via the Hades grinding script (`generate_params_poseidon.sage`)
— not freely chosen. This is the load-bearing detail for cross-validation.

### Dependency on slot 292 (BN254 scalar field)

A correct Poseidon-BN254 needs:
1. `Fr` arithmetic (add, mul, square, pow, inv) modulo BN254's scalar prime
   `21888242871839275222246405745257275088548364400416034343698204186575808495617`.
2. Constant-time / Montgomery form ideal for perf, but plain `big.Int` works
   for correctness-first day-1.

Slot 292 (`new-elliptic-curves`) and slot 291 (`new-modular-arithmetic`) note
zero ECC and zero residue-class-ring abstractions. **Order matters:** ship
`crypto/field/bn254` (Fr only — no curve points needed for hashing!) **before**
`crypto/poseidon/bn254`. The field package is small (~300 LOC for plain
`big.Int`-backed Fr; ~700 LOC for Montgomery). Hashing does not need the curve
group, just the scalar field.

### R-MUTUAL-CROSS-VALIDATION 3/3 opportunities (high-confidence pins)

Algebraic hashes have **the best test-vector landscape of any cryptographic
primitive**: every production ZK library publishes deterministic test vectors
because circuit/native parity is mission-critical.

| Pin | Source A | Source B | Source C |
|---|---|---|---|
| Poseidon-BN254 t=3 (2-to-1 hash) | circomlib `poseidon.js` | Filecoin `neptune` Rust | StarkWare `starkex-resources` |
| Poseidon-BN254 t=5 | iden3 `go-iden3-crypto` | Aztec `barretenberg` | Polygon Hermez ref |
| Poseidon-BLS12-381 t=3 | Filecoin `neptune` | Aleo `snarkVM` (legacy) | Zcash Halo2 gadgets |
| Poseidon-Goldilocks t=12 (RP-64) | Risc0 `zeth`/SDK | Plonky2 `poseidon.rs` | Polygon zkEVM Plonky2 fork |
| Rescue-Prime BLS12-381 | Aleo SDK | Toposware ref impl | original paper Sage script |
| MiMC-7 BN254 | Zcash Sapling | iden3 `circomlib/mimc7` | original paper |

Each row gives 3 independent reference implementations — exact byte-for-byte
parity of the field-element output is achievable and is the strongest possible
saturation pin (no tolerance: equality over `Fr`).

### Cheapest day-1 PR (concrete)

**~330 LOC total**, in tree:

```
crypto/
  field/
    bn254/
      fr.go              // ~300 LOC Fr add/sub/mul/square/pow/inv (plain big.Int)
      fr_test.go
  poseidon/
    sponge.go            // ~80 LOC generic sponge over a Permutation interface
    poseidon.go          // ~150 LOC permutation: full+partial rounds, MDS, ARK, S-box
    params/
      bn254_t3.go        // ~1 KB constants: round consts + MDS matrix
      bn254_t5.go        // optional
    poseidon_test.go     // golden vectors from circomlib
    testdata/
      poseidon_bn254_t3.json   // 30 vectors per CLAUDE.md rule
```

Exposed API (target):
```go
package poseidon

type Permutation interface {
    Permute(state []Fr)        // in-place, fixed width
    Width() int
    Capacity() int
}

func Hash(perm Permutation, in []Fr) Fr   // sponge with rate=Width-Capacity
func HashTwo(perm Permutation, a, b Fr) Fr  // 2-to-1 Merkle compression
```

That single PR enables zkmark Fiat-Shamir, Merkle compression, and is the
substrate for every other ZK-rollup-flavored primitive Reality might add.

### Parameter table size note

Poseidon-BN254 t=3 needs:
- `(R_F + R_P) × t = (8 + 57) × 3 = 195` round constants (each a 32-byte Fr).
- `t × t = 9` MDS matrix entries.
- Total: ~6.5 KB of constants.

Poseidon-BN254 t=5: ~24 KB. Each (curve, width) pair gets its own file.
Recommend `crypto/poseidon/params/<curve>_t<n>.go` with `//go:generate`
directive pointing at the official Sage script for reproducibility audit.
**Do not** bake the script into the repo — it's a HadesMiMC grinder that
needs SageMath. Document the seed and the commit of the upstream script.

### Cross-link consumers

| Consumer | Why it needs algebraic hash |
|---|---|
| **zkmark** (slots 146-150, 175, 200) | Fiat-Shamir transcript — every non-interactive ZK proof. Without this, zkmark is unsoundly random-oracle-modeled. |
| **In-SNARK Merkle** | Inclusion proofs for state commitments. Dominant cost in zkVMs. |
| **Plonk lookup tables** | Plookup / cq lookup arguments use a hash for permutation arguments. |
| **STARK FRI** | Merkle commitment of evaluation domain. Goldilocks Poseidon is canonical here. |
| **VRF construction** | Some VRFs (Algorand-style) use algebraic hashes for in-circuit verification. |
| **Commit-reveal schemes** in slot 196 (color-info synergy?) | unrelated, ignore. |

The chain is: **Fr arithmetic (slot 291/292) → Poseidon permutation → Sponge →
zkmark Fiat-Shamir + Merkle**. Without the bottom of that stack, zkmark cannot
ship a sound proof system.

### Pitfalls / footguns the day-1 PR must avoid

1. **Round-constant byte order.** circomlib uses little-endian Montgomery
   internally but JSON test vectors are decimal strings. Pick decimal-string
   golden files; document endianness explicitly.
2. **MDS matrix sourcing.** There are at least 3 published MDS matrices for
   BN254 t=3 in the wild (Filecoin's, circomlib's, original paper's).
   Filecoin and circomlib **disagree**. Pick circomlib's (it has the largest
   ecosystem) and document.
3. **Sponge padding.** `pad10*1` vs zero-pad to capacity vs domain-tag-prefix:
   each library disagrees. Match circomlib's "absorb-then-squeeze, no
   padding for fixed-arity" convention for 2-to-1 compression and document a
   separate `HashVarLen` for variable-length.
4. **R_P optimization.** The 2019 paper's R_P had a security bug (Beyne
   et al. 2020); use the **patched** values (R_P=57 for BN254 t=3, not 56 or
   60). Several open-source libs still ship the unsafe value. circomlib is
   fixed; Filecoin neptune is fixed; verify before copying.
5. **Domain separation.** Production deployments tag the capacity element
   with `(arity, length)` to make different uses non-colliding. Iden3 uses
   `cap = 0`; Filecoin uses a domain tag. Pick one (recommend domain-tag like
   Filecoin) and document — it's load-bearing for security.
6. **Goldilocks needs a different Fr** (slot 292 won't help). Defer Plonky3
   support to a separate PR after BN254 lands.

## Concrete recommendations

1. **Block on slot 291/292 Fr arithmetic for BN254** before opening
   `crypto/poseidon`. Hashing without a correct field is meaningless.
2. **Day-1 PR (~330 LOC + golden files):** generic `Sponge` over a
   `Permutation` interface + Poseidon-BN254 t=3 permutation + circomlib
   golden vectors. Produces an immediately-useful 2-to-1 hash for in-circuit
   Merkle and a sponge for variable-length input.
3. **Day-2 PR:** Poseidon-BN254 t=5 (5-to-1 absorb is common in Merkle of
   higher arity). Same permutation core, new parameter table.
4. **Day-3 PR:** Poseidon-BLS12-381 t=3 (depends on a separate `Fr`
   package for that curve — slot 292 may stage this).
5. **Day-4 PR:** Poseidon2 with the "external/internal" matrix split.
   Strictly additive; new parameter sets.
6. **Day-5 PR:** Rescue-Prime BN254. Symmetric S-box gives ~2× cheaper
   constraints in some Plonk arithmetizations; native eval is slower. Worth
   shipping as a benchmark companion.
7. **Defer indefinitely:** Griffin, Anemoi, Vision, Reinforced Concrete.
   Frontier; ship only when a Reality consumer asks for one specifically.
   Document the names in `crypto/poseidon/doc.go` so users know we know about
   them.
8. **Defer to coordinated PR:** MiMC. Legacy compat only; ~100 LOC; ship
   alongside any Zcash Sapling-compat work, not before.
9. **Parameter generation:** Do **not** auto-generate the parameter tables.
   Copy circomlib's exact table, run a one-shot test that re-derives them
   via the upstream Sage script, then commit the table as a static `.go`
   file. The Sage script depends on SageMath — keep it out of CI.
10. **Cross-validation strategy:** for each (curve, width) tuple, encode a
    JSON test-vector file with at least 30 inputs (per CLAUDE.md rule),
    sourced from circomlib + neptune + starkex. Tolerance is `==` over Fr —
    no float tolerance. This is the cleanest R-MUTUAL-CROSS-VALIDATION 3/3
    in the entire repo.
11. **Sponge construction lives in `crypto/sponge/`** (not nested under
    poseidon), so the same sponge can later wrap Rescue, Griffin, Keccak-f,
    etc. Make `Permutation` the only required interface.
12. **Document constraint counts** in `crypto/poseidon/doc.go`: ~150 R1CS for
    Poseidon-BN254 t=3 2-to-1 hash, vs ~25k for SHA-256, vs ~1500 for
    Pedersen. Reality is a math library, but ZK consumers will compare these
    numbers and that comparison is the entire reason the package exists.
13. **No-allocation hot path:** sponge takes a caller-provided state slice
    of length `t`; permutation operates in place; never escape to heap.
    Per CLAUDE.md rule 3 ("no allocations in hot paths").
14. **Tag domains explicitly:** export `DomainTag` enum or constant per
    use case (Merkle leaf, Merkle internal, Fiat-Shamir transcript). Iden3
    and Filecoin both ship these as named constants — do the same.

## Sources

### Repo files inspected
- `C:\limitless\foundation\reality\crypto\hash.go` — non-crypto only (FNV/Murmur).
- `C:\limitless\foundation\reality\crypto\{prime.go, modular.go, rng.go}` — number theory, no hash.
- `C:\limitless\foundation\reality\zkmark\zkmark.go` — no Fiat-Shamir / no transcript / no hash.
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\324-dive-msm.md` — peer (multi-scalar mul).
- `C:\limitless\foundation\reality\reviews\overnight-400\MASTER_PLAN.md:325` — assignment line.

### Reference papers (specs)
- Albrecht, Grassi, Rechberger, Roy, Tiessen 2016. *MiMC: Efficient Encryption
  and Cryptographic Hashing with Minimal Multiplicative Complexity.* ASIACRYPT 2016.
- Grassi, Khovratovich, Rechberger, Roy, Schofnegger 2019. *Poseidon: A New
  Hash Function for Zero-Knowledge Proof Systems.* ePrint 2019/458 (USENIX 2021).
- Aly, Ashur, Ben-Sasson, Dhooghe, Szepieniec 2020. *Design of Symmetric-Key
  Primitives for Advanced Cryptographic Protocols.* IACR ToSC 2020/3.
  (Rescue, Rescue-Prime, Vision.)
- Beyne, Canteaut, Dinur, Eichlseder, Leander, Leurent, Naya-Plasencia,
  Perrin, Sasaki, Todo, Wiemer 2020. *Out of Oddity — New Cryptanalytic
  Techniques against Symmetric Primitives Optimized for Integrity Proof
  Systems.* CRYPTO 2020. (R_P fix.)
- Grassi, Khovratovich, Rønjom, Schofnegger 2022. *Reinforced Concrete: A
  Fast Hash Function for Verifiable Computation.* CCS 2022.
- Grassi, Hao, Rechberger, Schofnegger, Walch 2023. *Horst Meets Fluid-SPN:
  Griffin for Zero-Knowledge Applications.* CRYPTO 2023.
- Bouvier, Briaud, Chaidos, Kiltz, Lyubashevsky, Roy, Schofnegger,
  Tiessen 2023. *New Design Techniques for Efficient Arithmetization-Oriented
  Hash Functions: Anemoi Permutations and Jive Compression Mode.* CRYPTO 2023.
- Grassi, Khovratovich, Schofnegger 2023. *Poseidon2: A Faster Version of
  the Poseidon Hash Function.* AFRICACRYPT 2023.

### Reference implementations (test-vector sources)
- iden3 `circomlibjs/src/poseidon.js` and `circomlib/circuits/poseidon.circom`.
- Filecoin `neptune` (Rust): `https://github.com/filecoin-project/neptune`.
- StarkWare `starkex-resources/crypto/starkware/crypto/signature/pedersen_params.json`
  (companion test vectors for StarkNet Poseidon variant).
- Polygon `plonky2/src/hash/poseidon.rs` (Goldilocks variant).
- Risc0 `risc0/zkp/src/core/hash/poseidon.rs` (RP-64).
- Aleo `snarkVM/algorithms/src/crypto_hash/poseidon.rs`.
- Aztec `barretenberg/cpp/src/barretenberg/crypto/poseidon2/`.

### Cross-references (other slots)
- 322 (slot just done) — pushed SHA-256 as cheapest day-1 hash PR. **This
  slot complements, does not replace.** SHA-256 covers external interop;
  Poseidon covers in-circuit.
- 291 (`new-modular-arithmetic`), 292 (`new-elliptic-curves`) — Fr
  arithmetic prerequisite. Cannot ship Poseidon without BN254 Fr.
- 200 (zkmark) — primary downstream consumer. Fiat-Shamir is mandatory.
- 175 (synergy-zkmark-crypto) — likely already noted this dependency.
- 214 (`new-pairings`) — separate ZK primitive (curve-side); Poseidon is
  scalar-side. Independent PR streams.
- 324 (`dive-msm`) — peer ZK primitive (multi-scalar multiplication);
  also gated on slot 292 Fr/G1 work.
