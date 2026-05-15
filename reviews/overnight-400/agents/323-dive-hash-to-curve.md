# 323 — dive-hash-to-curve (RFC 9380: expand_message / hash_to_field / map_to_curve / cofactor / BLS12-381)

## Headline
reality v0.10.0 has zero hash-to-curve surface (no EC, no SHA-2, no SWU, no Elligator); the cheapest day-1 PR is `expand_message_xmd` + `hash_to_field` (~160 LOC) atop slot 322's SHA-256, but the headline win is a joint slot 292 + 323 BLS12-381 G1 hash-to-curve composed PR (~720 LOC end-to-end, 8 RFC-9380 vectors bit-exact, R-MUTUAL-CROSS-VALIDATION 3/3).

## Findings

### F1. Surface confirmation: zero hash-to-curve, zero EC, zero SHA-*
- Grep across the entire repo for `HashToCurve|MapToCurve|ExpandMessage|HashToField|Elligator|SWU|ShallueVanDeWoestijne|WahbyBoneh|hash_to_curve|map_to_curve|expand_message` returns matches **only inside** `reviews/overnight-400/agents/*.md` — pure review noise, no code.
- `crypto/hash.go:25-216` ships only **non-cryptographic** hashes: FNV1a32/64, MurmurHash3, ConsistentHash, SituationHash. No SHA-2, no SHAKE, no HMAC.
- `crypto/{modular.go,prime.go,rng.go}` are number-theoretic only — **no elliptic-curve point type, no field-element type, no projective/Jacobian arithmetic**. Slot 292 (`new-elliptic-curves.md`) and slot 213 (`new-isogeny.md`) confirm same.
- "BLS12" / "381" appears **only** in agent review markdown — not a single Go file references it.
- Therefore RFC-9380 surface = ∅. Every RFC-9380 primitive is a greenfield write.

### F2. RFC-9380 (Faz-Hernandez, Scott, Sullivan, Wahby, Wood, IRTF, Aug 2023) — primitive ladder
RFC 9380 decomposes hash-to-curve into a **strict 4-stage pipeline**:

```
input msg ──► expand_message_{xmd|xof}  (uniform bytes,  step §5.3)
            │
            ▼
            hash_to_field(msg, count)  (count·m·L bytes → F_p^m field elts, §5.2)
            │  count=2 for hash_to_curve (random oracle), count=1 for encode_to_curve
            ▼
            map_to_curve(u_i)  (deterministic field→curve, §6 — pick one of: SSWU / SvdW / Elligator2 / WB19)
            │
            ▼
            P = map(u0) + map(u1)        (group-add, RO mode only)
            │
            ▼
            clear_cofactor(P)  (multiply by h_eff to land in prime-order subgroup, §7)
            │
            ▼
            curve point in G_q
```

This **decomposition is the invariant** — every RFC-9380 ciphersuite (P-256, P-384, P-521, Curve25519, Curve448, secp256k1, BLS12-381 G1, BLS12-381 G2) instantiates the same pipeline with different (hash, expand, map, cofactor) tuples.

### F3. expand_message_xmd is HKDF-Expand stripped down (~80 LOC over SHA-256)
- §5.3.1 of RFC-9380. Inputs: `msg`, `DST` (≤ 255 bytes), `len_in_bytes`. Outputs: pseudorandom bytes.
- Algorithm: `b_0 = H(Z_pad || msg || I2OSP(len_in_bytes,2) || I2OSP(0,1) || DST_prime)`, then `b_i = H(strxor(b_0, b_{i-1}) || I2OSP(i,1) || DST_prime)`, concatenate.
- **Pure SHA-2 invocation** — no field math, no curve math. Composes directly on slot 322 SHA-256. **Day-1 unblocked** the moment slot 322 lands. Estimated 60-80 Go LOC including the `Z_pad` zero-block, `DST_prime` length-prefix, and `len_in_bytes ≤ 255 · b_in_bytes` precondition check.
- `expand_message_xof` (SHAKE128/SHAKE256) is the alternative — defers to a Keccak/SHA-3 PR (slot 322 noted SHA-3 as "T2 ~120 LOC after SHA-2").

### F4. hash_to_field is reduce-to-field after expand (~80 LOC, mod-bigint arithmetic)
- §5.2. For each of `count` field elements, take `L = ceil((ceil(log2(p)) + k)/8)` bytes (k = security parameter, e.g. 128 for BLS12-381 → L=64), interpret as big-endian integer, reduce mod p. For F_{p^m} (twist field), `m` such reductions per element.
- For BLS12-381 G1: count=2, m=1, L=64 → 128 bytes total from expand_message.
- For BLS12-381 G2: count=2, m=2, L=64 → 256 bytes total.
- The k=128 over-extraction is what makes the reduction **statistically uniform mod p** — bias ≤ 2^{-k}.
- Composes on `expand_message_xmd` (F3) + a `BigInt → F_p` reducer. The reducer is a **prerequisite**: it requires slot 292's `Fp` field type. So F4 cannot land before T1 of slot 292.

### F5. map_to_curve has 4 industrial choices — pick by curve shape, not by taste
| Method | Year | Curves | Constant-time? | Notes |
|---|---|---|---|---|
| Try-and-increment | folklore | any | **NO** (variable) | Forbidden by RFC 9380 for live protocols |
| Icart | Icart 2009 | y² = x³+ax+b, p ≡ 2 (mod 3) | yes | Cube-root based, doesn't fit BLS12-381 (p ≡ 1 mod 3) |
| Shallue–van de Woestijne 2006 | SvdW06 | **all** Weierstrass over F_p | yes | Generic fallback. Used by NIST P-256 in some modes. |
| **Simplified SWU (SSWU)** | Brier-Coron-Icart-Madore-Randriam-Tibouchi 2010 | y² = x³+Ax+B, A·B ≠ 0, char ≠ 2,3 | yes | RFC-9380's default for NIST P-256/384/521, secp256k1 (via isogeny) |
| **Wahby-Boneh 2019 (WB19)** | TCHES 2019 | BLS12-381 G1 (A=0) **and** G2 | yes | SSWU-on-isogenous-curve + 11-isogeny eval. ~9% of non-CT speed. **The production standard.** |
| Elligator 2 | Bernstein-Hamburg-Krasnova-Lange CCS'13 | Montgomery (Curve25519, Curve448) | yes | For Edwards/Montgomery only. Requires non-trivial 2-torsion. |

Key constraint: **BLS12-381 G1 has A=0 in y² = x³ + 4** → vanilla SSWU does not apply (needs A·B ≠ 0). WB19's contribution is "map to a 11-isogenous curve E' with nonzero A', then evaluate the 11-isogeny back to E." Cost of the isogeny eval ≈ ⅓ of one F_p square root, dominated by the rest of SSWU.

### F6. clear_cofactor is the most curve-specific stage
- For BLS12-381 G1: cofactor h = 0x396c8c005555e1568c00aaab0000aaab. RFC 9380 §H.1 gives a fast `h_eff = (1 - z) · h` formula (z = -0xd201000000010000 the BLS parameter). Evaluated as a single scalar mult — ~50 LOC given a working scalar-mult primitive (slot 292 T2).
- For BLS12-381 G2: h_eff is a length-3 chain of Frobenius + scalar mults (Budroni-Pintore "Endomorphisms for faster cofactor clearing"). RFC 9380 §H.2. ~80 LOC.
- For Curve25519: h = 8, just multiply by 8 (~5 LOC).
- For NIST curves: h = 1 (prime order), clear_cofactor is identity.

### F7. BLS12-381 G2 hash is roughly 2× the work of G1 (twist field F_{p²})
- G2 lives over F_{p²} (the twist field). Need: F_{p²} arithmetic (slot 292 T1), F_{p²} square root (Adj-Rodriguez-Henriquez), 3-isogeny on E'_{F_{p²}} → E_{F_{p²}}, faster clear_cofactor (Budroni-Pintore).
- RFC 9380 §8.8.2 ciphersuite: `BLS12381G2_XMD:SHA-256_SSWU_RO_`. WB19 §5 gives the explicit 3-isogeny coefficients (3 of x_num, x_den, y_num, y_den polynomials of degrees 3,2,3,2 over F_{p²}).
- This is ~150-200 LOC even with an existing F_{p²} type. Without F_{p²} (current state), it's a non-starter — cleanly defer until slot 292 T2 (G1 + scalar mult) AND slot 292 T3 (F_{p²}) ship.

### F8. The composition makes hash-to-curve a *consumer* of crypto, not a standalone package
The minimal dependency closure for `BLS12381G1_XMD:SHA-256_SSWU_RO_`:
1. `crypto.SHA256` (slot 322)
2. `crypto.HMAC` (not strictly required for XMD; XMD uses raw H, not HMAC — but most stacks share infra)
3. `crypto/ecc.Fp` for BLS12-381 base field (slot 292)
4. `crypto/ecc.G1Point` Jacobian/projective coordinates (slot 292)
5. `crypto/ecc.G1.ScalarMul` (slot 292) — for clear_cofactor
6. `crypto/ecc.G1.Add` (slot 292) — for combining the two SSWU outputs
7. `crypto/ecc.Fp.Sqrt` (slot 292) — SSWU's central operation; tonelli-shanks or `(p+1)/4` Bernstein for p ≡ 3 mod 4 (BLS12-381 base p IS 3 mod 4 → cheap closed-form sqrt)

Without all 7, hash-to-curve cannot be tested end-to-end. The **placement** matters: this should be `crypto/ecc/h2c` (sub-package), composing both `crypto` and `crypto/ecc`.

### F9. 8 official RFC-9380 test vectors are bit-exact regression
RFC 9380 §J gives test vectors per ciphersuite. For BLS12381G1_XMD:SHA-256_SSWU_RO_ (§J.9.1):
- 5 messages: empty, "abc", "abcdef0123456789", "q128_..." (128-byte), "a512_..." (512-byte)
- For each: u0, u1 (F_p), Q0, Q1 (curve points before cofactor clear), P (final).
- Plus expand_message §K.1 vectors (SHA-256, SHA-512) — 6 lengths × 2 hashes = 12 entries.
- And hash_to_field §K vectors per ciphersuite.

These are **gold standard** — every RFC-9380 implementation (zkcrypto/bls12_381, blst, herumi/mcl, kwantam/bls12-381_hash, armfazh/h2c-{rust,go}-ref, cloudflare circl) reproduces exactly the same bytes. Bit-exact reproduction is **R-MUTUAL-CROSS-VALIDATION** trivially — five independent reference implementations, identical outputs, free saturation.

### F10. Reference implementations to mutually validate against
- `armfazh/h2c-go-ref` — Go reference, RFC 9380 normative (Faz-Hernandez is RFC author)
- `armfazh/h2c-rust-ref` — Rust mirror by same author
- `kwantam/bls12-381_hash` — Wahby's own reference C implementation
- `cloudflare/circl` — production Go (`circl/group/group.go`)
- `zkcrypto/bls12_381` — Rust, used by Filecoin/Zcash
- `supranational/blst` — production C/asm, used by Ethereum consensus
- `bytemare/hash2curve` — pure-Go RFC 9380

Five+ independent implementations → **R-MUTUAL-CROSS-VALIDATION 3/3 saturates trivially** by picking any 3 and the RFC test vectors.

## Concrete recommendations

### R1. Tier ladder for hash-to-curve in reality

| Tier | Primitive | LOC | Composes | Day-N |
|---|---|---|---|---|
| **T0** | `expand_message_xmd[SHA-256]` | ~80 | slot-322 SHA-256 | Day-2 (after 322 lands) |
| **T0b** | `expand_message_xmd[SHA-512]` | trivial param | T0 | Day-2 |
| **T0c** | `expand_message_xof[SHAKE128/256]` | ~80 | needs SHA-3 | Day-3 (after SHA-3) |
| **T1** | `hash_to_field[F_p]` (m=1) | ~80 | T0 + slot-292 Fp | Day-3 (after 292 T1) |
| **T1b** | `hash_to_field[F_{p²}]` (m=2) | +30 | T1 + slot-292 Fp2 | Day-5 |
| **T2** | `map_to_curve_simple_swu` (A·B≠0) | ~150 | slot-292 Fp.Sqrt + Weierstrass | Day-3 |
| **T3** | `map_to_curve_sswu_iso` (A=0 via isogeny) | ~120 | T2 + 11-isogeny eval | Day-4 |
| **T4** | `clear_cofactor_bls12_381_g1` | ~50 | slot-292 G1.ScalarMul | Day-4 |
| **T5** | **`BLS12381G1_XMD:SHA-256_SSWU_RO_`** (composition) | ~80 | T0+T1+T3+T4 | Day-4 |
| **T6** | `map_to_curve_simple_svdw` (generic) | ~150 | slot-292 Fp.Sqrt | Day-5 |
| **T7** | `clear_cofactor_bls12_381_g2` (Budroni-Pintore) | ~80 | slot-292 G2 | Day-7 |
| **T8** | **`BLS12381G2_XMD:SHA-256_SSWU_RO_`** | ~150 | T1b+T3(over Fp2)+T7 | Day-7 |
| **T9** | `map_to_curve_elligator2` (Curve25519/448) | ~120 | Montgomery curve type | Day-8 (lower priority, no ZK-rollup pull) |
| **T10** | `secp256k1_XMD:SHA-256_SSWU_RO_` | ~80 | T1+T3 (with secp256k1 isogeny) | Day-9 |

### R2. Cheapest day-1-after-322 PR: T0 + T1 standalone (~160 LOC)
- Lands `crypto/h2c/expand.go` and `crypto/h2c/field.go` with golden-file vectors from RFC 9380 §K.1, §K.2, §K.3.
- Validates against `armfazh/h2c-go-ref` and `cloudflare/circl` — R-MUTUAL-CROSS-VALIDATION 3/3 trivial.
- **Independent of slot 292** (T1 only needs `*big.Int` mod p for a hardcoded BLS12-381 p — does not need a full Fp type yet).
- Forms the foundation that every subsequent ciphersuite reuses.

### R3. Headline composed PR: full BLS12381G1_XMD:SHA-256_SSWU_RO_ as a joint 292+323 PR
- Bundle slot-292 phase-A (`Fp`, BLS12-381 curve, `G1.{Add,ScalarMul}`, `Fp.Sqrt`) **with** slot-323 T0+T1+T3+T4+T5.
- Total estimate: 720 LOC (Fp ~200, G1 ~200, SSWU+isogeny ~150, expand+h2f ~160, plus tests).
- Single PR ships **8 official RFC-9380 G1 test vectors** as JSON golden files, plus 12 expand_message §K.1 vectors, plus h2f §K.3 vectors.
- This unblocks **every BLS-signature consumer** (zkmark, ETH consensus, Filecoin-style aggregate signatures, Boneh-Lynn-Shacham short signatures, KZG commitments).

### R4. R-MUTUAL-CROSS-VALIDATION 3/3 pin opportunities (concrete, free)
1. **3/3-vector**: pick **3** independent reference implementations (`armfazh/h2c-go-ref`, `cloudflare/circl`, `kwantam/bls12-381_hash`), reproduce **all 5** RFC-9380 §J.9.1 BLS12-381 G1 messages bit-exact, including the intermediate u0, u1, Q0, Q1, and final P. Five messages × five intermediate values × three implementations = 75 cross-cross-validated assertions in one test file.
2. **3/3-invariant**: assert (a) `y² ≡ x³ + 4 (mod p)` for output points (curve membership); (b) `cofactor · P = O` (subgroup membership — this is a positive form: `r · P = O` where r is BLS12-381 G1 group order); (c) `P_returned == P_expected_RFC` (test-vector match). Three independent regression invariants, all trivially derivable.
3. **3/3-property**: round-trip through `hash_to_curve(m1) + hash_to_curve(m2) ≠ hash_to_curve(m1||m2)` (collision-resistance smoke test); `hash_to_curve(m)` is **deterministic** (idempotent on identical input); `DST` separation — different DSTs yield different outputs with same `m`.

### R5. Cross-link to consumers
- **BLS aggregate signatures** (zkmark, ETH 2.0 consensus, Filecoin, Drand, Chia): σ = sk · H(m), aggregate σ = Σ σ_i, verify via pairing e(g, Σσ_i) = e(Σ pk_i, H(m)). Hash-to-curve is **the** non-pairing core primitive.
- **VRF (RFC 9381)**: ECVRF uses hash-to-curve as `H(pk || alpha)` → curve point. Slot 292 cousin.
- **PAKE (CPace, OPAQUE, SPAKE2+)**: requires hash-to-curve for password→point.
- **KZG polynomial commitments**: setup uses random G1/G2 points; some variants hash-to-curve for SRS generation (powers-of-tau).
- **OPRF (RFC 9497)**: voprf-base uses `BLS12381G1_XMD:SHA-256_SSWU_RO_` directly.
- **ZK-rollup signature schemes**: every Plonk/Halo2/Groth16 verifier with on-chain BLS aggregate signatures.

### R6. CLAUDE.md / packaging guidance
- **Place at `crypto/h2c/`** (not `crypto/`) — keeps the 22-package toplevel uncluttered. RFC 9380 is sufficiently large (every ciphersuite is its own surface) to warrant a sub-package.
- Or alternatively `crypto/ecc/h2c/` if `crypto/ecc/` becomes the elliptic-curve sub-package per slot 292 recommendation.
- Treat each ciphersuite as a **named registered constant + factory** — `h2c.Suite{Hash, Expand, Map, Cofactor, DST}` with `BLS12381G1_XMD_SHA256_SSWU_RO`, `P256_XMD_SHA256_SSWU_RO`, etc. Composition over inheritance.
- Cite RFC 9380 §6.6.2 (SSWU full pseudocode) and §6.6.3 (SSWU isogeny variant) directly in the file headers — design rule 4 (every function cites its source).

### R7. Sequencing (in days, assuming 1 PR/day with passing CI)
- D1: slot 322 SHA-256 lands.
- D2: T0 (expand_message_xmd) — 80 LOC, RFC §K.1 vectors. Independent.
- D3: slot 292 T1 (Fp, point, scalar mult) lands. T1 hash_to_field stacked on it.
- D4: T2/T3 SSWU + clear_cofactor + T5 G1 hash. **Joint PR with slot 292 phase-A.**
- D5+: slot 214 (pairings) lands → BLS aggregate signature unlocks → zkmark consumers can pull.
- D7: T8 G2 hash (after Fp2 in slot 292 T3).
- D8+: Elligator 2, secp256k1 SSWU, P-256/P-384 SSWU.

### R8. Anti-recommendation
- **Do NOT ship try-and-increment.** It is variable-time, was the only method available 1990-2009, and is explicitly forbidden by RFC 9380 §10 for live protocols. Every modern review (Cloudflare's Faz-Hernandez specifically) treats T&I as a footgun.
- **Do NOT wrap an existing library** (CLAUDE.md design rule 6). All 5 RFC-reference implementations are MIT/BSD-compatible — read them, learn from them, write reality's own from the RFC, validate against them via JSON golden files (CLAUDE.md design rule 1).

## Sources

### Repo (zero-surface confirmation)
- `C:/limitless/foundation/reality/crypto/hash.go:1-216` — only FNV/Murmur/Consistent/Situation hashes. **No SHA-2.**
- `C:/limitless/foundation/reality/crypto/{modular.go,prime.go,rng.go}` — number-theoretic only, **no EC types.**
- Repo-wide grep `HashToCurve|MapToCurve|ExpandMessage|HashToField|Elligator|SWU|ShallueVanDeWoestijne|WahbyBoneh|hash_to_curve|map_to_curve|expand_message` matches **only** in `reviews/overnight-400/agents/*.md` (review noise). **Zero code surface.**
- `C:/limitless/foundation/reality/reviews/overnight-400/agents/322-dive-aes-vs-poly1305.md` — slot 322, SHA-256 day-1 PR.
- `C:/limitless/foundation/reality/reviews/overnight-400/agents/292-new-elliptic-curves.md` — slot 292, BLS12-381 phase-A.
- `C:/limitless/foundation/reality/reviews/overnight-400/agents/214-new-pairings.md` — slot 214, optimal-Ate.
- `C:/limitless/foundation/reality/reviews/overnight-400/agents/213-new-isogeny.md` — slot 213, isogeny machinery (informs T3 SSWU-iso).
- `C:/limitless/foundation/reality/reviews/overnight-400/agents/321-dive-finite-field.md` — F_p prerequisite.

### Specifications
- RFC 9380, "Hashing to Elliptic Curves," Faz-Hernandez, Scott, Sullivan, Wahby, Wood. IRTF, August 2023. https://datatracker.ietf.org/doc/rfc9380/ — normative.
- §5.2 hash_to_field; §5.3 expand_message_{xmd,xof}; §6 map_to_curve (SvdW, SSWU, Elligator); §7 clear_cofactor; §8 ciphersuites; §J test vectors; §K expand vectors.

### Foundational papers
- Wahby, Boneh. "Fast and simple constant-time hashing to the BLS12-381 elliptic curve." TCHES 2019. https://eprint.iacr.org/2019/403 — the production standard for BLS12-381 G1/G2 hash, 11-isogeny.
- Brier, Coron, Icart, Madore, Randriam, Tibouchi. "Efficient Indifferentiable Hashing into Ordinary Elliptic Curves." CRYPTO 2010. https://eprint.iacr.org/2009/340 — simplified SWU for char ≠ 2,3 Weierstrass.
- Shallue, van de Woestijne. "Construction of Rational Points on Elliptic Curves over Finite Fields." ANTS-VII, LNCS 4076, 2006. — generic SvdW06 fallback.
- Bernstein, Hamburg, Krasnova, Lange. "Elligator: Elliptic-curve points indistinguishable from uniform random strings." CCS 2013. https://eprint.iacr.org/2013/325 — Elligator 1/2 for Curve25519/448.
- Icart. "How to Hash into Elliptic Curves." CRYPTO 2009. — original deterministic encoding (subsumed by Brier et al.).

### Reference implementations (R-MUTUAL-CROSS-VALIDATION pool)
- armfazh/h2c-go-ref — RFC author's own Go reference. https://github.com/armfazh/h2c-go-ref
- armfazh/h2c-rust-ref — Rust mirror. https://github.com/armfazh/h2c-rust-ref
- kwantam/bls12-381_hash — Wahby's own C reference of the WB19 paper. https://github.com/kwantam/bls12-381_hash
- cloudflare/circl — production Go, `circl/group`. https://github.com/cloudflare/circl
- zkcrypto/bls12_381 — Rust, Filecoin/Zcash production.
- supranational/blst — C/asm, Ethereum consensus production.
- bytemare/hash2curve — pure-Go RFC 9380. https://github.com/bytemare/hash2curve

### Test-vector sources
- RFC 9380 §J (per-ciphersuite full vectors), §K (expand_message vectors).
- ethereum/bls12-381-tests — JSON test vectors used by ETH consensus clients. https://github.com/ethereum/bls12-381-tests
- input-output-hk/bls-e2e-testvectors — Cardano BLS bindings tests. https://github.com/input-output-hk/bls-e2e-testvectors

### Cross-references in this overnight review
- 057-crypto-missing.md, 058-crypto-sota.md, 059-crypto-api.md — crypto package gap analysis.
- 175-synergy-zkmark-crypto.md — zkmark consumer of BLS hash-to-curve.
- 200-synergy-zkmark-info.md — KZG / pairing-based ZK.
- 212-new-pq-signatures.md — non-overlap (PQ signatures don't use hash-to-curve, but share `expand_message_xof` over SHAKE).
- 263-new-quasi-mc.md — unrelated, but Sobol QMC is independent of hash-to-curve.
