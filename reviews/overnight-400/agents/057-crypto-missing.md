# 057 — crypto-missing

**Topic:** crypto: missing — EdDSA, BLS, Schnorr, Pedersen commitments, Shamir SS, VSS, oblivious transfer, garbled circuits, ECIES.

**Premise from 056:** the `crypto/` package is 884 LOC of number-theory + non-crypto hashing + deterministic PRNGs (Miller-Rabin / ModPow / ExtGCD / CRT / FNV-1a / MurmurHash3 / Jump-consistent / MT19937-64 / PCG-XSH-RR / Xoshiro256**). **Zero** cryptographic primitives ship today. This audit enumerates what is missing, scoped against `reality`'s "math primitives, no AEAD/network protocol, zero deps, golden-file validated, MIT" constraints (CLAUDE.md §1–§6).

**Scope filter — what counts as "in-scope" for `reality`:**
- IN: pure mathematical primitives — finite-field arithmetic, polynomial arithmetic, modular operations, lattice basis ops, deterministic algorithms with golden vectors.
- IN: signature/commitment math (the polynomial-evaluation / scalar-mul / pairing / inner-product math), NOT the random-nonce generation or wire formats.
- BORDERLINE: hash compression functions (Davies-Meyer, sponge round function) — the round function is math; the streaming API is engineering.
- OUT: AEAD modes (GCM/CCM/SIV/OCB), TLS, key wrapping protocols, CSPRNG seeding from OS entropy, X.509/PKCS#8 serialization, network protocol state machines.
- OUT: anything requiring `crypto/rand` / `getrandom(2)` or operating-system entropy — `reality` is deterministic by design.
- OUT: AES round transforms (S-box / ShiftRows / MixColumns) — these are byte-level operations not "math" in the calculus / linalg / number-theory sense `reality` ships; same for ChaCha20.
- HOWEVER: GHASH GF(2^128) multiplication, Poly1305 mod-(2^130-5) arithmetic, and Keccak-f permutation on a 5×5×64 lane lattice ARE math primitives and are in-scope as algebraic objects.

---

## Headline

`crypto/` is a number-theory + PRNG kit. To reach the topic-listed primitives (EdDSA, BLS, Schnorr, Pedersen, Shamir, VSS, OT, GC, ECIES) requires an estimated **8,000–12,000 LOC across 5 missing layers**, none of which exist today: (1) **big-integer arithmetic** (currently uint64-only — Curve25519 scalars are 256-bit, BLS12-381 base field is 381-bit, RSA-2048 is 2048-bit), (2) **finite-field tower** (Fp / Fp2 / Fp4 / Fp6 / Fp12 — needed for any pairing-friendly curve), (3) **elliptic-curve point arithmetic** (Edwards / Weierstrass / Montgomery models, all absent), (4) **standard hash-function primitives** (no SHA-2, SHA-3, BLAKE, Poseidon — every signature scheme depends on at least one), (5) **constant-time conditional-select / Montgomery-ladder discipline** (the package has zero side-channel hygiene per 056). The topic asks specifically for Tier-1/2/3 enumeration of EdDSA/BLS/Schnorr/Pedersen/Shamir/VSS/OT/GC/ECIES — of these nine, **zero** are reachable without first landing the five missing layers. Shamir Secret Sharing is the cheapest entry point (polynomial interpolation in a prime field, ~150 LOC including a Lagrange evaluator), and is the recommended Tier-1 anchor; the remaining eight all depend on either curve arithmetic or hash functions that themselves don't exist.

---

## Missing primitives — Tier 1 (foundation, must ship before anything else)

These are not "topic-listed" primitives but are **prerequisites** for every topic-listed primitive. Without them, EdDSA/BLS/Schnorr/etc. are unreachable.

### T1-BIGINT — Big-integer arithmetic (≈400–600 LOC, blocks everything)
Currently the package is uint64-only. RSA needs 2048-bit, P-256 needs 256-bit, BLS12-381 needs 381-bit base field, Ed25519 needs 255-bit. **Reality must ship its own multi-precision integer** (Go's `math/big` is in scope as a stdlib import per CLAUDE.md §2 "only the language's standard math library", but using `math/big.Int` ties the cross-language golden contract to Go's specific algorithm choices — Python's `int` is unbounded, C++ has no native bigint, C# uses `BigInteger`). Decision needed: (A) wrap `math/big.Int` in Go and require Python/C++/C# to match its bit-exact output, or (B) implement a 64-bit-limb multi-precision type from scratch with documented limb layout. Option (B) is the cross-language-friendly choice and is what `blst`, `arkworks`, and `dalek` all do.
- ADD/SUB/MUL/DIV with limb-array storage.
- MOD via Barrett reduction OR Montgomery reduction (Montgomery is preferred for repeated mod ops in the same field).
- ModExp for variable-base / fixed-base.
- ModInverse via extended Euclidean OR Fermat's little theorem (a^(p-2) mod p) for prime moduli.
- ConstantTimeSelect, ConstantTimeCompare, ConstantTimeNeg.

### T1-FIELD — Generic prime-field Fp (≈300 LOC)
On top of T1-BIGINT, a `Field{p}` type with Add/Sub/Mul/Square/Inverse/Sqrt/Pow/Neg/Zero/One. **Tonelli-Shanks** for square roots in odd-prime fields (used by every curve compression decode). **Cipolla** as fallback for p ≡ 1 mod 4 with high 2-adicity. Frobenius endomorphism for fields of characteristic > 2.

### T1-FIELDTOWER — Fp2 / Fp4 / Fp6 / Fp12 extension fields (≈400 LOC)
Required for pairings on BLS12-381 / BN254. Tower construction: Fp → Fp2 (i² = non-residue), Fp2 → Fp6 (v³ = (i + something)), Fp6 → Fp12 (w² = v). Karatsuba and Toom-Cook for extension multiplication. Frobenius coefficients precomputed.

### T1-EC — Generic elliptic curve point group (≈500 LOC)
`Point{X, Y, Z}` in projective Jacobian coordinates. Add (mixed and full), Double, Negate, IsOnCurve, IsInfinity, ToAffine, FromAffine. **Two curve models needed:**
- Short Weierstrass (P-256, P-384, P-521, secp256k1, BN254, BLS12-381 G1).
- Twisted Edwards (Ed25519, Ed448) — complete addition law, no special cases.
- Montgomery (Curve25519, Curve448) — for X25519 ladder.

Constant-time scalar multiplication via Montgomery ladder OR fixed-window with point-table conditional-select. **No double-and-add-with-branching** — that's the textbook timing leak.

### T1-HASH — SHA-2, SHA-3, BLAKE2b/3, HMAC (≈600 LOC)
Every signature scheme listed in the topic depends on a hash. Currently absent.
- **SHA-256 / SHA-512** — FIPS 180-4 round function (math: 64-round / 80-round Ch/Maj/Σ/σ functions over 32/64-bit words).
- **SHA-3 / Keccak-f[1600]** — sponge construction over a 5×5×64-lane state, 24 rounds of θ/ρ/π/χ/ι (math: GF(2) linear maps + a single nonlinear χ step). Pure math, definitely in-scope.
- **HMAC(H, K, M)** — math: H((K⊕opad) || H((K⊕ipad) || M)). 30 LOC once H is available.
- **BLAKE2b / BLAKE3** — chacha-style mixing. BLAKE3 has Merkle tree internal structure (math).
- **HKDF** — explicitly listed in topic; HMAC-based, ~40 LOC once HMAC ships. RFC 5869.

### T1-CT — Constant-time discipline kit (≈100 LOC, blocks any "secret-handling" claim)
Per 056, the package has zero constant-time guarantees. Before any topic-listed primitive ships:
- `ConstantTimeByteEq(a, b byte) byte`
- `ConstantTimeSelect(cond int, a, b uint64) uint64` — branch-free.
- `ConstantTimeCompare([]byte, []byte) int` — length-dependent leak documented.
- `ConstantTimeLessOrEq(x, y uint64) int`
- Document explicitly which functions are CT and which are explicitly variable-time (e.g., signature verification is publicly variable-time).

---

## Missing primitives — Tier 1 (topic-listed core)

After T1 foundation is in place:

### T1-SCHNORR — Schnorr signature math (≈80 LOC over T1-EC + T1-HASH)
The simplest signature in the topic list. Σ-protocol: prover commits R = k·G, challenge c = H(R || P || m), response s = k + c·x. Verify: s·G == R + c·P. **In-scope:** the equation. **Out-of-scope:** the `k` nonce derivation (RFC 6979 deterministic-k IS in-scope; HMAC-DRBG IS in-scope; ad-hoc `crypto/rand` is OUT). BIP-340 Schnorr (Bitcoin Taproot, secp256k1, x-only pubkeys) is the canonical modern target.

### T1-EDDSA — Ed25519 + Ed448 signature math (≈250 LOC over T1-EC-Edwards + SHA-512/SHAKE256)
RFC 8032. Deterministic-by-design (no nonce — k = H(prefix || m)). Specifically:
- Ed25519: SHA-512, Edwards25519 group, cofactor 8.
- Ed448: SHAKE-256, Edwards448 group, cofactor 4.
- Verification equation: [s]B = R + [k]A.
- Cofactor-clearing during pubkey decode is mandatory and was the source of multiple 2020-2024 implementation bugs (ZIP-215 vs strict).
- **Cross-implementation hazard:** the spec has known ambiguities (cofactored vs non-cofactored verification); reality must pick one and golden-test it against `crypto/ed25519` (Go stdlib) AND `pyca/cryptography` AND OpenSSL.

### T1-ECDSA — ECDSA signature math (≈150 LOC over T1-EC + T1-HASH)
P-256, P-384, secp256k1. Sign: k random in [1, n-1], r = (k·G).x mod n, s = k^-1·(z + r·d) mod n. Verify: w = s^-1, u1 = z·w, u2 = r·w, check (u1·G + u2·Q).x == r. **Required:** RFC 6979 deterministic k (HMAC_DRBG over (d, m)) — this is the deterministic-friendly version `reality` should target instead of the random-k spec.

### T1-SHAMIR — Shamir Secret Sharing (≈150 LOC, **cheapest topic primitive**)
Pure polynomial arithmetic in Fp. Split: pick random a_1...a_{k-1} ∈ Fp, set f(x) = secret + a_1·x + ... + a_{k-1}·x^{k-1}, output (i, f(i)) for i = 1..n. Reconstruct: Lagrange interpolation at x=0 from any k shares. **Math only — Lagrange interpolation already lives in `linalg/` or `calculus/`, just needs a finite-field variant.** No curves, no hash, no big-int (well — for 256-bit secrets you need T1-BIGINT, but for `reality`'s test mode a 64-bit prime field is sufficient to ship the algorithm with golden vectors). **Recommended Tier-1 starting point.**

### T1-PEDERSEN — Pedersen commitments (≈80 LOC over T1-EC, after Schnorr)
C = g^v · h^r where g, h are independent generators with unknown discrete-log relationship. Hiding (perfect, given uniform r) and binding (computational, under DLOG hardness). Math: two scalar-mults and a point-add. Used as building block for VSS, range proofs, ring signatures.

---

## Missing primitives — Tier 2 (topic-listed advanced)

### T2-VSS — Verifiable Secret Sharing (≈200 LOC over Shamir + Pedersen)
- **Feldman VSS:** dealer publishes commitments C_j = g^{a_j} for j=0..k-1; share verifier checks g^{f(i)} == ∏ C_j^{i^j}. Discrete-log assumption only. ~80 LOC.
- **Pedersen VSS:** dealer publishes C_j = g^{a_j} h^{b_j} where b_j is a second random polynomial; provides information-theoretic hiding of secret. ~120 LOC.
- Both are pure math (group exponentiation + polynomial evaluation in the exponent).

### T2-FROST — Threshold Schnorr signatures (≈400 LOC over Schnorr + VSS)
RFC 9591 (2024). Two-round threshold signing for Schnorr/EdDSA. **Math:** distributed key generation (Pedersen DKG), nonce commitments, share aggregation, partial signature combination via Lagrange coefficients. **In-scope:** the math. **Out-of-scope:** the network rounds — but reality can ship the per-participant deterministic functions and let consumers wire transport.

### T2-BLS — BLS signatures (≈600 LOC over T1-EC G1/G2 + T1-FIELDTOWER + pairing)
RFC 9380 / IRTF draft. σ = [sk]H(m) on G1; verify e(σ, g_2) == e(H(m), pk). **Aggregation:** σ_agg = ∑ σ_i. **Requires pairing** (next item). **Hash-to-curve** (next-next item). The smallest-on-the-wire signature scheme — 48 bytes for BLS12-381 G1 — and the most math-heavy.

### T2-PAIRING — Optimal Ate pairing on BLS12-381 / BN254 (≈800 LOC over T1-FIELDTOWER + T1-EC)
Miller's algorithm + final exponentiation. **The single most complex math primitive in this entire enumeration.** Components:
- Miller loop: line-function evaluation at each doubling/addition during scalar-mult-by-(t-1).
- Frobenius twist for G2 ⊂ E'(Fp2) → E(Fp12) embedding.
- Final exponentiation: easy part (Frobenius) + hard part (cyclotomic).
- Subgroup checks (G1 prime-order check, G2 subgroup check).
- ~3000 LOC in `blst`, ~2000 in `arkworks` after deduplication. A `reality`-quality from-scratch impl is closer to 800–1000 LOC.

### T2-HASH-TO-CURVE — RFC 9380 hash-to-curve (≈300 LOC over T1-EC + T1-HASH + T1-FIELD)
- **SSWU** (Simplified Shallue-van de Woestijne-Ulas) for Weierstrass curves with non-zero a, b.
- **Icart** for cubic-curve compatibility.
- **Elligator 2** for Curve25519 / Edwards.
- BLS12-381–specific isogeny-based suite (G1 SSWU + 11-isogeny, G2 SSWU + 3-isogeny).
- expand_message_xmd / expand_message_xof primitives.

### T2-MSM — Multi-scalar multiplication (≈250 LOC over T1-EC)
**Pippenger's bucket method.** Given points P_1...P_n and scalars s_1...s_n, compute ∑ s_i·P_i in O(n / log n) time vs naive O(n) scalar-mults. Used in zk-SNARK setup, proof generation, batch signature verification. The algorithmic core is bucketing scalars by digit windows — pure combinatorics + group ops. **Overlap risk with `linalg/` sparse-MV** — different group, but algorithmic shape is similar enough that a shared "windowed accumulation" abstraction may emerge.

### T2-ECIES — ECIES encryption math (≈150 LOC over T1-EC + T1-HASH + T1-KDF)
SEC1. Sender: ephemeral keypair (r, R=r·G), shared = r·Q (recipient pubkey), derive K = KDF(shared || ...), c = AES_K(m), τ = MAC_K(c). **In-scope:** the ECDH part + KDF derivation. **Out-of-scope:** the AES encryption itself (AES is not math per the scope filter; but the ECIES *math layer* is in-scope and should be golden-tested at the K = KDF(...) boundary, leaving AES as a consumer concern).

### T2-ELGAMAL / T2-PAILLIER — Homomorphic encryption math (≈300 LOC combined)
- **ElGamal in Fp\*** or on an elliptic curve: c = (g^r, m·h^r). Multiplicatively homomorphic on the second coordinate.
- **Paillier:** c = g^m · r^n mod n². Additively homomorphic. Decryption requires λ = lcm(p-1, q-1) and Carmichael L function. ~200 LOC, all pure modular arithmetic, no curves.
- Both are explicitly math primitives and have well-defined deterministic-given-randomness golden-vector forms.

---

## Missing primitives — Tier 3 (advanced, niche, or partial-overlap)

### T3-OT — Oblivious Transfer (≈300 LOC, multi-protocol)
- **1-out-of-2 OT:** Naor-Pinkas (DDH-based), Chou-Orlandi (CCS '15, simplest). Pure group operations. ~150 LOC.
- **OT extension:** IKNP, KOS, SoftSpokenOT — bit-matrix manipulations + correlation-robust hash. ~300 LOC, math-heavy.
- **OT extension underpins** garbled circuits and most MPC protocols.

### T3-GC — Garbled Circuits (≈500 LOC, partly engineering)
- **Yao's garbled circuit** with point-and-permute, free-XOR, half-gate optimization, fixed-key AES garbling.
- The garbling math is in-scope (label generation, gate encoding); the circuit IR and AES round function are engineering / out-of-scope.
- Recommended scope: ship the per-gate garbling math, leave circuit-frontend to consumers.

### T3-POLYCOMMIT — Polynomial commitment schemes (≈600 LOC, overlap with `zkmark/`)
- **KZG (Kate-Zaverucha-Goldberg):** trusted setup [τ^i]G_1, commit(f) = f(τ)·G_1. Pairing-based. ~150 LOC over T2-PAIRING.
- **FRI (Fast Reed-Solomon Interactive Oracle):** STARK-friendly, no trusted setup, no pairings. Reed-Solomon code distance argument. ~400 LOC over `signal/` FFT + Merkle.
- **IPA (Inner Product Argument):** Bulletproofs-style, no trusted setup, logarithmic proof size. ~300 LOC over T1-EC + T2-MSM.
- **Bulletproofs range proofs** sit on IPA: ~200 LOC additional.

**Overlap caveat:** `reality` has no `zkmark/` package today. If one is planned, polynomial commitments belong there, not in `crypto/`. Recommend deferring T3-POLYCOMMIT until the reality package map decides where ZK math lives.

### T3-NTT — Number-Theoretic Transform (≈200 LOC)
FFT over a finite field, used in Falcon (lattice signatures), Kyber/Dilithium (PQ KEM/sig), Plonk-style ZK proofs. Cooley-Tukey radix-2 with twiddle factors that are ω = primitive 2^k-th root of unity in Fp. **Overlaps `signal/` FFT** — same algorithm shape, different ring. Recommended: implement once in `linalg/` or `crypto/` and have the other reference it.

### T3-LATTICE — Lattice-crypto primitives (≈800 LOC for a minimal Kyber/Dilithium math layer)
- Module-LWE / Module-SIS arithmetic over Rq = Zq[X]/(X^n + 1).
- **Polynomial multiplication via NTT** (T3-NTT).
- Sampling: centered binomial, uniform mod q, rejection sampling — but most of these need entropy and are partly out-of-scope. The deterministic NTT-based polynomial arithmetic is in-scope.
- **Falcon** adds Gaussian sampling over lattice basis — borderline (the math is well-defined; the sampler is the hard part).
- NIST PQC standardized: ML-KEM (Kyber), ML-DSA (Dilithium), SLH-DSA (SPHINCS+), Falcon. All four have substantial math layers.

### T3-ALGEBRAIC-HASH — Poseidon, Rescue, Anemoi (≈400 LOC)
Hash functions designed for ZK-circuit-friendliness — every operation is a low-degree polynomial in Fp. **Pure math, ideal for `reality`.** Used in Plonk, Halo2, StarkWare ecosystem.
- Poseidon: full-round / partial-round structure with x^5 S-box.
- Rescue / Rescue-Prime: x^5 forward, x^{1/5} backward.
- Anemoi: Flystel construction.

### T3-VECTORCOMMIT — Vector / Verkle commitments (≈400 LOC)
Verkle trees use KZG or IPA per-node instead of Merkle. Math is the polynomial-commitment layer above + tree construction. Mostly engineering once T3-POLYCOMMIT lands.

### T3-DLOG — Discrete log algorithms (≈200 LOC)
- **Baby-step giant-step:** O(√n) time/space.
- **Pollard rho:** O(√n) time, O(1) space.
- **Pohlig-Hellman:** reduces DLOG to subgroup DLOGs via CRT — uses existing `ChineseRemainder`.
- These are *attack* algorithms, useful for parameter validation (verify a generator's order, check embedding degree) and pedagogy.

### T3-SYMMETRIC-MATH — Poly1305, GHASH, ChaCha20 quarter-round (≈250 LOC)
The math primitives only:
- **Poly1305:** evaluation of a polynomial mod (2^130 - 5) at a key-derived point. Pure modular arithmetic. ~80 LOC.
- **GHASH:** multiplication in GF(2^128) = GF(2)[X]/(x^128 + x^7 + x^2 + x + 1). Pure binary-field arithmetic, in-scope. ~60 LOC.
- **ChaCha20 quarter-round:** ARX (add/rotate/xor) on 32-bit words. Marginally in-scope; the round function is math, the streaming-mode XOR is engineering.
- AES S-box / MixColumns: the S-box is multiplicative inverse in GF(2^8) followed by an affine transform — in-scope as an algebraic object, not as a "cipher."

### T3-RNG-CRYPTO — CSPRNG-shape APIs (≈200 LOC, partial scope)
- **HMAC-DRBG** (NIST SP 800-90A): deterministic given a seed. Pure function of seed → byte stream. **In-scope** under `reality`'s deterministic-everywhere model — feed it a seed, get a byte stream, golden-validate against NIST CAVP test vectors.
- **CTR-DRBG, Hash-DRBG:** same pattern.
- **ChaCha20-DRBG / Fortuna:** out-of-scope (depend on AES/ChaCha cipher).
- This is the single biggest "fake gap" — `reality` already has deterministic PRNGs (MT/PCG/Xoshiro), but those are explicitly NOT cryptographic. HMAC-DRBG fills the gap with the same deterministic API shape AND the cryptographic guarantee.

---

## Recommended ordering (Tier-by-Tier blockers)

1. **T1-BIGINT** — must land first. Decision: wrap `math/big` (cheap, 100 LOC wrapper) vs roll-your-own (cross-language-friendly, 400 LOC). Recommend wrap+expose for v0.x, replace with native limbs in v1.
2. **T1-CT** — constant-time kit. 100 LOC. Lands alongside BIGINT so every primitive built on top inherits CT discipline.
3. **T1-FIELD + T1-HASH (SHA-256, SHA-512, HMAC, HKDF)** — parallel tracks, both depend only on T1-BIGINT.
4. **T1-EC (Weierstrass first: P-256, secp256k1)** — depends on T1-FIELD + T1-CT.
5. **T1-SHAMIR** — pure Fp polynomial arithmetic, can ship in parallel with T1-EC. **Cheapest topic-listed primitive, recommended as the first user-facing crypto win.**
6. **T1-ECDSA** — over T1-EC + T1-HASH + RFC 6979 deterministic k.
7. **T1-EC-Edwards (Ed25519 group)** + **T1-EDDSA** — depends on Edwards curve + SHA-512.
8. **T1-SCHNORR + T1-PEDERSEN** — small additions over T1-EC.
9. **T2-VSS** — Feldman + Pedersen, ~200 LOC over Shamir + Pedersen.
10. **T2-FROST** — threshold Schnorr, ~400 LOC over Schnorr + VSS.
11. **T1-FIELDTOWER (Fp2/Fp6/Fp12)** — required before BLS / pairings.
12. **T2-PAIRING (Optimal Ate on BLS12-381)** — the single largest crypto-math expense in the entire enumeration.
13. **T2-HASH-TO-CURVE** — required for BLS hashing.
14. **T2-BLS** — over pairing + hash-to-curve.
15. **T2-MSM (Pippenger)** — perf accelerator for BLS aggregate verify and ZK setup.
16. **T2-ECIES + T2-ELGAMAL + T2-PAILLIER** — small leaf additions.
17. Tier 3 — schedule by consumer demand.

---

## Cross-language golden-file feasibility per tier

CLAUDE.md §1 mandates 20+ JSON golden vectors per function, validated across Go/Python/C++/C#. Per primitive:

| Primitive | Reference impl | Cross-lang feasibility |
|---|---|---|
| T1-BIGINT | Go `math/big` / Python `int` / C# `BigInteger` / C++ — needs vendored bigint | Easy if all 4 use the same wire format (big-endian byte array) |
| T1-FIELD / Fp | Built on T1-BIGINT | Easy |
| T1-FIELDTOWER | Custom; arkworks `ark-bls12-381` | Medium — coefficient ordering conventions vary |
| T1-EC Weierstrass | Go `crypto/elliptic`, Python `cryptography`, C `OpenSSL` | Easy — RFC 6979 nails determinism |
| T1-EC Edwards (Ed25519) | RFC 8032 has 5 official test vectors; need 20+ — extend deterministically | Easy |
| T1-HASH SHA-2/3 | NIST CAVP test vectors (>1000 each) | Trivial |
| T1-SHAMIR | No standard test vectors — generate from `reality` Go canonical | Medium — must publish vectors + Lagrange-coefficient convention |
| T1-ECDSA (RFC 6979) | RFC 6979 §A.2.5 / .6 / .7 has official vectors | Trivial |
| T1-EDDSA | RFC 8032 §7 has official vectors per curve | Trivial |
| T1-SCHNORR (BIP-340) | BIP-340 official test vectors | Trivial |
| T2-PAIRING | EIP-2537 / `blst` interop test suite | Hard — pairing intermediate values are non-canonical, need final-output fixtures only |
| T2-HASH-TO-CURVE | RFC 9380 §J appendix | Easy — official vectors |
| T2-BLS | IRTF BLS-signature draft test vectors + Ethereum 2.0 spec tests | Easy |
| T3-NTT | No std vectors — generate from canonical | Medium |
| T3-Poseidon | StarkWare reference impl + Plonky2 | Medium — many parameter choices |
| T3-HMAC-DRBG | NIST CAVP DRBG test vectors | Trivial |

---

## What NOT to add (out-of-scope confirmation)

Marked **OUT** with rationale:
- **AES block cipher / round function impl** — byte-level Sbox/MixColumns is engineering, not the math `reality` ships. The GF(2^8) inverse + affine transform that DEFINES the S-box IS in-scope as algebraic content, but the round-function streaming code is not.
- **AEAD modes** (GCM, CCM, SIV, OCB, ChaCha20-Poly1305) — protocol composition, not primitive math.
- **TLS / Noise / DTLS / QUIC handshake math** — protocol state machines.
- **X.509 / PKCS#1/8/12 / PEM parsers** — serialization, not math.
- **CSPRNG seeded from OS entropy** — `reality` is deterministic.
- **PBKDF2 / Argon2 / scrypt** — borderline. PBKDF2 is HMAC iteration (math, in-scope). Argon2/scrypt are memory-hard functions whose definition is math but whose validation requires a memory model — schedule under Tier 3 if at all.
- **Password hashing schemes** generally — out-of-scope, consumers should use a hardened library.
- **Side-channel-resistant pairing batch verification** — reachable as a perf delta after T2-PAIRING ships, but not a separate primitive.
- **Post-quantum signature/KEM full impls** (ML-KEM / ML-DSA / SLH-DSA / Falcon) — schedule as T3 with substantial math layers; out-of-scope for the topic-listed nine.

---

## Recommended Tier-1 anchor PR sequence

**Sequence A (signatures-first, 2000 LOC over 3 months):**
1. T1-BIGINT (math/big wrapper) + T1-CT — 200 LOC.
2. T1-FIELD + T1-HASH (SHA-256, SHA-512, HMAC, HKDF) + golden vectors from NIST CAVP — 700 LOC.
3. T1-EC Weierstrass (P-256, secp256k1) + ECDSA + RFC 6979 — 500 LOC.
4. T1-EC Edwards (Edwards25519) + Ed25519 + RFC 8032 vectors — 400 LOC.
5. T1-SCHNORR (BIP-340) + T1-PEDERSEN + T1-SHAMIR — 200 LOC.

This sequence delivers ECDSA, Ed25519, Schnorr, Pedersen, Shamir — five of nine topic-listed primitives — without requiring pairings.

**Sequence B (pairings-first, 3000 LOC):**
1. T1-BIGINT + T1-CT + T1-FIELD + T1-FIELDTOWER + T1-EC — 1500 LOC.
2. T2-PAIRING (Optimal Ate on BLS12-381) — 1000 LOC.
3. T2-HASH-TO-CURVE + T2-BLS — 500 LOC.

This sequence delivers BLS but defers the simpler signatures.

Recommend **Sequence A first** because it covers more topic-listed primitives at lower cost, deferring BLS until the consumer (likely a future ZK or threshold-sig consumer) materializes.

---

## Non-overlap

- 056 (crypto-numerics) covers correctness of what's there and the package-name-vs-contents trap. 057 (this) covers what's missing.
- API-shape audit (`crypto-api`?) deferred. Naming/import-path consequences of adding 8000 LOC of crypto under `crypto/` should be a future agent — possibly `crypto/` becomes a meta-package with subpackages `crypto/field`, `crypto/curve`, `crypto/sig`, `crypto/hash`, etc.
- Performance audit deferred — pairing perf and MSM perf are dominant costs and merit their own agent.
- Polynomial commitment schemes (KZG/FRI/IPA) overlap with a hypothetical `zkmark/` package; flagged but not duplicated here.
- Hash function audit (the proposed T1-HASH layer) overlaps with whatever package owns SHA-2/3 — currently nobody. Possibly `crypto/` is the right home; possibly a new `hash/` package.

Report length: 388 lines (under 400 cap).
