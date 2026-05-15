# 058 — crypto-sota

**Topic:** crypto: compare with libsodium, RustCrypto, Crypto++, dalek, Arkworks (+ BoringSSL/NSS, Bouncy Castle, ZK-frame primitives, Pasta curves, Mersenne primes).

**Premise (carried from 056/057):** `crypto/` is a 884-LOC number-theory + non-crypto-hash + deterministic-PRNG kit. 056 owns numerical correctness of what's there. 057 owns the missing-primitive map (T1 BIGINT/FIELD/FIELDTOWER/EC/HASH/CT foundations, T1/T2/T3 algorithms). **058 owns architecture, hygiene, and what zero-dep Go can borrow from each peer.** No primitive enumeration here — only library-design transferability.

---

## Headline

The nine SOTA peers diverge on *everything except one principle*: **the layer above raw arithmetic must be a typed algebraic object the user cannot misuse**. libsodium fuses key-type-and-bytes, RustCrypto fuses trait-and-key, Crypto++ fuses template-pipeline-stage, dalek fuses curve-point-and-equality-via-canonical-encoding, arkworks fuses field-and-trait. Reality's crypto/ today is *pure-functional uint64-in-uint64-out* with no algebraic carrier — the same shape as `math/bits`. **The single highest-leverage architectural decision before any primitive lands (057's T1-FIELD, T1-EC) is whether `Fp`, `Curve`, `Point`, `Scalar`, `Signature`, `Commitment` are typed values or `[]byte` aliases**, because every misuse-prevention trick from every SOTA peer is downstream of that decision. Recommendation: **typed values with explicit `Bytes()` and `FromBytes(...) (T, error)` boundaries, modeled on dalek's encoding-canonicality discipline + RustCrypto's trait minimalism**, costs ~150 LOC of type machinery and pre-empts at least ten classes of bug that bit OpenSSL/PyCrypto/early-Bouncy-Castle in the 2010-2020 decade.

---

## Per-library: architecture, misuse-prevention trick, what reality can borrow

### A. libsodium (Frank Denis, 2013–2026; NaCl-derived)

**(1) Headline architecture.** Single C library, ~25k LOC, "the cryptography library that doesn't make you cry." Inherits Bernstein's NaCl design: ~25 high-level functions covering box/secretbox/sign/auth/aead/kx/pwhash + a handful of low-level primitives (Curve25519, Ed25519, Poly1305, ChaCha20, BLAKE2b, Argon2). Every function is opinionated to a single algorithm (no AES-vs-ChaCha selection, no SHA-1-vs-SHA-256 selection at the call site — opinions are baked at the function-name level: `crypto_box_curve25519xsalsa20poly1305` is the explicit name; `crypto_box` is the alias to "the chosen one for this version"). Entire library compiles to a single static binary; no dynamic dispatch, no plugin system, no algorithm registry.

**(2) Misuse-prevention trick.** Three layered tricks:
- **Type-tagged byte buffers via length constants** — `crypto_box_PUBLICKEYBYTES` / `crypto_box_SECRETKEYBYTES` / `crypto_box_NONCEBYTES` are distinct compile-time constants. A C caller who swaps a public key for a secret key gets the same `unsigned char *` shape (no static type protection in C), but the *compile-time length mismatch when assigning to a fixed-size array* catches half the cases at the call site. Bindings (Python, Go, Rust) lift these to actual types.
- **`sodium_memzero` + `sodium_mlock`** — explicit secret-buffer hygiene API. The library has a *concept* of secret memory and forces callers to think about it.
- **No nonce-reuse rope to hang yourself with** — `crypto_secretbox_easy` requires the caller to provide a unique nonce, but the docs *and example code in the readme* say "use `randombytes_buf` for it." There's a *secondary* misuse-resistant variant `crypto_secretbox_xchacha20poly1305_easy` with a 192-bit nonce so accidental reuse is statistically negligible. The default constant choice (XChaCha20-Poly1305 over ChaCha20-Poly1305) sacrifices 4 bytes of bandwidth per message to remove a real footgun.

**(3) What reality can borrow.**
- **Length constants as the discoverability surface.** Reality's future crypto API should expose `Ed25519PublicKeyBytes = 32`, `Ed25519SignatureBytes = 64`, `BLS12381G1CompressedBytes = 48` as named constants. CLAUDE.md's "every function cites its source" generalizes to "every byte boundary cites its size." ~30 LOC of constants for the full T1 surface.
- **Opinion at the function-name level, not at a `mode` argument.** Don't ship `Sign(algo, key, msg)` taking an `algo` enum. Ship `Ed25519Sign(key, msg)` and `EcdsaP256Sign(key, msg)` as separate names. Every SOTA peer that did the enum approach (OpenSSL `EVP_*`, Bouncy Castle `Cipher.getInstance("AES/CBC/PKCS5Padding")`) ships a string-typo-as-CVE history. Reality's flat-namespace go convention favors this naturally; 058's contribution is *don't fight it later*.
- **DO NOT borrow** the high-level `box`/`secretbox` opinion-stack. Reality is a math primitive library, not a protocol library; opinion-stacking is for the consumer (aicore? a future `secrets/` package?). 057's scope filter explicitly puts AEAD / boxing OUT.
- **DO borrow** the convention that `Bytes()` always returns the *canonical* encoding (libsodium has one encoding per type and rejects everything else). Decode is strict; encode is single-form. Pre-empts the "two valid encodings of the same point" attack class (see dalek section).

### B. RustCrypto (workspace, 2019–2026; ~80 crates)

**(1) Headline architecture.** A *trait-based composition system* in pure Rust. The base traits live in `digest`, `cipher`, `signature`, `aead`, `kdf`, `mac`, `password-hash`, `elliptic-curve`. Concrete implementations are crate-per-algorithm (`sha2`, `sha3`, `aes`, `chacha20`, `ed25519-dalek`, `k256`, `p256`, `bls12_381`). The composition primitive is *zero-cost generic functions over traits*: `fn sign<S: Signer>(s: &S, msg: &[u8])` compiles to a direct call when monomorphized, indistinguishable from a hand-rolled `Ed25519::sign(...)`. Const-fn primitives mean field-element constants and curve-parameters can be evaluated at compile time, eliminating a class of "library was built with a different curve than expected" bugs.

**(2) Misuse-prevention trick.** Four layered:
- **Typestate for protocol stages.** A `Hasher` is consumed by `.finalize()` returning `Output<Self>`. You cannot accidentally call `.finalize()` twice or hash-after-finalize — the type system forbids it. Same for `Signer::sign(self, msg)` consuming the signer in protocols where keys are single-use (Schnorr-with-deterministic-k *intentionally* takes `&self` to allow reuse; Picnic/SPHINCS+ takes `self` to enforce one-shot).
- **`zeroize::Zeroize` derive macro** automatic on Drop. Any struct holding secret bytes implements zeroization without the author having to write it.
- **`subtle::Choice` and `CtOption`** — branch-free conditional types. The whole API forbids `if secret { ... } else { ... }` at the type level; you can only express "select one of two values based on a Choice with no branching emitted by LLVM."
- **`elliptic-curve` trait family** — `Curve` has associated types for `Scalar`, `AffinePoint`, `ProjectivePoint`, `FieldBytes`. A function generic over `<C: Curve>` cannot accidentally mix scalars-from-curve-A with points-from-curve-B; the type system rules it out. Every "I scalar-multiplied a P-256 point by a P-384 scalar" bug is impossible.

**(3) What reality can borrow.**
- **`subtle::Choice` analog in Go.** Go has no const-time guarantee, but `crypto/subtle` ships `ConstantTimeSelect`, `ConstantTimeEq`, `ConstantTimeByteEq`, `ConstantTimeCompare`. Reality should *re-export and document them* (or wrap into a `crypto/ct` sub-package as 057's T1-CT proposes). Borrow specifically: a `Choice` value type with `If(c Choice, a, b uint64) uint64` semantics, not a free function. Cost: ~50 LOC.
- **Per-curve type families.** When 057's T1-EC lands, do `type P256Point struct {...}`, `type P256Scalar [4]uint64`, `type Ed25519Point struct {...}`, `type Ed25519Scalar [4]uint64` as distinct types — *not* a single `type Point struct { Curve *Curve; X, Y *big.Int }`. Go has no associated types but it has named types and methods; the same misuse-prevention works syntactically. The cost is ~5x type duplication; the benefit is the type checker rejecting `Ed25519Verify(p256pubkey, sig, msg)` at compile time.
- **DO borrow** const-generic fixed-size byte arrays for serialization (`[32]byte` for Ed25519 keys, `[64]byte` for Ed25519 sigs) instead of `[]byte`. Go does `[N]byte` natively; use it. Pre-empts `len(sig) != 64` runtime checks.
- **DO NOT borrow** the trait/generic substrate itself. Go generics (1.18+) are powerful enough but the Rust trait system is a 10-year ecosystem mature; replicating it in Go ends up looking like java's `Cipher.getInstance(string)` factory pattern by accident. Reality should *not* design a generic `Signer`/`Verifier` interface and parameterize over it; ship the concrete types and let consumers wrap if they need polymorphism.
- **DO borrow** `Drop`-style explicit zeroization: a `(*Scalar).Zero()` method that overwrites the receiver bytes. Go has no destructor, so the discipline is convention-by-method-name; document it.

### C. Crypto++ (Wei Dai, 1995–2026)

**(1) Headline architecture.** C++ template metaprogramming for compile-time wiring. Algorithms are template classes with policy-pattern parameters: `CBC_Mode<AES>::Encryption` is a *type* whose constituent operations are statically dispatched at compile time. Pipeline composition uses `BufferedTransformation` — `source >> filter >> filter >> sink` with operator overloading. The `>>` operator triggers template instantiation that inlines the pipeline. The library is the largest C++ crypto library in mature use (~200k LOC), covers everything from AES to BLS to PQC, and ships ~150 algorithms.

**(2) Misuse-prevention trick.**
- **Compile-time policy composition** prevents wiring incompatibilities: `HMAC<SHA256>` is a type; `HMAC<NotAHash>` doesn't compile.
- **Pipeline ownership via `auto_ptr`-style filters** — each filter takes ownership of the next, so you can't double-attach or detach mid-stream.
- **`SecByteBlock` zeroizes on destruction.** Same idea as `zeroize` but in C++ via destructor.
- **`OAEP<SHA256>::Decoder`** — explicitly named, no string-keyed factory, no "OAEP without specifying the hash" footgun.

**(3) What reality can borrow.**
- **DO NOT borrow** the template metaprogramming substrate; Go has no equivalent and shouldn't pretend.
- **DO borrow** the *naming discipline*: `HMAC[SHA256]`-style Go generics (post-1.18) for `HMAC[H Hash]` parameterization is the closest analog and is in scope. ~40 LOC for the generic HMAC wrapper once the underlying hashes ship.
- **DO borrow** the principle that *pipeline parameters are types, not strings*. `RFC6979SignatureScheme[Curve, Hash]` is a typed configuration; `Sign("ecdsa", "sha256", "rfc6979", key, msg)` is a typo waiting to happen. Go generics support this for nullary type parameters.
- **DO NOT borrow** the BufferedTransformation pipeline operator-overloading idiom. Go has no operator overloading, and even if it did, this pattern conflates "stream transform" with "crypto primitive" — reality is the latter, not the former.
- **DO borrow** the breadth-coverage discipline as a quality signal — Crypto++ has ~150 algorithms with consistent naming. Reality's eventual surface should pick the *naming convention* (Algorithm-Mode-Hash, e.g., `EcdsaP256Sha256`) and apply it to all new primitives uniformly. Crypto++'s `<Mode><Cipher>` pattern (`CBC_Mode<AES>`) is the canonical reference for "config compiles into the type name."

### D. dalek-cryptography (Isis Lovecruft + Henry de Valence, 2017–2026)

**(1) Headline architecture.** Curve25519/Ed25519/Ristretto255 in pure Rust, no_std-compatible. Three crates: `curve25519-dalek` (group ops), `ed25519-dalek` (signatures), `x25519-dalek` (DH). The most carefully constant-time elliptic-curve library in production use. Architecture choice: **only one curve, but every refinement of it.** Edwards25519, Curve25519 (Montgomery view), Ristretto255 (prime-order quotient group eliminating cofactor confusion), and the "scalar" type living in `[u64; 4]` packed-radix-26 form for fast multiplication. Constant-time table lookups via per-byte conditional-select. 4-bit windowed scalar multiplication with precomputed point tables.

**(2) Misuse-prevention trick.** Three:
- **Ristretto255 itself is the trick.** The Edwards25519 curve has cofactor 8: there are 8 points that all encode "the same logical group element," and naive equality checks on encoded points are wrong. Ristretto255 is a quotient construction that yields a *prime-order group* with **canonical encoding**: there is exactly one encoding per group element, byte-for-byte equality on the encoding implies group equality, and decoding rejects non-canonical inputs at the `from_bytes` boundary. **The library forces you to use Ristretto255 if you want a prime-order group**, and the `RistrettoPoint::from_bytes_canonical` method *fails* on any of the 7 non-canonical encodings of the same logical point.
- **`Scalar::from_bytes_mod_order` vs `Scalar::from_canonical_bytes`** — two explicit constructors with different security postures, named after their semantics. The unsafe one isn't even named "raw"; it says exactly what it does.
- **No `Default` impl for `Scalar` or `EdwardsPoint`** — you cannot accidentally create a zero scalar or identity point and use it as a key. Construction is always explicit.

**(3) What reality can borrow.**
- **The canonical-encoding discipline.** Every reality crypto type that has a `Bytes()` and `FromBytes(...)` boundary must enforce: (a) `Bytes()` returns the unique canonical encoding; (b) `FromBytes(...)` accepts ONLY the canonical encoding and rejects all others with a typed error. This is the single highest-impact transferable principle from dalek and pre-empts 50% of the curve-implementation CVE class. Cost: ~5 LOC per type, ~20 type-tests per type in the golden file. Reality's golden-file infrastructure makes this *cheaper* than in dalek because the cross-language test vectors include the rejected non-canonical encodings as negative cases.
- **Distinct types for distinct group views.** When 057's Ed25519 lands, ship `EdwardsPoint`, `MontgomeryPoint` (for X25519), and `Ristretto255Point` as three distinct types with explicit conversion methods. The mathematical relationship (one underlying curve, three views) is documented; the type system prevents accidental cross-use.
- **Two-named-constructors discipline.** For every type that can be constructed from bytes, ship `FromBytes` (canonical, strict) and `FromBytesUnchecked` (skip validation, document footgun). Never ship a single ambiguous `FromBytes` that does best-effort. Cost: 2x the function count; benefit: every user has to make an explicit safety choice. Borrow this even for non-curve types (e.g., `Field.FromBytes` should reject non-canonical = `≥ p` representations).
- **Constant-time table-lookup pattern.** When 057's T1-EC lands with windowed scalar multiplication, the table-lookup must be branch-free across *all* table entries (read every entry, conditional-select via subtle.Choice). dalek's reference impl is the cleanest in any language; port the algorithm shape directly. ~40 LOC.

### E. Arkworks (Pratyush Mishra et al, 2020–2026)

**(1) Headline architecture.** Rust-based ZK/cryptographic library, originally for the SNARK ecosystem. Generic over `F: PrimeField`, `G: AffineCurve`, `P: Pairing`. The core trait `PrimeField` provides all field operations; concrete instantiations include BLS12-381 (`Fr`, `Fq`), BN254, MNT4/6, Pasta (Pallas/Vesta), Mersenne/Baby-Bear/Goldilocks (the "ZK-friendly" small fields). Pairings via the `PairingEngine` trait. Polynomial commitment schemes (KZG/IPA) generic over the curve. The library is the de-facto standard for ZK research code.

**(2) Misuse-prevention trick.**
- **`PrimeField` trait derives FFT support.** Implementations declare a `TWO_ADIC_ROOT_OF_UNITY` constant; any FFT/NTT routine can ask the field for its root of unity, never hardcodes it. Wrong-curve-with-wrong-root bugs become impossible.
- **`AffineCurve` vs `ProjectiveCurve` separation.** Affine points (cheap to compare, expensive to add) and projective points (cheap to add, expensive to compare/serialize) are distinct types. Conversions are explicit (`into_projective`, `into_affine`). You cannot accidentally serialize a projective point with garbage Z-coordinate.
- **`SerializationError` is a typed enum.** Compressed vs uncompressed encoding, validity-check policy, are all type-level choices.

**(3) What reality can borrow.**
- **The `PrimeField` interface as the canonical Fp shape.** Reality's future T1-FIELD should expose: `Add/Sub/Mul/Square/Inverse/Negate/Pow/Sqrt/IsZero/IsOne/FromBytes/ToBytes/ Random(seed)/TwoAdicRootOfUnity/CharBitLen`. Every ZK-friendly field is just a parameter pack of those operations. Cost: ~40 LOC of interface + per-field instantiation. Once shipped, an NTT routine generic over `PrimeField` is ~80 LOC and works across BLS-Fr / Goldilocks / Mersenne-31 / BabyBear without rewrite.
- **Affine vs Projective separation.** When T1-EC lands, ship both representations as named types with explicit conversion. Most operations live on Projective; equality and serialization live on Affine. Cost: ~60 LOC extra per curve.
- **DO borrow** the small-prime ZK-friendly field family. 057 lists Pasta (Pallas/Vesta) and Mersenne-31 / Goldilocks / BabyBear as future targets; arkworks is the reference impl for all four. Reality should land `Field32` (Mersenne-31), `Field64` (Goldilocks: `2^64 - 2^32 + 1`), `BabyBear` (`2^31 - 2^27 + 1`) as separate concrete types. Goldilocks is *especially* attractive: its modulus fits in a u64, reduction is `bits.Mul64`+three-conditional-subs (~10 LOC), and it's the standard field for Plonky2/Plonky3 ZK proofs. Cost: ~150 LOC per small field including FFT.
- **DO NOT borrow** arkworks's recent `r1cs-std` / circuit-DSL substrate. That's a ZK frontend, not a primitive library; out of reality's scope (would belong in a hypothetical `zkmark/`).
- **DO borrow** the *design pattern* of "library is generic over `F: PrimeField`"; even without Go traits, reality can express it via interface satisfaction with the interface defined narrowly enough that monomorphization-style codegen via `go generate` or simply per-field concrete code-gen yields zero-overhead specialization.

### F. BoringSSL / Mozilla NSS

**(1) Headline architecture.** BoringSSL is Google's OpenSSL fork (~600k LOC) used in Chrome/Android/gRPC. NSS is Mozilla's TLS stack (~400k LOC) used in Firefox/Thunderbird. Both are *huge protocol libraries with crypto math at the bottom*; the math is in `crypto/fipsmodule/` (BoringSSL) or `lib/freebl/` (NSS). The math layer is what reality cares about: P-curves field arithmetic, EC point ops, RSA bignum, AES, hashes — all hand-tuned assembly per architecture, with a portable C fallback.

**(2) Misuse-prevention trick.** Both rely on the "hide the unsafe primitives behind an opinion-stacked API" discipline:
- BoringSSL: low-level `BIGNUM`, `EC_POINT`, `EVP_*` APIs are *internal*; consumers are pushed to `SSL_*` (TLS) or higher-level helpers.
- NSS: PKCS#11 token abstraction means every secret is in a *handle*, never a raw byte buffer. The library cannot give you the bits of a private key without explicit unwrap, and unwrap is gated on the token's policy.
- **Both ship FIPS 140-2/3 validated modes** with self-tests on startup.

**(3) What reality can borrow.**
- **The math/protocol separation discipline.** Reality is "the math at the bottom of an SSL stack" by design. The borrow is: *don't let the math layer leak protocol assumptions* (e.g., don't bake "TLS-record-style nonce derivation" into a primitive; that's a protocol concern). Reality already does this by being opinion-free; the borrow is *don't lose this discipline as crypto primitives accrue*.
- **Hand-tuned assembly per architecture.** Out of scope for reality (zero-dep, single-Go-source). But the *interface shape* of "scalar functions with a portable fallback and per-arch acceleration" is something Go does naturally via build tags. When 057's T1-FIELD lands for P-256, the `Add` function should be implementable via build-tagged amd64 assembly later without changing the API. Borrow the *forward-compatibility shape*; defer the assembly.
- **Self-test on initialization.** Reality's golden-file tests already cover this in CI; FIPS-style runtime self-tests are out of scope. But: a single `crypto.Verify()` function that runs all golden vectors at runtime *is* in scope and is ~50 LOC over the existing test infrastructure. Useful for security-sensitive consumers.
- **DO NOT borrow** the BIGNUM / EC_POINT C-API shapes; they're optimized for hand-rolled allocation lifecycles and are alien to Go. Reality should design Go-idiomatic types and let allocation discipline come from value-type struct semantics, not malloc/free patterns.

### G. Bouncy Castle (Java/C#, 1999–2026)

**(1) Headline architecture.** Most-comprehensive open-source crypto library by raw breadth (~500 algorithms across symmetric, asymmetric, signature, KEM, PQC). Heavy use of the JCA (Java Cryptography Architecture) factory pattern: `Cipher.getInstance("AES/CBC/PKCS5Padding")`. Provider-based plugin system. Algorithm registration via OIDs (Object Identifiers).

**(2) Misuse-prevention trick.** Bouncy Castle is the *cautionary tale*, not a model:
- The string-keyed factory has a 20-year history of CVEs from typos and ambiguous strings (`AES/ECB/NoPadding` is silently insecure for most uses; the API doesn't warn).
- The provider system allows malicious-or-buggy provider injection.
- The OID-based dispatch caused multiple "wrong algorithm under the same OID" bugs.

**(3) What reality can borrow.**
- **DO NOT borrow** the factory pattern. Period.
- **DO NOT borrow** the provider/SPI architecture; it's a security smell.
- **DO borrow** the OID and standardized-algorithm-name registry as *documentation*. When reality ships ECDSA-P256, the godoc should cite "OID 1.2.840.10045.4.3.2 (ecdsa-with-SHA256)" so consumers wiring this into X.509 / CMS / JWS know what they have. Cost: ~1 doc-line per primitive.
- **DO borrow** the breadth-of-coverage *as a long-term benchmark*: Bouncy Castle has shipped almost every standardized primitive ever published. Reality's 057 Tier-1+2+3 ladder gets to maybe 30% of BC's surface; that's fine — but knowing the comparison size-orients the effort.

### H. ZK-frame primitives: Plonky3 / Halo2 / Risc0 / Sp1

**(1) Headline architecture.** Modern ZK systems (2023–2026) layered as: small-prime field (Goldilocks / Mersenne-31 / BabyBear) → polynomial ring → polynomial commitment (FRI / IPA / KZG) → constraint system (Plonkish / R1CS) → frontend DSL → zkVM (Risc0/Sp1 execute RISC-V; Plonky3 is field-and-IOP only). The math primitives reality could expose:
- **Small-prime fields**: Goldilocks `2^64 - 2^32 + 1`, Mersenne-31 `2^31 - 1`, BabyBear `2^31 - 2^27 + 1`, KoalaBear `2^31 - 2^24 + 1`. Each has FFT/NTT-friendly structure.
- **Algebraic hashes**: Poseidon2 (the 2023 successor with halved rounds), Rescue-Prime Optimized, Anemoi, Griffin, Monolith, Skyscraper. All are ZK-circuit-friendly.
- **Polynomial commitments**: FRI for STARK-friendly (no trusted setup), KZG for SNARK-friendly (trusted setup, smaller proofs), IPA for Halo2 (no trusted setup, larger proofs).
- **Lookup arguments**: Plookup, Caulk, Halo2's lookup gadget — pure polynomial-identity manipulations.

**(2) Misuse-prevention trick.** ZK primitives mostly operate on *public, deterministic* data (the proof, the public inputs); secret data is the witness which is generated locally. The misuse class is different:
- **Soundness bugs from incorrect challenge derivation** (Fiat-Shamir variants). Recent CVEs (Plonky2 Frozen-Heart 2023) came from missing public-input commitment in the transcript. The fix is *transcript discipline*: a `Transcript` object that absorbs labeled public values and squeezes challenges; type-system prevents reordering.
- **Field-mismatch in cross-curve proofs.** Recursive SNARKs use two curves with different fields; mixing them silently is a soundness break. Type-distinct `Fp` and `Fr` solves this.

**(3) What reality can borrow.**
- **Small-prime fields are the highest-leverage ZK-friendly addition.** Goldilocks `2^64 - 2^32 + 1` modular reduction is ~10 LOC of `bits.Mul64` + conditional subs, fits in a single `uint64` (no big-int needed), supports FFT of size 2^32, is the canonical Plonky2/Plonky3 field. **Reality could ship Goldilocks before any other crypto primitive at ~150 LOC including NTT — the zero-dep zero-bigint zero-curve crypto win.**
- **Mersenne-31 `2^31 - 1`** is even simpler (modular reduction is `(x & MASK) + (x >> 31)`, trivial), supports Plonky3's "Mersenne-31 with circle FFT" innovation. ~80 LOC.
- **Poseidon2 is the Tier-1 ZK-hash entry point.** The round constants are public, the S-box is x^7 (or x^5 depending on field), the MDS matrix is small. ~400 LOC for a Goldilocks instantiation. Cross-tested against Plonky3's reference vectors.
- **Transcript abstraction.** A `Transcript` type with `AbsorbField(label, value)`, `AbsorbBytes(label, bytes)`, `SqueezeField() Field`, `SqueezeBytes(n) []byte` methods backed by a duplex sponge (Keccak-f[1600] or a Poseidon variant). ~80 LOC. Pre-empts the entire Frozen-Heart / weak-Fiat-Shamir CVE class.
- **DO NOT borrow** the Plonkish constraint system itself; that's a ZK frontend (zkmark/ scope), not a primitive.

### I. Pasta curves (Pallas / Vesta) — Halo2's curve cycle

**(1) Headline architecture.** Pallas and Vesta are a "cycle of curves": each curve's base field equals the other's scalar field. This makes recursive SNARK verification cheap (you can verify a Pallas proof inside a Vesta circuit cheaply, and vice versa). Designed by Daira Hopwood for Zcash's Halo. Both are short Weierstrass with `a=0`, `b=5`, prime order. ~256-bit field.

**(2) Misuse-prevention trick.**
- The two curves are deliberately *named differently* so cross-curve mistakes are syntactically obvious (no `pasta::Curve<0>` vs `pasta::Curve<1>` confusion).
- Each curve has its own crate with its own `Fp`/`Fq` types.

**(3) What reality can borrow.**
- **Cycle-of-curves naming convention.** When reality eventually ships pairing-friendly or recursion-friendly curves, name them after their *purpose* (`Pallas`, `Vesta`) rather than enumerated indices. Anti-pattern: `Curve0` / `Curve1`. Best-practice: distinct identifiers per curve. ~free.
- **Pasta as a Tier-2 entry point** for recursive SNARK math. Reality probably defers this until consumer demand materializes; flagged for the future map.

### J. Mersenne primes for ZK (Plonky3 circle-FFT, 2024)

**(1) Headline architecture.** Plonky3's recent (2024) innovation: use `Fp = 2^31 - 1` (Mersenne-31), and instead of standard NTT (which fails because `2^31 - 1` is a Mersenne *prime* with no large 2-adic subgroup), use *circle FFT* over the Pythagorean curve `x^2 + y^2 = 1` in `Fp x Fp`. The "frequency domain" lives on this circle, which has a large 2-adic subgroup of order `2^31`.

**(2) Misuse-prevention trick.** The circle-FFT framework is mathematically subtle (cosets of the circle subgroup, anti-diagonal mappings). The Plonky3 implementation hides the math behind a `RadixDecoupledPoly` type with explicit `eval_at_point`, `low_degree_extension` methods. The user manipulates polynomials, not circle points.

**(3) What reality can borrow.**
- **Mersenne-31 itself** as a small field (~80 LOC reduction).
- **DO NOT** ship circle FFT yet — too research-frontier for a math primitive library; flagged as a Tier-3 future direction once consumer demand exists.
- **DO borrow** the lesson: *small fields can use FFT-shaped algorithms over non-obvious algebraic objects*; reality's signal/ FFT is over complex numbers, but the *algorithmic shape* (radix-2, twiddle factors, butterfly) is reusable across rings. Once T3-NTT lands per 057, factoring out the butterfly so the same code works for `Fp` (Goldilocks NTT) and `Cf` (signal FFT) avoids duplicating ~200 LOC. ~40 LOC of refactor cost up front.

---

## Cross-library architectural patterns reality should adopt

Distilling from the 10 libraries, four transferable principles:

### P1 — **Typed algebraic carriers, not byte slices.**

Borrow from: dalek, RustCrypto, arkworks. Costs ~150 LOC of type machinery (`Scalar`, `Point`, `Field`, `Commitment`, `Signature` types per primitive) at the T1 layer. Pre-empts the entire "wrong-key-type" / "wrong-curve" CVE class. **The single highest-leverage architectural decision.**

### P2 — **Canonical encoding discipline at every byte boundary.**

Borrow from: dalek, libsodium. Every `FromBytes` is strict; every `Bytes()` returns the unique canonical form; non-canonical inputs are rejected with a typed error. Cost ~5 LOC per type, mostly in golden-file negative cases.

### P3 — **Constant-time-by-construction discipline.**

Borrow from: RustCrypto (`subtle::Choice`), dalek (table lookups), BoringSSL (assembly fallbacks). Reality should:
- Re-export `crypto/subtle` primitives or wrap as `crypto/ct`.
- Document explicitly per-function which is CT and which is variable-time (verification is publicly variable-time; signing is CT).
- Use branch-free conditional-select for table lookups.

This is 057's T1-CT layer; 058's contribution is *insisting it is non-negotiable* before any secret-handling primitive ships.

### P4 — **Opinion at the function name, not at an enum/string parameter.**

Borrow from: libsodium, RustCrypto. Reject Bouncy Castle's factory pattern and Crypto++'s template-string-tag patterns. **`Ed25519Sign` is a name; `Sign(ALGO_ED25519, ...)` is a CVE.** Cost ~free; preserve the discipline as the surface grows.

### P5 — **Small ZK-friendly fields are the cheapest crypto-math win available.**

Borrow from: arkworks, Plonky3. Goldilocks (`2^64 - 2^32 + 1`) and Mersenne-31 (`2^31 - 1`) require no big-int, no curve, no constant-time discipline (public ZK-proof data, not secrets), and have well-defined NIST/IETF-shaped golden-vector references via Plonky3's reference impl. Recommend Goldilocks as the **first non-trivial crypto-adjacent primitive reality ships**, before Tier-1 signature work even starts. ~150 LOC including FFT. Cross-language: Plonky3 (Rust), Plonky2 (Rust), and the upcoming Goldilocks-aware Halo2 fork all share a canonical reference; Python (galois library) and C++ (libff/arkworks-cpp ports) already exist.

### P6 — **Transcript object for any Fiat-Shamir-style protocol.**

Borrow from: ZK-frame primitives. ~80 LOC. Pre-empts the Frozen-Heart / weak-FS CVE class. Reality won't ship a SNARK frontend, but it can ship the math primitive *transcript* type that future ZK / Schnorr / Fiat-Shamir consumers all use. Provides a canonical "commit public values, derive challenge" discipline.

---

## Architecture-level recommendations (what reality's crypto/ should look like in v1)

1. **Sub-package the crypto/ tree** as 057 forecasted (8000+ LOC of primitives don't fit in a flat 5-file package): `crypto/ct`, `crypto/field`, `crypto/curve`, `crypto/hash`, `crypto/sig`, `crypto/kdf`, `crypto/poly`, `crypto/zk`. Adopt RustCrypto's per-primitive crate-equivalent split.
2. **Ship Goldilocks `Field64` + Mersenne-31 `Field32` + Poseidon2 + Transcript first.** ~600 LOC, no big-int, no curves, no constant-time, no CSPRNG dependency, deterministic everywhere, golden-vector cross-validateable against Plonky3. **This is the cheapest "real cryptographic primitive" surface reality can ship in 1 month.** Once it lands, reality's crypto/ has gone from "non-cryptographic kit" to "ZK-grade math primitive library" with a single coherent story. The signature work (057's Sequence-A) follows in months 2-3.
3. **Adopt typed values everywhere** (P1). No `[]byte` keys, no `*big.Int` curve points. The 30%-LOC overhead pays for itself in the first CVE not shipped.
4. **Adopt canonical-encoding discipline** (P2) from day 1; it's free if done up front, expensive to retrofit.
5. **Document the constant-time policy per-function** (P3). Even if some functions remain variable-time (and they should: verification is public-data-only and benefits from variable-time speedups), document which is which. Reality's CLAUDE.md §5 "precision documented, not assumed" extends naturally to "side-channel posture documented, not assumed."

---

## Per-library borrow-summary table

| Library | Architecture borrow | Misuse-prevention borrow | NOT borrow |
|---|---|---|---|
| libsodium | Length constants as discoverable surface | Opinion at function-name level | High-level box/secretbox protocol layer |
| RustCrypto | Per-curve typed families | `subtle::Choice` discipline; typed CtOption | Trait-based generic substrate (use Go generics narrowly) |
| Crypto++ | Generic `HMAC[H]`-style type-parameterized config | Compile-time policy compose | Template-metaprogramming substrate; pipeline operator overloading |
| dalek | Affine/Projective/canonical-encoding split | Two-named-constructors (`FromBytes`/`FromBytesUnchecked`); Ristretto255 prime-order quotient | Edwards25519-only mono-curve focus (reality covers more) |
| arkworks | `PrimeField` interface; small ZK-friendly fields | Affine-vs-Projective type separation | r1cs-std circuit DSL (out of reality scope) |
| BoringSSL/NSS | Math/protocol layer separation; build-tagged assembly forward-compat shape | Self-test on init (optional) | BIGNUM/EC_POINT C-API shapes; PKCS#11 handle abstraction |
| Bouncy Castle | OID/standardized-name doc registry | (none — cautionary) | Factory pattern; provider/SPI architecture |
| Plonky3/Halo2/Risc0/Sp1 | Goldilocks/Mersenne-31 small fields; Poseidon2; Transcript | Type-distinct `Fp`/`Fr` for cross-curve proofs | Plonkish constraint system (zkmark/ scope) |
| Pasta (Pallas/Vesta) | Cycle-of-curves naming (purpose-named, not indexed) | Distinct types per curve in cycle | Recursive SNARK frontend |
| Mersenne-31 / Plonky3 circle-FFT | Mersenne-31 field; algorithmic-shape factoring (NTT/FFT shared butterfly) | (none specific) | Circle-FFT framework itself (research frontier) |

---

## Engineering-elegance ranking (058's editorial)

dalek > arkworks > RustCrypto > libsodium > Plonky3 > Crypto++ > BoringSSL > NSS > Bouncy Castle.

dalek wins for the *most* careful side-channel discipline at the *smallest* surface area. arkworks wins for the cleanest abstraction over a vast surface. RustCrypto wins for the most disciplined trait/type ecosystem. libsodium wins for the most opinion-stacked user-facing surface. Reality should aim for *dalek's careful arithmetic + arkworks's typed-field abstraction + libsodium's named-function discipline + RustCrypto's constant-time-by-type primitives*, in that priority order.

---

## Non-overlap with sibling agents

- **056 (numerics):** owned correctness of present functions + the package-name-vs-contents issue. 058 inherits the "package will accumulate real crypto" premise but adds *zero* primitive correctness analysis.
- **057 (missing):** owned the primitive-by-primitive map and Tier-1/2/3 ladder. 058 **never duplicates** the LOC estimates; 058 covers *how* primitives should be structured (typed values, canonical encoding, CT discipline, transcript), not *which* primitives to ship.
- **Future API agent:** ergonomics, naming, error contracts of the eventual crypto API. 058 surfaces *architectural* patterns (P1-P6); the future API agent owns the Go-signature-shape choices that fall out of those patterns.
- **Future perf agent:** assembly fallbacks, hand-tuned field arithmetic, MSM Pippenger window-size tuning, pairing batching. 058 flags forward-compatibility shape (build-tagged amd64 hooks) but defers all perf detail.

Report length: 380 lines (under 400 cap).
