# 322 — dive-aes-vs-poly1305 (AES / ChaCha20 / Poly1305 / GHASH / Hashes audit)

## Headline

reality v0.10.0 ships **zero symmetric-cipher primitives and zero cryptographic hashes**: `crypto/hash.go` exposes only the three non-cryptographic hashes FNV1a32/FNV1a64/MurmurHash3_32 + ConsistentHash + the bespoke SituationHashWithStructure (`crypto/hash.go:25-204`); there is no AES, ChaCha20, Poly1305, GHASH, Salsa20, HMAC, SHA-1, SHA-2, SHA-3/Keccak, BLAKE2, or BLAKE3 anywhere — repo-wide grep on those tokens returns hits only in review markdown (056/057/058/060, 175, 211, 304, 320, 321) and a `[32]byte CorpusSHA` shape parameter in `zkmark/zkmark.go:114` that the package itself never computes (the SHA is callee-injected). Reality also does not import a single `crypto/sha*`, `crypto/aes`, `crypto/cipher`, `crypto/hmac`, or `golang.org/x/crypto/*` symbol anywhere (verified: zero matches over `**/*.go`). The slot-057 scope filter explicitly puts AES/ChaCha20 byte-shuffling OUT and the algebraic substrates GHASH·GF(2^128), Poly1305·mod(2^130-5), Keccak-f[1600] sponge, AES S-box·GF(2^8) IN — but none of the four IN-scope algebraic objects ship today either. **The cheapest day-1 win is not a primitive at all: it is a 30-line CLAUDE.md amendment that codifies the "stdlib defers to `crypto/sha256` + `golang.org/x/crypto/{chacha20,poly1305,blake2b,sha3}` for streaming/keyed crypto; reality owns only the algebraic substrates" rule, then a single Tier-0 R-MUTUAL-CROSS-VALIDATION 3/3 pin landing SHA-256 (FIPS 180-4) as a from-scratch reference impl golden-validated against `crypto/sha256.Sum256` (~280 LOC + 30 vectors) — saturating cross-language hash determinism in one PR and unblocking zkmark's Fiat-Shamir transform.**

## Findings

- **F1 — Repo-wide symmetric-crypto surface = zero.** Grep `(?i)AES|ChaCha|Poly1305|GHASH|HMAC|SHA-?(1|256|512|3)|Keccak|BLAKE|Salsa|Rijndael` returns 41 files, **all** in review docs except `zkmark/zkmark.go:111-114` (`CorpusSHA [32]byte` declared as a parameter shape with the comment "echoes the 32-byte corpus SHA the proof was generated against" — caller computes it, zkmark never does) and `zkmark/zkmark_test.go:44-45` (test-only stub `copy(corpusSHA[:], []byte("0123…"))`). The string "SHA" in `crypto/` exists nowhere outside test names like `TestSituationHashWithStructure_*` (`crypto/structural_hash_test.go:9-99`).
- **F2 — `crypto/hash.go` is non-cryptographic-only.** `crypto/hash.go:25-46` FNV-1a 32/64 (Fowler-Noll-Vo 1991, hash-table grade — trivial offline collision construction); `:61-110` MurmurHash3_32 x86 variant (Appleby 2011, seeded keyed hash but NOT a MAC — known HashDoS attacks since 2012); `:129-144` Lamping-Veach 2014 jump-consistent hash (sharding only, not a checksum); `:171-204` `SituationHashWithStructure` + `StructuralDescriptor` (FNV1a64 XOR-fold for observation fingerprinting). None is collision-resistant in the cryptographic sense.
- **F3 — No CSPRNG, by 304's diagnosis.** `crypto/rng.go:16-192` ships MT19937-64, PCG-XSH-RR, xoshiro256\*\*, splitmix64 — all *deterministic statistical* PRNGs, all explicitly NOT cryptographic. ARCHITECTURE.md `:54` references a stale `XorShift64` that no longer exists. The package is named `crypto` and contains zero cryptographic generators. ChaCha20-as-CSPRNG (RFC 7539 stream-cipher mode) is the natural addition and is **already-shipped in Go's `golang.org/x/crypto/chacha20`**.
- **F4 — No stdlib-crypto imports anywhere.** Grep `crypto/sha|crypto/aes|crypto/cipher|crypto/hmac|x/crypto|crypto/rand` over `**/*.go` returns **zero hits** (verified). Reality is currently a 100% from-scratch math library in the literal sense — but at the cost of having no cryptographic guarantees of any kind.
- **F5 — Slot-057 scope decision (already authoritative).** The 057-crypto-missing review §"Scope filter" already adjudicated:
  - **OUT**: AES round transforms (S-box / ShiftRows / MixColumns) at the byte-shuffle level, AEAD modes (GCM/CCM/SIV/OCB/ChaCha20-Poly1305 composition), TLS handshake state, X.509 serialization, OS-entropy CSPRNG seeding.
  - **IN**: GF(2^8) inverse + affine transform that *defines* the AES S-box (algebraic content); GHASH = GF(2^128) multiplication mod `x^128 + x^7 + x^2 + x + 1`; Poly1305 = polynomial evaluation mod `2^130 − 5` at a key-derived point; Keccak-f[1600] permutation = θ/ρ/π/χ/ι rounds on a 5×5×64 lane lattice (mostly GF(2)-linear maps + one nonlinear χ step); ChaCha20 quarter-round (ARX over 32-bit words — borderline).
  - **IN as primitive even though "byte-level"**: SHA-256/SHA-512 round functions (FIPS 180-4 Ch/Maj/Σ/σ over 32/64-bit words — math, ~280/300 LOC); HMAC = `H((K⊕opad) ‖ H((K⊕ipad) ‖ M))` (RFC 2104, 30 LOC over a hash); HKDF = HMAC iteration (RFC 5869, 40 LOC over HMAC).
  - 057's recommended Tier-1 anchor: `T1-HASH` = SHA-256 + SHA-512 + HMAC + HKDF + SHA-3/Keccak-f[1600] + BLAKE2b/BLAKE3, ~600 LOC.
- **F6 — RFC anchor map (sources for any future PR).**
  - **AES**: Daemen-Rijmen *Rijndael* (1998 AES submission) → FIPS 197 (NIST 2001-11-26). Block size 128, key 128/192/256, 10/12/14 rounds. S-box = (multiplicative inverse in GF(2^8) with poly `0x11d`) ∘ affine transform. MixColumns = matrix mul over GF(2^8). Standard reference implementations: `Go crypto/aes` (constant-time, table-based + AES-NI fallback), Bernstein-Schwabe bitsliced, `BearSSL` ct-table-free.
  - **ChaCha20**: Bernstein 2008 ("ChaCha, a variant of Salsa20") → RFC 7539 (Nir-Langley 2015) → RFC 8439 (2018, supersedes 7539, identical algorithm). 20-round ARX permutation on 16×uint32 state, 256-bit key, 96-bit nonce, 32-bit block counter. Quarter-round = `a += b; d ^= a; d = ROTL(d, 16); c += d; b ^= c; b = ROTL(b, 12); a += b; d ^= a; d = ROTL(d, 8); c += d; b ^= c; b = ROTL(b, 7)`.
  - **Poly1305**: Bernstein 2005 ("The Poly1305-AES message-authentication code") → RFC 7539 §2.5 → RFC 8439 §2.5. One-time MAC: clamp 256-bit key to (r, s) where r is masked to 22 bytes with the famous mask `0x0ffffffc0ffffffc0ffffffc0fffffff`, evaluate polynomial `(c_1 r^q + c_2 r^{q-1} + … + c_q r^1) mod (2^130 − 5)`, add `s mod 2^128`. **Pure modular arithmetic in F_p with p = 2^130 − 5** — algebraically the cleanest AEAD MAC ever shipped. ~80 LOC of the 130-bit modular ops + clamping.
  - **GHASH**: NIST SP 800-38D §6.4 (McGrew-Viega 2007 GCM mode). MAC built on multiplication in GF(2^128) = GF(2)[X] / (x^128 + x^7 + x^2 + x + 1). Hardware accel via PCLMULQDQ (Intel 2008, Gueron-Kounavis 2010). The single field-mul is the entire spec. ~60 LOC of carryless multiply + modular reduction.
  - **HMAC**: Krawczyk-Bellare-Canetti, RFC 2104 (1997). `HMAC(K,M) = H((K' ⊕ opad) ‖ H((K' ⊕ ipad) ‖ M))` with `opad = 0x5C × blocksize`, `ipad = 0x36 × blocksize`. ~30 LOC over any cryptographic hash. Used everywhere: TLS, IPsec, JWT, RFC 6979 deterministic-k ECDSA (already in 057's Tier-1).
  - **SHA-2 (SHA-256, SHA-512)**: NIST FIPS 180-4 (2015). Merkle-Damgård + Davies-Meyer compression. SHA-256 = 64 rounds × {Ch, Maj, Σ_0, Σ_1, σ_0, σ_1} over 32-bit words, 8×32-bit IV. SHA-512 = 80 rounds, 64-bit words. **CAVS test vectors** = >1000 vectors per algorithm, the gold standard for cross-language pinning.
  - **SHA-3 (Keccak)**: NIST FIPS 202 (2015), Bertoni-Daemen-Peeters-Van-Assche 2007. Sponge construction over 1600-bit state (5×5 lanes × 64 bits), 24 rounds × {θ, ρ, π, χ, ι}. **Different from SHA-2** — sponge, not Merkle-Damgård. Resists length-extension attacks SHA-2 is vulnerable to. Available in Go via `golang.org/x/crypto/sha3` (added to stdlib `crypto/sha3` in Go 1.21).
  - **BLAKE2**: Aumasson-Neves-Wilcox-O'Hearn-Winnerlein 2013, RFC 7693 (2015). ChaCha-style mixing (G function = 4 ARX rounds), `BLAKE2b` 64-bit words, 12 rounds, up to 64-byte output, optional keyed mode (built-in MAC, no HMAC needed).
  - **BLAKE3**: O'Connor-Aumasson-Neves-Wilcox-O'Hearn 2020. Reduced from 12→7 rounds, Merkle-tree-internal (parallelizable), single-mode for hash/MAC/KDF/XOF. Fastest software hash on commodity CPUs. No NIST/RFC standardization yet (2026); Cloudflare-maintained reference at github.com/BLAKE3-team/BLAKE3.
- **F7 — Cross-language alignment.** All seven primitives above have **bit-exact reference vectors** that every conformant impl must reproduce:
  - SHA-2 / SHA-3: NIST CAVS (Cryptographic Algorithm Validation Suite) — thousands of vectors, free download. **Tolerance = 0** (exact bytes). Go canonical = `crypto/sha256.Sum256` and `golang.org/x/crypto/sha3.Sum256`. Python: `hashlib.sha256` / `hashlib.sha3_256`. C++: `OpenSSL EVP_sha256`. C#: `System.Security.Cryptography.SHA256`. **All four trivially agree** on every byte.
  - HMAC: RFC 2104 §A and RFC 4231 (HMAC-SHA-2 vectors).
  - ChaCha20 / Poly1305: RFC 7539 §A.1-A.4 and RFC 8439 §2.6-2.8 ship official test vectors.
  - GHASH: NIST SP 800-38D §B.1 ships AES-GCM vectors that include intermediate GHASH state.
  - BLAKE2 / BLAKE3: official `kat` (known-answer-test) vectors maintained by the spec authors.
- **F8 — Slot 304 already filed the CSPRNG gap.** 304 §recommendation 4 (`crypto/csprng.go`, ~150 LOC ChaCha20-PRNG per RFC 7539). Slot 320 / 321 cover the GF(2^m) substrate that AES MixColumns + GHASH share. Slot 057 §T3-SYMMETRIC-MATH (~250 LOC) bundles Poly1305 + GHASH + ChaCha20 quarter-round. **All three slots agree** that the *math layer* of these primitives is in-scope; the *cipher streaming* layer is not. This dive does not change that conclusion — it concretizes it.
- **F9 — Why "defer to stdlib" is the right answer for AES specifically.** `crypto/aes` in Go stdlib is (i) constant-time (CT-table-free fallback + AES-NI on amd64/arm64 with hardware extensions), (ii) FIPS-validated by Go release engineering, (iii) 100% NIST-AESAVS conformant. Reimplementing AES from first principles per CLAUDE.md design rule 6 produces, by definition, a non-FIPS-validated impl that has to track Go's constant-time discipline manually. **Net cost-benefit**: ~600 LOC of AES + 600 LOC of GCM/CTR/CBC modes + a constant-time audit + AES-NI assembly fallback, vs. one `import "crypto/aes"` line. CLAUDE.md design rule 6 ("Reimplement from first principles. Do not wrap existing libraries.") was written for *math* primitives (Cooley-Tukey, Tonelli-Shanks, Karatsuba) where the algorithm IS the artifact. AES is an *engineering* primitive where the standard's value is its non-malleability across implementations. **Recommend: amend rule 6 to exclude crypto-engineering primitives explicitly.** Reality ships the GF(2^8) S-box as algebra (per slot 057 IN-scope); does not ship an AES streaming impl.
- **F10 — Why HMAC/SHA-256 should still be reimplemented.** Counter-argument to F9: HMAC and SHA-256 are *small* (300 LOC and 30 LOC respectively), have *thousands* of CAVS test vectors (cross-language pinning is trivial), and are *load-bearing for zkmark/Fiat-Shamir/Merkle/RFC-6979/HKDF*. The cost of from-scratch SHA-256 is ~280 LOC + 30 vectors; the benefit is **R-MUTUAL-CROSS-VALIDATION 3/3 saturation in one PR** because Go canonical can be golden-tested against `crypto/sha256.Sum256` directly (1-line `assert.Equal`), Python against `hashlib.sha256`, C++ against OpenSSL EVP, C# against `SHA256`. Four witnesses, all stdlib, all bit-exact — this is the cleanest hash R-MUTUAL pin in the entire library map.
- **F11 — Consumer impact map.**
  - `zkmark/`: declares `CorpusSHA [32]byte` (`zkmark.go:111-114`) and the test stub fakes `corpusSHA = "0123…"` (`zkmark_test.go:44-45`). The package literally cannot compute the SHA itself today. Any real Fiat-Shamir transform requires reality (or its caller) to have a real SHA-256. **Highest-priority consumer.**
  - **Pistachio** (foundation asset checksum, per CLAUDE.md context): if it computes SHA over assets to verify integrity, it currently must do so itself. A reality-shipped SHA-256 with golden vectors is the first cross-language artifact-checksum primitive.
  - **golden-file integrity**: CLAUDE.md mandates JSON test vectors shared across Go/Python/C++/C#. Any of those JSONs being tampered with is currently undetectable from inside reality. A keyed BLAKE2b (built-in MAC mode) on each `testdata/*.json` would fix this in <50 LOC at testutil layer.
  - **aicore** (per dependency position `Consumer Apps -> Services -> AI (aicore) -> reality -> math stdlib`): if aicore ever needs a deterministic content-addressable store for prompts/contexts, it needs SHA-256 and currently has to reach to its own crypto layer.
- **F12 — R-MUTUAL-CROSS-VALIDATION 3/3 ranking.** Of the seven primitives in F6, the *cheapest* ones to saturate the 3/3 cross-language pin (Go-canonical + Python port + C++ port, with C# as bonus 4th axis) are, in order:
  1. **SHA-256** — 280 LOC, 30 NIST CAVS vectors, exact-bytes tolerance, 4 stdlib witnesses (Go `crypto/sha256`, Python `hashlib`, OpenSSL, .NET) all available for zero-effort validation. **Recommended Tier-0 win.**
  2. **HMAC-SHA-256** — 30 LOC over SHA-256, RFC 4231 vectors, same 4 stdlib witnesses.
  3. **BLAKE2b** — 200 LOC, RFC 7693 KAT vectors, exact-bytes tolerance. Bonus: built-in keyed MAC mode means the testdata-integrity use case (F11) drops to 0 extra LOC.
  4. **SHA-3 / Keccak-f[1600]** — 350 LOC, FIPS 202 vectors, sponge construction is genuinely *math* (5×5×64 GF(2)-linear maps), good fit for `crypto/`.
  5. **Poly1305** — 80 LOC, RFC 8439 vectors, cleanest *algebraic* AEAD MAC (mod 2^130-5).
  6. **GHASH** — 60 LOC over GF(2^128) (which slot 321 already scoped). Free once 321 lands.
  7. **ChaCha20-block** — 100 LOC ARX permutation, RFC 8439 vectors. Useful as CSPRNG core (closes 304's gap).

## Concrete recommendations

1. **CLAUDE.md `:114-117`** — amend Quick Reference + design rules. Add a new §"What we don't ship" subsection with: "Reality does NOT ship: AES block cipher (use `crypto/aes`), AEAD modes (use `crypto/cipher`), TLS/Noise/QUIC handshakes, X.509/PKCS serialization, OS-entropy CSPRNG seeding (use `crypto/rand`). Reality DOES ship the algebraic substrates these protocols compose: GF(2^8) inverse + affine (AES S-box), GF(2^128) multiplication (GHASH), F_{2^130-5} polynomial evaluation (Poly1305), Keccak-f[1600] permutation, ARX quarter-round (ChaCha20-block), and the cryptographic hashes SHA-256/SHA-512/SHA-3/BLAKE2b/BLAKE3 + HMAC/HKDF as math primitives with golden-file validation against NIST CAVS / RFC test vectors." ~30 LOC of CLAUDE.md edits. **Cheapest possible PR. Ship first.**
2. **CLAUDE.md design rule 6 — narrow scope.** Current text reads "Reimplement from first principles. Do not wrap existing libraries." Amend to: "Reimplement *math* primitives from first principles (FFT, Tonelli-Shanks, Karatsuba, etc.). Cryptographic-engineering primitives where standardization itself is the artifact (AES rounds, AES-GCM, AES-CTR, AEAD-mode composition) MAY defer to Go's `crypto/*` and `golang.org/x/crypto/*` stdlib; reality ships only the *algebraic* substrates of those primitives, validated against the same RFC/NIST vectors the stdlib targets." ~10 LOC edit.
3. **`crypto/sha2.go` (NEW, ~280 LOC) + `testdata/crypto/sha256.json` (≥30 vectors).** Implement SHA-256 (FIPS 180-4) from scratch. Padding, length-encoding, 64-round Davies-Meyer compression, 8×32-bit IV `0x6a09e667…0x5be0cd19`, K constants table, Ch/Maj/Σ_0/Σ_1/σ_0/σ_1. Validate every Go test against `crypto/sha256.Sum256` (R-MUTUAL 3/3 pin axis 2 = stdlib bit-equality). Add 30 NIST CAVS test vectors as golden JSON: empty string, "abc", `'a' × 1_000_000`, RFC 4634 §8 messages, all 256 single-byte messages. Tolerance 0. **R-MUTUAL-CROSS-VALIDATION 3/3 saturated in one PR** because Python/C++/C# stdlib all reproduce the bytes exactly. SHA-512 is a 1-day follow-up (~300 LOC, same shape, 64-bit words).
4. **`crypto/hmac.go` (NEW, ~30 LOC).** Generic `HMAC(h func()hash.Hash, key, msg []byte) []byte` over the `hash.Hash` interface (or, to avoid stdlib dep, a `crypto.Hash` interface declared inline with `Reset() / Write([]byte) / Sum([]byte) []byte / BlockSize() int / Size() int`). RFC 2104 + RFC 4231 vectors. Unblocks RFC 6979 deterministic-k ECDSA (slot 057 T1-ECDSA), HKDF, JWT-equivalent constructions.
5. **`crypto/blake2b.go` (NEW, ~200 LOC) + `testdata/crypto/blake2b.json`.** BLAKE2b-512 with optional keyed mode (RFC 7693). Pin against `golang.org/x/crypto/blake2b.Sum512` for R-MUTUAL axis 2. *Use the keyed mode immediately* for `testutil/integrity.go` to MAC every `testdata/**/*.json` (F11).
6. **`crypto/sha3.go` (NEW, ~350 LOC).** Keccak-f[1600] permutation (5×5×64-lane state, 24 rounds × {θ, ρ, π, χ, ι}). SHA3-256 / SHA3-512 / SHAKE128 / SHAKE256 wrappers. FIPS 202 vectors. Genuinely *math* primitive (one nonlinear step χ, all else GF(2)-linear) and a fit candidate for the `crypto/` package's mathematical character. Pin against `golang.org/x/crypto/sha3` (or stdlib `crypto/sha3` on Go ≥1.21).
7. **`crypto/poly1305.go` (NEW, ~80 LOC) + golden vectors.** Pure F_{2^130-5} polynomial evaluation. **Algebraic primitive only — does not include the AEAD-mode composition** (that's protocol-layer, OUT per slot 057). Vectors from RFC 8439 §2.5.2 ("Cryptographic" example). Slot 057 T3-SYMMETRIC-MATH-Poly1305 line item. Useful as standalone one-time MAC.
8. **`crypto/ghash.go` (NEW, ~60 LOC) — defer until slot 321 GF(2^128) lands.** Once `coding/galois/gf2m.go` (slot 320 keystone C3) ships GF(2^128) multiplication mod `x^128 + x^7 + x^2 + x + 1`, GHASH = repeated `{Y_i = (Y_{i-1} ⊕ X_i) · H}`. RFC NIST SP 800-38D §6.4. ~60 LOC over the GF(2^128) substrate. Vectors from SP 800-38D §B.1.
9. **`crypto/chacha20_block.go` (NEW, ~100 LOC).** Just the 20-round permutation on the 16×uint32 state. **Not** the streaming XOR (that's protocol-layer). Becomes the core of slot 304's missing CSPRNG — wrap with a counter and you have RFC 7539 keystream generation for `crypto.NewChaCha20PRNG`. Vectors from RFC 8439 §2.3.2.
10. **`crypto/aes_sbox.go` (NEW, ~80 LOC) — algebraic S-box only.** Define `func AESSbox(b byte) byte = affineTransform(GF256Inverse(b))` where `GF256Inverse` uses slot 320's `Galois{Modulus: 0x11d}` Exp/Log table. Ship this as an *algebraic curiosity* (FIPS 197 §5.1.1) — not a cipher. Verifies against `crypto/aes` indirectly (encrypt one block of zeros under all-zero key, check round-1 state). **Tracks the slot 057 IN-scope decision** that "the GF(2^8) inverse + affine transform that DEFINES the S-box IS in-scope as algebraic content."
11. **DO NOT ship**: full AES round function, AES-CTR/CBC/GCM modes, ChaCha20 streaming XOR, ChaCha20-Poly1305 AEAD composition, AES-GCM, Salsa20, RC4, DES/3DES, IDEA, Blowfish/Twofish, Serpent, Camellia, ARIA, GOST. **Defer to `crypto/aes` + `crypto/cipher` + `golang.org/x/crypto/chacha20poly1305`.** Document this explicitly in CLAUDE.md per recommendation 1.
12. **`testutil/integrity.go` (NEW, ~50 LOC)** — once BLAKE2b ships (rec 5), MAC every `testdata/**/*.json` and store the MACs in a single `testdata/_macs.json` file with the BLAKE2b key derived from a static reality-version constant. Detects testdata corruption / tampering across the 4 language ports. Free benefit of having BLAKE2b's keyed mode.
13. **R-MUTUAL-CROSS-VALIDATION 3/3 pin opportunities (this dive's contribution to the saturation map):**
    - **HASH-SHA256**: 4 axes (Go canonical, Go stdlib `crypto/sha256.Sum256`, Python `hashlib.sha256`, C++ OpenSSL `EVP_sha256`). Bonus 5th = C# `SHA256`. Tolerance 0. **Saturates in one PR (recommendation 3).**
    - **HASH-SHA512**: same 4 axes, same tolerance, ~300 LOC follow-up.
    - **HASH-SHA3-256 / SHAKE-256**: 4 axes (reality, `golang.org/x/crypto/sha3` (stdlib `crypto/sha3` since Go 1.21), `hashlib.sha3_256`, OpenSSL EVP_sha3_256, C# SHA3 (.NET 8+)).
    - **MAC-HMAC-SHA256**: trivially follows SHA-256.
    - **MAC-POLY1305**: 4 axes (reality, `golang.org/x/crypto/poly1305`, `cryptography.hazmat.primitives.poly1305`, OpenSSL `EVP_MAC` Poly1305).
    - **CIPHER-CHACHA20-BLOCK** (just the permutation): 4 axes (reality, `golang.org/x/crypto/chacha20`, `cryptography.hazmat.primitives.ciphers.ChaCha20`, OpenSSL `EVP_CIPHER_CTX_ctrl(EVP_CTRL_CHACHA20_*)`).
    - **PERMUTATION-AES-SBOX**: 257 vectors (all bytes 0x00…0xFF + identity check) cross-validated against `crypto/aes` round-key-zero one-block encrypt (extracts post-S-box state). Tolerance 0.
    - **All seven** are exact-bytes (tolerance 0), all have stdlib witnesses in 4 languages, all ship with NIST/RFC official test vectors. **The crypto-hash track is the densest R-MUTUAL 3/3 surface in the entire reality library map** — every vector pin is a 4-axis saturation simultaneously, because crypto stdlib parity is universal.
14. **Update `ARCHITECTURE.md`** — replace the stale `XorShift64` reference at `:54` (per slot 304 rec 13). After recommendations 3-9 land, also document the new "Reality `crypto/` is *algebra-of-cryptography*, not *cryptography-of-protocols*" framing.
15. **Issue map for the slot-322 PR sequence (cheapest first):**
    - PR-322a: CLAUDE.md + ARCHITECTURE.md amendments (recs 1, 2, 14). ~50 LOC docs. **Land today; unblocks the rest of the conversation.**
    - PR-322b: SHA-256 + 30 CAVS vectors (rec 3). ~280 LOC + golden JSON. **Single highest-leverage primitive.**
    - PR-322c: HMAC + RFC 4231 vectors (rec 4). ~30 LOC. **Unblocks RFC 6979 ECDSA in slot 057.**
    - PR-322d: BLAKE2b + integrity MAC (recs 5, 12). ~250 LOC.
    - PR-322e: SHA-3/Keccak-f (rec 6). ~350 LOC. Most genuinely *math* of the bunch.
    - PR-322f: Poly1305 (rec 7). ~80 LOC. Algebraic AEAD MAC.
    - PR-322g: ChaCha20-block + tie into slot 304 CSPRNG (rec 9). ~100 LOC.
    - PR-322h (after slot 320/321 ship): GHASH + AES-S-box (recs 8, 10). ~140 LOC over GF(2^m) substrate.
    - **Total: ~1,280 LOC of from-scratch crypto, all R-MUTUAL 3/3-pinned, all NIST/RFC-validated, zero stdlib runtime deps.** AES streaming + AEAD modes remain explicit consumer-imports stdlib (per rec 11).

## Sources

- **Repo files**:
  - `C:/limitless/foundation/reality/crypto/hash.go:25-204` — FNV-1a 32/64, MurmurHash3_32, ConsistentHash, SituationHashWithStructure (non-cryptographic only)
  - `C:/limitless/foundation/reality/crypto/rng.go:16-192` — MT19937-64, PCG-XSH-RR, xoshiro256\*\* (no CSPRNG)
  - `C:/limitless/foundation/reality/crypto/modular.go:20-135` — ModPow, ModInverse, ChineseRemainder (uint64 prime-field)
  - `C:/limitless/foundation/reality/crypto/prime.go:26-264` — Miller-Rabin, prime factorisation, GCD/LCM/ExtendedGCD
  - `C:/limitless/foundation/reality/zkmark/zkmark.go:111-114` — `CorpusSHA [32]byte` parameter shape only (callee computes)
  - `C:/limitless/foundation/reality/CLAUDE.md` — design rules 1, 2, 6 (golden files, zero deps, reimplement from first principles)
  - `C:/limitless/foundation/reality/reviews/overnight-400/agents/056-crypto-numerics.md` — package-name-vs-contents trap diagnosis
  - `C:/limitless/foundation/reality/reviews/overnight-400/agents/057-crypto-missing.md` — scope filter (lines 7-14), Tier-1 T1-HASH (lines 50-57), Tier-3 T3-SYMMETRIC-MATH (lines 180-186), DO-NOT-add list (lines 244-254)
  - `C:/limitless/foundation/reality/reviews/overnight-400/agents/058-crypto-sota.md` — libsodium / RustCrypto / dalek / arkworks architecture comparison
  - `C:/limitless/foundation/reality/reviews/overnight-400/agents/175-synergy-zkmark-crypto.md` — zkmark × crypto dependency map (zkmark has zero crypto today)
  - `C:/limitless/foundation/reality/reviews/overnight-400/agents/304-dive-rng-quality.md` — PRNG audit, ChaCha20-CSPRNG gap (lines 4, 35)
  - `C:/limitless/foundation/reality/reviews/overnight-400/agents/320-dive-error-correction.md` — GF(2^8) keystone for AES MixColumns
  - `C:/limitless/foundation/reality/reviews/overnight-400/agents/321-dive-finite-field.md` — GF(2^m) Mul/Inv/Sqrt audit, Itoh-Tsujii, Rabin/Ben-Or irreducibility
- **Standards / RFCs**:
  - FIPS 197 (2001) — AES / Rijndael (Daemen-Rijmen)
  - FIPS 180-4 (2015) — SHA-1 / SHA-2 family (NIST)
  - FIPS 202 (2015) — SHA-3 / Keccak (Bertoni-Daemen-Peeters-Van-Assche)
  - NIST SP 800-38D (2007) — GCM/GMAC; GHASH at §6.4 + test vectors at §B.1 (McGrew-Viega)
  - RFC 2104 (1997) — HMAC (Krawczyk-Bellare-Canetti)
  - RFC 4231 (2005) — HMAC-SHA-2 test vectors
  - RFC 5869 (2010) — HKDF (Krawczyk)
  - RFC 7539 (2015) → RFC 8439 (2018) — ChaCha20-Poly1305 (Nir-Langley); ChaCha20 §2.3, Poly1305 §2.5, AEAD §2.8
  - RFC 7693 (2015) — BLAKE2 (Aumasson-Neves-Wilcox-O'Hearn-Winnerlein)
  - RFC 6979 (2013) — Deterministic-k DSA / ECDSA (Pornin)
  - BIP-340 (2020) — Schnorr signatures over secp256k1 (Wuille-Nick-Ruffing)
- **Academic / vendor sources**:
  - Bernstein, "ChaCha, a variant of Salsa20" (2008-01-28)
  - Bernstein, "The Poly1305-AES message-authentication code" (FSE 2005)
  - Daemen-Rijmen, "AES Proposal: Rijndael" (1998 NIST AES submission); "The Design of Rijndael" Springer 2002
  - Aumasson et al., "BLAKE2: simpler, smaller, fast as MD5" (ACNS 2013)
  - O'Connor-Aumasson-Neves-Wilcox-O'Hearn, "BLAKE3: one function, fast everywhere" (2020)
  - Gueron-Kounavis, "Intel® Carry-Less Multiplication Instruction and its Usage for Computing the GCM Mode" (Intel white paper, 2010)
  - Käsper-Schwabe, "Faster and Timing-Attack Resistant AES-GCM" (CHES 2009)
  - NIST CAVS (Cryptographic Algorithm Validation Suite) test vector repository — csrc.nist.gov/projects/cryptographic-algorithm-validation-program
- **Library references for cross-language R-MUTUAL pinning**:
  - Go stdlib `crypto/sha256`, `crypto/sha512`, `crypto/aes`, `crypto/cipher`, `crypto/hmac` (Go ≥1.0)
  - Go stdlib `crypto/sha3` (Go ≥1.21); `golang.org/x/crypto/{chacha20,poly1305,blake2b,blake2s,sha3}` (all versions)
  - Python `hashlib` (SHA-2/3, BLAKE2 since 3.6), `cryptography.hazmat.primitives.{hashes,hmac,poly1305,ciphers}`
  - C/C++ OpenSSL `EVP_*` family (canonical reference for Poly1305, AES-GCM, ChaCha20)
  - .NET `System.Security.Cryptography.{SHA256,SHA512,SHA3_256,HMACSHA256,AesGcm,ChaCha20Poly1305}` (.NET 6 / 8)
  - libsodium (Frank Denis) — primary reference for ChaCha20-Poly1305 and BLAKE2b conformance

Report length: 232 lines (under 400 cap).
