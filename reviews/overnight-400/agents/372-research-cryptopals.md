# 372 — research-cryptopals (crypto competitions: what wins, why, lessons for reality)

## Headline
Across 30 years of open crypto competitions, simple algebraic permutations with strong bitsliceability and constant-time-by-construction (Rijndael, Keccak, ChaCha20, Ascon, ML-KEM/Kyber NTT) consistently beat clever ad-hoc designs — reality's `crypto` package should mirror this: prefer round-function permutations, NTT-style structured arithmetic, and zero data-dependent branches/memory access.

## Survey

### 1. AES (NIST 1997–2001) — winner: Rijndael
Open call Sept 1997; 15 submissions → 5 finalists (MARS, RC6, Rijndael, Serpent, Twofish). Rijndael (Daemen & Rijmen, Belgium) selected Oct 2000, FIPS 197 published Nov 2001. Won on the rare quadrant of "best in software AND hardware AND constrained AND smartcard," with low memory and uniform round structure (SubBytes/ShiftRows/MixColumns/AddRoundKey). Decisive design lesson: a single algebraic S-box (inversion in GF(2^8)) plus an MDS-style linear layer gives provable diffusion bounds and bitsliceable software. Serpent had bigger margin but lost on speed; Twofish lost on simplicity. **Lesson for reality:** clean field-arithmetic decomposition beats hand-tuned heuristics; one well-analyzed primitive > many ad-hoc tricks.

### 2. NESSIE (EU 2000–2003)
European parallel to AES. 42 submissions → 17-algorithm portfolio (Feb 2003). Selected: AES (Rijndael), Camellia, MISTY1, SHACAL-2, RSA-OAEP, RSA-PSS, ECDSA, etc. Notable rejects: every stream cipher submission was rejected (none met the bar) — explicitly motivating eSTREAM. **Lesson:** rigorously transparent processes can return "none of the above" — and a public null result is itself a contribution. reality's golden-file regime is the analogue: a vector that fails to converge is a documented finding.

### 3. CRYPTREC (Japan, 2000–present)
Japanese government cipher list, ongoing rather than one-shot. e-Government recommended list (2003, revised 2013, 2023) ratifies AES, Camellia, SHA-256/-3, ECDSA, RSA-PSS. Most domestic Japanese ciphers (CIPHERUNICORN, Hierocrypt, MISTY1) demoted to "Candidate" not for cryptanalytic weakness but **lack of deployment** — a lesson reality cites: math without integration into real software is a museum piece.

### 4. eSTREAM (ECRYPT 2004–2008)
Stream-cipher response to NESSIE's stream gap. Two profiles: software (Profile 1) and hardware (Profile 2). Software portfolio: HC-128, Rabbit, Salsa20/12, SOSEMANUK. Hardware: Grain, MICKEY, Trivium. Salsa20 and its descendant ChaCha (D. J. Bernstein, 2008) became the universal soft-stream cipher — TLS, WireGuard, QUIC/HTTP3, Linux `getrandom`. Won by being a constant-time ARX (add-rotate-xor) permutation with zero S-boxes, zero table lookups → no cache-timing side channels by construction. **Lesson for reality:** ARX > S-boxes when constant-time is mandatory; pure 32-bit arithmetic on 4×4 state is portable, vectorizable, and trivially auditable.

### 5. SHA-3 (NIST 2007–2012) — winner: Keccak
64 submissions → 5 finalists (BLAKE, Grøstl, JH, Keccak, Skein). Keccak (Bertoni/Daemen/Peeters/Van Assche) won Oct 2012; FIPS 202 published 2015. Selected for "elegant design, large security margin, hardware efficiency, flexibility." Introduced the **sponge construction**: a fixed permutation `f` over a state, alternately absorbing and squeezing — one primitive yields hash, MAC, XOF, AEAD, PRF. Pure permutation (no Davies-Meyer, no compression-function gymnastics). **Lesson for reality:** one permutation can power many functions; design a *single* well-analyzed object and reuse it (sponge ≈ "interpreter" for crypto).

### 6. CAESAR (2014–2019) — winners: Ascon, AEGIS, OCB, ChaCha20-Poly1305
Authenticated-encryption competition organized by Bernstein. 57 submissions → 6 finalists across 3 use-case portfolios. Selections (Feb 2019):
- **Lightweight**: Ascon-128, Ascon-128a (primary).
- **High-performance**: AEGIS-128, OCB.
- **Defense in depth**: Deoxys-II, COLM.
ChaCha20-Poly1305 was named separately (RFC 8439) and IETF-standardized in parallel. AES-GCM remained the de facto incumbent. **Lesson:** different deployment regimes legitimately need different primitives; reality's `crypto` shouldn't try to ship one-size-fits-all AEAD — it should ship the *components* (permutation, polynomial MAC, NTT) and let consumers compose.

### 7. NIST LWC (2019–2023, finalized 2025) — winner: Ascon
Lightweight competition for ≤ 2 kB ROM, IoT/embedded targets. 57 submissions → 10 finalists (TinyJambu, Xoodyak, GIFT-COFB, Romulus, ISAP, etc.) → **Ascon family** selected Feb 2023. SP 800-232 final published Aug 2025: Ascon-AEAD128, Ascon-Hash256, Ascon-XOF128, Ascon-CXOF128. Same sponge philosophy as Keccak but with a smaller (320-bit) permutation and a simpler S-box (5-bit). Won for second time (CAESAR + LWC) by breadth: one permutation does AEAD + hash + XOF. **Lesson for reality:** primitive reuse — `signal.FFT` is to `crypto.NTT` what Ascon's permutation is to all four standardized functions. Reuse the permutation; don't proliferate.

### 8. NIST PQC (2016–2024) — winners: ML-KEM, ML-DSA, SLH-DSA, FN-DSA
82 submissions → 4 KEM + 3 signature finalists in round 3. Standards (Aug 2024):
- **FIPS 203 ML-KEM** (was Kyber) — module-LWE + NTT over Z_q (q=3329, n=256).
- **FIPS 204 ML-DSA** (was Dilithium) — module-LWE/SIS + Fiat-Shamir-with-aborts.
- **FIPS 205 SLH-DSA** (was SPHINCS+) — stateless hash-based, no number-theoretic assumption.
- **FIPS 206 FN-DSA** (Falcon, draft) — NTRU + FFT over Z[X]/(X^n+1) with Gaussian sampling.
Lattice losers: Saber (round-LWR), NTRU, NTRU Prime — all viable, lost on margins and IP. SIKE killed by Castryck-Decru classical attack mid-process — a defining cautionary tale. **Lesson for reality:** the NTT (number-theoretic transform) is the unifying primitive — same Cooley-Tukey butterfly structure as `signal.FFT`. reality should expose `NTT(x, q, ω, n)` as a first-class library function; ML-KEM/ML-DSA/FN-DSA all reduce to it. Constant-time modular reduction (Barrett, Montgomery) is a sibling primitive.

### 9. NIST Additional-Signature On-Ramp (2023–ongoing)
Open call closed June 2023; 40 submissions → 14 second-round candidates (Oct 2024): CROSS, FAEST, HAWK, LESS, MAYO, Mirath, MQOM, PERK, QR-UOV, RYDE, SDitH, SNOVA, SQIsign, UOV. Goal: a non-lattice, non-stateless-hash signature that beats SLH-DSA. Drives multivariate-quadratic, code-based, and isogeny revivals. **Lesson:** "diversification of cryptographic assumptions" is a meta-design goal — reality should not bake in a single hardness assumption either; expose primitives at the algebra level (rings, lattices, codes) so consumers can swap.

### 10. Password Hashing Competition (2013–2015) — winner: Argon2
24 submissions → 9 finalists → Argon2 (Biryukov/Dinu/Khovratovich) won July 2015; special recognition to Catena, Lyra2, yescrypt, Makwa. RFC 9106 (2021) standardized Argon2id. Won on memory-hardness with explicit time/memory/parallelism parameters and resistance to both GPU/ASIC parallelism and tradeoff attacks. **Lesson for reality:** parameterizable cost (Argon2's `(t, m, p)`) is the right knob — not magic constants. Echoes reality's per-function precision tolerances rather than one global epsilon.

### 11. ZPrize 2022/2023 — open ZK acceleration competition
Industry-funded ($1.5M+ in 2023) prizes for fastest implementations of:
- **MSM** (multi-scalar multiplication on BLS12-377/BN254) — FPGA, GPU, WASM tracks.
- **NTT/FFT** over prime fields (Goldilocks, BabyBear, Mersenne31).
- **Plonk / poseidon-hash / ECDSA verification** throughput.
2023 winners included Cysic (FPGA MSM), Hardcaml/Jane Street (FPGA NTT), and split awards in ECDSA verification. **Lesson:** the industry is implementation-bound, not theory-bound — clean reference NTT and finite-field arithmetic in reality would be heavily reusable for ZK builders. The hot fields are 31-bit Mersenne (Plonky3, Stwo) and Goldilocks (2^64 - 2^32 + 1).

### 12. Cryptopals (Matasano/NCC, ongoing)
Not a competition — 8 sets of progressively harder hands-on attack exercises (frequency analysis, CBC padding oracle, MT19937 cloning, length extension, Bleichenbacher, GCM nonce reuse, Set 8 elliptic-curve invalid-curve attacks, lattice attacks on biased nonces). Maintained by Sean Devlin / NCC. Pedagogical lesson: **you don't understand a primitive until you've broken its misuse.** Direct mirror to reality's golden-file edge cases (NaN, ±0, subnormal, ULP-boundary): you don't understand a function until you've tested its pathological inputs. Cryptopals' insistence on implementing attacks (not just theorizing) is the same epistemics as reality's "golden files are the proof."

## Cross-link to reality slots

| Competition primitive | reality home | Status |
|---|---|---|
| AES round function (GF(2^8) inversion + MixColumns) | `crypto/aes` | Not present (potential addition) |
| SHA-3 / Keccak-f[1600] permutation | `crypto/sha3` or `crypto/keccak` | Not present |
| ChaCha20 ARX permutation | `crypto/chacha` | Not present |
| Poly1305 GF(2^130 - 5) MAC | `crypto/poly1305` | Not present |
| Ascon-p permutation | `crypto/ascon` | Not present |
| NTT (Cooley-Tukey over Z_q) | `signal.FFT` already exists; mirror for prime fields → `crypto/ntt` or `linalg/ntt` | Adjacent (FFT only over C) |
| Modular arithmetic (Barrett, Montgomery) | `crypto/modular` (already in `crypto`) | Partial — extend for NTT primes |
| Argon2-style memory-hard hash | `crypto/argon2` | Not present |
| Lattice basis reduction (LLL/BKZ) | `linalg` extension | Not present (research candidate) |
| Multi-scalar multiplication (Pippenger) | `crypto/ec` + `linalg` | Not present |

reality currently exposes `crypto` (primality, modular, PRNGs, hash). The "modern competition" gap is: no AEAD, no sponge permutation, no NTT, no lattice arithmetic. Each is a discrete, well-specified primitive with abundant golden vectors (Wycheproof, NIST CAVP).

## Lessons reality should adopt

1. **Constant-time-by-construction is non-negotiable.** ARX (ChaCha) and bitsliced S-box (AES, Ascon) designs win because they have no data-dependent memory access or branches. reality should adopt a project-wide "no `if x[i] then` over secret data" rule for any future `crypto` additions, validated by a `dudect`-style timing test in CI.

2. **One permutation, many functions.** Keccak (sponge → hash, MAC, XOF, AEAD, PRF) and Ascon (same) are the deepest design idea of the last 20 years. Apply this to reality's API: a single well-analyzed `keccak.F1600(state)` should back `Hash`, `XOF`, and `AEAD`, not three independent implementations.

3. **Algebraic structure beats heuristic mixing.** Rijndael's GF(2^8) inverse, ChaCha's quarter-round, Kyber's NTT — winners are *describable in three lines of math*. reality's "every function cites its source" mandate is the same value: an opaque table-of-magic-numbers function should not pass review.

4. **NTT is the new FFT.** ML-KEM, ML-DSA, FN-DSA, Plonky3, Stwo, Risc0, Hyperplonk all share the same Cooley-Tukey butterfly over a prime field. reality has `signal.FFT` over ℂ; it should generalize to `NTT(x, q, ω, n)` over Z_q. This single addition unlocks PQC, ZK, and FHE consumers.

5. **Per-function tolerance, not global.** Argon2's parameterized (t, m, p) and PHC's "set realistic goals" lesson maps directly onto reality's per-function golden-file tolerances. Resist the temptation to globalize.

6. **Diversify hardness assumptions.** SIKE's classical break mid-NIST-PQC and the additional-signature on-ramp prove single-assumption baskets are dangerous. reality should expose primitives at the algebra layer (ring, lattice, code, isogeny) so consumers can hot-swap.

7. **Open process + reproducible vectors = trust.** AES, SHA-3, PQC all won the *trust* battle by being publicly attackable for years before standardization. reality's golden-file vectors play the same role: every claimed identity is independently verifiable in 4 languages. Keep the bar there.

8. **Document failure modes loudly.** SIKE, NTRU-HRSS Round-2 attacks, GCM nonce-reuse forgery — these are the most-cited "lessons" papers. reality should treat documented failure modes (range, NaN, overflow, accumulation drift) as first-class API surface, not appendix.

9. **Implementation-bound > theory-bound.** ZPrize spending $500K on an MSM speedup tells you where the marginal value is. reality's `optim` and `linalg` should target *implementation* primitives (Pippenger, Karatsuba, Toom-Cook, NTT-Bailey-4-step) not just textbook algorithms.

10. **Educational counterpart matters.** Cryptopals turns crypto consumers into competent reviewers. reality's golden-file vectors + per-function provenance citations are the same thing for math: a learner can audit by re-deriving. Preserve and expand this stance — the "queryable provenance metadata" rule is a high-leverage differentiator.

## Sources

- [NIST: First 3 Finalized Post-Quantum Encryption Standards (Aug 2024)](https://www.nist.gov/news-events/news/2024/08/nist-releases-first-3-finalized-post-quantum-encryption-standards)
- [FIPS 203 ML-KEM](https://csrc.nist.gov/pubs/fips/203/final)
- [NIST IR 8528 — additional signature first-round status](https://nvlpubs.nist.gov/nistpubs/ir/2024/NIST.IR.8528.pdf)
- [NIST PQC Additional Digital Signature project page](https://csrc.nist.gov/projects/pqc-dig-sig)
- [Cloudflare blog: another look at PQ signatures](https://blog.cloudflare.com/another-look-at-pq-signatures/)
- [NIST AES Development archive](https://csrc.nist.gov/projects/cryptographic-standards-and-guidelines/archived-crypto-projects/aes-development)
- [NIST: Commerce Dept. announces Rijndael as AES (Oct 2000)](https://www.nist.gov/news-events/news/2000/10/commerce-department-announces-winner-global-information-security)
- [Wikipedia: AES process](https://en.wikipedia.org/wiki/Advanced_Encryption_Standard_process)
- [NIST: SHA-3 winner announcement (Oct 2012)](https://www.nist.gov/news-events/news/2012/10/nist-selects-winner-secure-hash-algorithm-sha-3-competition)
- [Wikipedia: SHA-3 / Keccak](https://en.wikipedia.org/wiki/SHA-3)
- [CAESAR competition home (Bernstein)](https://competitions.cr.yp.to/caesar.html)
- [Wikipedia: CAESAR Competition](https://en.wikipedia.org/wiki/CAESAR_Competition)
- [Ascon official site (TU Graz)](https://ascon.isec.tugraz.at/)
- [NIST SP 800-232 final — Ascon LWC standard (Aug 2025)](https://csrc.nist.gov/pubs/sp/800/232/final)
- [NIST LWC project page](https://csrc.nist.gov/Projects/Lightweight-Cryptography)
- [NESSIE crypto-competitions index](https://competitions.cr.yp.to/nessie.html)
- [Wikipedia: NESSIE](https://en.wikipedia.org/wiki/NESSIE)
- [CRYPTREC ciphers list](https://www.cryptrec.go.jp/en/list.html)
- [Wikipedia: CRYPTREC](https://en.wikipedia.org/wiki/CRYPTREC)
- [eSTREAM / Salsa20 (Wikipedia)](https://en.wikipedia.org/wiki/Salsa20)
- [DJB ChaCha family page](https://cr.yp.to/chacha.html)
- [Password Hashing Competition home](https://www.password-hashing.net/)
- [Wikipedia: Password Hashing Competition](https://en.wikipedia.org/wiki/Password_Hashing_Competition)
- [Argon2 reference impl (P-H-C)](https://github.com/P-H-C/phc-winner-argon2)
- [ZPrize 2023 winners announcement](https://www.zprize.io/blog/announcing-the-2023-zprize-winners)
- [zkSecurity ZPrize wrap-up](https://blog.zksecurity.xyz/posts/zprize-final/)
- [Cryptopals challenges](https://cryptopals.com/)
- [Bernstein: Cryptographic Competitions paper (2024)](https://cr.yp.to/papers/competitions-20240113.pdf)
