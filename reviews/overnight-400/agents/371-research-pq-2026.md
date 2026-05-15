# 371 — research-pq-2026 (NIST PQC standards 2026)

## Headline
By May 2026 NIST has finalized FIPS 203/204/205 (ML-KEM, ML-DSA, SLH-DSA), HQC has been picked as the round-4 backup KEM with draft standard targeted for 2026, and FIPS 206 (FN-DSA / Falcon) is in late-stage IPD with publication slated late-2026/early-2027 — the math primitive that unlocks all four lattice schemes in reality is the Number-Theoretic Transform (slot 293).

## Survey

### 1. FIPS 203 — ML-KEM (Module-Lattice KEM, ex-Kyber)
Finalized 13-Aug-2024. Effective date 14-Aug-2024. Three parameter sets: ML-KEM-512 (NIST L1 / AES-128 equivalent), ML-KEM-768 (L3, default for most TLS deployments incl. Cloudflare/Google Chrome hybrid X25519MLKEM768), ML-KEM-1024 (L5). Underlying hardness: Module-LWE over Rq = Z_q[x]/(x^256+1) with q = 3329, n = 256, k ∈ {2,3,4}. Public key ~800–1568 B, ciphertext ~768–1568 B. The arithmetic core is NTT-based polynomial multiplication mod 3329; secret/error coefficients are sampled from a centered binomial distribution (η ∈ {2,3}); SHAKE128/256 drives all expansion and hashing; Compress/Decompress lossy rounding handles ciphertext size. Per FIPS 203 §6 the "stored" representation of public keys is already in NTT domain. (csrc.nist.gov/pubs/fips/203/final)

### 2. FIPS 204 — ML-DSA (Module-Lattice Signature, ex-Dilithium)
Finalized Aug-2024. Parameter sets ML-DSA-44 (L2), ML-DSA-65 (L3, NIST recommended default), ML-DSA-87 (L5). Same ring as Kyber (Z_q[x]/(x^256+1)) but q = 8380417 (a 23-bit NTT-friendly prime). Fiat-Shamir-with-aborts construction; signatures 2420 / 3309 / 4627 B; public keys 1312 / 1952 / 2592 B. Reuses NTT, rejection sampling, SHAKE128/256, Decompose / HighBits / LowBits rounding helpers, and infinity-norm checks. Distinct math vs Kyber: requires `BitUnpack`/`SimpleBitUnpack` and modular reduction over a 23-bit prime, plus the "make-hint" / "use-hint" pair for the rounding of t. (csrc.nist.gov/pubs/fips/204/final)

### 3. FIPS 205 — SLH-DSA (Stateless Hash-Based Signature, ex-SPHINCS+)
Finalized Aug-2024. Pure hash-based — security rests on second-preimage / collision resistance of SHA-256 or SHAKE, no number-theoretic assumption. 12 parameter sets: {128s, 128f, 192s, 192f, 256s, 256f} × {SHA2, SHAKE} where 's' = small-signature/slow-sign, 'f' = fast-sign/large-signature. Built from WOTS+ one-time signatures, FORS few-time signatures, and a hyper-tree of Merkle trees. Signatures 7856–49856 B (huge), keys 32–128 B. Ideal for code-signing / firmware where verify dominates. Conservatism is the selling point: if all lattice assumptions ever break, SLH-DSA still stands. (csrc.nist.gov/pubs/fips/205/final)

### 4. FIPS 206 — FN-DSA (ex-Falcon), DRAFT 2026
NIST PQC conf Sep-2025 status: IPD basically written, awaiting public release. Publication target: late 2026 / early 2027. NTRU-lattice-based with GPV/Fast-Fourier sampling — produces the smallest PQ signatures (Falcon-512 ≈ 666 B; Falcon-1024 ≈ 1280 B). Requires double-precision IEEE-754 FFT for polynomial inversion plus a constant-time discrete Gaussian sampler over Z. The FP requirement is genuinely novel for a NIST crypto primitive and is the principal source of side-channel and reproducibility concerns. Verification side is integer-only and cheap. (csrc.nist.gov/presentations/2025/fips-206-fn-dsa-falcon)

### 5. HQC — Round-4 KEM selection (March 2025)
NISTIR 8545 (final, 2025) confirmed HQC as the second standardized KEM — code-based, structured Hamming Quasi-Cyclic codes, IND-CCA2 via HHK transform. Rationale: serve as the algorithmically-distinct backup if Module-LWE breaks. Draft standard expected 2026 (~year after announcement); final ~2027. Public key 2249 / 4522 / 7245 B; ciphertext ~2× pk; key-gen and decap dominated by polynomial mult over GF(2)[x]/(x^n−1) with n prime — uses bit-sliced/Karatsuba GF(2)-polynomial arithmetic, not NTT. Different primitive surface than ML-KEM. (csrc.nist.gov/news/2025/hqc-announced-as-a-4th-round-selection)

### 6. Classic McEliece — round-4 alternate, no NIST standard imminent
Conservative Goppa-code KEM with 50-year track record. Selected for round 4 but NIST has signaled it will not standardize federally; ISO/SC 27 is taking it. Killer attribute = ciphertexts only 128–240 B; killer drawback = public keys 261 KB to 1.36 MB. Suitable for static long-term endpoints (e.g., HSMs, enterprise CA roots), not TLS handshakes. Math primitives needed: binary-Goppa decoding (Patterson algorithm), GF(2^m) arithmetic, semi-systematic matrix reduction. (classic.mceliece.org/nist.html)

### 7. SHAKE128/256 + SHA-3 dependency
FIPS 202 (2015). Every NIST PQC scheme uses Keccak: ML-KEM and ML-DSA use SHAKE128 (matrix expansion / sample-in-ball XOF) and SHAKE256 (PRF, hashing of pk/messages); SLH-DSA has SHA-2 and SHAKE variants; FN-DSA uses SHAKE256 for the random nonce. The 1600-bit Keccak permutation with sponge construction is therefore a hard dependency for any reality PQC ambition. crypto/hash.go currently lacks SHA-3 / SHAKE. (nvlpubs.nist.gov/nistpubs/fips/nist.fips.202.pdf)

### 8. Number-Theoretic Transform (NTT) — the keystone primitive
NTT = DFT over Z_q where q is chosen such that a primitive 2n-th root of unity exists mod q. Reduces polynomial multiplication in Z_q[x]/(x^n+1) from O(n²) to O(n log n). ML-KEM uses an incomplete 7-layer NTT producing degree-1 residue polynomials (because 256 ∤ 3329−1) — so "pointwise" mul is actually base-mul of degree-1 polynomials. ML-DSA uses a complete NTT (q = 8380417 satisfies q ≡ 1 mod 2n with full splitting). Required butterflies: Cooley-Tukey forward (Gentleman-Sande inverse), Montgomery / Barrett reduction, precomputed twiddle table (zetas[]). Hardware studies (PQShield 2024) show NTT consumes 60–80% of ML-KEM cycle budget — this is THE primitive to get right. (eprint.iacr.org/2024/585)

### 9. Centered Binomial Distribution (CBD) sampling
Replaces discrete Gaussians in Kyber/Dilithium for noise/secret sampling. CBD_η returns a − b where a, b are sums of η Bernoulli(1/2) bits. For η=2 yields support [-2,+2] with probabilities {1,4,6,4,1}/16; for η=3 yields [-3,+3]. Implementation = unpack 2η·n bits from a SHAKE256 stream, count, subtract. Constant-time and trivially side-channel-safe. Why CBD over Gaussian: indistinguishable enough from Gaussian for LWE security under bound on the supported norm, but ~10× simpler to sample. ML-KEM-512 uses η=3, ML-KEM-768/1024 use η=2/2. (pq-crystals.org/kyber spec round3)

### 10. Discrete Gaussian sampling (FN-DSA only)
Gaussian over Z with σ ≈ 1.17·√(q/2N) — for Falcon-512 σ ≈ 1.55. Two competing approaches: cumulative-distribution-table sampler (constant-time, used by reference Falcon) and rejection sampling. Both must be statistically perfect to avoid key leakage. The signing-time sampler runs *fast Fourier sampling* recursively over the LDL decomposition of the NTRU public-basis tree — this needs IEEE-754 doubles for the FFT. Falcon's use of FP arithmetic is the single most controversial design decision in NIST PQC and is why FN-DSA shipped after the lattice cousins. (di.ens.fr/~prest/Publications/falcon.pdf)

### 11. Compress / Decompress (ML-KEM ciphertext shrinkage)
Lossy fixed-point rounding `Compress_d(x) = round(x·2^d / q) mod 2^d` with `Decompress_d(y) = round(y·q / 2^d)`. Used to drop low-order bits of ciphertext polynomials (d_u ∈ {10,11}, d_v ∈ {4,5}) reducing ciphertext from ~3 KB to ~1 KB while keeping decryption-failure probability < 2^-138. Requires constant-time integer division by q = 3329 — done via Barrett's `mulhi(x, ⌊2^32/q⌋) >> shift`. Tiny but mandatory primitive that an out-of-the-box modular library will get wrong (typical impls use slow `%`).

### 12. Migration timeline & deprecation calendar
NSA CNSA 2.0 (Sep-2022) requires: software/firmware signing PQ-only by 2025; new NSS systems CNSA-2.0 by Jan-2027; networking gear by 2030; full enforcement 2031; quantum-vulnerable algos retired 2035. NIST SP 800-131A schedule: RSA-2048 / ECDSA-P256 / DH-2048 disallowed after 2030 (deprecated in 2030, fully banned 2035). FIPS 140-2 → Historical on 21-Sep-2026 (only FIPS 140-3 modules certifiable). Hybrid TLS already deployed: X25519MLKEM768 in Chrome 124+, OpenSSL 3.5, BoringSSL. Signatures still mostly classical because PQ sigs are large (5–50× RSA-2048). (csrc.nist.gov/projects/post-quantum-cryptography ; CNSA 2.0)

## Reality positioning

reality is a math foundation, not a TLS stack — it should provide the *primitives* needed to build any of the standardized schemes, not the schemes themselves. That keeps the surface area small and aligns with the zero-deps / golden-file philosophy.

### Recommendation: target ML-KEM + ML-DSA primitives first
- Highest deployment momentum (already in TLS 1.3 hybrid handshakes), shared math (both use NTT, CBD, SHAKE), and the math is *clean integer arithmetic* — no IEEE-754 corner cases to standardize across Go/Python/C++/C# golden files.
- Ship in this order:
  1. **Slot 293 NTT** (already-keystoned in this review series) — generic forward / inverse NTT over Z_q with parameterized q, n, ψ. Provide both Kyber (q=3329, incomplete) and Dilithium (q=8380417, complete) as named instances. Constant-time Montgomery reduction.
  2. **SHA-3 / SHAKE128 / SHAKE256** in `crypto` package — required for all four lattice schemes and SLH-DSA. Currently missing from `crypto/hash.go`. Reuse Keccak-f[1600] core; expose absorb / squeeze / XOF interface.
  3. **CBD_η sampler** in `prob` (it IS a probability distribution) with golden vectors: input SHAKE stream → output coefficient polynomial.
  4. **Compress_d / Decompress_d** rounding in `crypto` (or a new `crypto/lattice` subpackage) — 6 lines, but a footgun.
  5. **Polynomial-mod-q ring ops** as a building block in `linalg` or new `crypto/poly` (add, sub, base-multiply, conditional sub).

### Defer / decline
- **SLH-DSA**: pure hash-based — once SHA-2/SHAKE exist, the rest is Merkle/WOTS+ tree plumbing, not "math". Belongs in an applications repo (aicore?), not reality.
- **FN-DSA / Falcon**: requires IEEE-754 FP discrete-Gaussian sampler. The FP contract makes Go/Python/C++/C# golden parity *very* hard (different libm, different FMA). Wait until FIPS 206 ships and a reference integer-only signing-time sampler appears (active research 2025–2026: "HAWK"-style integer Falcon variants).
- **HQC**: GF(2)-polynomial arithmetic over a *prime* n — different math surface, separate primitive ladder. Add only after ML-KEM/ML-DSA primitives are stable. Draft FIPS not until 2026; final 2027 — plenty of slack.
- **Classic McEliece**: Goppa decoding (Patterson) is a niche specialty; key sizes mean nobody will deploy it through reality. Decline.

### Cross-links
- **Slot 293 NTT** (`reviews/overnight-400/agents/293-new-ntt.md`) — the keystone; this slot's recommendations all funnel into 293's design.
- **Slot 211 lattice-crypto** (`reviews/overnight-400/agents/211-new-lattice-crypto.md`) — CBD, Compress/Decompress, polynomial ring; coordinate scope so 211 owns the ring-arithmetic primitives and 371 owns the standards-tracking guidance.
- **Slot 322 AES-vs-Poly1305** — AEAD primitives sit alongside the PQ KEM; once a hybrid handshake produces a shared secret, an AEAD seals the channel. SHAKE + KMAC could replace HMAC-SHA-2 in a PQ-native suite.
- `crypto/hash.go` currently has no SHA-3 — gating dependency for ALL of the above.

### Decision rule for reality
Implement primitives that:
1. are pure number-theoretic / integer math (no FP — protects the 4-language golden invariant),
2. are shared across ≥ 2 standardized PQC schemes (NTT, SHAKE, CBD),
3. are footguns when re-implemented (Compress/Decompress, Barrett-mod-3329, Montgomery-mod-8380417).

## Sources
- [FIPS 203 final (ML-KEM)](https://csrc.nist.gov/pubs/fips/203/final) — nvlpubs.nist.gov/nistpubs/fips/nist.fips.203.pdf
- [FIPS 204 final (ML-DSA)](https://csrc.nist.gov/pubs/fips/204/final)
- [FIPS 205 final (SLH-DSA)](https://csrc.nist.gov/pubs/fips/205/final)
- [FIPS 206 status update, NIST PQC Conf 2025 (FN-DSA)](https://csrc.nist.gov/presentations/2025/fips-206-fn-dsa-falcon)
- [HQC selected as round-4 KEM (Mar 2025)](https://csrc.nist.gov/news/2025/hqc-announced-as-a-4th-round-selection)
- [NIST IR 8545 — round-4 status report (2025)](https://nvlpubs.nist.gov/nistpubs/ir/2025/NIST.IR.8545.pdf)
- [NIST PQC FIPS-approved announcement (Aug 2024)](https://csrc.nist.gov/news/2024/postquantum-cryptography-fips-approved)
- [Federal Register, FIPS 203/204/205 issuance](https://www.federalregister.gov/documents/2024/08/14/2024-17956/announcing-issuance-of-federal-information-processing-standards-fips-fips-203-module-lattice-based)
- [Beginner Guide to NTT (eprint 2024/585)](https://eprint.iacr.org/2024/585.pdf)
- [Falcon original paper (Fouque et al.)](https://falcon-sign.info/falcon.pdf)
- [Kyber round-3 spec (CBD definition)](https://pq-crystals.org/kyber/data/kyber-specification-round3-20210131.pdf)
- [Classic McEliece NIST submission](https://classic.mceliece.org/nist.html)
- [FIPS 202 (SHA-3, SHAKE)](https://nvlpubs.nist.gov/nistpubs/fips/nist.fips.202.pdf)
- [NSA CNSA 2.0 timeline (postquantum.com)](https://postquantum.com/quantum-policy/nsa-cnsa-2-0-pqc/)
- [PQShield NTT hardware accelerator paper (Oct 2024)](https://pqshield.com/wp-content/uploads/2024/10/High-Performance-NTT-Hardware-Accelerator-to-Support-ML-KEM-and-ML-DSA.pdf)
