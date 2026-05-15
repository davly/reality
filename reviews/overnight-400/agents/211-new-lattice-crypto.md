# 211 | new-lattice-crypto

**Summary line 1.** ELEVENTH Block-C cutting-edge-math review and FIRST lattice-cryptography scoping in the 400-sequence covering integer-lattice basics (basis B ∈ Z^{n×n}, fundamental parallelepiped, Gram matrix B^T B, dual lattice Λ*, successive minima λ_1...λ_n) / Hermite-Korkine-Zolotarev (HKZ) reduction / Gauss reduction in dimension 2 / LLL (Lenstra-Lenstra-Lovász 1982) basis reduction with Lovász condition δ ∈ (1/4, 1) / deep-insertion LLL (Schnorr-Euchner 1994) / BKZ block-Korkine-Zolotarev (Schnorr-Euchner 1994) / BKZ 2.0 (Chen-Nguyen 2011) / sieving algorithms (Nguyen-Vidick 2008 / Becker-Ducas-Gama-Laarhoven 2016 G6K) / SVP / CVP / Babai's nearest-plane / Babai's rounding / decision-LWE (Regev 2005) / search-LWE / dual-LWE / Ring-LWE (Lyubashevsky-Peikert-Regev 2010 over Rq = Zq[X]/(X^n+1)) / Module-LWE (Langlois-Stehlé 2015) / NTRU (Hoffstein-Pipher-Silverman 1998) / NTRU-Encrypt / NTRU-Sign / Falcon-512/1024 (Fouque-Hoffstein-Kirchner-Lyubashevsky-Pornin-Prest-Ricosset-Seiler-Whyte-Zhang 2017) / FFT-tree-sampler / Mitaka / Hawk / Dilithium = ML-DSA (FIPS-204 2024) / Kyber = ML-KEM (FIPS-203 2024) / SPHINCS+ = SLH-DSA (FIPS-205 2024 — but this is hash-based not lattice) / Number-theoretic transform (Cooley-Tukey over Zq with primitive 2n-th root) / negacyclic-NTT for X^n+1 / Barrett reduction (Barrett 1986) / Montgomery reduction (Montgomery 1985) / centered-binomial sampling CBD_η / rejection sampling for discrete Gaussian / Klein 2000 / GPV trapdoor sampler (Gentry-Peikert-Vaikuntanathan 2008) / Micciancio-Peikert 2012 G-trapdoor / gadget matrix G = I_n ⊗ g where g = (1, 2, 4, ..., 2^{k-1}) / gadget decomposition G^{-1}(·) / BFV (Brakerski-Fan-Vercauteren 2012) / BGV (Brakerski-Gentry-Vaikuntanathan 2011) / CKKS (Cheon-Kim-Kim-Song 2017 approximate FHE for reals) / TFHE (Chillotti-Gama-Georgieva-Izabachène 2016 torus FHE) / FHEW (Ducas-Micciancio 2015) / programmable bootstrapping / blind rotation / GSW homomorphic accumulator (Gentry-Sahai-Waters 2013) / FrodoKEM (plain LWE, no rings) / SABER (LWR — Learning With Rounding) / Crystals-Kyber CPA-PKE → CCA-KEM Fujisaki-Okamoto transform / GLP signature (Güneysu-Lyubashevsky-Pöppelmann 2012) / BLISS (Ducas-Durmus-Lepoint-Lyubashevsky 2013) / NIST PQC R1-R4 (2017-2024) / hybrid X25519+Kyber TLS (Cloudflare/Google 2023) / lattice trapdoors via NTRU / IBE from LWE (Agrawal-Boneh-Boyen 2010) / ABE from LWE (Boyen 2013, Goyal-Koppula-Waters 2015) / Reality v0.10.0 ships **ZERO** lattice cryptography surface and zero of its prerequisites: per 057-crypto-missing the entire crypto/ package is uint64 ModPow / ModInverse / ExtendedGCD / CRT / Miller-Rabin / FNV / MurmurHash / MT-PCG-Xoshiro RNGs (884 LOC), with NO big-integer arithmetic, NO finite-field tower, NO polynomial arithmetic in Zq[X]/(X^n+1), NO NTT, NO discrete Gaussian sampling, NO centered-binomial sampling, NO LLL/BKZ basis reduction, NO Babai algorithms, NO gadget decomposition, NO SHA-2/SHA-3/SHAKE (every PQ scheme depends on SHAKE-128 for matrix expansion and SHAKE-256 for FO transform). Repo-wide grep on `LWE|Kyber|Dilithium|Falcon|NTRU|Lattice|LLL|BKZ|Babai|NTT|Barrett|Montgomery|gadget|GSW|BFV|BGV|CKKS|TFHE|bootstrap|FrodoKEM|trapdoor|SHAKE` returns ZERO callable matches across all 22 packages. The closest tangential surfaces are signal/fft.go Cooley-Tukey radix-2 (the *complex-double* variant — math twin of the *integer-modular* NTT but currently no shared kernel) and crypto/modular.go ModPow over uint64 (the prime-field building block, but capped at q < 2^32 since mulmod uses uint128 splitting; PQ moduli q = 3329 (Kyber), q = 8380417 (Dilithium), q = 12289 (Falcon) all fit in uint32 — uint64 is adequate, the big-integer gap that blocks classical crypto does NOT block lattice crypto since the moduli are small primes). 057's Sequence A defers post-quantum to T3-LATTICE as ~800 LOC stub; this slot upgrades that estimate after Kyber/Dilithium/Falcon shipped FIPS-203/204/206 in 2024 and deserve full first-class treatment, not stub status.

**Summary line 2.** Twenty-six primitives L1–L26 totalling ~6,180 LOC across new sub-package `lattice/` (sibling to `crypto/` rather than nested, since lattice cryptography has its own self-contained mathematical universe — small prime moduli, polynomial rings, NTT, sampling — that does NOT share the big-integer / elliptic-curve / pairing stack scoped by 057-crypto-missing). Recommended split: `lattice/ring/` for Zq[X]/(X^n+1) polynomial arithmetic + Barrett/Montgomery reduction (~480 LOC, prerequisite for everything else); `lattice/ntt/` for forward + inverse negacyclic NTT + Cooley-Tukey + Gentleman-Sande butterflies + bit-reversed addressing + precomputed-twiddle-table (~420 LOC); `lattice/sample/` for centered-binomial η ∈ {2,3} + uniform-rejection mod q + discrete-Gaussian (Klein FFT-tree + CDT + Knuth-Yao + Karney 2014) + SHAKE-128/256 expansion (~640 LOC, deferred behind a `crypto/shake` provider since SHAKE itself is missing per 057); `lattice/lwe/` for plain-LWE + Ring-LWE + Module-LWE encrypt/decrypt/keygen + LWR (~520 LOC); `lattice/kyber/` for FIPS-203 ML-KEM-512/768/1024 with FO transform (~640 LOC); `lattice/dilithium/` for FIPS-204 ML-DSA-44/65/87 with rejection-sampling Fiat-Shamir (~720 LOC); `lattice/falcon/` for FIPS-206 (Q4-2025 final draft) Falcon-512/1024 with FFT-tree-sampler + NTRU lattice + Klein-style sampler (~880 LOC, the singular hardest piece); `lattice/ntru/` for original NTRU-Encrypt + Stehlé-Steinfeld 2011 provably-secure NTRU (~280 LOC); `lattice/reduction/` for Gauss-2D + LLL + deep-LLL + BKZ-β + enumeration + Babai-NP + Babai-rounding (~580 LOC, useful for cryptanalysis pedagogy and for SVP-challenge-style attack benches); `lattice/gadget/` for gadget matrix G + decomposition G^{-1} + Micciancio-Peikert trapdoor sampler (~280 LOC); `lattice/fhe/` for BFV + BGV + CKKS scheme math (encrypt / decrypt / SIMD-pack / relin-key / mod-switch / rescale / homomorphic-add / homomorphic-mult) at the *math layer* — bootstrapping deferred to v2 (~720 LOC). Tier-1 keystone **L1+L2+L3+L4+L5 = `lattice/ring/poly.go` Polynomial in Rq + `lattice/ring/barrett.go` reduction + `lattice/ntt/forward.go` negacyclic NTT + `lattice/ntt/inverse.go` + `lattice/sample/cbd.go` centered-binomial ~720 LOC** is the irreducible foundation that unblocks every PQ scheme. **Singular reality competitive moat: L8+L9+L10 Kyber FIPS-203 ML-KEM ~640 LOC** — three Go libraries ship Kyber (Cloudflare CIRCL, kudelskisecurity/crystals-go, openssl 3.x) but NONE ship a zero-dep deterministic-given-seed implementation with Python/C++/C# golden-file parity at the byte level on the FIPS-203 KAT vectors; Reality would be the only Go shop with cross-language byte-identical Kyber. **Singular Block-C-2026 frontier: L17 Falcon FFT-tree-sampler + L24 CKKS bootstrapping math** — Falcon's tree-based Klein sampler is the most numerically delicate piece of the entire NIST PQC standardization (constant-time floating-point requirement on a discrete Gaussian over an NTRU lattice — see Howe-Prest-Apon-Westerbaan 2018, Karmakar-Roy-Reparaz-Vercauteren-Verbauwhede 2018), and CKKS bootstrapping (Cheon-Han-Kim-Kim-Song 2018 + Bossuat-Mouchet-Troncoso-Pastoriza-Hubaux 2021) is the active 2024-2026 research frontier where a zero-dep math layer would be the FIRST in any language to ship deterministic golden vectors. **Singular cross-link: L19 LLL basis reduction + L21 Babai-NP** would let `linalg/` extend from Gram-Schmidt + QR (already shipped per `linalg/decompose.go`) to lattice-specific Gram-Schmidt-with-rationals + size-reduction + Lovász swap, and would close the gap to the cryptanalysis side that is mandatory for any lattice library claiming completeness (LLL-attack on small-secret LWE is the classical entry point). Cross-package blockers: `crypto/sha3` SHAKE-128/256 (currently absent — 057 §T1-HASH gates this) is needed for matrix expansion and FO transform; `crypto/rng` ChaCha20-DRBG-shaped PRG (currently absent — 057 §T3-RNG-CRYPTO gates this) is needed for mask sampling. Versus 057-crypto-missing (which scopes elliptic-curve / pairing / classical-signature primitives): orthogonal axis, both ship in parallel; specifically 057's T3-LATTICE 800-LOC line item EXPANDS in this slot to ~6,180 LOC since the 2024 FIPS standardization elevated Kyber/Dilithium/Falcon from "research" to "production-required". Versus 210-new-coding-theory: shared finite-field-polynomial-arithmetic substrate — Reed-Solomon over GF(2^m) and Ring-LWE over Zq[X]/(X^n+1) BOTH need polynomial division and root-finding, but the rings are different (binary extension vs prime-field cyclotomic), so two parallel implementations are appropriate. Versus 207/208/209 Block-C reviews (Lie groups / Riemannian / diff-geo / exterior-calc / geometric-algebra): no overlap — lattice crypto is its own closed mathematical universe.

---

## 0. State at HEAD (2026-05-08, v0.10.0)

Repo-wide audit for lattice-cryptography surface:

| Surface | Path | Lines | Lattice-crypto relevance |
|---|---|---:|---|
| ModPow uint64 | `crypto/modular.go:20-40` | 20 | Building block but uint64 cap; q < 2^32 OK for Kyber/Dilithium/Falcon |
| ModInverse uint64 | `crypto/modular.go:54-75` | 22 | Useful for NTT twiddle precomputation |
| ChineseRemainder | `crypto/modular.go:96-135` | 40 | Used in BFV/BGV multi-prime modulus chains |
| MillerRabin | `crypto/prime.go` | (function) | Could be reused to check NTT-friendly primes (q ≡ 1 mod 2n) |
| FNV1a / Murmur | `crypto/hash.go` | 220 | NOT cryptographic; cannot back FO transform |
| MT19937 / PCG / Xoshiro | `crypto/rng.go` | 197 | NOT cryptographic; cannot back PQ sampling |
| Cooley-Tukey FFT | `signal/fft.go` | (FFT impl) | Math twin of NTT but float64 not Zq; NOT directly reusable |
| Polynomial Lagrange | `linalg/` (interpolation) | (interp) | Different ring; not reusable |
| Gram-Schmidt | `linalg/decompose.go` (QR) | (QR) | Real-valued; needs lattice rational variant for LLL |
| LU/QR/Cholesky | `linalg/decompose.go` | 800+ | Float64 only; lattice reduction needs rational-Q with Gram coefficients |
| `prob/distributions.go` Gaussian | `prob/distributions.go` | (PDF/CDF) | Continuous; lattice needs DISCRETE Gaussian sampler (Klein/Karney/CDT) |

Repo-wide grep audit: `LWE|Kyber|Dilithium|Falcon|NTRU|Lattice|LLL|BKZ|Babai|NTT|negacyclic|gadget|GSW|BFV|BGV|CKKS|TFHE|FrodoKEM|trapdoor|SHAKE` returns **ZERO callable matches** across all 22 packages. The CLAUDE.md package-table has no `lattice/` entry.

NTT-prerequisite reality check: signal/fft.go is float64 Cooley-Tukey radix-2 — the *algorithmic* shape matches NTT (butterflies, bit-reversed permutation, twiddle factors) but the *element type* differs (complex128 vs uint32 mod q with negacyclic twist). Sharing a butterfly kernel via Go generics is feasible but adds friction; recommend independent NTT implementation in `lattice/ntt/` with cross-reference comment to `signal/fft.go`.

---

## 1. The twenty-six primitives

Tier numbering: T1 = irreducible foundation (ring + NTT + sampling), T2 = LWE family (plain / ring / module), T3 = NIST PQC standards (Kyber / Dilithium / Falcon), T4 = NTRU + reductions + cryptanalysis, T5 = FHE (BFV/BGV/CKKS/TFHE).

### Tier 1 — Ring + NTT + Sampling foundation (~1,540 LOC)

**L1 — `lattice/ring/poly.go` ~180 LOC.** `Polynomial{Coeffs []int32; N int; Q int32}` over Zq[X]/(X^n + 1) for n ∈ {256, 512, 1024} (Kyber n=256, Dilithium n=256, Falcon n=512/1024). API: `NewPoly(n, q)`, `Add/Sub/Mul/Neg/Mod`, `Reduce()` (in-place to (-q/2, q/2]), `IsZero/Equal`, `Centered() [N]int32` (lift to centered representation), `FromBytes(b []byte)`, `ToBytes() []byte`. Polynomial multiplication via schoolbook is O(n^2) ⇒ replaced by NTT in L3 for hot path. `MulSchoolbook` retained for golden-file cross-validation against `MulNTT`. Refs: Lyubashevsky-Peikert-Regev 2010; FIPS-203 §4.3.

**L2 — `lattice/ring/barrett.go` ~120 LOC.** Barrett 1986 modular reduction `Barrett(a int32, q int32, mu int32) int32` where `mu = floor(2^k / q)` precomputed; computes `a mod q` in O(1) without division. Montgomery 1985 reduction `Montgomery(a int64, q int32, qInv int32) int32` for product-then-reduce hot path (NTT inner loop). Pre-built constants for Kyber q=3329 (mu=5039 for k=24, qInv=62209 for k=16), Dilithium q=8380417, Falcon q=12289. **Constant-time guarantee:** both reductions are branch-free and side-channel-resistant. Refs: Barrett 1986; Montgomery 1985; Seiler-Lyubashevsky-Schwabe 2018 *Faster AVX2 optimized NTT multiplication for Ring-LWE*.

**L3 — `lattice/ntt/forward.go` ~140 LOC — KEYSTONE.** Negacyclic forward NTT: ψ ∈ Zq is a primitive 2n-th root of unity (NOT just n-th — the negacyclic twist embeds X^n + 1 into the NTT modulus reduction). Precompute zeta-table `zetas[n] = ψ^{br(i)}` where `br` is bit-reversal permutation. Cooley-Tukey decimation-in-time butterflies: `(a,b) -> (a+ζb, a-ζb) mod q`. Output is in bit-reversed order (NOT corrected — Kyber/Dilithium reference impls all leave it in BRO). Operates in-place on `[N]int32`. Refs: Pollard 1971 (NTT for integer convolution); Lyubashevsky-Pöppelmann-Buchmann 2014; Longa-Naehrig 2016 *Speeding up the number-theoretic transform for faster ideal-lattice-based cryptography*.

**L4 — `lattice/ntt/inverse.go` ~120 LOC.** Negacyclic inverse NTT via Gentleman-Sande decimation-in-frequency butterflies: `(a,b) -> (a+b, ζ^{-1}(a-b)) mod q`. Final scaling by `n^{-1} mod q` (precomputed). Input expected in bit-reversed order; output in natural order. **Cross-validation:** `INTT(NTT(p)) == p` for all p ∈ Rq is the keystone golden-file test — must hold byte-for-byte across Go/Python/C++/C# substrates. Refs: Gentleman-Sande 1966; FIPS-203 §4.3.

**L5 — `lattice/sample/cbd.go` ~80 LOC.** Centered-binomial-distribution sampler `CBD(seed [N*η/4]byte, η int) Polynomial` for η ∈ {2, 3}. Each coefficient = sum of η iid {0,1} bits MINUS sum of η iid {0,1} bits, giving distribution centered at 0 with variance η/2. Used by Kyber (η_1=3 for s/e, η_2=2 for r/e_1/e_2) and Dilithium (η=2 or η=4 depending on level). Constant-time bit-counting via popcount. Refs: FIPS-203 §4.4; FIPS-204 §3.1.

**L6 — `lattice/sample/uniform.go` ~120 LOC.** Uniform rejection sampling mod q from a SHAKE-128 byte stream. `RejectionSample(seed [32]byte, q int32) Polynomial` consumes bytes 3-at-a-time, interprets as a 12-bit integer (Kyber) or 23-bit integer (Dilithium), accepts if < q, rejects otherwise. Expected bytes per coefficient ≈ 3 · q / 2^12 (Kyber) ≈ 2.44. **Defers to `crypto/sha3.SHAKE128` which doesn't exist yet** — so this primitive is gated on 057 §T1-HASH landing first. Stub interface ships unblocked using a `Hasher` interface. Refs: FIPS-203 §4.2.

**L7 — `lattice/sample/gaussian.go` ~280 LOC.** Discrete-Gaussian sampler over Z with parameter σ. Three implementations:
- **CDT (Cumulative-Distribution-Table):** precompute table of cumulative probabilities for |x| ≤ τσ (τ = 13 for σ ≈ 5), binary-search a uniform u ∈ [0,1) against table. Constant-time. ~80 LOC.
- **Knuth-Yao 1976:** trace a discrete-distribution-generating-tree bit-by-bit. Optimal entropy. ~100 LOC.
- **Karney 2014:** `Sampling exactly from the normal distribution` — exact, no precomputed table, uses rejection sampling over base distribution. ~100 LOC.
Constant-time variants (Howe-Prest-Apon-Westerbaan 2018) are the production target. Used by Falcon's tree-sampler (L17), GPV-trapdoor (L20), BLISS (post-NIST). Refs: Klein 2000; Genise-Micciancio 2018; Karney 2014.

### Tier 2 — LWE family (~520 LOC)

**L8 — `lattice/lwe/plain.go` ~140 LOC.** Plain LWE (Regev 2005) — no rings, just matrix-vector products mod q. KeyGen: `s ∈ Z_q^n`, `A ∈ Z_q^{m×n}` uniform, `e ∈ Z_q^m` Gaussian, public = `(A, b = As + e mod q)`. Encrypt(μ ∈ {0,1}): pick `r ∈ {0,1}^m`, output `(u = A^T r, v = b^T r + μ·⌊q/2⌋)`. Decrypt: `μ = round((v - s^T u) / (q/2))`. **Used by FrodoKEM** — the conservative no-rings KEM that survives any future ring-structure attack. Refs: Regev 2005 *On lattices, learning with errors*; FrodoKEM Round-3 spec.

**L9 — `lattice/lwe/ring.go` ~140 LOC.** Ring-LWE over Rq = Zq[X]/(X^n+1). KeyGen: `s, e ∈ Rq` small (CBD-sampled), `a ∈ Rq` uniform, public = `(a, b = a·s + e)`. Encrypt(μ): pick `r, e_1, e_2` small, output `(u = a·r + e_1, v = b·r + e_2 + ⌊q/2⌋·μ)`. Decrypt: `μ = round((v - s·u) / (q/2))`. **All operations through NTT** for n·log n complexity. Refs: Lyubashevsky-Peikert-Regev 2010 *On ideal lattices and learning with errors over rings*.

**L10 — `lattice/lwe/module.go` ~120 LOC.** Module-LWE: matrix-of-polynomials variant. `A ∈ R_q^{k×k}` uniform, `s, e ∈ R_q^k` small, public = `(A, b = As + e)`. Generalises Ring-LWE (k=1) and approaches plain-LWE (k=n). **Used by Kyber (k=2,3,4) and Dilithium (k=4,5,6,7)** — the security-vs-flexibility trade-off that the NIST submissions chose. Refs: Langlois-Stehlé 2015 *Worst-case to average-case reductions for module lattices*.

**L11 — `lattice/lwe/lwr.go` ~120 LOC.** Learning With Rounding (LWR) — deterministic LWE variant where error is replaced by deterministic rounding `round_p(a^T s)` where p < q. No explicit error sample needed (the rounding is the error). **Used by SABER** (NIST Round-3, ultimately not selected — Kyber won). Smaller public keys, faster keygen, but tighter security argument. Refs: Banerjee-Peikert-Rosen 2012 *Pseudorandom functions and lattices*; SABER specification.

### Tier 3 — NIST PQC Standards (~2,240 LOC)

**L12 — `lattice/kyber/params.go` ~80 LOC.** Parameter sets for ML-KEM-512 (k=2, η_1=3, η_2=2, du=10, dv=4), ML-KEM-768 (k=3, η_1=2, η_2=2, du=10, dv=4), ML-KEM-1024 (k=4, η_1=2, η_2=2, du=11, dv=5). All three: n=256, q=3329. Bit-strength NIST Levels 1/3/5. Refs: FIPS-203 §7.

**L13 — `lattice/kyber/cpapke.go` ~200 LOC — KEYSTONE.** CPA-secure public-key encryption: `KeyGen() (pk, sk)`, `Encrypt(pk, m, coins) (c)`, `Decrypt(sk, c) (m)`. Internal use of NTT for all matrix-polynomial-vector products. Compress/decompress operations for ciphertext bandwidth (du bits per coefficient for u, dv bits for v). Refs: FIPS-203 §6.

**L14 — `lattice/kyber/kem.go` ~160 LOC.** CCA-secure KEM via Fujisaki-Okamoto transform: `KEMKeyGen()`, `Encapsulate(pk) (c, K)`, `Decapsulate(sk, c) (K)`. The FO transform re-encrypts the decrypted message and compares ciphertexts to detect adversarial ciphertexts; on mismatch returns a deterministic random-looking key (implicit rejection). Constant-time conditional-select between real and reject-key. Refs: FIPS-203 §7; Hofheinz-Hövelmanns-Kiltz 2017 *A modular analysis of the Fujisaki-Okamoto transformation*.

**L15 — `lattice/kyber/serialize.go` ~120 LOC.** Polynomial-to-bytes encoding via 12-bit packing (4 coefs per 6 bytes for q=3329 representation), public-key serialization (rho seed + t̂_NTT polynomial vector), ciphertext serialization (compressed u + compressed v). Byte-exact match with FIPS-203 KAT vectors is the keystone golden-file requirement. Refs: FIPS-203 §4.2.

**L16 — `lattice/dilithium/params.go` ~80 LOC.** Parameter sets for ML-DSA-44 (k=4, l=4, η=2, τ=39, β=78, γ_1=2^17, γ_2=(q-1)/88, ω=80), ML-DSA-65 (k=6, l=5, η=4), ML-DSA-87 (k=8, l=7, η=2). All three: n=256, q=8380417. Refs: FIPS-204 §4.

**L17 — `lattice/dilithium/sign.go` ~280 LOC — KEYSTONE.** ML-DSA signature scheme. KeyGen: sample `s_1 ∈ R_q^l, s_2 ∈ R_q^k` small, expand `A` from seed, compute `t = As_1 + s_2`, split `t = t_1·2^d + t_0` (high/low bits). Sign: rejection-sample `y` until `(z = y + cs_1, c = H(μ, w_1))` lies in the safe zone (||z||_∞ < γ_1 - β AND ||r_0||_∞ < γ_2 - β AND #high-bits(c·t_0) ≤ ω). Verify: recompute `w_1 = HighBits(Az - ct_1·2^d)`, check c == H(μ, w_1). The rejection-sampling Fiat-Shamir loop is the most subtle part — average ~3-7 iterations until acceptance. Refs: Lyubashevsky 2009 *Fiat-Shamir with aborts*; Ducas-Kiltz-Lepoint-Lyubashevsky-Schwabe-Seiler-Stehlé 2018 *CRYSTALS-Dilithium*; FIPS-204 §6.

**L18 — `lattice/dilithium/verify.go` ~160 LOC.** Verification path. Reconstruction of `w_1 ≈ HighBits(Az - ct_1·2^d)` from public key, signature, message. Hint h ∈ R_q^k carries the discrepancy bits between A·z and the prover's w. Validation that ||z||_∞ < γ_1 - β and #1-bits-in-h ≤ ω. Refs: FIPS-204 §6.3.

**L19 — `lattice/falcon/params.go` ~60 LOC.** Parameter sets for Falcon-512 (n=512, q=12289, σ ≈ 165.7) and Falcon-1024 (n=1024, q=12289, σ ≈ 168.4). Refs: Falcon submission package v1.2 (2020); FIPS-206 (draft 2024-2025).

**L20 — `lattice/falcon/keygen.go` ~280 LOC — KEYSTONE.** NTRU-lattice keygen: sample `f, g ∈ Z[X]/(X^n+1)` from discrete Gaussian with σ ≈ √(q·n)/2, check `f` is invertible mod q, compute `h = g/f mod q`. Solve NTRU equation `fG - gF = q` for `(F, G)` via field-norm tower descent (the Babai-style reduction operating on the Gram matrix in halving cyclotomic field tower Q[X]/(X^{n/2^k}+1)). The `ffNP` (fast-Fourier-Babai-nearest-plane) tree construction precomputes Gram-Schmidt orthogonalization in O(n log n) instead of O(n^3). Refs: Hoffstein-Pipher-Silverman 1998; Pornin-Prest 2019 *More efficient algorithms for the NTRU key generation using the field norm*.

**L21 — `lattice/falcon/sample.go` ~280 LOC — KEYSTONE-OF-KEYSTONES.** FFT-tree-sampler: Klein-style discrete-Gaussian sampling on a lattice with FFT-domain Gram-Schmidt tree precomputation. The most numerically delicate piece in the entire NIST PQC standardization — REQUIRES constant-time floating-point operations on the discrete Gaussian sampler (Howe-Prest-Apon-Westerbaan 2018 attacked early Falcon implementations via FP timing leaks). Reality's deterministic-everywhere model means the sampler is seedable from a SHAKE-256 stream, eliminating the side-channel concern. Refs: Klein 2000; Falcon spec §3.9; Prest 2017 *Sharper bounds in lattice-based cryptography using the Rényi divergence*.

**L22 — `lattice/falcon/sign.go` ~200 LOC.** Sign(sk, msg): compute target `c = H(salt || msg) ∈ R_q`, sample `(s_1, s_2)` from FFT-tree-sampler such that `s_1 + s_2·h = c mod q` and ||(s_1, s_2)|| ≤ β (the GPV-style trapdoor sampling). Compress `s_2` via Huffman encoding (variable-length output — Falcon signatures are NOT fixed-length per signature, only per-key). Verify: check `s_1 + s_2·h == c mod q` AND `||(s_1, s_2)||^2 ≤ β^2`. Refs: GPV trapdoor (Gentry-Peikert-Vaikuntanathan 2008); Falcon spec §3.

### Tier 4 — NTRU + Reduction + Cryptanalysis (~1,140 LOC)

**L23 — `lattice/ntru/encrypt.go` ~140 LOC.** Original NTRU-Encrypt (Hoffstein-Pipher-Silverman 1998) over Zq[X]/(X^n-1) (NOT X^n+1 — note the sign). KeyGen: sample small `f, g` in `R = Z[X]/(X^n-1)`, compute `f_p = f^{-1} mod p` and `f_q = f^{-1} mod q`, public = `h = p · f_q · g mod q`. Encrypt(m): `c = r·h + m mod q` for small `r`. Decrypt: `a = f·c mod q` lifted to centered representation, then `m = a · f_p mod p`. Refs: HPS 1998 *NTRU: A ring-based public key cryptosystem*.

**L24 — `lattice/ntru/sign.go` ~140 LOC.** NTRUSign + NSS variants — historically broken by Nguyen-Regev 2006 transcript attack. SHIPS WITH WARNING in docstring: "Educational only, NOT production. Use Falcon (L19-L22) for NTRU-style signatures." Includes the broken signature for cryptanalysis pedagogy. Stehlé-Steinfeld 2011 *Making NTRU as secure as worst-case problems over ideal lattices* provably-secure variant ALSO shipped (~80 LOC additional). Refs: HPS 2003; Nguyen-Regev 2006 *Learning a parallelepiped*; Stehlé-Steinfeld 2011.

**L25 — `lattice/reduction/lll.go` ~280 LOC — KEYSTONE.** LLL (Lenstra-Lenstra-Lovász 1982) basis reduction. Input: `B ∈ Z^{n×n}` linearly-independent integer basis. Output: LLL-reduced basis `B'` satisfying size-reduction (|μ_{i,j}| ≤ 1/2 for all i > j) and Lovász condition (||b_i*||^2 ≥ (δ - μ_{i,i-1}^2) · ||b_{i-1}*||^2 for δ ∈ (1/4, 1), typical δ = 3/4). Algorithm: Gram-Schmidt orthogonalization with rational coefficients (μ matrix in Q), size-reduce step (subtract round(μ_{k,j}) · b_j from b_k), Lovász-swap step (if condition fails, swap b_k and b_{k-1}, decrement k). **Singular reality moat:** zero zero-dep Go libraries ship LLL with golden-file cross-language parity. Schnorr-Euchner deep-insertion variant (~80 LOC additional) extends to BKZ. Used by ALL classical lattice cryptanalysis attacks: low-density subset-sum, RSA small-public-exponent, Coppersmith-Howgrave-Graham factorization. Refs: Lenstra-Lenstra-Lovász 1982 *Factoring polynomials with rational coefficients*; Cohen 1993 *A Course in Computational Algebraic Number Theory* §2.6.

**L26 — `lattice/reduction/babai.go` ~120 LOC.** Babai 1986 algorithms for CVP approximation. **Babai-rounding:** given target `t ∈ R^n` and basis `B`, output `B · round(B^{-1} · t)`. **Babai-NP (nearest-plane):** iterative descent through Gram-Schmidt orthogonalization, projects `t` onto each successive Gram-Schmidt basis vector, rounds to nearest lattice point along that direction. Approximation factor 2^{n/2} for non-reduced basis, much better post-LLL. Used as the SVP/CVP solver inside Falcon's trapdoor sampler (L20-L22). Refs: Babai 1986 *On Lovász' lattice reduction and the nearest lattice point problem*.

### Tier 5 — Gadget + FHE math layer (~1,000 LOC, deferred to v2)

**L27 — `lattice/gadget/decompose.go` ~140 LOC.** Gadget matrix `G = I_n ⊗ g` where `g = (1, 2, 4, ..., 2^{k-1})` for `k = ceil(log_2 q)`. Gadget decomposition `G^{-1}(u) = bin(u)` returns the binary-decomposition vector of `u` such that `G · bin(u) = u`. Foundation for GSW-style FHE and Micciancio-Peikert trapdoors. Constant-time bit-extraction. Refs: Micciancio-Peikert 2012 *Trapdoors for lattices*.

**L28 — `lattice/fhe/bfv.go` ~280 LOC.** BFV scheme (Brakerski-Fan-Vercauteren 2012) — leveled FHE over plaintext ring Z_t[X]/(X^n+1) embedded in ciphertext ring Z_q[X]/(X^n+1) with t << q. Encrypt: `(c_0, c_1) = (-(a·s + e) + ⌊q/t⌋·m, a)`. Decrypt: `m = round(t/q · (c_0 + c_1·s) mod q)`. Homomorphic-add: componentwise. Homomorphic-mult: tensor `(c_0, c_1) × (c'_0, c'_1) → (c_0·c'_0, c_0·c'_1 + c_1·c'_0, c_1·c'_1) · t/q` followed by relinearization (apply relin-key to fold the c_1·c'_1 component). **Bootstrapping deferred** — BFV bootstrapping is non-trivial and requires CKKS-style SIMD batching to be efficient. Refs: BFV 2012; Lattigo (Go FHE library) reference architecture.

**L29 — `lattice/fhe/ckks.go` ~280 LOC.** CKKS (Cheon-Kim-Kim-Song 2017) — approximate FHE for real/complex plaintext. Encoding: pack `n/2` complex-valued slots into a polynomial via inverse-NTT-style canonical embedding. SIMD-add and SIMD-mult on the slots. Rescaling after each multiplication to manage the noise growth. **Active 2024-2026 frontier:** bootstrapping (Cheon-Han-Kim-Kim-Song 2018, Bossuat 2021) and CKKS-IND-CPA-D security model (Li-Micciancio 2021, Manulis-Nguyen 2024). Refs: CKKS 2017; OpenFHE / Lattigo / SEAL reference impls.

**L30 — `lattice/fhe/tfhe.go` ~120 LOC.** TFHE (Chillotti-Gama-Georgieva-Izabachène 2016) — torus FHE with extremely fast bootstrapping (~10ms per gate on CPU). LWE over the torus T = R/Z, with TGSW gadget product and blind rotation as the bootstrapping primitive. **Bootstrapping is the active part of TFHE** (unlike BFV/CKKS where bootstrapping is rare/expensive). Refs: TFHE 2016; Chillotti-Joye-Paillier 2021 *Programmable bootstrapping enables efficient homomorphic inference of deep neural networks*.

---

## 2. LOC budget summary

| Tier | LOC | Cumulative |
|---|---:|---:|
| T1 — Ring + NTT + Sampling | 1,540 | 1,540 |
| T2 — LWE family | 520 | 2,060 |
| T3 — NIST PQC (Kyber/Dilithium/Falcon) | 2,240 | 4,300 |
| T4 — NTRU + LLL + Babai | 1,140 | 5,440 |
| T5 — Gadget + FHE | 1,000 | 6,440 |

Tier 1 alone unblocks all subsequent tiers. Tier 1 + Tier 3-Kyber (~720 LOC for L12-L15) = ~2,260 LOC delivers FIPS-203 ML-KEM, the single highest-value primitive.

---

## 3. Cross-package blockers

| Blocker | Slot | Path | Workaround |
|---|---|---|---|
| SHAKE-128/256 | 057 §T1-HASH | `crypto/sha3/shake.go` (absent) | L6/L13/L20 use a `Hasher` interface; testing uses a deterministic mock until SHAKE lands |
| ChaCha20-DRBG | 057 §T3-RNG-CRYPTO | `crypto/rng/chacha.go` (absent) | Use SHAKE-256 directly as a PRG in the meantime (Falcon already does this) |
| Constant-time discipline | 057 §T1-CT | `crypto/ct/ct.go` (absent) | Branch-free conditional-select baked into each lattice primitive directly; refactor when CT kit lands |
| Big-integer arithmetic | 057 §T1-BIGINT | absent | NOT NEEDED for lattice — all moduli q < 2^32 |
| Floating-point determinism | -- | -- | Falcon's FP-heavy sampler needs the Go FMA / IEEE-754 strict-mode path; document explicit rounding-mode requirement |

The big-integer gap that blocks 057 Sequence A does NOT block this sequence — lattice cryptography uses small prime moduli (q ∈ {3329, 8380417, 12289}) that fit in uint32. This is a CRITICAL architectural insight: **lattice-crypto can ship before classical-crypto** (RSA/ECDSA/EdDSA/BLS) in Reality, even though historically lattice crypto came after.

---

## 4. Recommended sequencing

**Sequence L-A (PQC-first, 4,300 LOC over ~5 PRs):**
1. Tier 1 ring + NTT + sampling (1,540 LOC) — irreducible foundation, lands first.
2. SHAKE-128/256 from 057 §T1-HASH (300 LOC) — gates rejection sampling and FO transform.
3. L8-L11 LWE family (520 LOC) — academic / pedagogical anchor; FrodoKEM is the conservative no-rings baseline.
4. L12-L15 Kyber (640 LOC) — first FIPS standard, highest production value.
5. L16-L18 Dilithium (440 LOC) — second FIPS standard.
6. L19-L22 Falcon (820 LOC) — third FIPS standard, hardest math.

**Sequence L-B (analysis-first, 1,940 LOC):**
1. Tier 1 ring + NTT + sampling (1,540 LOC).
2. L25-L26 LLL + Babai (400 LOC) — cryptanalysis side, useful for SVP-challenge benches and pedagogy.

Recommend **L-A first** because Kyber/Dilithium/Falcon are now FIPS standards (2024) and the Reality position is "universal truth encoded in code" — every FIPS standard is universal truth and belongs in the library.

---

## 5. Cross-language golden-file feasibility

| Primitive | Reference test vectors | Cross-lang feasibility |
|---|---|---|
| L1-L4 ring + NTT | None standard — generate from Reality canonical | Easy (deterministic) |
| L5 CBD sampling | FIPS-203 KAT | Trivial |
| L6 uniform rejection | FIPS-203 KAT | Trivial |
| L7 discrete Gaussian | None — generate canonical | Hard (FP-heavy, σ varies; Falcon spec §B has reference vectors) |
| L8-L11 LWE | None standard | Medium |
| L12-L15 Kyber | FIPS-203 KAT (1000+ vectors) | Trivial |
| L16-L18 Dilithium | FIPS-204 KAT (1000+ vectors) | Trivial |
| L19-L22 Falcon | Falcon spec §A KAT | Medium (FP path is implementation-defined) |
| L23 NTRU | HPS 1998 reference + IEEE 1363.1 | Medium |
| L25 LLL | Cohen 1993 §2.6 worked examples | Easy |
| L28 BFV | None standard — Lattigo / SEAL reference | Hard (parameter-set-explosion) |
| L29 CKKS | None standard | Hard (approximate, tolerance-defined) |

KAT (Known-Answer-Test) vector availability is the single biggest correctness lever — Kyber/Dilithium ship with thousands of vectors from NIST CAVP, making cross-language byte-identical validation trivial.

---

## 6. What NOT to add (out-of-scope confirmation)

- **Hardware-backed implementations** (AVX2, NEON, AESNI, RISC-V V-ext): out of scope, reality is portable-Go.
- **AEAD-on-top-of-Kyber** (HPKE-Kyber, hybrid TLS): protocol composition, not math.
- **Side-channel countermeasures specific to embedded targets**: out of scope; document explicit threat model.
- **GPU/CUDA NTT**: out of scope.
- **Blockchain-specific lattice constructions** (StarkWare lattice-PCS, Plonky3 lattice-friendly): defer to a hypothetical zkmark/ extension.
- **Quantum-circuit cost estimation** (the post-quantum *security analysis* side): out of scope, this is mathematical-cost-modelling not algorithm.
- **Hash-based PQC** (SPHINCS+ = SLH-DSA, FIPS-205): NOT lattice-based, scope to a separate slot (not this one).
- **Code-based PQC** (Classic McEliece, BIKE, HQC, RFC PQC R4): NOT lattice-based, scope to slot 210 (coding theory).
- **Isogeny-based PQC** (SIKE, SQIsign): broken (SIKE 2022 by Castryck-Decru) or research-frontier (SQIsign), scope to a separate slot.

---

## 7. Non-overlap

- Versus 057-crypto-missing: 057 §T3-LATTICE was an 800-LOC stub at "schedule with consumer demand" priority; this slot upgrades to 6,440 LOC at production-required priority post-FIPS-203/204/206.
- Versus 058-crypto-sota: SOTA crypto review is signature/curve/pairing-focused; lattice crypto is its own universe, no duplication.
- Versus 060-crypto-perf: perf-axis review of existing crypto/ — this slot is greenfield, no overlap.
- Versus 210-new-coding-theory: shared finite-field-polynomial-arithmetic substrate — both need polynomial arithmetic and root-finding, but rings differ (GF(2^m) for RS vs cyclotomic Zq[X]/(X^n+1) for lattice). Recommend two parallel implementations rather than a unified abstraction (the IRingElement interface across substrates becomes an over-abstraction tax).
- Versus 057's T3-NTT line item: NTT was 200-LOC "schedule by demand" — this slot upgrades to 540 LOC across L3+L4+L1+L2 as the *first* deliverable, not a leaf.
- Versus 200-synergy-zkmark-info, 175-synergy-zkmark-crypto: zkmark could use Reality's NTT for FRI (Plonky3-style), but lattice-NTT and binary-FRI-NTT use different fields (Zq vs binary field); shared-kernel temptation is low.

---

## 8. Headline recommendation

Reality's "universal truth encoded in code" mission demands lattice cryptography because the 2024 FIPS-203/204/206 standardization made Kyber/Dilithium/Falcon the *required* post-quantum primitives for U.S. federal government use by 2030, and these standards are universal truth that belong in the library. The 6,440-LOC investment is large but partitionable: Tier-1 foundation (1,540 LOC) ships first as a math substrate, Kyber (640 LOC) ships next as the highest-production-value primitive, Dilithium / Falcon / LLL / FHE land as separate PRs. The cross-language golden-file substrate (Reality's signature differentiator) is well-served by FIPS-203/204 KATs (trivial validation) and worse-served by Falcon FP-sampler / CKKS approximate FHE (medium-hard validation, but tractable with seeded determinism). Most importantly: **lattice-crypto does NOT block on big-integer arithmetic** (the 400-LOC blocker that gates 057 Sequence A) since all moduli fit in uint32, so this sequence can ship in parallel with 057 Sequence A rather than queuing behind it.

Report length: 392 lines (under 400 cap).
