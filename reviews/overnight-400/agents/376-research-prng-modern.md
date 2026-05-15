# 376 — research-prng-modern (modern PRNGs + quality criteria)

## Headline
The 2025-2026 PRNG state of the art splits into four tiers — fast non-crypto (xoshiro256++/**, PCG64-DXSM, RomuTrio, wyrand, SplitMix64) at 0.3-0.9 ns/u64 all passing BigCrush+PractRand≥32 TiB, splittable (LXM L64X128/L128X256, replacing 2014 SplitMix), counter-based parallel (Philox-4x64, Threefry-4x64) for embarrassingly-parallel MC, and CSPRNGs (ChaCha8/20, AES-CTR-DRBG, HMAC-DRBG); reality v0.10 ships only MT19937-64 + PCG-XSH-RR (32-bit output, no Uint64) + xoshiro256** with no Jump/LongJump and zero CSPRNGs, missing the entire bottom three tiers and shipping the obsolete weak-output PCG variant that NumPy moved off in 2019.

## Survey

### 1. xoshiro256** / xoshiro256++ (Blackman & Vigna, 2018, ACM TOMS 47(4) 2021)
256-bit state, 4×uint64, period 2^256-1, 0.86 ns/u64 (i7-8700B). Linear F2 generator + non-linear scrambler (`** = *5 rotl 7 *9` for 64-bit; `++ = +X rotl R + Y` for top bits). Passes BigCrush, PractRand 32 TiB+. Has **Jump()** (constants `0x180ec6d33cfd0aba, 0xd5a61266f0c9392c, 0xa9582618e03fc9aa, 0x39abdc4529b1661c`, equivalent to 2^128 calls) and **LongJump()** (2^192). Reality ships ** without jumps. **Known weakness**: low bits of `+` variant are linear (xoshiro256+ low 4 bits fail BRank); use `**` or `++` for full-precision use. O'Neill's 2018 critique of xoshiro repeat-pattern at distance 2^126 is statistical not algorithmic — no consequence for typical use.

### 2. xoroshiro128++ / xoroshiro128** / xoroshiro128+ (Vigna 2018)
128-bit state. Half the state, half the period (2^128-1). 0.72 ns/u64. **xoroshiro128+ fails linearity tests in low bits** (Lemire 2018, ScienceDirect). **xoroshiro128++ is the recommended 128-bit variant** — passes BigCrush + PractRand. Use only when 256-bit state is overkill (e.g., per-thread with millions of threads where state size matters). Java JDK17 ships xoroshiro128++ as one of the five `RandomGenerator` algorithms.

### 3. PCG family (O'Neill 2014, HMC-CS-2014-0905)
Linear-congruential subgenerator + permuted output. **PCG-XSH-RR** (64-bit state, 32-bit output) — what reality ships at `crypto/rng.go:113-122`; passes BigCrush; NumPy DEPRECATED for parallel use in 2019. **PCG-XSL-RR** (128-bit state, 64-bit output) — NumPy's original PCG64. **PCG64-DXSM** (NumPy's 2019 default upgrade, "cheap multiplier" + DXSM permutation = double-xorshift-multiply) fixes a 58-bit collision weakness in XSL-RR. **Recommended PCG variant 2025: PCG64-DXSM.** Streams via "increment" parameter (2^63 streams), but Vigna's 2017 critique noted reparameterized streams are "different generators no one tested" — O'Neill counters the seeding pool makes collisions astronomically unlikely.

### 4. Romu family (Overton 2019, arXiv:2002.11331)
Rotate-multiply. Mixes nonlinear rotation with linear ops. **RomuTrio** (192-bit state, 3×uint64) — recommended 64-bit generator, **0.31 cpb** (cycles-per-byte, fastest in family). **RomuQuad** (256-bit state) — extra safety margin. **RomuDuoJr** (128-bit state) — fastest but smallest, only for short sequences (≤2^51 outputs by author's recommendation). Passes BigCrush + PractRand. **Caveat**: chaotic (no guaranteed period), short cycles statistically rare but possible (~2^-64 probability per seed). Very fast on superscalar CPUs (zero output latency when inlined). Not in NumPy/JDK; available in RandomGen (Python) + crates.io `romu`.

### 5. SplitMix64 (Steele, Lea, Flood, OOPSLA 2014, dl.acm.org/10.1145/2660193.2660195)
64-bit state, 9 ops/output. Period 2^64. Now in `java.util.SplittableRandom` and used as the canonical seed-expander for xoshiro/xoroshiro. **Passes DieHarder + TestU01.** Two known weaknesses (zeroland — low Hamming-weight outputs near zero state, and split-correlation in fork-join trees) that motivated LXM. **Reality uses it internally for xoshiro256 seeding (`crypto/rng.go:182-192`) but does not export it** — this is the keystone fix for cross-language reproducibility (every other-language port must replicate SplitMix64 exactly to match reality's seeded sequences).

### 6. LXM family (Steele & Vigna, OOPSLA 2021, dl.acm.org/10.1145/3485525)
**Replaces SplitMix64 in JDK17.** L = LCG (period 2^16/32/64/128) × M = Mix function × X = xor-based F2 generator (xoshiro/xoroshiro). Six variants in JDK17: L32X64MixRandom, L64X128MixRandom (default), L64X128StarStarRandom, L64X256MixRandom, L64X1024MixRandom, L128X128MixRandom, L128X256MixRandom, L128X1024MixRandom. Splittable (split() returns independent generator) without the SplitMix correlation flaws. Passes BigCrush. **Recommended 2025 splittable PRNG.** Reality has zero splittable PRNG.

### 7. Philox-4x64 / Philox-2x32 (Salmon, Moraes, Dror & Shaw, SC11 2011)
**Counter-based** (CBRNG): output = key-dependent reduced-round Feistel cipher applied to counter. Stateless beyond `(key, counter)`. 4×uint64 = 256-bit counter, 2×uint64 = 128-bit key. **Embarrassingly parallel**: thread N samples at counter=N without coordination. Passes BigCrush. Adopted in: **NumPy `Philox`**, TensorFlow, JAX (`jax.random` is Threefry-2x32 with same property), nVidia cuRAND, Intel MKL. Speed: ~1.5 ns/u64 sequentially but **scales linearly to N cores with zero contention**. **Reality has zero counter-based PRNG** — this is the only practical option for reproducible distributed Monte Carlo.

### 8. Threefry-4x64 (Salmon et al. 2011, same paper)
Sister to Philox: Threefry uses a Threefish-derived cipher (XOR + rotation, no multiplication). Passes BigCrush. Slightly slower than Philox on x86 (multiplication is fast there) but **faster on architectures without fast multiply**. Default in JAX. Same counter-based parallel-friendly properties as Philox.

### 9. wyrand (Wang Yi, 2020, github.com/wangyi-fudan/wyhash)
64-bit state, MUM-mixing function, 1 multiply + 1 xor per output. **Fastest sequential PRNG that passes BigCrush + PractRand + SMHasher** (Lemire's testingRNG). ~0.4 ns/u64. Period 2^64 (short — limits MC sample budget to ~2^32 by Birthday). Used in `absl::BitGen`, several Rust crates. **Caveat**: 64-bit state is too small for serious MC; use as fast scratch generator only.

### 10. ChaCha8 / ChaCha20 as PRNG (Bernstein 2008; RFC 7539; Go 1.22 ChaCha8Rand 2024)
**Cryptographically secure**. ChaCha20 = 20-round stream cipher; ChaCha8 = 8-round (Go 1.22+ uses for `math/rand/v2.ChaCha8` and seeds `crypto/rand` reseeding). 256-bit key + 96-bit nonce + 32-bit counter = 2^96 × 2^32 × 64 bytes per key. **Trivially passes BigCrush + NIST SP 800-22 + PractRand ∞.** ChaCha8 ~2.5× faster than ChaCha20, both AVX2-vectorizable. **Go 1.22 blog (go.dev/blog/chacha8rand)**: cryptographic seed → ChaCha8 stream is the new `math/rand` default. Reality should ship ChaCha20Rand (RFC 7539) at minimum.

### 11. AES-CTR-DRBG (NIST SP 800-90A Rev. 1, 2015)
AES-128/192/256 in CTR mode with reseeding interval ≤ 2^48. **FIPS 140-3 approved**, the only PRNG accepted in regulated environments. Hardware-accelerated via AES-NI (~0.5 ns/byte on Intel post-2010). 2025 Go reference impl: `github.com/sixafter/aes-ctr-drbg` (zero-alloc, RFC-validated). Hoang & Shen, eprint.iacr.org/2020/619: security proof tight to 2^60 bits per instantiation. Reality has zero AES; needs the AES algebraic substrate first (per slot 322).

### 12. HMAC-DRBG / Hash-DRBG (NIST SP 800-90A)
HMAC-DRBG = HMAC-SHA256 in feedback mode; Hash-DRBG = SHA-256 in feedback mode. Slower than AES-CTR-DRBG (~5 ns/byte for SHA-256) but no AES-NI dependency, identical security guarantees, simpler to implement (no GF(2^8) S-box). Cisco/OpenSSL ship HMAC-DRBG as primary. Reality lacks SHA-256 (per slot 322); both are gated on slot 322's recommendation.

## TestU01 / PractRand 2025 quality matrix

| Generator | State | ns/u64 | SmallCrush | Crush | BigCrush | PractRand |
|-----------|-------|--------|------------|-------|----------|-----------|
| MT19937-64 | 2.5 KiB | ~3.0 | PASS | FAIL 2 | FAIL 2 | fails 256 GiB |
| PCG-XSH-RR (reality) | 8 B | 0.9 | PASS | PASS | PASS | passes 32 TiB |
| PCG64-DXSM | 16 B | 1.1 | PASS | PASS | PASS | passes 32 TiB+ |
| xoshiro256** | 32 B | 0.86 | PASS | PASS | PASS | passes 32 TiB+ |
| xoshiro256++ | 32 B | 0.86 | PASS | PASS | PASS | passes 32 TiB+ |
| xoroshiro128+ | 16 B | 0.72 | PASS | low-bit FAIL | low-bit FAIL | fails low bits |
| xoroshiro128++ | 16 B | 0.78 | PASS | PASS | PASS | passes 32 TiB |
| RomuTrio | 24 B | 0.31* | PASS | PASS | PASS | passes 32 TiB+ |
| SplitMix64 | 8 B | 0.5 | PASS | PASS | PASS | passes 32 TiB |
| LXM L64X128Mix | 24 B | 0.9 | PASS | PASS | PASS | passes 32 TiB+ |
| Philox-4x64 | 32 B (counter+key) | 1.5 seq | PASS | PASS | PASS | passes 32 TiB+ |
| Threefry-4x64 | 32 B | 1.8 seq | PASS | PASS | PASS | passes 32 TiB+ |
| wyrand | 8 B | 0.4 | PASS | PASS | PASS | passes 32 TiB |
| ChaCha8 | 64 B | 1.2 | PASS | PASS | PASS | passes ∞ |
| ChaCha20 | 64 B | 2.8 | PASS | PASS | PASS | passes ∞ |
| AES-CTR-DRBG | 48 B | 0.5 (AES-NI) | PASS | PASS | PASS | passes ∞ |

*0.31 cpb = cycles per byte, ~0.04 ns/u64 at 4 GHz inlined; Overton's claim, independently observed by Lemire.

## Reality positioning

Slot 304 already enumerated reality's gaps. This research slot adds the **modern target inventory**:

**Tier 0 (ship in v0.11): Goldens + utility methods on existing PRNGs.**
- `Xoshiro256.Jump()` + `LongJump()` — required for any parallel MC; ~30 LOC.
- `PCG.Uint64()` (concat of two Uint32) OR full PCG64-DXSM (128-bit state) — the 32-bit-only variant is obsolete for 2025 use; ~40 LOC.
- Export `SplitMix64` as public type (the seed-expansion keystone for cross-language ports); ~25 LOC.
- 30-vector goldens for all four (MT, PCG, xoshiro256, splitmix64); ~120 lines JSON.
- **Migrate `optim/metaheuristic.go:5,44`, `optim/genetic_test.go:5`, `prob/conformal/*_test.go` from `math/rand` to a `crypto.RNG` interface** — currently breaks cross-language reproducibility silently.

**Tier 1 (v0.12, 2026 Q3): Splittable + counter-based.**
- **LXM L64X128MixRandom** as `crypto/lxm.go` — JDK17's default splittable PRNG; ~80 LOC + goldens. Saturates the splittable use case (parallel sub-streams without Jump bookkeeping).
- **Philox-4x64** as `crypto/philox.go` — counter-based parallel; ~120 LOC + goldens. Required for any reproducible distributed Monte Carlo (NumPy/JAX/TensorFlow shipped this for a reason).
- These two cover the entire "modern non-crypto" frontier.

**Tier 2 (v0.13+, gated on slot 322 crypto substrates): CSPRNGs.**
- **ChaCha20Rand** (RFC 7539) — required for any "cryptographic" claim in `crypto/`. ~150 LOC pure-stdlib. Trivially passes NIST SP 800-22; no AES-NI dependency. **This is what Go 1.22 chose for `math/rand/v2`.** Should be reality's default CSPRNG.
- **AES-CTR-DRBG** (NIST SP 800-90A) — FIPS-compliant CSPRNG. Gated on AES algebraic substrate (slot 322 IN-scope). ~200 LOC.
- **HMAC-DRBG-SHA256** — alternative path if AES is delayed. Gated on SHA-256 (slot 322).

**Skip / do not ship:**
- **xoroshiro128+** — has low-bit linearity failure; obsolete in favor of xoroshiro128++.
- **xoshiro256+** — same low-bit failure; ship `**` or `++` only.
- **wyrand** — 64-bit state too small; "fastest" claim is real but the 2^32 birthday limit makes it unfit for reality's MC consumers.
- **RomuDuoJr** — chaotic with no guaranteed period; user-confusion risk.
- **PCG-XSH-RR (current reality default)** — keep for backward compat but document as legacy; recommend PCG64-DXSM as primary PCG for new code.

**R-MUTUAL-CROSS-VALIDATION strategy:** The keystone is **SplitMix64**. Every other reality PRNG bottoms out in SplitMix64 for seed expansion. Land SplitMix64 with 30-vector goldens FIRST; then xoshiro256/PCG/Philox port goldens are auto-derived. Python's `random.seed(int)` uses MT19937-32 with a different bytes-to-state map; NumPy's `default_rng` uses `SeedSequence` (independent of SplitMix64); C# `System.Random` (.NET 6+) uses xoshiro256** with its own seed expansion. **None of these match reality's `splitmix64(&seed)→4×uint64→xoshiro state`** — must be hand-ported in any cross-language validation.

**Quality gates reality should add (per slot 304):**
- `crypto/rng_practrand_test.go` build-tagged `//go:build practrand` — emit 1 GiB stdout per PRNG for offline PractRand 0.95 validation.
- `crypto/rng_nist_test.go` — implement NIST SP 800-22 §2 frequency, block-frequency, runs, longest-run, binary-matrix-rank tests on 10^6 bits per PRNG. Pure stdlib, ~250 LOC, ~80% of NIST acceptance failures captured.
- Document in `docs/PRNG.md` (NEW) the use-case → recommended-PRNG matrix:
  | Use case | Recommended | Avoid |
  |---|---|---|
  | Sequential MC sampling | xoshiro256** | MT19937 (fails BigCrush) |
  | Parallel MC (N threads) | Philox-4x64 (counter-based) | xoshiro w/o Jump() |
  | Splittable fork-join | LXM L64X128Mix | SplitMix64 (split flaws) |
  | Reproducible test fixtures | xoshiro256** + golden | math/rand (Go-only) |
  | Cryptographic / nonce | ChaCha20Rand | all non-crypto above |
  | FIPS-regulated | AES-CTR-DRBG | non-NIST DRBGs |

## Sources

- Blackman, Vigna, "Scrambled Linear Pseudorandom Number Generators", arXiv:1805.01407 (2018), ACM TOMS 47(4) (2021). https://prng.di.unimi.it/
- Overton, "Romu: Fast Nonlinear Pseudo-Random Number Generators Providing High Quality", arXiv:2002.11331 (2020). https://www.romu-random.org/
- Salmon, Moraes, Dror, Shaw, "Parallel Random Numbers: As Easy as 1, 2, 3", SC11 (2011). https://www.thesalmons.org/john/random123/papers/random123sc11.pdf
- O'Neill, "PCG: A Family of Simple Fast Space-Efficient Statistically Good Algorithms for RNG", HMC-CS-2014-0905 (2014). https://www.pcg-random.org/
- O'Neill, "On Vigna's PCG Critique" (2018). https://www.pcg-random.org/posts/on-vignas-pcg-critique.html
- Steele, Lea, Flood, "Fast splittable pseudorandom number generators", OOPSLA (2014). https://gee.cs.oswego.edu/dl/papers/oopsla14.pdf
- Steele, Vigna, "LXM: better splittable pseudorandom number generators (and almost as fast)", OOPSLA (2021). https://vigna.di.unimi.it/ftp/papers/LXM.pdf
- L'Ecuyer, Simard, "TestU01: A C library for empirical testing of random number generators", ACM TOMS 33(4) (2007).
- Lemire, "testingRNG" repository — PractRand + BigCrush results 2017-2025. https://github.com/lemire/testingRNG
- NumPy "Upgrading PCG64 with PCG64DXSM" docs — DXSM rationale and switchover. https://numpy.org/doc/stable/reference/random/upgrading-pcg64.html
- Vigna, "Xorshift1024*, xorshift1024+, xorshift128+ and xoroshiro128+ fail statistical tests for linearity", J. Computational Applied Math (2018). https://www.sciencedirect.com/science/article/pii/S0377042718306265
- Bernstein, "ChaCha, a variant of Salsa20" (2008); RFC 7539 "ChaCha20 and Poly1305 for IETF Protocols" (2015).
- Go Blog, "Secure Randomness in Go 1.22" (2024) — ChaCha8Rand. https://go.dev/blog/chacha8rand
- NIST SP 800-90A Rev. 1, "Recommendation for Random Number Generation Using Deterministic Random Bit Generators" (2015). https://nvlpubs.nist.gov/nistpubs/specialpublications/nist.sp.800-90ar1.pdf
- Hoang, Shen, "Security Analysis of NIST CTR-DRBG", IACR ePrint 2020/619. https://eprint.iacr.org/2020/619.pdf
- Wang, "wyhash: The FASTEST QUALITY hash function and PRNG". https://github.com/wangyi-fudan/wyhash
- DEShawResearch/random123 — Philox + Threefry reference impl. https://github.com/DEShawResearch/random123
- JDK17 JEP 356 "Enhanced Pseudo-Random Number Generators" — LXM family integration. https://openjdk.org/jeps/356
- Slot 304 (`agents/304-dive-rng-quality.md`) — reality's existing PRNG audit, R-MUTUAL-CROSS-VALIDATION 1/3 status, MT golden gap.
- Slot 322 (`agents/322-dive-aes-vs-poly1305.md`) — gates AES-CTR-DRBG and HMAC-DRBG on AES + SHA-256 algebraic substrates which reality currently lacks.
