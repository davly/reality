# 304 — dive-rng-quality (TestU01 BigCrush + NIST SP 800-22 audit of reality's PRNGs)

## Headline
Three classical PRNGs (MT19937-64, PCG-XSH-RR, xoshiro256**) are correctly implemented and seedable, but reality has zero TestU01/PractRand/NIST 800-22 acceptance gates, no cryptographic CSPRNG (ChaCha20/AES-CTR) despite living in the `crypto` package, the PCG variant is 32-bit-output only and lacks Uint64, golden coverage is asymmetric (MT has 10 vectors — below the 20-vector minimum, PCG and xoshiro have zero goldens), and 16+ consumer files (`optim/metaheuristic.go`, `optim/genetic_test.go`, `prob/conformal/*_test.go`, `gametheory_test.go`, `timeseries/garch`, `topology/persistent`, `infogeo/mmd`, `changepoint/bocpd`, `info/lz`) bypass crypto entirely and use Go-stdlib `math/rand`, which silently breaks bit-identical cross-language reproducibility (Go's `math/rand` ≠ Python `random` ≠ C++ `std::mt19937` ≠ C# `Random`).

## Findings (existing PRNG audit)

- **`crypto/rng.go:16-39` — MT19937-64**: faithful Matsumoto-Nishimura 1998 reimplementation; period 2^19937 - 1; state 312×uint64 = 2.5 KiB; deterministic (`NewMersenneTwister(seed uint64)`). Tempering constants at `:51-54` match reference. Reference cited.
- **`crypto/rng.go:93-127` — PCG-XSH-RR (O'Neill 2014)**: 64-bit state / **32-bit output**; period 2^64; deterministic per `(seed, seq)` stream. Output function at `:113-122` is XSH-RR (correct for "PCG32"). **NO `Uint64()` method** — callers wanting 64-bit must call `Uint32()` twice; this is a footgun (entropy mixing across the 32-bit boundary is not equivalent to a true PCG64-XSL-RR). LCG multiplier `6364136223846793005` matches reference (line `:116`).
- **`crypto/rng.go:139-176` — xoshiro256**: Blackman-Vigna 2021 with SplitMix64 seeding (`:149-152`); period 2^256 - 1; state 4×uint64 = 32 B; deterministic. Constants `5,7,9` and rotations `17,45` at `:158-168` match reference. **No `Jump()` / `LongJump()` methods** for parallel streams — required for any parallel/embarrassingly-parallel MC use.
- **`crypto/rng.go:182-192` — splitmix64**: Steele 2014; used internally for xoshiro seeding only; not exposed as a PRNG type, even though it is the canonical seed-expander other languages need to match the goldens.
- **No ChaCha20 / AES-CTR / Salsa20 / Hash-DRBG**. The package is named `crypto` but contains zero cryptographic PRNGs. ARCHITECTURE.md `:54` claims `MersenneTwister, XorShift64` — `XorShift64` does not exist in the file (stale doc).
- **Golden coverage gap** (`testdata/crypto/`):
  - `mersenne_twister.json` — 10 vectors (CLAUDE.md mandates **≥20**, target 30). Only seed=42, no IEEE/edge cases for seed (0, 1, 2^64-1, etc.).
  - **No `pcg.json`, no `xoshiro256.json`** — PCG and xoshiro have only inline `KnownSequence` tests (5 vectors each in `crypto_test.go:658, :719`). Cross-language ports cannot validate against these.
- **`crypto/crypto_test.go:580-837` — local statistical gates**: 10-bin chi-squared on 10 000 samples per PRNG (df=9, threshold 27.88, p<0.001). This is the entire statistical-quality acceptance for the library. By comparison: TestU01 SmallCrush runs 10 tests on ~10^9 samples; Crush runs 96 tests on ~10^11; BigCrush runs 106 tests on ~10^12. **Reality runs 1 test on 10^4 samples.**
- **No NIST SP 800-22 gates** anywhere (frequency, runs, longest-run, binary-matrix-rank, DFT, non-overlapping-template, overlapping-template, Maurer's Universal, linear-complexity, serial, approximate-entropy, cumulative-sums, random-excursions, random-excursions-variant — 15 tests total).
- **`prob/` does NOT use crypto's PRNGs**: `prob/distributions.go` is pure CDF/PDF (no sampling); only `prob/conformal/*_test.go` uses `math/rand` for test data. Confirmed by grep — zero matches for `crypto/rand`, `math/rand`, `rand.New` in `prob/*.go` (non-test).
- **`optim/metaheuristic.go:5,44`** uses `*rand.Rand` (stdlib) — simulated annealing, GA. This is the highest-volume in-repo PRNG consumer and it bypasses crypto entirely.
- **TestU01 BigCrush expected results** (L'Ecuyer-Simard 2007, plus updated 2018-2024 results from Vigna's prng.di.unimi.it and PractRand 0.95):
  | Generator | SmallCrush | Crush | BigCrush | PractRand |
  |---|---|---|---|---|
  | MT19937-64 (current default) | PASS | FAIL 2 (LinearComp, MatrixRank — known) | FAIL 2 same | fails ≥256 GiB |
  | PCG32 (XSH-RR) | PASS | PASS | **2 known borderline** in BigCrush at >2^36 outputs (period exhaustion-class) | passes 32 TiB |
  | xoshiro256** | PASS | PASS | PASS | passes 32 TiB+ |
  | (missing) ChaCha20 / AES-CTR | PASS | PASS | PASS | passes ∞ (cryptographic) |
  | (stdlib `math/rand`, used in optim) | PASS | FAIL several (lagged-Fibonacci ALFG-like) | FAIL several | n/a |
- **R-MUTUAL-CROSS-VALIDATION 3/3 status**: **NOT saturated for any PRNG**. Required: same seed → bit-identical sequence in Go ≡ Python (port) ≡ C++ (port) ≡ C#. Currently only 1/3 axes (Go) is gated by goldens, and only for MT.

## Concrete recommendations

1. **`crypto/rng.go:127` — add `(*PCG).Uint64()`** that combines two `Uint32()` calls high|low (~6 LOC), OR add proper PCG64-XSL-RR (128-bit state, 64-bit output). The latter is ~40 LOC and is what BigCrush results in the literature actually refer to. Without this, the `PCG` type is asymmetric with the other two.
2. **`crypto/rng.go:197` — add `(*Xoshiro256).Jump()` and `LongJump()`** (Vigna constants `0x180ec6d33cfd0aba, 0xd5a61266f0c9392c, 0xa9582618e03fc9aa, 0x39abdc4529b1661c` and the long-jump set). ~30 LOC. Required for parallel MC chains and reproducible per-thread streams.
3. **`crypto/rng.go` — add `splitmix64` as exported `SplitMix64` type** (`type SplitMix64 struct{ state uint64 }`, `Uint64()`, `Float64()`). ~25 LOC. Already implemented internally; just needs a public wrapper. This is the canonical seeder other-language ports need.
4. **`crypto/csprng.go` (NEW, ~150 LOC) — add ChaCha20 stream-cipher PRNG** per RFC 7539. Constructor `NewChaCha20PRNG(key [32]byte, nonce [12]byte) *ChaCha20PRNG`. Method `Uint64()`, `Read([]byte)`, `Float64()`. Pure stdlib (math/binary). Trivially passes BigCrush + NIST SP 800-22; this is the only acceptable PRNG for any "secure" use. Currently the `crypto` package has zero CSPRNGs — a naming-vs-content mismatch.
5. **`testdata/crypto/mersenne_twister.json`** — extend from 10 → ≥30 vectors. Add seeds {0, 1, 2, 42, 12345, 0xdeadbeef, 2^64-1}, indices {0, 1, 311 (last in first twist), 312 (first after twist), 623 (last in second twist), 9999, 99999}. Tolerance must remain 0 (exact integers). ~40 lines JSON.
6. **`testdata/crypto/pcg.json` (NEW)** — ≥30 vectors covering `(seed, seq)` ∈ {(42,54), (0,0), (1,1), (0xdeadbeef, 0xcafebabe), (2^64-1, 2^63-1)} × indices {0, 1, 5, 100, 10000}. Tolerance 0. ~30 lines JSON.
7. **`testdata/crypto/xoshiro256.json` (NEW)** — same matrix; ≥30 vectors. The 5 inline expected values in `crypto_test.go:719-725` should move here. Tolerance 0.
8. **`testdata/crypto/splitmix64.json` (NEW)** — ≥30 vectors. Critical because every other reality PRNG that reseeds (or any port) bottoms out in SplitMix64.
9. **`crypto/rng_practrand_test.go` (NEW, build-tagged `//go:build practrand`, ~80 LOC)** — emit 1 GiB of each PRNG to stdout under a flag, for offline PractRand validation. Don't run by default (1 GiB × 3 PRNGs). This is the cheapest way to get genuine empirical statistical validation without porting TestU01.
10. **`crypto/rng_nist_test.go` (NEW, ~250 LOC)** — implement NIST SP 800-22 §2 (frequency, block-frequency, runs, longest-run, binary-matrix-rank). These five plus DFT cover ~80% of NIST acceptance failures. Run on 10^6 bits per PRNG. Pure stdlib.
11. **`docs/PRNG.md` (NEW or fold into ARCHITECTURE.md `:54`)** — document the precision/use-case matrix:
    | Use case | Recommended | Avoid |
    |---|---|---|
    | Stats sampling, MC integration | `Xoshiro256` | `MersenneTwister` (slow, fails 2 BigCrush) |
    | Reproducible test fixtures | `Xoshiro256` (with goldens) | `math/rand` (Go-only) |
    | Cryptographic / nonce generation | `ChaCha20PRNG` (TODO) | all of the above |
    | Massive parallel chains | `Xoshiro256` + `Jump()` (TODO) | `MersenneTwister`, `PCG32` |
12. **Migrate `optim/metaheuristic.go:44`** from `*rand.Rand` to a `crypto.RNG` interface (`Uint64() uint64; Float64() float64`). ~15 LOC change in metaheuristic + one-line interface in `crypto/rng.go`. Same pattern for `optim/genetic_test.go:5`. Not strictly required for v0.10 but enables cross-language reproducibility of optimization results — currently impossible.
13. **Update ARCHITECTURE.md `:54, :290`** — replace stale `XorShift64` references; add `PCG`, `Xoshiro256`, planned `ChaCha20PRNG` and `SplitMix64`.

## Cross-language reproducibility

- **Today**: 1/3 axes saturated. Go MT has goldens (10 vectors). PCG and xoshiro have zero goldens. Python's `random` defaults to MT19937-32 (different output bits than MT19937-64); C++ `std::mt19937_64` *does* match if seeded the same way (single uint64 seed → same key-init via `seed_seq`-or-not depending on constructor — landmine). C# `System.Random` is xoshiro256** in .NET 6+ but uses different seed-expansion (no SplitMix64 — instead a custom hashing of the int seed). NumPy's `Generator(PCG64())` is PCG64-XSL-RR, **not** PCG-XSH-RR — different output function, sequences will not match reality's PCG-XSH-RR.
- **R-MUTUAL-CROSS-VALIDATION 3/3 pin opportunity**:
  1. Generate 1024 outputs from each of `{SplitMix64, MT19937-64, PCG-XSH-RR (with explicit Uint64 = hi<<32|lo concat), Xoshiro256**}` for 5 seeds in Go.
  2. Hand-port each algorithm to Python (≤80 LOC each, pure Python no NumPy — NumPy's PRNGs are different variants), C++17 (`uint64_t` arithmetic only, no `<random>`), and C# (`ulong`).
  3. Assert bit-identical sequences in CI (or in checked-in vector artifacts). 4/4 axes ≡ 4 ports + Go canonical.
  - Land in this order to saturate 3/3: **Go-canonical golden generation (already partial for MT) → Python port + golden validation → C++ port + golden validation**. C# is 4th axis (bonus).
- **Most important: SplitMix64 seeding is the keystone.** Python's `random.seed(int)` does not produce the same internal state as reality's `NewXoshiro256(seed)` because reality uses `splitmix64(&seed)` four times. Document this loud or every cross-language port will silently produce different sequences from the same seed.

## Cheapest day-1 PR (≤300 LOC, ~4 hours)

1. Add `(*PCG).Uint64()` (6 LOC) + tests (10 LOC).
2. Add `Xoshiro256.Jump()` and `LongJump()` (30 LOC) + tests (20 LOC).
3. Export `SplitMix64` type (25 LOC) + tests (15 LOC).
4. Generate `testdata/crypto/{pcg,xoshiro256,splitmix64}.json` with 30 vectors each via Go's `math/big` at 256-bit precision (zero deps; existing testutil pattern).
5. Extend `mersenne_twister.json` to 30 vectors.
6. Add `TestGoldenPCG`, `TestGoldenXoshiro256`, `TestGoldenSplitMix64` paralleling existing `TestGoldenMersenneTwister` at `crypto_test.go:594`.
7. Update `ARCHITECTURE.md` PRNG entries.

This is pure infra work, zero algorithmic risk, +1 deterministic golden axis per PRNG. Saturates the Go side of R-MUTUAL-CROSS-VALIDATION (1/3 → 1/3 with stronger gates). Day-2 PR adds Python port (1/3 → 2/3); day-3 adds C++ port (2/3 → 3/3 saturated).

## Sources

- `C:\limitless\foundation\reality\crypto\rng.go` (full file, 197 LOC)
- `C:\limitless\foundation\reality\crypto\crypto_test.go:579-837` (PRNG tests)
- `C:\limitless\foundation\reality\testdata\crypto\mersenne_twister.json` (10 vectors)
- `C:\limitless\foundation\reality\ARCHITECTURE.md:54, :290` (stale `XorShift64` reference)
- `C:\limitless\foundation\reality\docs\STRUCTURE.md:72` (lists `MersenneTwister, NewMersenneTwister, PCG, NewPCG, Xoshiro256, NewXoshiro256` — confirms inventory)
- `C:\limitless\foundation\reality\optim\metaheuristic.go:5,:44` (consumer using stdlib `math/rand` instead of crypto PRNGs — bypass)
- `C:\limitless\foundation\reality\prob\conformal\*_test.go:5` (3 files using `math/rand`)
- L'Ecuyer & Simard, "TestU01: A C library for empirical testing of random number generators", ACM TOMS 33(4), 2007 — defines SmallCrush/Crush/BigCrush.
- Matsumoto & Nishimura, "Mersenne Twister: A 623-Dimensionally Equidistributed Uniform PRNG", ACM TOMS 1998 — MT19937 reference.
- O'Neill, "PCG: A Family of Simple Fast Space-Efficient Statistically Good Algorithms for RNG", Tech Report HMC-CS-2014-0905, 2014.
- Blackman & Vigna, "Scrambled Linear Pseudorandom Number Generators", ACM TOMS 47(4), 2021. https://prng.di.unimi.it/
- Steele, Lea & Flood, "Fast splittable pseudorandom number generators", OOPSLA 2014 — SplitMix64.
- Bernstein, "ChaCha, a variant of Salsa20" (2008); RFC 7539 (2015).
- NIST SP 800-22 Rev. 1a, "A Statistical Test Suite for Random and Pseudorandom Number Generators for Cryptographic Applications" (2010).
- PractRand 0.95 results (Doty-Humphrey, public test data) — MT19937 fails ≥256 GiB; xoshiro256** passes 32 TiB+.
- NumPy `numpy.random.PCG64` docs — confirms NumPy ships PCG64-XSL-RR (not XSH-RR like reality).
- .NET 6+ `System.Random` source — confirms xoshiro256** under the hood but with .NET-specific seed expansion (not SplitMix64).
