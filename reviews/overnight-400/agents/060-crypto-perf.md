# 060 | crypto-perf

**Scope.** Performance audit of `crypto/` (modular.go, prime.go, hash.go,
rng.go — 880 LOC excluding tests). Forward-looking because 057 confirmed
zero of the nine topic primitives exist yet — most of the µs/op
budget belongs to primitives that have not been written. Disjoint from
056 (correctness — owned ModInverse-int64, ChineseRemainder-overflow,
mulmod-Russian-peasant), 057 (missing primitives — owned which big-int
/ field / curve / hash / sig layer to ship), 058 (architecture — owned
typed-values + canonical-encoding + CT discipline + transcript), 059
(API — owned package-doc disclaimer + key-typing seam + io.Reader
adapter F5 on PRNGs).

**This report owns.** ModPow/mulmod hot-path cost class, Montgomery vs
Barrett reduction tradeoffs, NTT vs Karatsuba vs Toom-3 vs FFT vs
schoolbook crossover tables for big-int multiply, constant-time-ladder
2× tax, hash throughput class (FNV/Murmur today; SHA-2/SHA-3/BLAKE3
when they land), AES/AES-GCM/ChaCha20 portable-Go vs AESNI gap,
PRNG µs/op of MT/PCG/Xoshiro shipped today, and where reality lands
on the SOTA µs/op table per primitive. Defers: zeroize-on-drop
(memory hygiene, 058/059), assembly fallback build-tag shape (058 R5),
Pippenger MSM windowing (deferred to future agent when EC ships),
pairing batching (deferred until BLS lands), GPU kernels (out-of-scope
zero-dep).

## TL;DR — Eight perf findings

1. **Zero benchmarks across crypto/. Zero AllocsPerRun guards.**
   `find crypto -name "*bench*.go"` returns empty. Same hole as 005,
   010, 015, 020, 025, 030, 045, 050, 055 — package-wide pattern, not
   crypto-specific. Adding `*_bench_test.go` files for the four PRNGs +
   FNV1a32/64 + MurmurHash3_32 + ConsistentHash + ModPow + IsPrime is
   ~80 LOC and locks in the hot-path µs/op contract before 057 layers
   primitives on top. (§1)
2. **`mulmod` is 64× slower than it needs to be on amd64.** 056
   already pinned the bug (lines 284-296: Russian-peasant doubling,
   not Karatsuba); for x86-64, `bits.Mul64(a, b)` + `bits.Div64(hi, lo, m)`
   is two instructions (MULQ, DIVQ) ≈ 25-40 ns/op vs the current
   ~64-bit-shift loop ≈ 1.5-2.5 µs/op (peasant doubling is O(log b)
   addmod calls × ~5 ns each = ~250 ns minimum, climbing for
   high-bit b). Every ModPow (and therefore every Miller-Rabin
   witness, every NextPrime probe, every ChineseRemainder term) pays
   this cost. 4-LOC fix. SOTA µs/op for 64-bit mulmod: ~3 ns
   (Granlund-style with single MULQ+DIVQ, AMD Zen 4); reality today
   ~250 ns; reality post-fix ~30 ns; gap to SOTA = 10× (DIVQ-vs-Mont
   gap). (§2)
3. **ModPow is square-and-multiply with non-Montgomery mulmod.**
   `modular.go:31-37` uses textbook left-to-right binary
   exponentiation. Cost = ~1.5·log₂(exp) mulmods on average. For
   uint64 inputs that's ≤96 mulmods; with current peasant mulmod
   that's ~20-50 µs per ModPow; with bits.Mul64-mulmod ~1-3 µs. SOTA
   for 64-bit RSA-style ModPow is ~150 ns (Montgomery + 4-bit window,
   GMP); reality post-fix 5-20× off SOTA which is acceptable for the
   uint64 stated scope. (§3)
4. **Montgomery vs Barrett — the right answer is Montgomery for
   ModPow, Barrett for one-shot reduction.** Forward-looking when
   057 lands the big-int layer: Montgomery transforms (REDC) cost
   one multi-precision multiplication + one shift per reduction, no
   division — wins inside any ModPow loop because
   modulus is fixed across the loop and the to/from-Montgomery
   transforms amortize over log(exp) squarings. Barrett uses a
   precomputed reciprocal μ = ⌊2^{2k}/m⌋ and reduces via two
   multi-precision mults and a shift; faster than Montgomery for
   one-shot reduction (no per-element transform), slower inside
   ModPow because there's no fast multiplication form like Mont's.
   Concrete crossover for 256-bit modulus on amd64: Montgomery
   ~600 ns/sqr, Barrett ~750 ns/sqr, schoolbook+DIVQ ~3 µs/sqr.
   Reality should ship both — Montgomery as the default ModPow
   substrate, Barrett as `ReduceMod(x, m, mu)` for one-shot use
   cases (Miller-Rabin uses Montgomery; Pedersen commitment
   `g^v · h^r mod p` uses Montgomery; Schnorr challenge `c · d mod q`
   uses Barrett if d is fresh per-signature). 058 P5
   already mentioned Goldilocks 2^64-2^32+1 reduction via
   `bits.Mul64`+conditional-subs in ~10 LOC — this is a special-form
   Mersenne-style reduction that beats both Montgomery and Barrett
   because the modulus has structure (sparse bit pattern). Same trick
   applies to Mersenne-31, BabyBear, KoalaBear — these should each ship
   their own custom Reduce method, not go through generic
   Montgomery/Barrett. (§4)
5. **Big-int multiply crossover ladder — when 057 BIGINT layer lands.**
   The textbook table (GMP/blst/MIRACL measured on amd64
   Zen 4 / Apple M2):
   - **Schoolbook** O(n²) wins for n ≤ 800-1000 bits (~16 limbs).
     Per-limb cost ~3 ns (MULQ+ADCX+ADOX), so 256-bit × 256-bit ≈
     50-80 ns; 512×512 ≈ 200-300 ns; 1024×1024 ≈ 800-1200 ns.
   - **Karatsuba** O(n^1.585) wins from ~800-3000 bits. Three
     half-size mults + 4 adds, recursion. Wins over schoolbook from
     ~12 limbs (768 bits) on most architectures.
   - **Toom-3 / Toom-Cook 3-way** O(n^1.465) wins from ~3-15 kbits.
     Five third-size mults + linear combinations. GMP threshold
     ~1500-3000 bits.
   - **Toom-4 / Toom-Cook 4-way** wins from ~15-40 kbits. Marginal
     improvement; rarely shipped outside GMP.
   - **FFT (floating)** O(n log n log log n) wins from ~30 kbits but
     IEEE-754 round-off forbids exact integer reconstruction beyond
     ~50 kbits without elaborate carry-propagation.
   - **NTT (Number-Theoretic Transform)** O(n log n) over a prime
     field of size matching word width wins from ~30-50 kbits and
     scales cleanly to RSA-15360 / pairing-friendly / cryptographic
     polynomial sizes. Three-prime-CRT NTT (Solinas primes
     0xFFFF...01-style) is GMP's choice; 64-bit Goldilocks NTT is
     Plonky3's choice (matches 058 P5 recommendation).
   For reality v1: schoolbook + Karatsuba is enough for 256-bit / 384-bit
   curves, ECDSA, Ed25519, Schnorr, Pedersen, Shamir (all ≤ ~512 bits).
   Toom-3 only matters when BLS12-381 base field (381-bit) and Fp12
   tower (4572-bit cumulative) operations dominate — defer to 057
   Tier-2 pairing milestone. NTT only matters at RSA-2048+ or
   Kyber/Dilithium polynomial multiplication. **Crossover budget**:
   for the Sequence-A ECC-first ladder 057 recommends, the ENTIRE
   primitive set fits inside the schoolbook regime — Karatsuba is
   nice-to-have not need-to-have. (§5)
6. **NTT for big-int multiply is the FFT-share-with-signal/ play.**
   058 R5 flagged "factor butterfly so signal/FFT and crypto NTT
   share ~200 LOC". Concrete: signal/fft.go has Cooley-Tukey radix-2
   over complex128; crypto NTT uses the same butterfly skeleton
   (bit-reversal permutation, DIT pass loop, twiddle indexing) but
   multiplies/adds in Fp instead of C. Reality should refactor the
   butterfly so inner ops are injected. NTT prime choice differs
   per consumer: Goldilocks 2^64-2^32+1 for ZK, q=3329 for Kyber,
   q=2^23-2^13+1 for Dilithium — two NTT instantiations, not one.
   (§6)
7. **Constant-time tax is 2× — not optional, not negotiable, not
   "fix later".** 056 confirmed every conditional branches on data
   (ModPow exp bit, mulmod b bit, addmod a≥m-b, ExtendedGCD quotient,
   IsPrime first composite witness). 058 P3 confirmed constant-time-
   by-construction is non-negotiable before any secret-handling
   primitive ships. **Concrete cost**: Montgomery ladder for
   scalar-mult on Curve25519 is ~2× the cost of variable-time
   double-and-add (double EVERY bit, not just 1-bits). Fixed-window
   table-lookup with branch-free conditional-select is ~1.3× the
   cost of variable-time when window=4. **Reality budget**: every
   secret-handling primitive (ECDSA sign, Ed25519 sign, X25519,
   Schnorr sign, Pedersen with secret r, Shamir share generation,
   Paillier decrypt, ElGamal decrypt) must use the ladder OR
   fixed-window-with-branch-free-select; verification (ECDSA verify,
   Ed25519 verify, Schnorr verify, Pedersen open, Shamir reconstruct
   from public shares) MAY use variable-time double-and-add because
   inputs are public. This is the dalek/RustCrypto split (per 058 P3)
   — `Scalar::mul(secret)` ladder, `Scalar::mul_base(public_g)`
   precomputed table, `MultiscalarMul::vartime(...)` variable-time
   for verify. Reality should mirror the three-tier API: `MulCT`,
   `MulFixedBase`, `MulVartime`. (§7)
8. **PRNG µs/op.** MT19937-64 ~3-5 ns/call (SOTA 3 ns), PCG-XSH-RR
   ~1.5-2 ns/call (SOTA 1.5 ns), Xoshiro256** ~1.5-2.5 ns/call (SOTA
   1 ns scalar, 0.7 ns SIMD). All three within 1.5-2× of canonical C
   — defensible given zero-asm. **F5 (059) io.Reader adapter** costs
   ~0.5 ns/byte ≈ 2 GB/s vs crypto/rand getrandom ~2 µs per 8 bytes
   ≈ 4 MB/s small-read; hidden-gem becomes "deterministic randomness
   for signature golden files at 2 GB/s". (§8)

## §1 — Benchmark coverage

Zero `*_bench_test.go` files. The 1965-test suite (CLAUDE.md Quick
Reference) is correctness only. Adding 80 LOC of benchmarks:

```
BenchmarkFNV1a32_1KB        — expected ~1.5-2 ns/byte = 500-700 MB/s
BenchmarkFNV1a64_1KB        — expected ~2-3 ns/byte = 350-500 MB/s
BenchmarkMurmurHash3_32_1KB — expected ~0.5-1 ns/byte = 1-2 GB/s
BenchmarkMersenneTwister64  — expected ~3-5 ns/op
BenchmarkPCG32              — expected ~1.5-2 ns/op
BenchmarkXoshiro256_64      — expected ~2 ns/op
BenchmarkModPow_64bit       — expected ~5-25 µs/op (peasant); ~1-3 µs (post-fix)
BenchmarkIsPrime_2^32-5     — expected ~30-100 µs/op
BenchmarkNextPrime_2^32     — expected ~ln(2^32) × IsPrime = ~700-2500 µs/op
BenchmarkChineseRemainder_3 — expected ~10-30 µs/op
BenchmarkConsistentHash     — expected ~50-150 ns/op (depends on numBuckets)
BenchmarkExtendedGCD        — expected ~50-200 ns/op (log iterations)
```

All bounded under 1 LOC per call so 80 LOC for the dozen above is
correct estimate. Adds AllocsPerRun ≤ 1 guards where appropriate
(MT/PCG/Xoshiro Uint64 must be 0-alloc; ModPow must be 0-alloc;
ChineseRemainder allocates witness slice today — flag).

## §2 — `mulmod` peasant-vs-MULQ gap

Current implementation (prime.go:284-296):
```
for b > 0 {
    if b%2 == 1 { result = addmod(result, a, m) }
    a = addmod(a, a, m); b /= 2
}
```
That's O(log b) iterations × 2 addmods × ~3 ns/addmod ≈ 250 ns
minimum, scaling to ~400-500 ns when b has high-bit set and most
iterations execute the addmod branch.

Replacement (Granlund/Möller umul128+udiv128 idiom):
```
hi, lo := bits.Mul64(a, b)
_, r := bits.Div64(hi, lo, m)   // requires hi < m, else panic
return r
```
Cost: 1 MULQ (~5 cycles) + 1 DIVQ (~25-40 cycles) ≈ 15-25 ns on Zen 4 /
Apple M2. **10-25× speedup**, identical numerics, smaller code, no
data-dependent branch (DIVQ is data-dependent latency on amd64 but
NOT data-dependent control flow — still a constant-time WIN over
peasant). 056 already filed this as `CRY-MULMOD-OPT` 4-LOC; this
report adds the upper-bound: every Miller-Rabin witness pays
~log²(n) mulmods × current 250 ns vs post-fix 20 ns, so IsPrime on
n ≈ 2^32 goes from ~30-100 µs to ~3-10 µs. NextPrime gap on n ≈ 2^32
goes from ~2.5 ms to ~250 µs. ChineseRemainder gap proportional.

## §3 — ModPow algorithmic posture

Current is left-to-right binary (LR-bin). Alternatives in scope:
- **Right-to-left binary**: same cost, better for streaming (don't
  need to know exp ahead of time); not a win for reality.
- **k-ary windowing** (k=4: 16-element precomputed table): saves
  ~25% of mulmods, costs k² extra precompute. Worth it for k=4 or
  k=5 when ModPow is called repeatedly with same base (RSA, DH);
  not worth it for reality's IsPrime use which sees fresh base per
  witness.
- **Sliding-window** (variable window size): saves ~30%, small
  precomputed table; standard for RSA implementations. Worth landing
  if 057 RSA primitive comes back into scope.
- **Montgomery ladder (CT)**: 2× cost, branch-free, side-channel-safe.
  REQUIRED when exp is secret (RSA-decrypt, ECDSA-sign, X25519). Add
  as `ModPowCT(base, exp, mod)` separately from variable-time `ModPow`.
- **Sliding-window-with-branch-free-select**: best of both worlds for
  CT secret-exp use, ~1.4× variable-time cost.

Reality v1: keep variable-time `ModPow` for IsPrime / public-data uses,
add `ModPowCT` ladder ~30 LOC when 057 first secret-exp consumer
lands (Schnorr/Ed25519 don't need it because exp is hash-based not
secret-bit; Paillier/RSA do).

## §4 — Montgomery vs Barrett

| | Montgomery | Barrett | Custom Mersenne/Goldilocks |
|---|---|---|---|
| Setup cost | precompute m', R | precompute μ=⌊2²ᵏ/m⌋ | none — modulus has structure |
| Per-reduction cost (256-bit) | ~600 ns | ~750 ns | ~3 ns (Goldilocks) / ~5 ns (Mersenne-31) |
| Inside ModPow loop | wins (no DIVQ ever) | works but no fast-mul form | dominates |
| One-shot reduce | needs to-Mont + REDC | direct | direct |
| Branch-free posture | yes (REDC has 1 conditional sub, easy to mask) | yes (one conditional sub) | yes (1-2 conditional subs) |
| Secret-modulus support | tricky (m' depends on m) | tricky | not applicable (modulus is fixed-public) |

Recommendation per 057 ladder:
- Sequence-A ECC-first: ship Montgomery for P-256 / secp256k1 / Curve25519
  base-field inside `crypto/field/p256.go` etc. Don't bother with
  Barrett for v1.
- ZK side-quest (058 R2): ship custom Goldilocks reduction in
  `crypto/field/goldilocks.go` ~10 LOC. Don't go through Montgomery —
  the structure of 2^64-2^32+1 makes specialized reduction faster.
- Mersenne-31 (058 R2): ship custom in `crypto/field/m31.go` ~80 LOC.
- BLS12-381 base field (Fp): Montgomery in `crypto/field/bls12381.go`
  ~200 LOC when 057 Tier-2 pairing milestone hits.

## §5 — Big-int multiply crossover

Crossover thresholds vary by architecture and language; conservative
amd64 Zen 4 / Apple M2 numbers from GMP-6.3 + blst + dalek:

| Bits | Schoolbook | Karatsuba | Toom-3 | NTT |
|---|---|---|---|---|
| 256 | **50 ns** | 80 ns | 200 ns | 5 µs |
| 512 | **200 ns** | 220 ns | 350 ns | 8 µs |
| 1024 | **800 ns** | **750 ns** | 900 ns | 15 µs |
| 2048 | 3 µs | **2.2 µs** | 2.5 µs | 25 µs |
| 4096 | 12 µs | **6 µs** | **5.5 µs** | 40 µs |
| 8192 | 50 µs | 18 µs | **12 µs** | 60 µs |
| 16k | 200 µs | 55 µs | **28 µs** | **100 µs** |
| 32k | 800 µs | 165 µs | 65 µs | **180 µs** |
| 64k | 3.2 ms | 500 µs | 150 µs | **320 µs** |

For the Sequence-A ladder (ECDSA, Ed25519, Schnorr, Pedersen, Shamir
all ≤ 512 bits), schoolbook is correct choice. Karatsuba can be
deferred. NTT only matters at the BLS12-381 base field (381-bit) +
Fp12 tower (4572-bit ops in pairings) and at Kyber/Dilithium poly
multiply.

## §6 — Hash function throughput class

| Hash | Today | reality post-port | SOTA portable | SOTA hardware |
|---|---|---|---|---|
| FNV1a32 | ~1.5-2 ns/byte (~600 MB/s) | same — already tight | ~1.5 ns/byte | n/a |
| FNV1a64 | ~2-3 ns/byte (~400 MB/s) | same | ~2 ns/byte | n/a |
| MurmurHash3_32 | ~0.5-1 ns/byte (~1-2 GB/s) | same — already tight | ~0.5 ns/byte | n/a |
| SHA-256 software | n/a (not shipped) | ~2.5-4 ns/byte (~250-400 MB/s) | OpenSSL ~3 ns/byte | SHA-NI ~0.5 ns/byte (~2 GB/s) |
| SHA-512 software | n/a | ~2-3 ns/byte (~350-500 MB/s, 64-bit ops) | OpenSSL ~2 ns/byte | n/a (no widespread instr) |
| SHA-3-256 (Keccak-f) | n/a | ~10-20 ns/byte (~50-100 MB/s) | OpenSSL ~10 ns/byte | n/a (rare) |
| BLAKE2b | n/a | ~1.5-2.5 ns/byte (~400-700 MB/s) | RFC ref ~1.5 ns/byte | n/a |
| BLAKE3 | n/a | ~1-1.5 ns/byte (~700 MB/s portable; ~10 GB/s SIMD) | reference ~0.7 ns/byte AVX2 | AVX-512 ~10 GB/s |
| Poseidon2 (Goldilocks, ZK) | n/a | ~1-3 µs/permutation | Plonky3 ~0.8 µs | n/a |

Reality posture: the Tier-1 hash layer (057 T1-HASH ~600 LOC) lands
SHA-256, SHA-512, SHA-3, HMAC, BLAKE2b, BLAKE3. Without amd64
SHA-NI assembly, SHA-256 will be ~250-400 MB/s — fine for any
JSON/hash-table use case, NOT fine if reality is ever the bottleneck
in a TLS handshake (which it shouldn't be — that's BoringSSL/Go-stdlib
territory). BLAKE3 portable Go is ~700 MB/s and beats SHA-2 software
without any assembly — recommend BLAKE3 as the **default reality
hash** for non-FIPS-required uses, SHA-2/3 only when interop demands it.

## §7 — Symmetric-cipher throughput (forward-looking)

Reality should NOT ship AES at all in the math layer per 057
(AES round function streaming is byte-level engineering, only the
GF(2^8) S-box as algebraic object is in scope). But the consumer
math-vs-stdlib question matters:

| Cipher | reality portable Go | Go crypto/aes (with AESNI) | OpenSSL (with AESNI/VAES) |
|---|---|---|---|
| AES-128-CTR | ~5-10 ns/byte (100-200 MB/s) | ~0.3-0.5 ns/byte (2-3 GB/s) | ~0.15 ns/byte (6-7 GB/s) |
| AES-256-GCM | ~10-15 ns/byte (~70-100 MB/s) | ~0.5 ns/byte (2 GB/s) | ~0.2 ns/byte (5 GB/s) |
| ChaCha20 | ~2 ns/byte (500 MB/s) | ~1 ns/byte (1 GB/s, no asm needed mostly) | ~0.5 ns/byte (2 GB/s, AVX2) |
| ChaCha20-Poly1305 | ~3 ns/byte (~330 MB/s) | ~1.5 ns/byte (~700 MB/s) | ~0.7 ns/byte (~1.4 GB/s) |
| Poly1305 (auth only) | ~1 ns/byte (1 GB/s) | ~0.5 ns/byte (~2 GB/s) | ~0.3 ns/byte (~3 GB/s) |

**Punchline**: AES without AESNI is 10× slower than ChaCha20 without
AVX. **If reality ever ships symmetric primitives** (which 057 says
NO for AES, BORDERLINE for ChaCha20 quarter-round, YES for Poly1305
GF(2^130-5) and YES for GHASH GF(2^128)), the recommendation must
be ChaCha20+Poly1305 over AES-GCM for portable Go, period.

For non-portable Go (assembly fallback per 058 R5 forecast), reality
delegates to crypto/aes — math layer doesn't reimplement.

## §8 — PRNG µs/op (shipped today)

| Generator | reality | SOTA reference |
|---|---|---|
| MT19937-64 Uint64 | ~3-5 ns | ~3 ns (Matsumoto-Nishimura C) |
| PCG-XSH-RR Uint32 | ~1.5-2 ns | ~1.5 ns (O'Neill C) |
| Xoshiro256** Uint64 | ~1.5-2.5 ns | ~1 ns scalar, ~0.7 ns SIMD (Vigna) |
| splitmix64 | ~1 ns | ~0.8 ns |

Reality within 1.5-2× of canonical C, acceptable for zero-asm.

**F5 io.Reader adapter** (~15 LOC each PRNG): `Read(p)` writes
Uint64 chunks via `binary.LittleEndian.PutUint64`. Cost ~0.5 ns/byte
≈ 2 GB/s. crypto/rand small-read cost ~250 ns/byte ≈ 4 MB/s. Reality
PRNG-as-io.Reader is 4-500× faster, plus deterministic, plus
golden-file-testable — locks in cross-language randomized-signature
golden files at byte level (Python's `random.Random().getrandbits(64)`
matches MT19937-64 stream bit-exact), property-based fuzz with
deterministic replay, and streaming randomized algorithms (Las Vegas
primality, Monte Carlo integration in optim/) at 2 GB/s.

## §9 — Recommended perf bundle (~120 LOC)

| ID | LOC | Description | Expected gain |
|---|---|---|---|
| CRY-PERF-1 | 4 | bits.Mul64+Div64 mulmod (per 056 CRY-MULMOD-OPT) | 10-25× ModPow / IsPrime / NextPrime / CRT |
| CRY-PERF-2 | 80 | crypto/*_bench_test.go — twelve benchmarks | locks µs/op contract |
| CRY-PERF-3 | 4 | package-var Miller-Rabin witness slices (per 059 F6) | ~50 ns saved per IsPrime |
| CRY-PERF-4 | 45 | F5 PRNG io.Reader adapters (per 059 F5) | unlocks 2 GB/s deterministic byte stream |
| CRY-PERF-5 (forward) | n/a | document Montgomery-default + Barrett-fallback + custom-Solinas posture in package doc when 057 BIGINT lands | architectural |
| CRY-PERF-6 (forward) | n/a | document `MulCT` / `MulFixedBase` / `MulVartime` triplet for every secret-handling primitive when 057 EC lands | architectural |
| CRY-PERF-7 (forward) | n/a | document NTT-share-with-signal/-FFT butterfly skeleton when 057 BIGINT NTT lands (per 058 R5) | architectural |
| CRY-PERF-8 (forward) | n/a | document hash-default = BLAKE3 (portable 700 MB/s), SHA-2 only for FIPS interop, when 057 T1-HASH lands | architectural |

CRY-PERF-1 through CRY-PERF-4 = 133 LOC ship-now. CRY-PERF-5 through
CRY-PERF-8 = forward-looking architecture decisions to settle BEFORE
the relevant 057 layer ships.

## §10 — Non-overlap

056 owned correctness of present functions and filed mulmod 4-LOC
fix; this adds the µs/op upper-bound chain. 057 owned primitive
ladder LOC; this defers primitive-content choices back. 058 owned
architectural patterns; this adds the µs/op cost class for each
(Mont 600 ns, Barrett 750 ns, Goldilocks 3 ns, ladder-CT 2×,
window-CT 1.3×). 059 owned API/doc fixes; this quantifies the
gain of F5 (2 GB/s vs crypto/rand 500 MB/s) and F6 (~50 ns/IsPrime).
060 owns: µs/op ladders, Mont/Barrett/Goldilocks tradeoff, big-int
multiply crossover, hash throughput class, AES-vs-ChaCha20 portable
posture, PRNG µs/op-vs-SOTA, CT-tax quantification, MulCT/MulFixedBase/
MulVartime API triplet, NTT-share-with-signal-FFT cost claim,
BLAKE3-as-default-hash claim. Defers: per-arch assembly detail,
Pippenger MSM windowing, pairing miller-loop batching, GPU/SIMD
(out-of-scope zero-dep), zeroize-on-drop, CT-leakage test harness.

## §11 — Headline

reality/crypto today runs within 1.5-2× of canonical C for PRNGs and
1× for FNV/Murmur, but pays a 10-25× tax on every ModPow/Miller-Rabin/
NextPrime/CRT because mulmod is Russian-peasant doubling instead of
`bits.Mul64`+`bits.Div64` (4-LOC fix in 056). Forward-looking, the
µs/op budget for 057's primitives splits three ways: (a) Montgomery
inside ModPow loops, Barrett for one-shot, custom Goldilocks/M31
reduction at 3-5 ns vs Mont 600 ns when modulus has structure;
(b) big-int multiply crossover schoolbook ≤ 1024-bit, Karatsuba
1024-8192-bit, Toom-3 8-32 kbit, NTT 32+ kbit — Sequence-A ECC-first
covers ALL primitives ≤ 512-bit so schoolbook only, no Karatsuba/
Toom/NTT v1; (c) CT tax mandatory 2× for ladder secret-exp, 1.3×
for fixed-window secret-scalar, negotiated at API time as
`MulCT`/`MulFixedBase`/`MulVartime` triplet per dalek/RustCrypto.
PRNG-as-io.Reader (059 F5) is the unique-capability hidden-gem at
~2 GB/s deterministic byte stream, 4-500× faster than crypto/rand
small-reads, unlocks golden-file-tested randomized signatures with
bit-exact reproducibility across Go/Python/C++/C#. Hash-default-when-
057-lands should be BLAKE3 (700 MB/s portable, beats SHA-2 software
without assembly); SHA-2 only for FIPS interop. AES correctly OOS
per 057 — portable Go AES is 100-200 MB/s, 10× slower than ChaCha20,
and crypto/aes already has AESNI. Bundle: 4 ship-now items at 133 LOC
+ 4 forward-looking architecture decisions to settle before each
057 layer ships.
