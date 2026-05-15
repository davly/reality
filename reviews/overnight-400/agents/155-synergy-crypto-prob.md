# 155 | synergy-crypto-prob

**Topic:** crypto × prob — randomness extractors, leftover hash lemma, entropy bookkeeping.
**Block:** B (cross-package synergies).
**Date:** 2026-05-08.
**Scope:** capabilities that emerge ONLY when `crypto/` and `prob/` are composed; not what either lacks in isolation (056-060 / 116-120 own those).

## Two-line summary

`crypto/` ships three deterministic PRNGs (MT19937-64, PCG-XSH-RR, xoshiro256\*\*) with no entropy meter, and `prob/` ships seven distributions plus `KLDivergenceNumerical` with no min-entropy / collision-entropy / extractor — both use entropy concepts in opposite halves of the seam (PRNG output is "uniform" by assumption; Bayesian samplers consume "uniform" floats by assumption) and **nothing audits the bridge**. **Twelve synergy primitives (X1-X12) totalling ~1320 LOC of pure glue** stand up the leftover-hash lemma, NIST SP 800-90B health tests, statistical RNG batteries, differential-privacy noise mechanisms, and a PRNG-state audit trail on top of two stable bases; cheapest one-day unlock is **X1 MinEntropy + CollisionEntropy** (~60 LOC, three lines of math each, used everywhere downstream), and the highest-leverage architectural change is **X4 Extract**: a Toeplitz universal-hash extractor that turns any of the three PRNGs into a near-uniform bitstream with a quantified ε (the leftover-hash lemma is the bridge between `crypto.Xoshiro256` and `prob.NormalQuantile`, and reality currently asserts it without proving it).

---

## Bases — what each package exposes today

`crypto/` (4 files, ~880 LOC of source):
- **`prime.go`** — `IsPrime`, `MillerRabin` (deterministic witnesses, n < 3.317×10²⁴), `PrimeFactors`, `NextPrime`, `GCD/LCM/ExtendedGCD`. Number-theory.
- **`modular.go`** — `ModPow`, `ModInverse`, `ChineseRemainder`. Modular arithmetic.
- **`hash.go`** — `FNV1a32/64`, `MurmurHash3_32`, `ConsistentHash`, `SituationHashWithStructure`. **All non-cryptographic** (per 056 the package name promises crypto and delivers number theory + hash table-grade hashing — confirmed; SHA-2/3, BLAKE, HMAC are entirely absent, see 057 §T1-HASH).
- **`rng.go`** — `NewMersenneTwister(seed) → MersenneTwister.{Uint64, Float64}`, `NewPCG(seed, seq) → PCG.{Uint32, Float64}`, `NewXoshiro256(seed) → Xoshiro256.{Uint64, Float64}`, helper `splitmix64`. Three deterministic PRNGs, no API for "extract n cryptographic bits", no health tests, no DRBG, no reseeding hook.

`prob/` (~14 files, ~5500 LOC of source):
- **Distributions** — `NormalPDF/CDF/Quantile`, `BetaPDF/CDF`, `ExponentialPDF/CDF/Quantile`, `UniformPDF/CDF`, `BetaPDF/CDF`, `PoissonPMF/CDF`, `GammaPDF/CDF`, `BinomialPMF/CDF`. Acklam quantile for Normal.
- **Inference** — `KLDivergenceNumerical` (trap-rule), `JeffreysKLDivergence` (Bernoulli), `LogLoss`, `BrierScore`, `LogOddsPool`, `WilsonConfidenceInterval`, hypothesis tests, Markov chain `MarkovSimulate`.
- **Information theory** — **NONE in `prob/`**. `ShannonEntropy`, `JointEntropy`, `ConditionalEntropy`, `MutualInformation`, `KLDivergence`, `CrossEntropy` live in `compression/entropy.go` (see compression/entropy.go:27-176). H₁ (Shannon) is the only Rényi order present in the whole repo. **No H_∞, no H₂, no smooth min-entropy, no Rényi family.**

**Cross-imports today:** `prob/` does not import `crypto/`. `prob/markov.go:115` rolls its own internal LCG (`seed = seed*6364136223846793005 + 1442695040888963407`) instead of consuming `crypto.NewPCG` or `crypto.NewXoshiro256`. Tests in `prob/conformal/*_test.go` use `math/rand` directly (`rand.New(rand.NewSource(2026))`). The two packages are siblings under `reality/` and **cannot prove a single uniformity claim about each other's outputs**.

## The conceptual unlock — entropy IS the seam

Every `crypto.*PRNG.Float64()` produces a bitstring that `prob.NormalQuantile` etc. **assume** is uniform-on-[0,1) to within float64 precision. That assumption has three failure modes that need primitives to detect or repair:

1. **Min-entropy of the seed is below the security parameter.** `splitmix64` of a low-entropy seed (e.g., `time.Now().UnixNano() & 0xFFFF`) gives outputs that pass the χ² uniformity test but fail any cryptographic min-entropy estimator. Reality has no `H_∞` estimator anywhere.
2. **The PRNG state has been observed.** MT19937-64 state is recoverable from 624 outputs, PCG-XSH-RR from ~64, xoshiro256\*\* from 4. After observation the next "Float64" is no longer extractable. There is no audit trail flagging this.
3. **The sampler consumes more bits than the seed entropy.** `MarkovSimulate(steps=10⁹, seed=42)` consumes more uniform variates than the 64 bits of seed entropy — every output past bit 64 of seed entropy is deterministic given the seed. Reality has no entropy bookkeeping primitive that surfaces this.

The leftover-hash lemma is the *exact* mathematical bridge: if a source X has min-entropy H_∞(X) ≥ k and h is a 2-universal hash family with output length m ≤ k − 2log(1/ε), then h(X) is ε-close to uniform. **Reality ships every primitive needed to state this** (`crypto.FNV1a64` is even ε-biased-universal, just not 2-universal in the strict sense; a Toeplitz extractor is ~80 LOC) — never the wire-up.

---

## X1 — `MinEntropy` + `CollisionEntropy` (~60 LOC; cheapest)

```go
// MinEntropy returns H_∞(p) = -log2(max_i p_i), the least-favourable Rényi entropy.
// Output range: [0, log2(n)]; equals 0 on a δ-distribution, log2(n) on uniform.
// Reference: Rényi (1961); central to the leftover hash lemma.
func MinEntropy(probs []float64) float64

// CollisionEntropy returns H_2(p) = -log2(sum_i p_i^2). Used in privacy
// amplification and as a tight lower bound on H_1 (Shannon).
// Identity: H_∞(p) ≤ H_2(p) ≤ H_1(p) ≤ log2(n).
// Reference: Rényi (1961); Bennett-Brassard-Crépeau-Maurer (1995).
func CollisionEntropy(probs []float64) float64

// RenyiEntropy(probs, alpha) — generalises both above. alpha=1 reduces to
// Shannon (use compression.ShannonEntropy for that), alpha=2 to collision,
// alpha=+Inf to min-entropy (with limit handling).
func RenyiEntropy(probs []float64, alpha float64) float64
```

Pure math, no PRNG dependency, ~60 LOC. **Unblocks every other entry below** because the leftover-hash lemma, NIST SP 800-90B, and DP all parameterise on H_∞ (or H_2), not H_1. Place: `prob/info.go` (companion file to absent prob/info.go today) — or extend `compression/entropy.go` with the Rényi family. **Recommendation: new `prob/info.go`** because compression/ is positioned as a coding-theory package and Rényi entropy is squarely an inference primitive.

Golden vectors trivial: `MinEntropy([0.5,0.5]) = 1`, `MinEntropy([0.9,0.1]) = log2(1/0.9) = 0.152003…`, `CollisionEntropy([0.25,0.25,0.25,0.25]) = 2`, etc.

## X2 — `EmpiricalMinEntropy` from a byte stream (~80 LOC)

NIST SP 800-90B §6.3 "most-common-value estimator" — count the empirical frequency of each symbol in a sample, take the most-common, derive the lower-bound on H_∞ via the binomial confidence interval. This is what NIST actually requires for IID entropy sources.

```go
// EmpiricalMinEntropy estimates H_∞ from a byte stream using NIST
// SP 800-90B §6.3 ("most common value" estimator) with a 99% confidence
// upper bound on the most-frequent-symbol probability.
// Returns the lower bound on per-byte H_∞ in bits.
func EmpiricalMinEntropy(data []byte) float64

// CompressionEntropyEstimator (§6.3.4) — alternative estimator using a
// Maurer-style universal compression test. Cross-validates §6.3.
func CompressionEntropyEstimator(data []byte) float64
```

Bridges `crypto.MersenneTwister.Uint64()` output bytes ↔ a quantified entropy claim. **Cross-validation pattern matches recent commit 6a55bb4 (audio onset 3-detector cross-validation saturating R-MUTUAL-CROSS-VALIDATION 3/3):** running both NIST §6.3 and §6.3.4 on the same MT/PCG/xoshiro byte stream and checking they agree to within their stated CIs is a 3/3 saturation of the rule on the seam.

## X3 — NIST SP 800-90B health tests (~150 LOC; live entropy meter)

```go
// RepetitionCountTest — SP 800-90B §4.4.1. Cutoff C such that
// P[run of ≥C identical samples | H_∞ = h] = α. Returns true on failure.
func RepetitionCountTest(samples []byte, hMin float64, alpha float64) bool

// AdaptiveProportionTest — SP 800-90B §4.4.2. Sliding window of W samples,
// flag if any symbol appears ≥ cutoff times.
func AdaptiveProportionTest(samples []byte, hMin float64, W int) bool

// ContinuousHealthTester — stateful struct that wraps a *crypto.PCG (or any
// io.Reader-shaped source) and runs both tests on every call to Read.
// Implements the "continuous reseed" discipline.
type ContinuousHealthTester struct {
    src     RandomSource  // adapter over crypto.PCG / Xoshiro256 / etc.
    hMin    float64
    state   *healthState
    failures uint64
}
func (h *ContinuousHealthTester) Read(p []byte) (int, error)
```

This is the live entropy meter that turns a `crypto.*PRNG` into an auditable
bitstream. ~150 LOC, no math beyond binomial tail bounds (already available
via `prob.BinomialCDF`).

## X4 — `ExtractToeplitz`: leftover-hash-lemma extractor (~120 LOC; keystone)

The leftover-hash lemma in operational form:

> Let X be a random source over {0,1}ⁿ with H_∞(X) ≥ k. Let h: {0,1}ⁿ → {0,1}ᵐ be drawn uniformly from a 2-universal family H. If m ≤ k − 2log(1/ε), then (h, h(X)) is ε-close in statistical distance to (h, U_m).

The Toeplitz family {T_s : s ∈ {0,1}^(n+m-1)} is the canonical 2-universal hash family used in QKD privacy amplification and Intel/AMD on-die conditioners. T_s(x) = T·x where T is the (n+m-1)-vector slid into an m×n Toeplitz matrix.

```go
// ExtractToeplitz applies a Toeplitz-matrix universal hash to extract m
// near-uniform bits from a high-min-entropy source. The seed must contain
// at least n+m-1 fresh bits.
//
// Leftover hash lemma: if H_∞(input) ≥ m + 2log(1/ε), output is ε-close to
// uniform in statistical distance. Caller is responsible for the entropy
// claim (use X2 EmpiricalMinEntropy to estimate it).
//
// Reference: Impagliazzo-Levin-Luby (1989), Bennett-Brassard-Robert (1988),
// NIST SP 800-90B §3.1.5.4.
func ExtractToeplitz(input []byte, seed []byte, m int) []byte

// ExtractVonNeumann applies the von Neumann debiaser: pairs (0,1)→0,
// (1,0)→1, (0,0) and (1,1) discarded. Achieves H_∞ = 1/bit on output (true
// unbiased) at cost of throughput log2(p(1-p)) bits/input bit. Constructive
// proof of the leftover hash lemma at m=1.
// Reference: von Neumann (1951) "Various techniques used in connection with
// random digits."
func ExtractVonNeumann(input []byte) []byte

// ExtractUniversalHash — alternative extractor using Carter-Wegman matrix-
// vector product over GF(2^64). Faster than Toeplitz, requires more seed.
func ExtractUniversalHash(input []byte, seed []byte, m int) []byte
```

**This is the keystone primitive.** It directly bridges `crypto.MersenneTwister.Uint64()` (low-quality but reproducible) to `prob.NormalQuantile` (assumes high-quality uniform). The cost is the seed bits (use a fresh `crypto.NewXoshiro256` instance) and the proof is the leftover-hash lemma with parameters from X1+X2.

~120 LOC: 60 for Toeplitz matrix-vector product over GF(2), 30 for von Neumann (trivial), 30 for documentation of the ε-bound.

## X5 — `HMAC_DRBG` and `Hash_DRBG` over reality's existing hashes (~250 LOC, blocked)

NIST SP 800-90A specifies three DRBGs: Hash_DRBG, HMAC_DRBG, CTR_DRBG. Each composes a hash (or block cipher) into a stateful uniform-bits source that supports `Reseed` and `Generate(num_bits)` with backtracking resistance.

**Blocked on 057's T1-HASH** (no SHA-2/3, no HMAC in `crypto/` today). Once SHA-256 lands per 057 §T1-HASH, the DRBG layer is ~250 LOC of state machine + reseed logic. Hash_DRBG is the cheapest (no HMAC needed); HMAC_DRBG adds ~80 LOC; CTR_DRBG adds ~120 LOC and requires AES (which 057 §T1 explicitly puts OUT of scope).

**Recommendation pinned for post-057:** ship Hash_DRBG-SHA256 first. Pre-shipped, document that `crypto.NewXoshiro256` is **NOT** SP 800-90A compliant and must be paired with an X4 extractor before being used in any DP/MCMC where uniformity is a load-bearing assumption.

## X6 — `BBS` (Blum-Blum-Shub) on top of `crypto.IsPrime` + `crypto.ModPow` (~50 LOC)

BBS is the textbook-cryptographic PRNG: choose primes p, q ≡ 3 (mod 4), N = p·q, then x_{i+1} = x_i² mod N, output the parity bit (or last log2(log2 N) bits). Its security reduces to the quadratic residuosity assumption — the only PRNG in the field with a *complexity-theoretic* security proof.

`crypto/` has every primitive: `IsPrime` (deterministic Miller-Rabin), `ModPow`, `mulmod`. ~50 LOC.

```go
// NewBBS creates a Blum-Blum-Shub PRNG with primes p, q ≡ 3 (mod 4) and
// initial seed x_0 coprime to p*q. Returns nil if p or q is composite or
// not ≡ 3 (mod 4), or if gcd(x_0, p*q) ≠ 1.
//
// Security: parity-bit output is provably as hard as factoring N=pq under
// the quadratic residuosity assumption. Slow (~100x xoshiro256**).
//
// Reference: Blum, Blum, Shub (1986) "A Simple Unpredictable Pseudo-Random
// Number Generator." SIAM J. Computing.
func NewBBS(p, q, x0 uint64) *BBS

func (b *BBS) Bit() uint8       // one parity bit
func (b *BBS) Uint64() uint64   // 64 squarings — slow on purpose
```

Note the uint64 modulus is the bottleneck per 056 — for serious BBS the user wants 2048-bit N which gates on 057's T1-BIGINT. **Ship the uint64 BBS now as the pedagogical anchor; document the 2048-bit version blocked on T1-BIGINT.**

## X7 — `Fortuna`-style entropy accumulator (~200 LOC, partially blocked)

Fortuna (Schneier-Ferguson 2003) is the "32 entropy pools, hash each pool when reseeding pool i every 2^i reseeds" architecture used in FreeBSD's `/dev/random`. The entropy bookkeeping is the whole point: every entropy event goes to one of 32 pools in round-robin, the pools accumulate H_∞ over time, and the reseed schedule is provably resistant to a continuous attacker who poisons k of the 32 pools.

Reality should ship a Fortuna-shaped accumulator even before SHA-2 lands — using FNV-1a (X3 health-checked) or a 4-round Keccak-f stub as the placeholder hash. ~200 LOC for the pool machinery; the hash slot is plug-replaceable when 057 §T1-HASH ships.

This is the **entropy bookkeeping primitive** explicitly called for by the topic prompt: every uniform sample drawn by `prob.MarkovSimulate` or `prob.BayesianUpdate` should be debit-able against a tracked H_∞ budget. Today there is no such primitive anywhere in reality; an aicore consumer wiring `crypto.NewPCG(time.Now().UnixNano(), 0)` into a 10⁶-step MCMC has no way to know they've exceeded the seed entropy by 14 orders of magnitude.

## X8 — Statistical RNG test battery (~250 LOC; NIST SP 800-22 subset)

NIST SP 800-22 "A Statistical Test Suite for Random and Pseudorandom Number Generators" (Rukhin et al. 2010) ships 15 tests. Reality should ship the **subset that requires only existing prob/ primitives**:

| Test | Primitive needed | LOC | prob/ primitive used |
|---|---|---|---|
| Frequency (monobit) | NormalCDF | 20 | NormalCDF |
| BlockFrequency | ChiSquaredCDF (private!) | 30 | mathutil.go:187 — promote to public |
| Runs | NormalCDF | 30 | NormalCDF |
| LongestRun | ChiSquaredCDF | 40 | (after promotion) |
| BinaryMatrixRank | linalg.Rank over GF(2) | 50 | linalg dependency |
| FFTSpectral | signal.FFT + threshold | 30 | signal/ dependency |
| ApproximateEntropy | log + windowed counts | 30 | stdlib only |
| CumulativeSums | NormalCDF | 30 | NormalCDF |
| **Total** | | ~260 | |

Excluded as too heavy or out of scope: NonOverlappingTemplate (template DB), Universal (Maurer), LinearComplexity (Berlekamp-Massey), Serial, RandomExcursions{,Variant}, OverlappingTemplate.

**Cheap one-day standalone unlock: Frequency + Runs + ApproximateEntropy.** Three tests, ~80 LOC, all just NormalCDF or stdlib. Run on every `crypto.*PRNG.Uint64()` stream in a `TestPRNG_PassesNIST_SP800_22_Subset` table-test as the cross-validation witness.

## X9 — Differential privacy noise on the seam (~150 LOC)

DP noise mechanisms are the cleanest synergy of all because they live on the package boundary by definition: the *mechanism* is a function that consumes uniform crypto bits and emits a draw from a specific `prob.Distribution`.

```go
// LaplaceMechanism returns f(x) + Laplace(0, sensitivity/epsilon) where the
// Laplace draw consumes log2(precision) uniform bits from prng.
//
// Privacy: ε-DP for any neighbouring x, x' with ||f(x)-f(x')||_1 ≤ sensitivity.
// Reference: Dwork-McSherry-Nissim-Smith (2006) "Calibrating Noise to
// Sensitivity in Private Data Analysis."
func LaplaceMechanism(value, sensitivity, epsilon float64, prng RandomSource) float64

// GaussianMechanism returns f(x) + N(0, sigma^2) with
// sigma = sqrt(2 ln(1.25/delta)) * sensitivity / epsilon. Consumes Acklam
// quantile of a uniform draw — uses prob.NormalQuantile.
//
// Privacy: (ε,δ)-DP under L2-sensitivity. Reference: Dwork-Roth (2014).
func GaussianMechanism(value, sensitivity, epsilon, delta float64, prng RandomSource) float64

// DiscreteLaplaceMechanism — geometric-difference variant for integer
// outputs. Avoids floating-point side channels (Mironov 2012).
func DiscreteLaplaceMechanism(value int64, sensitivity int64, epsilon float64, prng RandomSource) int64

// ExponentialMechanism — score-based mechanism for non-numeric outputs.
// Reference: McSherry-Talwar (2007).
func ExponentialMechanism(scores []float64, sensitivity, epsilon float64, prng RandomSource) int
```

~150 LOC total. **Sits in `prob/dp.go`** (consumer-side; takes `RandomSource` interface so it works against `crypto.*PRNG` and `math/rand` alike). Privacy budget composition (the ε-accountant) is a separate ~80 LOC bookkeeping primitive that pairs naturally with X7 Fortuna.

**Critical numerical detail (Mironov CCS 2012):** the naïve "uniform − uniform" Laplace and the naïve "Box-Muller" Gaussian both leak privacy via floating-point representation. The discrete and snapping-mechanism variants are mandatory for any reality consumer claiming ε-DP. Document this prominently or the primitive is a footgun.

## X10 — Entropy-conditioned MCMC: rejection rate as live entropy meter (~80 LOC)

Metropolis-Hastings rejection rate is mathematically related to the KL divergence between proposal and target: high acceptance means proposal ≈ target (low information gain per step); low acceptance means high information gain per accept (high effective H_∞ per accepted sample). The **realised** entropy of an MCMC chain is bounded by a function of the rejection rate (Roberts-Gelman-Gilks 1997 for Gaussian target, generalised by Latuszyński-Roberts 2013).

```go
// MetropolisHastings runs MH with a proposal kernel and target log-density.
// Returns chain + acceptance rate. The acceptance rate is a live entropy
// meter: target ≈ 0.234 for high-d Gaussian (Roberts-Gelman-Gilks 1997).
func MetropolisHastings(target func(float64) float64, proposal func(float64, RandomSource) float64,
    x0 float64, steps int, prng RandomSource) (chain []float64, acceptRate float64)

// EffectiveSampleSize — autocorrelation-based ESS (Geyer 1992). Combined
// with X1 MinEntropy on a histogram of `chain`, gives realised H_∞/sample.
func EffectiveSampleSize(chain []float64) float64
```

This is the synergy that makes MCMC auditable: an aicore consumer running 10⁶ MH steps with `crypto.NewPCG(seed)` can now call `EffectiveSampleSize` and `MinEntropy(histogram(chain))` and know whether they have *actually* drawn from the target or just churned the seed. ~80 LOC.

## X11 — `RandomSource` interface: the architectural lift (~30 LOC, blocks X3-X10)

Today every PRNG in `crypto/` is its own concrete type with a different API surface (`MersenneTwister.Uint64`, `PCG.Uint32`, `Xoshiro256.Uint64`). `prob/` does not consume any of them. **The single highest-leverage refactor is one ~30-LOC interface.**

```go
package crypto  // or a new package crypto/rand if naming-cleanup of 056 happens first

// RandomSource is the minimal contract for a deterministic uniform-bits
// source. Every PRNG in this package implements it; consumers in prob/,
// chaos/, optim/, etc. should accept *RandomSource* not concrete types.
type RandomSource interface {
    Uint64() uint64
    Float64() float64
}

// All three PRNGs satisfy this trivially. PCG.Uint64() is the only adapter:
func (p *PCG) Uint64() uint64 { return uint64(p.Uint32())<<32 | uint64(p.Uint32()) }
```

**This is the MarkovSimulate fix** — `prob/markov.go:115` rolls a private LCG today; with `RandomSource`, it becomes `func MarkovSimulate(..., src RandomSource)` and the three crypto PRNGs are drop-in. Same for every `prob/conformal/*_test.go` that uses `math/rand`. Same for X9 DP mechanisms. Same for X10 MH. **All twelve synergy primitives in this report depend on this 30-LOC interface.**

Recommendation: ship X11 in the same PR as X1, before any of X2-X10. It is the seam itself.

## X12 — Entropy bookkeeping audit trail (~150 LOC)

```go
// EntropyAccountant tracks the running H_∞ debit of a RandomSource against
// the seed entropy claim. Every Uint64() call debits 64 from the running
// budget (assuming seed entropy is fully extractable, which is conservative
// for non-extractor seeds and exact post-X4 Toeplitz extraction).
//
// Panics (or returns error) when budget < 0 — the sentinel for "this PRNG
// has emitted more bits than its seed contains."
type EntropyAccountant struct {
    src       RandomSource
    seedH     float64  // claimed seed min-entropy (bits)
    consumed  float64  // running debit
    onExhaust func()   // hook for re-seed (X3 ContinuousHealthTester compat)
}

func (e *EntropyAccountant) Uint64() uint64
func (e *EntropyAccountant) Float64() float64
func (e *EntropyAccountant) Remaining() float64

// SHA3Sponge — when 057 T1-HASH lands, replace ad-hoc accountant with a
// Keccak-f sponge whose absorb capacity == claimed entropy and whose
// squeeze rate == output rate. Post-quantum extractor + DRBG in one
// primitive. Until then, document the gap.
```

This is the explicit answer to the topic prompt's "What's the audit trail?" — today there is **none**. Reality has no concept of "entropy debit" anywhere. ~150 LOC ships the bookkeeping; downstream consumers (aicore, Pistachio when it does Monte Carlo lighting) get a runtime guard.

---

## Composition map: what stacks on what

```
                     X11 RandomSource (30 LOC, FOUNDATION)
                     │
                     ├── X1 MinEntropy/CollisionEntropy (60)  ◄─ pure, no PRNG
                     │   └── X2 EmpiricalMinEntropy (80) ◄─ NIST §6.3
                     │       └── X3 SP 800-90B health tests (150)
                     │
                     ├── X4 ExtractToeplitz/VonNeumann (120) ◄─ leftover hash lemma
                     │   └── X5 Hash_DRBG/HMAC_DRBG (250)  [BLOCKED on 057 §T1-HASH]
                     │       └── X7 Fortuna accumulator (200)
                     │           └── X12 EntropyAccountant (150)
                     │
                     ├── X6 BBS (50, uint64; 2048-bit blocked on 057 §T1-BIGINT)
                     │
                     ├── X8 NIST SP 800-22 subset (260)
                     │
                     ├── X9 DP Mechanisms (150) ◄─ Laplace/Gaussian/Exponential
                     │
                     └── X10 Metropolis-Hastings (80) ◄─ rejection rate as H meter
```

**Total connective tissue: ~1320 LOC** (1730 with X5 Hash_DRBG once 057 unblocks it). All twelve primitives are ZERO new mathematics — every theorem is 1949-2010 vintage, every formula has a Wikipedia page. The work is wire-up.

## Cheapest first PR — X1 + X11 + X8.frequency_only (~150 LOC)

- **X1 MinEntropy + CollisionEntropy + RenyiEntropy** in `prob/info.go` (~60 LOC).
- **X11 RandomSource interface** in `crypto/rng.go` next to existing PRNGs, plus `PCG.Uint64()` adapter (~30 LOC).
- **X8 NIST §2.1 Frequency monobit test** wired against all three PRNGs (~60 LOC).

Outcome: every PRNG in `crypto/` gets a quantitative uniformity certificate, every `prob/` distribution gets a Rényi-family entropy primitive, and the seam `RandomSource` is named for the first time. This is the smallest cohesive PR that delivers all three of "entropy estimator", "extractor lemma" (vacuously, since X8 monobit ε ≈ 0 for good PRNGs), and "audit trail" (one named interface).

## Highest-leverage one-day unlock — X4 ExtractToeplitz (~120 LOC)

The single primitive that turns reality from "uses PRNGs" into "proves uniformity via the leftover-hash lemma." Without X4, every claim downstream is "we hope this is uniform"; with X4, every claim is "this is ε-close to uniform with explicit ε from the LHL given seed min-entropy ≥ k from X1." It is the architectural lift the topic prompt demands.

## Cross-validation calibration pair (R-MUTUAL-CROSS-VALIDATION 3/3 candidate)

Three independent estimators of the same H_∞ on the same byte stream:

1. **X1 `MinEntropy(histogram(stream))`** — direct combinatorial estimate.
2. **X2 `EmpiricalMinEntropy(stream)`** — NIST §6.3 most-common-value with 99% CI.
3. **X8 ApproximateEntropy + X3 RepetitionCountTest** — independent windowed estimator.

Run all three on outputs of `crypto.NewMersenneTwister(42)`, `crypto.NewPCG(42, 1)`, `crypto.NewXoshiro256(42)` for n = 10⁶ samples. **All three should agree on H_∞ ≈ 8 bits/byte to within their stated confidence intervals.** This is exactly the R-MUTUAL-CROSS-VALIDATION pattern recently saturated by commit 6a55bb4 (audio onset 3-detector cross-validation) and 365368a (copula × autodiff Clayton log-PDF gradient pin) — three independent paths to the same number, golden-tested as a triplet.

If any of the three reports H_∞ < 7.5 for a non-pathological seed, that PRNG fails for cryptographic use. **MT19937-64 with seed = 0 is the known-bad calibration case** (the all-zero state is degenerate per Matsumoto-Nishimura 1998); the suite should detect this on the first batch.

## Placement recommendations (non-prescriptive)

- **`prob/info.go`** — X1 (MinEntropy, CollisionEntropy, RenyiEntropy), X2 (EmpiricalMinEntropy, CompressionEntropyEstimator). Why prob/: Rényi entropy is an inference primitive; compression/ is a coding-theory package and Shannon entropy there is for encoding-bound calculations. ~140 LOC.
- **`crypto/rng.go`** — X11 RandomSource interface + Uint64 adapter on PCG. ~30 LOC.
- **`crypto/extract.go`** — X4 (ExtractToeplitz, ExtractVonNeumann, ExtractUniversalHash), X6 (BBS uint64 stub). ~170 LOC.
- **`crypto/health.go`** — X3 (RepetitionCountTest, AdaptiveProportionTest, ContinuousHealthTester), X12 EntropyAccountant. ~300 LOC.
- **`crypto/drbg.go`** — X5 (Hash_DRBG, HMAC_DRBG). BLOCKED on 057 §T1-HASH. ~250 LOC future.
- **`crypto/fortuna.go`** — X7 Fortuna accumulator. Partially blocked on 057 hash; FNV stub usable. ~200 LOC.
- **`prob/dp.go`** — X9 DP mechanisms (Laplace, Gaussian, Discrete, Exponential). Imports `crypto.RandomSource` only. ~150 LOC.
- **`prob/mcmc.go`** — X10 MetropolisHastings, EffectiveSampleSize. Imports `crypto.RandomSource`. ~80 LOC.
- **`crypto/randtest.go`** — X8 NIST SP 800-22 subset. Imports `prob.NormalCDF` (one-way, prob → crypto). ~260 LOC.

## Single architectural risk

**Cycle hazard prob ↔ crypto.** X8 NIST SP 800-22 needs `prob.NormalCDF`, X9-X10 need `crypto.RandomSource`. If `prob/` imports `crypto/` AND `crypto/` imports `prob/`, the import graph cycles. Resolution options ordered by cost:

1. **Cheap:** put X8 in `prob/randtest.go` not `crypto/randtest.go`. The "random tests use NormalCDF" direction wins, "tests live next to the PRNGs they test" loses. Same outcome semantically; one-way import.
2. **Cheap+:** create `crypto/info` sub-package depending on neither parent. X1 / X2 / X8 live there. Both `crypto/` and `prob/` can use it without cycling.
3. **Expensive:** factor a `random` package above both that re-exports interfaces — adds a third import root for consumers. Reject; matches 153's precedent of "place new synergy in the consumer-shaped package."

**Recommendation: Option 1.** All synergy primitives live in the `prob/` half (consumer of randomness) except those that *enrich* `crypto/` directly (X4 extractors, X6 BBS, X12 accountant). Aligns with 151's `spectral/` recommendation (synergy in the consumer-shaped place, not the producer).

## Distinct from prior agents

- **056 crypto-numerics** owns ModInverse uint64 sign bug, ChineseRemainder overflow bug, Miller-Rabin error bound — *isolation correctness*, not synergy.
- **057 crypto-missing** owns SHA-2/3, EdDSA, BLS, Schnorr, Pedersen, Shamir, big-int, field tower — *isolation completeness*. Specifically blocks X5 Hash_DRBG, X7 Fortuna's hash slot, and 2048-bit BBS.
- **058 crypto-sota** owns libsodium/RustCrypto/dalek architectural transferability — *isolation typing*. Note: 058's "every primitive must be a typed algebraic object the user cannot misuse" applies forcefully to RandomSource: it should be an interface, not a `[]byte` channel.
- **059 crypto-api** / **060 crypto-perf** — owns API discoverability and PRNG throughput — *isolation*.
- **116-120 prob-***— owns prob/ correctness, missing distributions, SOTA, API, perf in isolation.
- **087 info-missing** — owns info-theoretic gaps in `compression/` (the only place Shannon entropy currently lives). Cross-check: this report's X1 RenyiEntropy is the natural extension flagged but not enumerated by 087.
- **151 synergy-signal-prob** — synergy template, no overlap.
- **153 synergy-prob-infogeo** — closes prob ↔ infogeo seam (Fisher, Bregman, natural gradient). Orthogonal to this report's prob ↔ crypto seam.

This report sits exactly between 057 (which flagged SHA-2/3 absence as the bottleneck for cryptographic primitives) and 116/117 (which noted the absence of H_∞ and DP in prob/) without doubling either. The synergy is the entropy seam, owned by neither isolation report.
