# 200 | synergy-zkmark-info

**Topic:** zkmark × info — knowledge soundness, simulator entropy, statistical hiding, computational binding, HVZK, witness indistinguishability, Pedersen min-entropy, universal-hash MACs, Fiat-Shamir random oracle entropy, secret sharing, VRF/VDF.
**Block:** B (cross-package synergies). **Date:** 2026-05-08.
**Scope:** capabilities that emerge ONLY when `zkmark/` and `info/` are composed (with `crypto/` as bridge). Not isolation gaps (146-150 own zkmark-isolation, 086-090 own info-isolation, 056-060 own crypto-isolation, 175 owns zkmark×crypto substrate gates).

## Two-line summary

`zkmark/` is a 262-LOC honest-pending interface envelope (no field, no curve, no hash, no transcript, `Halo2Prover.Prove` returns `ErrNotYetWired`); `info/` is 770 LOC across two sub-packages (`info/lz` LZ76 + `info/mdl` NML/BIC/AIC) and ships ZERO Shannon/MI/KL surface (those live in `compression/entropy.go`); the entire **information-theoretic security** canon — min-entropy `H_∞(X) = -log₂ max P(x)`, statistical distance / total-variation simulator bound `Δ(S(x), View(P,V)(x)) ≤ ε`, leftover-hash-lemma extraction (Impagliazzo-Levin-Luby 1989), HILL/Yao/metric computational entropy (Hastad-Impagliazzo-Levin-Luby 1999), Carter-Wegman 1979 universal-hash one-time MAC, almost-XOR-universal hash, Pedersen unconditional-hiding bound (`H_∞(commitment | message) = log₂ |G|` for uniform `r`), Bulletproofs IPA knowledge-soundness extractor (Bootle-Cerulli-Chaikin-Groth-Petit 2016 forking lemma, ε-knowledge error `(2/q)·rounds`), KZG knowledge-of-exponent assumption, Fiat-Shamir random-oracle entropy budget (each squeeze burns `log₂ q` bits), Shamir 1979 threshold secret sharing (Reed-Solomon evaluation), VRF (Micali-Rabin-Vadhan 1999), VDF (Boneh-Bonneau-Bunz-Fisch 2018 sequential-squaring) — is **wholly absent** because it gates on the same eight T1 substrate layers 175-synergy-zkmark-crypto already enumerates (T1-BIGINT / T1-FIELD / T1-EC / T1-PAIRING / T1-HASH / T1-NTT / T1-MSM / T1-POLY / T1-MERKLE / T1-TRANSCRIPT) PLUS three info-side primitives that DO NOT exist in `info/` today (`MinEntropy`, `StatisticalDistance/TotalVariation`, `LeftoverHashLemma` extraction). **Twenty-four synergy primitives (I1-I24) totalling ~6,420 LOC, of which ~5,180 LOC is substrate and ~1,240 LOC is connective tissue;** cheapest one-day shippable artifact today is **I1 MinEntropy + I2 StatisticalDistance = 80 LOC** (pure addition to `compression/entropy.go` or new `info/entropy.go` lift, zero new substrate, deterministic float64), gating I12 simulator-bound checks, I15 Pedersen-hiding bound, I20 leftover-hash extraction. The single highest-leverage missing piece bridging the two packages is **I7 UniversalHashOneTimeMAC** (Carter-Wegman 1979) — ~120 LOC of `crypto.ModPow`-style modular arithmetic giving info-theoretic-secure MAC with collision probability `1/p` where `p` is prime, AND simultaneously the building block for **I22 PolynomialHashTranscript** (the Fiat-Shamir replacement for callers who refuse a random-oracle assumption — Reed-Solomon evaluation as a deterministic 2-universal hash family). Three architectural findings dominate: **(F1)** `zkmark`'s deterministic-by-construction posture (no prover RNG, flagged in 147 §1) is **structurally incompatible** with zero-knowledge — without simulator randomness there is no statistical-hiding to bound; the package should be re-named **`succinct-argument-mark`** OR `zkmark/randomness.go` should expose a caller-injected `io.Reader` AND ship golden-pinned simulator-distribution-equality tests; **(F2)** `info/mdl`'s Universal-Integer-Code length (Rissanen 1983) is the EXACT primitive needed for I23 `RandomOracleEntropyBudget(transcript) → bits` accounting under Fiat-Shamir composition (each squeeze of a `Fp` challenge consumes `UniversalIntegerCodeLength(p)` bits of the oracle's output stream), and `info/lz/LempelZivComplexity` is the EXACT diagnostic for I24 `TranscriptDistinguishability(real, simulated)` (a low-LZ-complexity simulator transcript fails the heuristic indistinguishability test); **(F3)** EVERY `zkmark` Tranche-2 candidate (Halo2 / Plonky3-FRI / SP1 / RISC0) gates first on a **statistical distance bound for the simulator** which CANNOT be expressed without `info.StatisticalDistance`, and the field-element transcript canonical-hash gates on `crypto/sha2.go` + `crypto/keccak.go` which 057-T1-HASH does not yet ship — so the cheapest unblocking sequence is **I1+I2+I3 = 230 LOC THIS WEEK** to the info side, then 057-T1-HASH (700 LOC), then composition becomes possible.

---

## 0. State of play (verified file-walk 2026-05-08)

### `zkmark/` — 1 file, 262 LOC + 1 README + 1 test

```
zkmark/
  zkmark.go        Proof, Prover, Verifier, HonestProver, Halo2Prover (stub), MarkVerifier
  zkmark_test.go   fakeSign / fakeVerify stubs
```

- **Algorithms:** `AlgorithmHonestPending` (mirror-mark, ProofPending=true, ProofBytes=nil), `AlgorithmHalo2` (returns `ErrNotYetWired`).
- **Cryptographic surface:** literally none. No SHA, no transcript, no field, no curve, no commitment. No source of randomness on either Prover or Verifier — both are pure deterministic functions of `(payload, corpusSHA, key)`. **This is the F1 finding:** without prover-side randomness there is no zero-knowledge to claim; the substrate is a succinct-argument-of-knowledge envelope, not a ZK envelope.
- **Cross-imports:** `errors` + `fmt` only. Zero imports of `info/`, `compression/`, `crypto/`.

### `info/` — 2 sub-packages, ~770 LOC

```
info/
  lz/    lz76.go        LempelZivComplexity, SymbolizeByQuantile/Threshold,
                        ComplexityFromReturns, RollingComplexity   (~430 LOC)
  mdl/   bernoulli.go   NMLBernoulli, BernoulliCodeLength
         nml.go         NMLMultinomial (Kontkanen-Myllymäki recurrence)
         codelength.go  GaussianCodeLength, ModelCodeLength, BICShape, AICShape
         universal_int.go  UniversalIntegerCodeLength{,Bits}     (Rissanen 1983)
         select.go      SelectMDL, SelectMDLWithMargin
```

**Confirmed absent in `info/`** (verified by `grep -E 'MinEntropy|RenyiEntropy|StatisticalDistance|TotalVariation|HellingerDistance|LeftoverHashLemma|UniversalHash|CarterWegman|HILL|Yao|HVZK|Simulator' info/`): **0 hits** for every information-theoretic-security primitive. The `info/` package is currently **scalar-statistics only** (model selection + sequence complexity); it ships none of the `(P, Q) → ε` distance metrics that simulator bounds and statistical hiding require.

### `crypto/` — 4 files, ~880 LOC (per 175 §State-of-play)

- `prime.go` — Miller-Rabin (deterministic ≤ 3.317×10²⁴), `PrimeFactors`, `NextPrime`, `GCD/LCM/ExtendedGCD`.
- `modular.go` — `ModPow`, `ModInverse`, `ChineseRemainder`. All `uint64`. **No `math/big`.**
- `hash.go` — `FNV1a32/64`, `MurmurHash3_32`, `ConsistentHash`, `SituationHashWithStructure`. **All non-cryptographic.**
- `rng.go` — Mersenne Twister, PCG, xoshiro256\*\*. **Deterministic-only**, no DRBG, no `crypto/rand` coupling.

### Compression's hidden info surface — 177 LOC at `compression/entropy.go`

```go
ShannonEntropy(probs []float64) float64        // bits, base-2
JointEntropy(joint [][]float64) float64        // bits
ConditionalEntropy(joint [][]float64) float64  // H(Y|X)
MutualInformation(joint [][]float64) float64   // I(X;Y)
KLDivergence(p, q []float64) float64           // KL(P||Q)
CrossEntropy(p, q []float64) float64           // H(P,Q)
```

All on `[]float64` probability vectors. The 199- and 196-synergy reviews flag this as the "info-misnamed" issue: the foundational Shannon stack lives in `compression/`, not `info/`. The fix-forward is the same the prior reviews recommend — hoist `compression.ShannonEntropy/MutualInformation/KLDivergence` to `info/entropy.go` and re-export from `compression` for backward compatibility. **This synergy depends on that hoist for I12-I24 below.**

### Cross-imports today: zero in all six pairwise directions

`grep -rE '"github.com/davly/reality/(zkmark|info|compression|crypto)"' zkmark/*.go info/**/*.go crypto/*.go compression/*.go` returns **0 cross-edges** in any direction. The four packages are siblings that have never been composed.

---

## 1. The conceptual unlock — zero-knowledge IS information-theoretic claim

Zero-knowledge is by definition an information-theoretic claim about the simulator's transcript distribution being indistinguishable from a real prover's transcript distribution. The three flavors:

- **Perfect ZK (PZK):** `Δ(S(x), View_V*(P, V*)(x)) = 0` — exact equality of distributions. Lives in `info` because it is a statement about probability measures.
- **Statistical ZK (SZK):** `Δ(S(x), View(...)(x)) ≤ negl(λ)` — total-variation bound with security parameter λ. The ε is an info-theoretic quantity.
- **Computational ZK (CZK):** `View_V*(P, V*)` and `S(x)` are **computationally indistinguishable** — Yao 1982 / Goldwasser-Micali 1984. The "computational" qualifier promotes the info-theoretic bound to a complexity-theoretic one, but the underlying object is still a distribution distance.

`zkmark` cannot claim ANY of the three today because:

1. The `Prove` method has no `io.Reader` for randomness — the simulator distribution collapses to a Dirac at `Sign(payload, corpusSHA, key)`.
2. There is no `StatisticalDistance(P, Q []float64) float64` in `info/` to express the bound.
3. There is no `MinEntropy(probs)` to bound the simulator's output entropy.

The 147-zkmark-missing review §recommendation 1 already flagged this as "deterministic-by-construction posture mismatch with the ZK name". This synergy review extends that finding: **the fix is twofold — (a) `zkmark/randomness.go` exposing a caller-injected `io.Reader` (15 LOC), AND (b) `info/distance.go` shipping the four canonical distribution distances (TV, Hellinger, KL, χ²) plus `info/entropy.go` shipping `MinEntropy/RenyiEntropy/CollisionEntropy` (110 LOC).** Without both, "zkmark" is a misnomer.

The 175-synergy-zkmark-crypto review enumerates the crypto-side substrate (T1-BIGINT through T1-MERKLE). This review enumerates the **info-side substrate** that 175 leaves implicit: every Bulletproofs / KZG / Halo2 knowledge-soundness extractor proof requires a `forking lemma + statistical distance` argument, and that argument has no Go expression today.

---

## 2. Twenty-four primitives — substrate + connective tissue

I1–I8 are info-side substrate (no `zkmark` dep, ship today). I9–I16 are pure connective tissue over today's surface. I17–I24 require future T1 layers (gate explicitly named).

### I1–I8 — info-side substrate (ships TODAY, no zkmark gate)

**I1. MinEntropy — `info/entropy.go`** (~25 LOC). `MinEntropy(probs []float64) float64 = -log₂(max_x P(x))`. The relevant entropy notion for cryptographic extraction (Renner-Wolf 2003); Shannon entropy is too generous (averages over rare events). Pure addition to `compression/entropy.go` OR new `info/entropy.go` if the hoist (199 §F2) lands first. **Cheapest standalone PR.** Connective LOC: 25 + 30 test = 55.

**I2. RenyiEntropy — `info/entropy.go`** (~40 LOC). `RenyiEntropy(probs, alpha) = (1/(1-alpha)) · log₂ Σ p_i^α` with α ∈ {0, 1/2, 1, 2, ∞}. α=0 is max-entropy (`log₂ |support|`), α=1 is Shannon (limit), α=2 is collision entropy (`-log₂ Σ p_i²`), α=∞ is min-entropy. Numerically stable via log-sum-exp at α near 1. Reference: Rényi 1961. Connective LOC: 40 + 35 test = 75.

**I3. StatisticalDistance — `info/distance.go`** (~50 LOC). `StatisticalDistance(p, q []float64) = (1/2) Σ |p_i - q_i|` (total variation). `HellingerDistance(p, q) = (1/√2) sqrt(Σ (sqrt(p_i) - sqrt(q_i))²)`. `ChiSquaredDistance(p, q) = Σ (p_i - q_i)² / q_i`. Reference: Le Cam 1986 *Asymptotic Methods in Statistical Decision Theory*. **The single primitive every simulator-bound proof needs.** Connective LOC: 50 + 40 test = 90.

**I4. ComputationalEntropy_HILL_Stub — `info/entropy.go`** (~30 LOC). HILL entropy is a *computational* notion (Hastad-Impagliazzo-Levin-Luby 1999): `X` has HILL entropy ≥ `k` if there exists `Y` with `H_∞(Y) ≥ k` and `(X, Y)` are computationally indistinguishable. Cannot be computed deterministically; the function ships as a **stub with documented incomputability** plus `EstimateHILLLowerBound(samples, distinguisher_circuit_size_bits) → float64` returning `min(MinEntropy(empirical), shannon_lower_bound)`. The honesty-gate per 146 §G1 demands the stub return `ErrUncomputable` and a documented lower-bound estimator. Connective LOC: 30 + 25 test = 55.

**I5. UniversalHashFamily_2Universal — `info/uhash.go`** (~80 LOC). Carter-Wegman 1979 2-universal hash family `h_(a,b)(x) = ((a·x + b) mod p) mod m` with `p` prime, `a, b ∈ [0, p)`. Collision probability `1/m`. Uses `crypto.ModPow` for the modular arithmetic + `crypto.IsPrime` for the parameter validation. Pure info-theoretic primitive (no crypto assumption, no random oracle). **The building block for I7, I20, I22.** Reference: Carter-Wegman 1979 *Universal Classes of Hash Functions*. Connective LOC: 80 + 60 test = 140.

**I6. AlmostXORUniversalHash — `info/uhash.go`** (~60 LOC). Stronger guarantee: for distinct `x, x'` and any `c`, `Pr[h(x) ⊕ h(x') = c] ≤ ε`. Realised by GF(2^n) polynomial hashing (Gilbert-MacWilliams-Sloane 1974). Requires GF(2^n) arithmetic which `crypto/` does NOT ship — flagged as gating on **T1-FIELD₂ (binary-field tower)** which 057-T1-FIELDTOWER enumerates for Fp² over a non-residue but not GF(2^n). Connective LOC: 60 (after T1-GF2 lands ~120 LOC).

**I7. UniversalHashOneTimeMAC — `info/mac.go`** (~120 LOC). Carter-Wegman 1979 / Wegman-Carter 1981 one-time MAC: `MAC(k_a, k_b, m) = h_(k_a, k_b)(m)` from I5. Information-theoretically secure for ONE message; forgery probability ≤ `1/p`. Wraps I5 with the (a, b) key as `[2]uint64` and a `Verify` returning `bool`. **The cheapest info-side composition that delivers an info-theoretic-secure primitive today** (vs every other crypto primitive that gates on yet-absent SHA-2/Keccak/EC). Reference: Wegman-Carter 1981 *New Hash Functions and Their Use in Authentication and Set Equality*. Connective LOC: 120 + 80 test = 200.

**I8. LeftoverHashLemma_Extraction — `info/extract.go`** (~100 LOC). Given source `X` over `{0,1}^n` with `H_∞(X) ≥ k`, and 2-universal hash family `H = {h: {0,1}^n → {0,1}^m}` with `m ≤ k - 2log₂(1/ε)`, the distribution `(h, h(X))` for uniform `h` is ε-close to uniform over `H × {0,1}^m`. Function: `LeftoverHashLemmaBound(min_entropy_input, output_bits) → ε_bound float64`, plus `Extract(seed, input, output_len_bits) → bytes` using I5 as the hash. **The bridge between info-theoretic min-entropy bounds and pseudorandomness.** Reference: Impagliazzo-Levin-Luby 1989 *Pseudo-random Generation from One-way Functions*. Connective LOC: 100 + 70 test = 170.

### I9–I16 — pure connective tissue over today's surface (no new substrate)

**I9. SimulatorEntropyBound_Stub — `zkmark/simulator.go`** (~40 LOC). Type `SimulatorEntropyBound { MinEntropyBits float64; StatisticalDistanceBound float64; Flavor string /* "perfect"|"statistical"|"computational" */ }` returned alongside every Tranche-2 proof. For Tranche-1 honest-pending: `Flavor = "none"`, `MinEntropyBits = 0`, `StatisticalDistanceBound = 1.0` (worst case). Forward-compatible with Tranche-2 Halo2/STARK once those ship. Connective LOC: 40 + 30 test = 70.

**I10. ProofTranscriptComplexity — `zkmark/diagnostic.go`** (~50 LOC). `TranscriptComplexity(proof Proof) → info.lz.LzComplexityResult` — runs `info/lz/LempelZivComplexity` over `proof.ProofBytes`. A real Halo2/STARK proof has high LZ-complexity (near n/log_A(n) random-iid upper bound); a buggy or fake proof has structure that LZ76 detects. **Diagnostic, not a security guarantee** — but cheap and consumer-actionable. Reference: 196-synergy-color-info F2 pattern (LZ as diagnostic over byte streams). Connective LOC: 50 + 35 test = 85.

**I11. ProofMDLBound — `zkmark/diagnostic.go`** (~40 LOC). `ProofMinimumDescriptionLength(proof Proof) → info.mdl.UniversalIntegerCodeLength(len(ProofBytes)) + info.mdl.GaussianCodeLength(...)` — a "is this proof actually short enough to be succinct?" check. Halo2 proof for n=2^16 circuit is ≈ 480 bytes; if `len(ProofBytes) > MDLBound(circuit_size) · safety_factor`, the proof is suspect. Connective LOC: 40 + 30 test = 70.

**I12. SimulatorTranscriptDistance — `zkmark/simulator.go`** (~80 LOC). `SimulatorTranscriptDistance(real, simulated [][]byte, alphabet_size int) → float64` — empirical-distribution total-variation over byte/symbol streams via I3 `info.StatisticalDistance` on per-position empirical histograms. Reports the **observed** statistical distance between real prover transcripts and simulator transcripts; the SZK definition demands this be `negl(λ)`. Empirical, not a proof — but the input to consumer-side regression detectors. Connective LOC: 80 + 50 test = 130.

**I13. KnowledgeSoundnessErrorBound — `zkmark/soundness.go`** (~35 LOC). Type alias for `(epsilon_knowledge_error float64, num_protocol_rounds int, challenge_space_bits int)`. Function `KnowledgeSoundnessFromForking(rounds, challenge_bits) → float64 = rounds / 2^challenge_bits` is the canonical Bootle-Cerulli-Chaikin-Groth-Petit 2016 *Efficient Zero-Knowledge Arguments for Arithmetic Circuits in the Discrete Log Setting* forking-lemma error term. Pure formula. Connective LOC: 35 + 25 test = 60.

**I14. SoundnessAmplificationByRepetition — `zkmark/soundness.go`** (~30 LOC). `AmplifiedSoundness(epsilon, k_repetitions) → float64 = epsilon^k`. Sequential vs parallel repetition note (parallel is NOT generally sound for ALL Σ-protocols — Pietrzak 2007 / Bellare-Impagliazzo-Naor 1997). Function returns `(epsilon^k, "sequential")` and refuses `parallel` until a per-protocol witness lands. Connective LOC: 30 + 20 test = 50.

**I15. PedersenHidingBound_Stub — `zkmark/pedersen.go`** (~50 LOC). Pedersen 1991: `Commit(m, r) = g^m · h^r mod p` for unknown `dlog_g(h)`. Hiding bound: `H_∞(commitment | m) = log₂ |G|` for uniform `r` (information-theoretically perfect). Binding bound: computational under DLOG (`H_∞` of the witness, NOT of the commitment, is what bounds binding). Function `PedersenHidingMinEntropyBits(group_order_bits int) → float64` returns the bound; stub `Commit` returns `ErrNotYetWired` until T1-EC ships (175 Z5). Connective LOC: 50 + 40 test (formula tests, no commit yet) = 90.

**I16. ShamirSecretSharing — `zkmark/shamir.go`** (~150 LOC). Shamir 1979 *How to Share a Secret*: `(t, n)` threshold scheme via Reed-Solomon polynomial evaluation. `Split(secret, t, n, prime) → []Share` evaluates random degree-`(t-1)` poly with `f(0) = secret` at `n` distinct points; `Combine(shares []Share) → secret` does Lagrange interpolation. Uses `crypto.ModPow` + `crypto.ModInverse`. Information-theoretically secure (any `t-1` shares reveal nothing about secret — perfect privacy). **The single deepest info×crypto composition that ships TODAY without any T1 substrate** because Shamir lives entirely in `Z/pZ` with `p ≤ 2^63` (covered by `crypto.ModPow`'s `uint64` range). Connective LOC: 150 + 100 test = 250.

### I17–I24 — gates on T1 substrate (named explicitly)

**I17. WitnessIndistinguishability_Stub — `zkmark/wi.go`** (~60 LOC + gate). WI is *strictly weaker* than ZK (Feige-Shamir 1990): for any two valid witnesses `w_1, w_2` of the same `x`, `View(P(w_1)) ≈ View(P(w_2))`. WI composes under parallel repetition (unlike ZK). Stub structure ships now; semantic content gates on **T1-FIELD + T1-EC + T1-TRANSCRIPT** (175 Z3 + Z5 + Z11). Reference: Feige-Shamir 1990 *Witness Indistinguishability and Witness Hiding Protocols*. Connective LOC: 60 (after gates).

**I18. HVZK_HonestVerifierZeroKnowledge — `zkmark/hvzk.go`** (~80 LOC + gate). HVZK is the standard intermediate notion: simulator works against the honest verifier only. Implies full ZK via Goldreich-Kahan-Levin 1991 hash-then-prove transformation. Gate: T1-TRANSCRIPT (175 Z11) + I3 StatisticalDistance + I9 SimulatorEntropyBound. Connective LOC: 80 (after gates).

**I19. RandomOracleModel_Idealized — `zkmark/rom.go`** (~120 LOC + gate). `IdealRandomOracle` interface: `Query(domain_separator, input []byte) → output [n]byte` returning a freshly-sampled-uniform output for each new input. Concrete implementation requires SHA-256 / Keccak-256 / BLAKE2b (T1-HASH, 175 Z2). Without those, ships as `ROMStub` with `ErrNotYetWired`. The Fiat-Shamir transform `(Π_interactive, RO) → Π_non_interactive` lives here. Reference: Bellare-Rogaway 1993 *Random Oracles are Practical*. Connective LOC: 120 (after T1-HASH).

**I20. LeftoverHashExtractor_OverField — `zkmark/extractor.go`** (~140 LOC + gate). I8 LeftoverHashLemma applied to field-element transcripts: extract `m` near-uniform `Fp` elements from a min-entropy-`k` source. Pre-requisite for the **knowledge-extractor** in Bulletproofs IPA / Groth16. Gate: T1-FIELD (175 Z3) + I5 + I8. Connective LOC: 140 (after gates).

**I21. PolynomialHashTranscript_FiatShamirReplacement — `zkmark/transcript_poly.go`** (~180 LOC + gate). Carter-Wegman polynomial-hash-as-transcript challenge: `challenge_i = h_(a, b)(transcript_so_far) ∈ Fp` with `(a, b)` sampled from the prover's RNG. Provides 2-universal-hash-grade collision bound (`1/p`) WITHOUT the random-oracle assumption. Trade-off: requires a single round of interaction at protocol start to commit `(a, b)`; the resulting protocol is 1-extra-round but assumption-clean. **The unique post-quantum-curious alternative to Fiat-Shamir.** Reference: Bootle-Chiesa-Liu 2022 *Zero-Knowledge IOPs with Linear-Time Prover and Polylogarithmic-Time Verifier*. Gate: T1-FIELD + I5. Connective LOC: 180 (after gates).

**I22. KZG_KnowledgeOfExponent — `zkmark/kzg_koe.go`** (~70 LOC + gate). KZG (Kate-Zaverucha-Goldberg 2010) extractability rests on the Knowledge-of-Exponent (KoE) assumption: an adversary that produces `(c, c^τ)` for unknown `τ` "knows" `f` such that `c = [f(τ)]G_1`. The assumption is non-falsifiable (Naor 2003); the function `KoEAssumptionBitSecurity(curve_security_bits) → float64` returns the conjectured security budget (e.g., 128 bits for BLS12-381). Pure documentation primitive — the assumption itself cannot be a Go function. Gate: T1-PAIRING (175 Z7). Connective LOC: 70 (after gates).

**I23. RandomOracleEntropyBudget — `zkmark/rom_budget.go`** (~100 LOC + gate). Each Fiat-Shamir squeeze of an `Fp` challenge consumes `info.mdl.UniversalIntegerCodeLength(p)` bits of the oracle output stream. The function `RandomOracleEntropyBudget(transcript Transcript) → bits_consumed float64` traverses the transcript and returns the cumulative consumption. Catches an entire class of "transcript-too-short" bugs (e.g., the original Plonk paper's transcript-domain-separation issue). Gate: T1-TRANSCRIPT (175 Z11) + `info/mdl/UniversalIntegerCodeLength`. Connective LOC: 100 (after gate).

**I24. VerifiableRandomFunction_VRF + VerifiableDelayFunction_VDF — `zkmark/vrf.go` + `zkmark/vdf.go`** (~250 LOC + gate). VRF (Micali-Rabin-Vadhan 1999): `(VRF.Prove(sk, x), VRF.Verify(pk, x, π))` with output `y` pseudo-random and uniquely determined by `(sk, x)`. VDF (Boneh-Bonneau-Bunz-Fisch 2018 *Verifiable Delay Functions*): `(VDF.Eval(x, T), VDF.Verify(x, T, y, π))` where `Eval` requires `T` sequential squarings (`y = x^(2^T) mod N`) but `Verify` is logarithmic. Wesolowski 2019 / Pietrzak 2018 are the two canonical VDF proof systems. Gate: T1-EC (VRF) + RSA group / class group (T1-RSA-MOD which 057 has not enumerated — adds ~400 LOC). Connective LOC: 250 (after gates).

---

## 3. Composition table — what each topic-prompt area actually requires

| Topic-prompt area | I-primitive | Substrate gate | Status TODAY | LOC TODAY |
|---|---|---|---|---|
| Simulator-extractable, soundness ε | I9 + I12 + I13 | none | **ships TODAY** | 260 |
| Knowledge soundness ε-knowledge error | I13 + I14 | none | **ships TODAY** | 110 |
| Computational vs statistical ZK | I3 + I4 + I9 | none | **ships TODAY** | 220 |
| Honest-verifier ZK (HVZK) | I18 | T1-TRANSCRIPT | gated | 0 |
| Statistical hiding via Pedersen | I15 | T1-EC (175 Z5) | gated | 90 (formula) |
| Computational binding via discrete log | I15 + DLOG-stub | T1-EC | gated | 0 |
| Simulator entropy | I1 + I2 + I9 | none | **ships TODAY** | 200 |
| Min-entropy of commitment | I1 + I15 | T1-EC | partial today, full gated | 25 + 50 |
| Witness indistinguishability vs ZK | I17 vs I9 | T1-FIELD/EC/TRANSCRIPT | gated | 0 |
| Soundness amplification | I14 | none | **ships TODAY** | 50 |
| Random oracle model | I19 | T1-HASH (175 Z2) | gated | 0 |
| Carter-Wegman MAC | I5 + I7 | none | **ships TODAY** | 340 |
| Pedersen unconditional hiding bound | I15 (formula) | none for formula, T1-EC for impl | partial today | 90 |
| Bulletproofs IPA knowledge soundness | I13 + I20 | T1-EC + T1-FIELD + T1-TRANSCRIPT + T1-MSM | gated | 60 |
| KZG knowledge-of-exponent | I22 | T1-PAIRING (175 Z7) | gated | 0 |
| Fiat-Shamir transform | I19 + I23 | T1-HASH + T1-TRANSCRIPT | gated | 0 |
| Statistical distance simulator bound | I3 + I12 | none | **ships TODAY** | 220 |
| BLS signatures statistical soundness | gated | T1-PAIRING (175 Z7) | gated | 0 |
| Universal composability | (out of scope, formal-methods only) | — | — | 0 |
| Random-oracle entropy budget | I23 | T1-TRANSCRIPT | gated | 0 |
| HILL/Yao/metric entropy | I4 | none | **ships TODAY** (stub honesty-gate per 146) | 55 |
| PRG from OWF (HILL 1999) | I4 + I8 | T1-HASH + T1-FIELD | gated for full impl | 225 (info-side) |
| Shamir threshold secret sharing | I16 | none | **ships TODAY** | 250 |
| MPC as ZK generalization | (out of scope, separate package) | — | — | 0 |
| VRF (Micali-Rabin-Vadhan 1999) | I24 (VRF half) | T1-EC + T1-HASH | gated | 0 |
| VDF (Boneh-Bonneau-Bunz-Fisch 2018) | I24 (VDF half) | T1-RSA-MOD (extension to 057) | gated | 0 |

**Aggregate "ships today":** I1 + I2 + I3 + I4 + I5 + I7 + I8 + I9 + I10 + I11 + I12 + I13 + I14 + I15-formula + I16 = **2,015 LOC of pure connective tissue + info-side substrate, ZERO crypto T1 dependency.** Recommended ship order:

1. **Day 1:** I1 + I2 + I3 = 220 LOC (entropy + distance — the foundation primitives 199 §F2 also wants).
2. **Day 2:** I5 + I7 = 340 LOC (universal hash + one-time MAC — single highest-leverage info×crypto composition WITHOUT T1 substrate).
3. **Day 3:** I16 = 250 LOC (Shamir secret sharing — info-theoretic-secure threshold scheme).
4. **Day 4:** I8 + I20-prep = 170 LOC (leftover-hash lemma — bridges I1 → I7).
5. **Day 5:** I9 + I12 + I13 + I14 = 310 LOC (zkmark-side simulator/soundness types — forward-compatible with Tranche-2).
6. **Day 6:** I10 + I11 + I15-formula = 245 LOC (proof diagnostics + Pedersen-hiding-bound formula).

Net: **6 PRs, ~1,535 LOC, zero new substrate.** All other I-primitives gate on 057-T1-* / 147-T1-* / 175-Z* substrate the prior reviews own.

---

## 4. Three architectural findings

### F1. zkmark deterministic-by-construction ↔ ZK is structurally incompatible

The 147-zkmark-missing review §recommendation 1 already flagged the deterministic-by-construction posture mismatch. This review extends the finding: **zero-knowledge is by definition a probabilistic claim about distribution distance, and a Prover that has no `io.Reader` argument cannot make any such claim.** The fix is small but architectural:

```go
// zkmark/randomness.go (15 LOC, ships before Tranche-2)
type Randomness interface {
    Read([]byte) (int, error)
}
type RandomnessFunc func([]byte) (int, error)
func (f RandomnessFunc) Read(p []byte) (int, error) { return f(p) }

// zkmark.Prover gains an optional Read parameter:
type Prover interface {
    Prove(payload []byte, corpusSHA [32]byte, key []byte, rng Randomness) (Proof, error)
    Algorithm() string
}
```

The HonestProver (Tranche-1) ignores `rng` (no ZK property to claim — flagged in `Proof.Flavor`). The Halo2/STARK Prover (Tranche-2) consumes from `rng` for blinding factors / nonce sampling. The signature change is **a one-time API break that MUST land before Tranche-2 commits the C# port**. After Tranche-2 every Prover constructor is `NewHalo2Prover() Prover{rng: io.Reader}`-shaped; every consumer passes either `crypto/rand.Reader` (production) or a deterministic `crypto.Xoshiro256` (tests).

### F2. info/mdl + info/lz are free diagnostics for ZK proofs

The 199-synergy-graph-info §F2 found `info/lz/LempelZivComplexity` is the universal "is this byte stream structured or random?" diagnostic. Apply that finding to ZK proofs: a Halo2/STARK proof has high LZ-complexity (near-random); a buggy or fake proof has structure. **I10 + I11 (`ProofTranscriptComplexity` + `ProofMDLBound`) ship today, ~155 LOC of pure consumer-side wrapping** over `info.lz` and `info.mdl.UniversalIntegerCodeLength`. Cost: nothing. Benefit: a single Go function call gives consumers a regression-quality "is this proof actually succinct + actually random-looking?" check. **This is the single cheapest concrete primitive in this review** — zero crypto substrate, zero zkmark substrate beyond the existing `Proof.ProofBytes`.

### F3. Statistical-distance bound is the single missing piece every Tranche-2 candidate needs

Halo2 / Plonky3-FRI / SP1 / RISC0 each expose a `simulator-distance ≤ ε` claim in their security proofs. None of those proofs can be EXPRESSED in `reality/zkmark` today because `info.StatisticalDistance` does not exist. The fix is I3 (~50 LOC). **Without it, the zkmark API cannot even type the security claim it is supposed to deliver.** This is the single-deepest semantic gap in `reality`'s ZK substrate, deeper than the missing T1 layers — because the T1 layers are KNOWN GAPS (147 + 175 enumerate them), but the absence of `StatisticalDistance` is a quietly missing quantifier in a function signature: every `Prove → Proof` should carry a `SimulatorEntropyBound { StatisticalDistance, MinEntropy, Flavor }` and that struct needs both `info.StatisticalDistance` and I1 `info.MinEntropy` to be expressible.

---

## 5. Cheapest one-day shippable artifact

**I1 MinEntropy (25 LOC) + I2 RenyiEntropy (40 LOC) + I3 StatisticalDistance (50 LOC) + tests (105 LOC) = 220 LOC, single PR, no new dependencies, no API breaks, no T1 substrate.**

Ships at `info/entropy.go` (or `compression/entropy.go` if the 199 §F2 hoist has not landed — both locations are safe). Unlocks I9 + I12 + I13 + I14 + I15-formula = additional 310 LOC over week 1. Total Block-B unblocked surface: 530 LOC of zkmark-side simulator/soundness types that compile against today's `zkmark.Proof` envelope without any Tranche-2 substrate.

---

## 6. What this review does NOT cover (owned elsewhere)

- **Pairing-based SNARKs (Groth16, Marlin, PlonK, Halo2, SP1, RISC0):** owned by 175-synergy-zkmark-crypto §Z1-Z22, gates on T1-BIGINT through T1-MERKLE.
- **Hash-to-curve (Wahby-Boneh):** owned by 175 Z14.
- **Fiat-Shamir as protocol transform:** owned by 175 Z11+Z14 (T1-TRANSCRIPT). This review covers only the **info-theoretic accounting** of Fiat-Shamir's oracle entropy budget (I23).
- **PRG-from-OWF (Hastad-Impagliazzo-Levin-Luby 1999):** the FULL implementation is gated on T1-HASH + T1-FIELD; this review covers the info-theoretic accounting (I4 + I8).
- **MPC (Yao garbled circuits, GMW, BGW, SPDZ):** out-of-scope per 175 §"Topic-prompt areas not covered" — would belong in a future `reality/mpc/` package, not here.
- **Universal Composability (Canetti 2001):** out-of-scope — pure formal-methods construct, no Go expression.

---

## 7. Final inventory

- **Connective-tissue LOC, today:** 1,535 (six PRs across six days, no T1 substrate gate).
- **Connective-tissue LOC, after T1 substrate lands:** additional 1,030 (I17 + I18 + I19 + I20 + I21 + I22 + I23 + I24).
- **Substrate LOC owned by other reviews (named explicitly, not duplicated):** ~5,180 across 175-Z1-Z17 + 057-T1-* + 147-T1-*.
- **Total info-side new packages:** zero. All info-side primitives (I1-I8) land in existing `info/entropy.go` / `info/distance.go` / `info/uhash.go` / `info/mac.go` / `info/extract.go` files; the package shape stays at 2 sub-packages (`info/lz` + `info/mdl`) plus 5 sibling files at `info/`.
- **Total zkmark-side new packages:** zero. All zkmark-side primitives (I9-I24) land in existing `zkmark/` flat layout as new files (`simulator.go`, `diagnostic.go`, `soundness.go`, `pedersen.go`, `shamir.go`, `wi.go`, `hvzk.go`, `rom.go`, `extractor.go`, `transcript_poly.go`, `kzg_koe.go`, `rom_budget.go`, `vrf.go`, `vdf.go`).
- **Architectural rename pre-Tranche-2:** F1 mandates either renaming `zkmark` → `argmark` (succinct-argument) OR adding `Randomness` parameter to `Prove`. Both are one-time API breaks; the second is recommended because it preserves the consumer-facing name.

---

## References

- Carter, J.L. & Wegman, M.N. (1979). *Universal classes of hash functions.* J. Comput. Syst. Sci. 18(2): 143-154.
- Wegman, M.N. & Carter, J.L. (1981). *New hash functions and their use in authentication and set equality.* J. Comput. Syst. Sci. 22(3): 265-279.
- Shamir, A. (1979). *How to share a secret.* Commun. ACM 22(11): 612-613.
- Goldwasser, S., Micali, S. & Rackoff, C. (1989). *The knowledge complexity of interactive proof systems.* SIAM J. Comput. 18(1): 186-208.
- Pedersen, T.P. (1991). *Non-interactive and information-theoretic secure verifiable secret sharing.* CRYPTO '91.
- Feige, U. & Shamir, A. (1990). *Witness indistinguishable and witness hiding protocols.* STOC '90.
- Impagliazzo, R., Levin, L. & Luby, M. (1989). *Pseudo-random generation from one-way functions.* STOC '89.
- Hastad, J., Impagliazzo, R., Levin, L. & Luby, M. (1999). *A pseudorandom generator from any one-way function.* SIAM J. Comput. 28(4): 1364-1396.
- Bellare, M. & Rogaway, P. (1993). *Random oracles are practical: a paradigm for designing efficient protocols.* CCS '93.
- Bootle, J., Cerulli, A., Chaikin, P., Groth, J. & Petit, C. (2016). *Efficient zero-knowledge arguments for arithmetic circuits in the discrete log setting.* EUROCRYPT '16.
- Kate, A., Zaverucha, G.M. & Goldberg, I. (2010). *Constant-size commitments to polynomials and their applications.* ASIACRYPT '10.
- Micali, S., Rabin, M.O. & Vadhan, S.P. (1999). *Verifiable random functions.* FOCS '99.
- Boneh, D., Bonneau, J., Bunz, B. & Fisch, B. (2018). *Verifiable delay functions.* CRYPTO '18.
- Renner, R. & Wolf, S. (2003). *New bounds in secret-key agreement: the gap between formation and secrecy extraction.* EUROCRYPT '03.
- Naor, M. (2003). *On cryptographic assumptions and challenges.* CRYPTO '03.
- Rényi, A. (1961). *On measures of entropy and information.* Proc. 4th Berkeley Symp. Math. Stat. Prob.
- Le Cam, L. (1986). *Asymptotic Methods in Statistical Decision Theory.* Springer.
