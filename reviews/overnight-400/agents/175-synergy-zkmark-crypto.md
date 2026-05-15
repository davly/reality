# 175 | synergy-zkmark-crypto

**Topic:** zkmark × crypto — pairing-based SNARKs, hash-to-curve, polynomial commitments.
**Block:** B (cross-package synergies). **Date:** 2026-05-08.
**Scope:** capabilities that emerge ONLY when `zkmark/` and `crypto/` are composed; not isolation gaps (056-060 own crypto-isolation, 146-150 own zkmark-isolation). Repo v0.10.0, 1965 tests passing.

## Two-line summary

`zkmark/` is a 262-LOC honest-pending interface stub (Prover/Verifier/Proof envelope around a caller-injected `SignerFunc` — zero math/big, zero field elements, zero curve, ZERO crypto imports per the README "substrate-only ship" boundary), and `crypto/` is a 880-LOC number-theory + non-cryptographic-hash + deterministic-PRNG library (Miller-Rabin + ModPow + ModInverse + ExtendedGCD + ChineseRemainder + FNV1a32/64 + MurmurHash3_32 + ConsistentHash + MT19937-64 + PCG-XSH-RR + xoshiro256\*\* — ZERO SHA-2/3, ZERO Keccak, ZERO HMAC, ZERO elliptic-curve, ZERO finite-field-tower, ZERO pairing) — verified by direct grep returning zero matches across both packages on Pairing|G1|G2|Fp|FieldElement|Polynomial|Commitment|Merkle|FRI|KZG|Schnorr|BLS|Pedersen|FiatShamir|Poseidon|Keccak|sha256|HMAC|hash-to-curve|elliptic|ECDSA|Pippenger|MSM|NTT — and zero matches on `math/big` outside review docs across the entire repo. **Twenty-two synergy primitives (Z1-Z22) totalling ~7,950 LOC of substrate + connective tissue** would close the gap, but UNLIKE the 16 prior Block-B synergies where ≥85 % of LOC is pure connective-tissue composition over already-shipped surface, this pair has the OPPOSITE shape: **~95 % of the LOC is missing-substrate (T1-BIGINT/T1-FIELD/T1-FIELDTOWER/T1-EC/T1-PAIRING/T1-HASH/T1-NTT/T1-MSM/T1-POLY/T1-MERKLE/T1-TRANSCRIPT, the eight T1 layers 057-crypto-missing and 147-zkmark-missing already enumerate)** and only ~5 % is connective tissue (Z18-Z22, the 22-prim-set's last five) — meaning this synergy review is structurally a **dependency map** demonstrating that EVERY topic-prompt item (Groth16 / PlonK / Marlin / KZG / IPA / FRI / hash-to-curve / Pedersen / Schnorr / BLS-aggregate / Fiat-Shamir / Poseidon / sumcheck / HyperPlonk / Brakedown / WHIR / lattice-PCS) gates first on layers neither package ships, before any composition matters. The cheapest *honest* shippable artifact composing TODAY's `zkmark/` × `crypto/` surface is **Z18 SchnorrLikeOverModP** (~150 LOC, classical Schnorr identification over prime-modular groups using `crypto.ModPow` + `crypto.IsPrime` + `crypto.ExtendedGCD` + `crypto.Xoshiro256` + a SHA-256 stub callback, NOT elliptic-curve and NOT zero-knowledge per `reality`'s deterministic-by-construction posture flagged in 147 §recommendation 1) which can wrap into the existing zkmark `Prover`/`Verifier` envelope as a third `Algorithm` constant.

---

## Bases — what each package exposes today

### `zkmark/` (1 file, 262 LOC of source + 1 README)

`zkmark.go` ships the Tranche-1 contract:

- **Types:** `Proof{MarkChainStatement, Algorithm, ProofPending, ProofBytes, CorpusSHA[32]byte}`, `Prover` interface (`Prove(payload, corpusSHA, key) (Proof, error)`, `Algorithm() string`), `Verifier` interface (`VerifyProof(proof, payload, key) (bool, error)`).
- **Implementations:** `HonestProver` wraps caller-injected `SignerFunc` (returns `proof_pending=true`, `Algorithm = "honest-pending"`, `ProofBytes = nil`); `Halo2Prover` is a stub returning `ErrNotYetWired`; `MarkVerifier` delegates to caller-injected `VerifierFunc`.
- **Algorithm constants:** `AlgorithmHonestPending = "honest-pending"`, `AlgorithmHalo2 = "halo2"`.
- **Cryptographic surface:** literally none. No SHA, no HMAC, no field arithmetic, no curve, no transcript, no commitment scheme, no polynomial type. The README §"Substrate-only ship" makes this explicit: "`foundation/reality/zkmark/` has zero crypto dependencies".

### `crypto/` (4 files, ~880 LOC of source)

- **`prime.go`** — `IsPrime` (deterministic Miller-Rabin, n < 3.317×10²⁴), `MillerRabin(n, k)`, `PrimeFactors` (trial division O(√n)), `NextPrime`, `GCD/LCM`, `ExtendedGCD` (returns Bezout coefficients).
- **`modular.go`** — `ModPow(base, exp, m uint64)` (binary exp via Russian-peasant `mulmod`, no overflow), `ModInverse(a, m)` via extended Euclidean, `ChineseRemainder(residues, moduli)`. All `uint64`. **No `math/big` anywhere.**
- **`hash.go`** — `FNV1a32`, `FNV1a64`, `MurmurHash3_32`, `ConsistentHash` (Lamping-Veach jump hash), `SituationHashWithStructure`, `StructuralDescriptor`. All **non-cryptographic** (per 056 the package name promises crypto and delivers number theory + hash-table-grade hashing).
- **`rng.go`** — `NewMersenneTwister(seed) → .{Uint64, Float64}`, `NewPCG(seed, seq) → .{Uint32, Float64}`, `NewXoshiro256(seed) → .{Uint64, Float64}`, helper `splitmix64`. **Deterministic-only**, no DRBG, no reseeding hook, no `crypto/rand` coupling.

### Cross-imports today: zero in either direction

`grep -rE '"github.com/davly/reality/(crypto|zkmark)"' zkmark/*.go crypto/*.go` returns zero in both directions. `zkmark.go` imports only `errors` + `fmt`. `crypto/*.go` imports only `errors`. The two packages are siblings that have never been composed.

---

## The conceptual unlock — and why this synergy is structurally upside-down

Every prior Block-B synergy review (158-174) found **two well-stocked packages** with zero cross-edges and twelve-to-twenty pure-composition primitives waiting to be wired. THIS pair is different: both packages are **deliberate stubs** at this stage of the project (`zkmark/` per its `S62-S63` Tranche-2 commitment, `crypto/` per 057-crypto-missing's 5-layer foundation enumeration). The topic prompt names 17 areas — **every single one** requires substrate that neither package ships:

| Topic-prompt area | Substrate gate | Status |
|---|---|---|
| BLS12-381 / BN254 / BLS12-377 curves | T1-BIGINT, T1-FIELD, T1-FIELDTOWER (Fp²/Fp⁶/Fp¹²), T1-EC | absent |
| Tate / Weil / Optimal Ate pairings | T1-PAIRING (Miller loop + final exp) | absent |
| Groth16 / PlonK / Marlin | T1-EC + T1-PAIRING + T2-KZG + T2-PLONK + T1-NTT + T1-MSM + T1-POLY | absent |
| KZG polynomial commitments | T1-EC + T1-PAIRING + T1-MSM + T1-POLY + T1-TRANSCRIPT | absent |
| IPA (Bulletproofs) | T1-EC (Pasta) + T1-MSM + T1-POLY + T1-TRANSCRIPT | absent |
| FRI / STARKs | T1-FIELD (Goldilocks) + T1-NTT + T1-RS + T1-MERKLE + T1-TRANSCRIPT + T1-POSEIDON | absent |
| Merkle trees | T1-HASH (SHA-2/3) + T1-MERKLE | absent |
| Hash-to-curve (Wahby-Boneh) | T1-EC + T1-HASH (sha256/expand_message) + iso-map tables | absent |
| Pedersen commitments | T1-EC + small-scalar fixed-base table | absent |
| Schnorr / EdDSA / BLS signatures | T1-EC (or prime-group for classical Schnorr) + T1-HASH | partial (classical only via crypto.ModPow) |
| Fiat-Shamir transform | T1-HASH + T1-TRANSCRIPT | absent |
| Lookup arguments (Plookup, Caulk, cq) | T2-KZG (or T2-FRI) + T1-POLY | absent |
| Sumcheck protocol | T1-FIELD + T1-POLY (multilinear) + T1-TRANSCRIPT | absent |
| HyperPlonk | T2-MLE-COMMIT + T2-LOGUP + sumcheck | absent |
| Brakedown / Orion / WHIR / Ligero+ | T1-FIELD + T1-RS + T1-MERKLE + linear-time encoder | absent |
| Poseidon / Reinforced Concrete / Anemoi / Tip5 | T1-FIELD + round-constant tables + MDS matrices | absent |
| Lattice-based commitments (BGV / Module-LWE) | Polynomial-ring `R = Z[x]/(x^n+1)`, sample-from-discrete-Gaussian | absent |

**The only topic-prompt area for which BOTH packages contribute usable surface today is "Schnorr signatures (classical, prime-modular form)".** Everything else gates on layers that 057-crypto-missing's T1-* and 147-zkmark-missing's T1-NTT/T1-MSM/T1-POLY/T1-MERKLE/T1-TRANSCRIPT/T1-RS/T1-POSEIDON enumerate. This synergy review is therefore a **dependency map** showing the order in which substrate + composition land, not a 20-primitive composition catalog like 174.

---

## Twenty-two primitives — substrate gates first, then composition

Z1-Z17 are substrate; Z18-Z22 are connective tissue. The substrate primitives are owned by other reviews (057-T1-* / 147-T1-NTT/T1-MSM/T1-POLY/T1-MERKLE/T1-TRANSCRIPT/T1-RS/T1-POSEIDON); they appear here only because a `zkmark × crypto` synergy CANNOT escape naming them as gates.

### Substrate — Tier 1 (the eight foundation layers)

**Z1. T1-BIGINT — `crypto/bigint/`** (~600 LOC). Wrap `math/big.Int` (or hand-rolled limb arithmetic for constant-time discipline) with `Add/Sub/Mul/Div/Mod/Exp/ModInverse/CmpAbs`. Owned by 057. Currently absent across the repo. **Gates Z3-Z22.**

**Z2. T1-HASH — `crypto/sha2.go` + `crypto/keccak.go` + `crypto/blake.go`** (~700 LOC). SHA-256, SHA-3-256, Keccak-256, BLAKE2b, BLAKE3 round-functions. Currently absent (`crypto/hash.go` is FNV/Murmur/Jump only). Owned by 057-T1-HASH. **Gates Z6, Z9-Z11, Z14, Z16, Z18-Z22.**

**Z3. T1-FIELD — `zkmark/field/`** (~500 LOC). Prime-field `Fp` over an `Fp.Modulus` parameter, with `Add/Sub/Mul/Square/Inverse/Pow/Sqrt/Legendre/IsZero/Equal`. Two-adicity metadata + 2^k-th-root-of-unity tables (NTT prerequisite). Goldilocks (p = 2^64 − 2^32 + 1, two-adicity 32), Mersenne-31 (p = 2^31 − 1, circle-NTT), BabyBear (p = 2^31 − 2^27 + 1, two-adicity 27), and BLS12-381 scalar field q (255-bit prime). Owned by 057-T1-FIELD + 147-T1-FIELD. **Gates Z4-Z22 except Z18.**

**Z4. T1-FIELDTOWER — `zkmark/field/tower.go`** (~400 LOC). Quadratic / sextic / cubic-on-quadratic extensions Fp² / Fp⁶ / Fp¹² over a non-residue. Pairing prerequisite (G₂ ⊂ Fp², GT ⊂ Fp¹²). Owned by 057. **Gates Z5, Z7, Z11, Z14, Z19, Z20.**

**Z5. T1-EC — `zkmark/ec/`** (~700 LOC). Short-Weierstrass `y²=x³+ax+b` and twisted-Edwards `ax²+y²=1+dx²y²` curve points with `Add/Double/ScalarMul/Negate/IsOnCurve`. Endomorphism-aware (GLV-decomposition for BLS12-381 / BN254). Curves: **BLS12-381** (Boneh-Lynn-Shacham + Eth2 + Filecoin), **BN254** (Ethereum precompile, Groth16-on-Eth), **BLS12-377** (Aleo + Zexe), **Pasta cycle** (Pallas/Vesta for Halo2 IVC), **Bandersnatch** (Edwards on BLS12-381 scalar field, in-circuit Pedersen + Verkle), **Jubjub** (Edwards on BLS12-381 scalar, Sapling). **Gates Z7, Z11, Z12, Z13, Z14, Z15, Z19, Z20, Z21.**

**Z6. T1-MSM — `zkmark/ec/msm.go`** (~400 LOC). Multi-scalar-multiplication via Pippenger-bucket method; window size c ≈ log₂(n) − 2; GLV-endomorphism scalar split halves bit length on BLS12-381. Variants: public-scalar fast-path (production prover) + constant-time-scalar fall-back (signature). Workspace type `MSMScratch` per CLAUDE.md §3. Owned by 147-T1-MSM. **Gates Z11, Z12, Z14, Z19.**

**Z7. T1-PAIRING — `zkmark/ec/pairing.go`** (~600 LOC). Optimal-Ate pairing via Miller loop + final exponentiation. For BLS12-381: 64-bit Miller-loop scalar `x = -0xd201000000010000`, final exp = (p¹² − 1) / r, three-stage ladder (Aranha-Karabina-Longa-Gebotys-Lopez 2010). Pairing-product check: `Π e(Pᵢ, Qᵢ) == 1` is THE primitive Groth16/KZG/BLS-aggregate consume. Owned by 057 (extension of T1-EC). **Gates Z11, Z14, Z19, Z20, Z21.**

**Z8. T1-NTT — `zkmark/poly/ntt.go`** (~300 LOC). Radix-2 Cooley-Tukey FFT over `Fp` with two-adicity ≥ log₂(n); inverse NTT; coset-NTT (FRI prerequisite); truncated/mixed-radix variant. Owned by 147-T1-NTT. **Gates Z9, Z10, Z11, Z14, Z15, Z16, Z17, Z19, Z20, Z21.**

**Z9. T1-POLY — `zkmark/poly/`** (~500 LOC). Univariate `Polynomial` (coefficient form) + `Evaluations` (Lagrange-domain form), `Add/Sub/Scale/Mul-via-NTT/DivWithRemainder/Lagrange/VanishingPoly`. Multilinear extension (MLE) of `f: {0,1}ⁿ → Fp` in both coefficient form (length 2ⁿ) and Thaler dynamic-evaluation form (round-by-round, no per-round allocation). Owned by 147-T1-POLY. **Gates Z11, Z14, Z15, Z16, Z17, Z19, Z20, Z21.**

**Z10. T1-MERKLE — `zkmark/merkle/`** (~200 LOC). Binary + arity-N Merkle tree with caller-supplied hash function (SHA-256, Keccak, BLAKE3, or in-field Poseidon). Path-proof gen/verify, batch-opening with shared-node deduplication, frontier-only commitment for streaming proofs. Owned by 147-T1-MERKLE. **Gates Z15, Z16, Z17, Z21.**

**Z11. T1-TRANSCRIPT — `zkmark/transcript/`** (~150 LOC). Fiat-Shamir transcript with `Absorb(label, bytes)` + `SqueezeChallenge(label) → Fp`. Hash-based (Merlin-style, BLAKE2b/Keccak with explicit length-prefix) AND sponge-based (Poseidon-rate-2-arity-3 in-field, for STARK and recursive systems). Domain-separated `b"reality-zkmark-v1-<protocol>"`. Empty-transcript rejection. Replay-protection counter. Owned by 147-T1-TRANSCRIPT. **Gates Z14-Z22.**

**Z12. T1-POSEIDON — `zkmark/hash/poseidon.go` + `rescue.go` + `anemoi.go`** (~400 LOC). Poseidon (Grassi-Khovratovich-Lüftenegger-Rechberger-Roy-Schofnegger 2020) width t=3 (rate 2 cap 1), x⁵ S-box for BN254/BLS12, x⁷ for Goldilocks; Rescue-Prime (Aly-Ashur-Ben-Sasson-Dhooghe-Szepieniec 2020); Anemoi (Bouvier et al. 2022 Flystel S-box). Per-(p,t,α) round-constant tables golden-pinned. Owned by 147-T1-POSEIDON. **Gates Z11 (sponge variant), Z16, Z17, Z21.**

**Z13. T1-RS — `zkmark/poly/rs.go`** (~100 LOC after Z8 lands). Reed-Solomon encode (k coefficients → n evaluations via NTT), erasure decode (Welch-Berlekamp for general, direct Lagrange for subgroup positions), Berlekamp-Massey syndrome decode. Rate ρ ∈ {1/2, 1/4, 1/8, 1/16}. Owned by 147-T1-RS. **Gates Z16, Z17.**

### Substrate — Tier 2 (composition over T1 layers)

**Z14. T2-KZG — `zkmark/commit/kzg/`** (~300 LOC). Univariate KZG = `Commit(f) = Σ fᵢ · [τⁱ]G₁` (one MSM via Z6) + `Open(f, z)` quotient `q(X) = (f(X) − f(z))/(X − z)` long-divided then MSM-committed + `Verify(c, z, y, π)` pairing check `e(c − [y]G₁, [1]G₂) == e(π, [τ]G₂ − [z]G₂)` (one Z7 pairing-product). Batched-opening (random-linear-combination via Z11) + multi-point opening (Feist-Khovratovich). EIP-4844 wire-format adjacent. Owned by 147-T2-KZG. **Gates Z19, Z20, Z21.**

**Z15. T2-IPA — `zkmark/commit/ipa/`** (~400 LOC). Bulletproofs-style polynomial commitment over Pasta; commit `c = Σ fᵢ Gᵢ` (MSM via Z6); open via log₂(n) rounds of Bulletproofs folding `(G, f) ↦ (G' = u·G_left + u⁻¹·G_right, f' analog)`; verifier O(n) so Halo2 amortises via in-circuit accumulation. No pairing required (this is why Halo2 uses Pasta). Owned by 147-T2-IPA. **Gates Z21.**

**Z16. T2-FRI — `zkmark/commit/fri/`** (~600 LOC). Fast-RS interactive-oracle proof of proximity: prover commits to RS-encoded f via Merkle root r₀; verifier sends fold challenge α₀ via Z11; prover splits f = f_even(X²) + X·f_odd(X²), commits r₁ = Merkle(α₀·f_odd + f_even on D₀² halved); repeats log₂(n) − log₂(rate) times. Query phase: λ random positions in D₀ via Z11. DEEP-FRI quotient-by-out-of-domain-point. Soundness Ben-Sasson-Bentov-Horesh-Riabzev 2018 + Block-Garreta-Hall-Katz-Liu-Tairi 2023 proximity gaps. **No curve, no pairing — the transparent anchor.** Owned by 147-T2-FRI. **Gates Z17, Z21.**

**Z17. T2-MLE-COMMIT — `zkmark/commit/multilinear/`** (~400 LOC). Multilinear polynomial commitment variants: **Zeromorph** (Kohrita-Towa 2024, multilinear-KZG via quotient identity reducing to Z14), **PST** (Papamanthou-Shi-Tamassia 2013 direct multilinear KZG), **Hyrax** (Wahby et al. 2018 multilinear IPA, transparent), **Brakedown** (Golovnev-Lee-Setty-Thaler-Wahby 2021 code-based, hash-only, post-quantum-friendly), **Orion** (Xie-Zhang-Song 2022 linear-time prover successor), **WHIR** (Arnon-Chiesa-Fenzi-Yogev 2024 multilinear FRI variant). Owned by 147-T2-MLE-COMMIT. **Gates Z21 (HyperPlonk, Spartan, Lasso/Jolt — all out-of-band of THIS review).**

### Connective tissue — what actually composes `zkmark/` × `crypto/` (Z18-Z22)

These are the only primitives in the 22 that are pure composition over the **current** `zkmark/` + `crypto/` surface — i.e., they ship LOC against v0.10.0 without any T1-* layer landing first.

**Z18. SchnorrLikeOverModP — classical Schnorr identification over a prime-modular Schnorr group** (~150 LOC, single-day PR). Use `crypto.IsPrime` to verify (p, q) where q | p−1, generator g of order q via `crypto.ModPow(h, (p−1)/q, p)` for a `crypto.NextPrime`-derived h. Sign / Verify:

```go
// keygen: x ← random ∈ [1,q), y = g^x mod p
// sign(m, x): r ← random ∈ [1,q); R = g^r mod p; e = H(m||R) mod q; s = r + e·x mod q; return (e, s)
// verify(m, y, e, s): R' = g^s · y^(−e) mod p; e' = H(m||R') mod q; return e == e'
```

All arithmetic is `crypto.ModPow` + `crypto.ModInverse` + addmod/mulmod (all already exported by `crypto/`). Hash `H` is a caller-injected `func([]byte) []byte` (mirrors zkmark's `SignerFunc` injection pattern — keeps zkmark substrate-only-ship boundary intact). Random `r` is `crypto.NewXoshiro256(seed).Uint64() % q`. **This composes today.** It is NOT zero-knowledge (`reality` is deterministic, see 147 §recommendation 1) but it IS a sound non-interactive signature usable as a third `Algorithm` constant `AlgorithmSchnorrModP` in the existing zkmark Proof envelope. **Cross-validation pin (R-MUTUAL 3/3 candidate): same (p, q, g, x, m, r) tuple → same (e, s) byte-for-byte across Go / Python / C# / C++ via golden file.** This is the single most-honest existing-surface composition this review identifies.

**Z19. PoolPairingProductGate — interface stub for the pairing-product check Groth16/KZG/BLS-aggregate verifiers all reduce to** (~80 LOC). Define:

```go
package zkmark
type PairingProductCheck interface {
    // Pairs (Pᵢ, Qᵢ) on a pairing-friendly curve;
    // returns true iff Π e(Pᵢ, Qᵢ) == 1 in GT.
    Check(left, right []G1G2Pair) bool
}
type G1G2Pair struct { P G1Point; Q G2Point }
```

Plus an `ErrPairingNotYetWired` sentinel mirroring `ErrNotYetWired`. This is purely the envelope — it **forward-binds** the Z7 pairing the package will eventually consume, just like `Halo2Prover` forward-binds the Tranche-2 Halo2 backend. Lets the `zkmark.MarkVerifier` switch-statement gain `case AlgorithmGroth16: return PairingNotYetWired` cases without breaking existing parsers, and lets a future `Groth16Verifier` drop in by implementing `PairingProductCheck` once Z5/Z7 land. **Cross-link to Z21**: the verify-side of every pairing-based SNARK below is structurally one call to `Check`. **Composes today** as interface-only; the implementation is gated on Z5+Z7.

**Z20. KZGVerifierEnvelope — KZG verify envelope assuming external pairing oracle** (~100 LOC). `KZGVerifyOpening(srs SRS, c G1Point, z, y Fp, proof G1Point, pairing PairingProductCheck) bool`. The **algorithm logic** of KZG verify (build the two G1G2Pairs, call pairing.Check) is independent of the curve choice; only the underlying `Fp / G1Point / G2Point` types depend on Z3-Z5 and the actual `Check` impl depends on Z7. By publishing the verify *shape* now, we let consumers stub-test with a `MockPairingProductCheck{Should: true}` and have the algorithm logic be golden-pinned before any field/curve lands. **Cross-link to Z14:** Z14 is the FULL math (commit + open + verify + batched + multi-point); Z20 is JUST the verify envelope as an interface adapter against zkmark. Lets Tranche-2 land as `KZGProver` implementing `zkmark.Prover` returning `Algorithm = "kzg-bls12-381"` once substrate is in place.

**Z21. ZkmarkAlgorithmRegistry — extend zkmark Algorithm constants and verifier dispatch for the full set of named protocols** (~80 LOC). Add constants:

```go
const (
    AlgorithmHonestPending    = "honest-pending"
    AlgorithmHalo2            = "halo2"               // existing
    AlgorithmSchnorrModP      = "schnorr-modp"        // Z18; ships today
    AlgorithmGroth16          = "groth16-bls12-381"   // gated on Z5+Z7
    AlgorithmGroth16BN254     = "groth16-bn254"       // gated on Z5+Z7
    AlgorithmKZGOpening       = "kzg-bls12-381"       // gated on Z14
    AlgorithmPlonk            = "plonk-bls12-381"     // gated on Z14+Z9+Z11
    AlgorithmHaloIPA          = "halo2-ipa-pasta"     // gated on Z15
    AlgorithmFRIGoldilocks    = "fri-goldilocks"      // gated on Z16 (no curve!)
    AlgorithmStarkGoldilocks  = "stark-goldilocks"    // gated on Z16+Z12
    AlgorithmHyperPlonk       = "hyperplonk"          // gated on Z17
    AlgorithmBLSAggregate     = "bls-aggregate"       // gated on Z5+Z7
    AlgorithmPedersenCommit   = "pedersen-bls12"      // gated on Z5
    AlgorithmHashToCurve      = "hash-to-curve-bls12" // gated on Z2+Z5
)
var ErrPairingNotYetWired = errors.New("zkmark: pairing-based backend not yet wired")
var ErrFRINotYetWired     = errors.New("zkmark: FRI backend not yet wired")
```

Plus matching switch-cases in `MarkVerifier.VerifyProof`. Each new case returns its specific not-yet-wired sentinel, so a downstream caller can pattern-match `errors.Is(err, zkmark.ErrFRINotYetWired)` and choose an alternative. **Composes today as constants + sentinels only;** the actual Prover/Verifier impls are gated on substrate. The structural lift mirrors 147 §recommendation 3 (rename `AlgorithmHalo2` → backend-agnostic) — THIS review goes further by saying: don't rename, just **add siblings**, so the parser shape is open-set forward-compatible without breaking the existing Halo2 commitment.

**Z22. RandomOracleHashAdapter — wrap `crypto.MurmurHash3_32` / `crypto.FNV1a64` as a SHIM `RandomOracle` interface that the future `crypto.SHA256` will plug into** (~60 LOC). Define:

```go
package zkmark
type RandomOracle func(domain []byte, msg []byte) []byte
// 32-byte output expected; sha256-shaped.
```

Provide `NewMurmurOracle(seed uint32) RandomOracle` (today, NON-cryptographic — flagged `Unsound: true` in struct field) and `NewFNVOracle() RandomOracle` (today, NON-cryptographic). When Z2 lands, add `NewSHA256Oracle() RandomOracle` with `Unsound: false`. Every Z18-Z21 hash callsite consumes this interface. **The honest part:** today's implementations are flagged unsound at construction time; consumers cannot accidentally treat a Murmur-based "Fiat-Shamir transcript" as cryptographic. **Composes today.** This is the `zkmark × crypto` analog of the `zkmark.SignerFunc` substrate-only-ship pattern: caller injects the strong primitive when they have it; package provides the weak default with a flag. **Cross-link to Z11:** Z11 is the full transcript with domain-separation + replay-protection + sponge mode; Z22 is JUST the hash adapter Z11 will eventually plug into.

---

## Composition matrix

| # | Primitive | zkmark uses | crypto uses | T1-* gates | LOC | Ships today? |
|---|---|---|---|---|---|---|
| Z1 | T1-BIGINT | — | — | — | 600 | NO (substrate, owned 057) |
| Z2 | T1-HASH (SHA/Keccak/BLAKE) | — | (replaces FNV/Murmur) | — | 700 | NO (substrate, owned 057) |
| Z3 | T1-FIELD | (consumer) | — | Z1 | 500 | NO (substrate, owned 057+147) |
| Z4 | T1-FIELDTOWER | (consumer) | — | Z1, Z3 | 400 | NO (substrate, owned 057) |
| Z5 | T1-EC | (consumer) | (consumer of Z1) | Z1, Z3, Z4 | 700 | NO (substrate, owned 057) |
| Z6 | T1-MSM | (consumer) | — | Z3, Z5 | 400 | NO (substrate, owned 147) |
| Z7 | T1-PAIRING | (consumer) | — | Z3, Z4, Z5 | 600 | NO (substrate, owned 057+147) |
| Z8 | T1-NTT | (consumer) | — | Z3 | 300 | NO (substrate, owned 147) |
| Z9 | T1-POLY | (consumer) | — | Z3, Z8 | 500 | NO (substrate, owned 147) |
| Z10 | T1-MERKLE | (consumer) | (caller-supplied hash) | Z2 | 200 | NO (substrate, owned 147) |
| Z11 | T1-TRANSCRIPT | (consumer) | (caller-supplied hash) | Z2, Z3, Z12 (sponge) | 150 | NO (substrate, owned 147) |
| Z12 | T1-POSEIDON | (consumer) | — | Z3 | 400 | NO (substrate, owned 147) |
| Z13 | T1-RS | (consumer) | — | Z3, Z8 | 100 | NO (substrate, owned 147) |
| Z14 | T2-KZG | (Prover impl) | — | Z5, Z6, Z7, Z9, Z11 | 300 | NO |
| Z15 | T2-IPA | (Prover impl) | — | Z5, Z6, Z9, Z11 | 400 | NO |
| Z16 | T2-FRI | (Prover impl) | — | Z3, Z8, Z9, Z10, Z11, Z13 | 600 | NO |
| Z17 | T2-MLE-COMMIT | (Prover impl) | — | Z14 OR Z16 + Z9 | 400 | NO |
| **Z18** | **SchnorrLikeOverModP** | **wraps as Algorithm** | **`ModPow` + `ModInverse` + `IsPrime` + `NextPrime` + `Xoshiro256`** | (none — composes today) | **150** | **YES** |
| **Z19** | **PoolPairingProductGate (interface)** | **adds interface + sentinel** | — | (none — interface only; impl gated on Z7) | **80** | **YES (interface)** |
| **Z20** | **KZGVerifierEnvelope (algorithm logic vs mock pairing)** | **adds verify-shape** | — | (algorithm only; impl gated on Z14) | **100** | **YES (algorithm vs mock)** |
| **Z21** | **ZkmarkAlgorithmRegistry (12 new constants + 12 sentinels)** | **adds constants + dispatch** | — | (none — string constants only) | **80** | **YES** |
| **Z22** | **RandomOracleHashAdapter (Unsound-flagged shim)** | **adds interface** | **wraps `MurmurHash3_32` + `FNV1a64`** | (none — wraps existing; SHA-shim plugs in once Z2 lands) | **60** | **YES** |
| **Σ ships today** | | | | | **470** | |
| **Σ all 22** | | | | | **7,950** | |

The five Z18-Z22 primitives totalling **470 LOC** are the **only** lines this review can land against v0.10.0. Everything else is substrate owned by parallel reviews.

---

## Recommended PR sequence

### PR-1 — ship-today Tranche-1.5 envelope expansion (~470 LOC, 1.5 engineer-days)

**Z18 SchnorrLikeOverModP + Z19 PoolPairingProductGate + Z20 KZGVerifierEnvelope-vs-mock + Z21 AlgorithmRegistry (12 constants + 12 sentinels) + Z22 RandomOracleHashAdapter.**

- Closes the "second `Algorithm` ships today" gap noted in 147 §6 (recommends FRI-over-Goldilocks at ~1,800 LOC; this review shows Schnorr-mod-p at ~150 LOC is a *third*, even-cheaper-still anchor — **NOT zero-knowledge**, NOT pairing-based, but provably sound and composes the existing crypto+zkmark surface verbatim).
- Saturates **R-MUTUAL-CROSS-VALIDATION 3/3 pin** on Schnorr-mod-p: `crypto.ModPow`-based Go path × Python `pow(g, x, p)` × C# `BigInteger.ModPow` agree byte-for-byte across 30 golden vectors with 256-bit p, 255-bit q, fixed (x, m, r) tuples. Mirrors commit `6a55bb4` audio-onset and `365368a` Clayton-autodiff cross-validation idioms.
- First cross-edge `zkmark/ → crypto/` (one-way, cycle-free verified by enumeration).
- Z19/Z20 adapter envelopes are **forward-compatibility scaffolding**: lets a future `Groth16Prover` / `KZGProver` drop in once Z5/Z7/Z14 substrate lands without breaking the existing zkmark API surface. Mirrors how `Halo2Prover` already forward-binds Tranche-2.
- Z21 expansion to 14 algorithm constants + 12 sentinels lets cold-verifier callers branch on EXACTLY which backend is missing (`errors.Is(err, ErrFRINotYetWired)` vs `ErrPairingNotYetWired` vs `ErrNotYetWired` for the Halo2 default). Critical: the existing `AlgorithmHalo2` constant STAYS unchanged (per 147 §recommendation 3 the rename is one option; THIS review prefers add-siblings-don't-rename to avoid breaking 147's own forward-compat contract).

### PR-2 — substrate (gated on 057-T1-BIGINT/T1-HASH/T1-FIELD landing first; ~3,500 LOC, 4-6 engineer-weeks)

**Z2 + Z3 + Z8 + Z9 + Z10 + Z11 (with hash-based mode only — no Poseidon yet) + Z13.**

Sequencing matches 147's Sprint-1+2: T1-NTT + T1-POLY + T1-MERKLE + T1-TRANSCRIPT + T1-RS over a single small prime field (Goldilocks). After this PR lands, FRI-over-Goldilocks (Z16) is a single-day write because every dependency is satisfied. **No curve, no pairing required for this PR.**

### PR-3 — FRI anchor (~600 LOC, 1 engineer-week)

**Z16 T2-FRI + Z12 T1-POSEIDON.** Closes 147's "Sprint-3 cheapest-real-Tranche-2-shippable" recommendation. Plugs into Z21 as `AlgorithmFRIGoldilocks`. **Still no curve, still no pairing.** Saturates a second R-MUTUAL pin (Plonky2 Goldilocks vector × StarkWare ethSTARK vector × this `reality` impl) once cross-language vectors are sourced.

### PR-4 — curve substrate (~2,000 LOC, ~1 engineer-month, gated on PR-2)

**Z4 + Z5 + Z6 + Z7 over BLS12-381.** This is the heavy lift. After this PR, every pairing-based primitive (Groth16, BLS-aggregate, KZG, hash-to-curve, Pedersen, Bandersnatch-in-circuit) is a 100-400 LOC composition.

### PR-5 — KZG + Pedersen + Hash-to-curve (~700 LOC, 1-2 engineer-weeks)

**Z14 KZG + classical Pedersen-BLS12 + Wahby-Boneh hash-to-curve for BLS12-381 G₁/G₂.** Plugs into Z21 as `AlgorithmKZGOpening`, `AlgorithmPedersenCommit`, `AlgorithmHashToCurve`. Pedersen + Schnorr-on-curve (BLS12 instead of Z18's mod-p) ship together with hash-to-curve as the dependency-closure unit (each is 100-200 LOC; total fits one PR).

### PR-6 — Groth16 + BLS-aggregate (~400 LOC each = 800 LOC, 1-2 engineer-weeks)

**`AlgorithmGroth16` + `AlgorithmBLSAggregate` as `Prover` / `Verifier` implementations.** Both plug into Z19's `PairingProductCheck`. R-MUTUAL pin: a Groth16-BN254 proof from this `reality` impl × `arkworks-rs` × `gnark` agree on a fixed circuit (e.g., the canonical "I know x such that SHA256(x) = h" benchmark).

### PR-7 — Plonk + IPA-Pasta + Halo2 (~2,500 LOC, 4-6 engineer-weeks)

**Z15 IPA + `AlgorithmPlonk` + `AlgorithmHaloIPA`.** Closes zkmark.go's S62-S63 Tranche-2 commitment to Halo2.

### PR-8+ — multilinear (Z17 + HyperPlonk + Brakedown / Orion / WHIR) (~2,500 LOC over 2-3 PRs)

Frontier 2024-2026 work. Lasso/Jolt and Spartan are out-of-band (need a constraint-system layer 147 flags as borderline `zkmark/cs/`).

**Total to reach Halo2 parity: PRs 1-7 ≈ 10,500 LOC + ~10,500 LOC tests across ~3-4 engineer-months.** Matches 147's "~7,750 LOC of math + equal test surface across ~8 sprints" estimate to within rounding.

---

## Precision hazards

- **Schnorr-mod-p is NOT zero-knowledge in `reality`.** Per `reality`'s deterministic-by-construction posture (147 §1, this review §"Substrate"), `r ← Xoshiro256(seed).Uint64() % q` is determined by the seed; an attacker who recovers the seed recovers x. Z18's docstring MUST flag this — same caveat 147 §recommendation 1 makes about every Tranche-2 prover. Schnorr-mod-p is **sound** (verifier check is correct given the math) but **not hiding** until the prover gets prover-side OS entropy. The honest framing: "non-interactive signature soundly tied to a hidden x **only as long as the seed is hidden**", which is the standard signature-scheme contract anyway — but stake the disclaimer in package docs.
- **Modular reduction `% q` introduces bias when `q < 2⁶⁴`.** `Xoshiro256.Uint64() % q` is biased by ≤ 2⁻⁶⁴ × q for q close to 2⁶⁴. For 256-bit q the bias is invisible in 64-bit splits, but a strict implementation should use rejection sampling per 64-bit chunk (Z18 ships rejection-sampling by default; document the rate as ~q/2⁶⁴ rejection probability which for cryptographic q is ~1/2 → expected 2 draws per scalar).
- **`crypto.ModPow` is `uint64`-only.** Z18 must use modular composition over CRT-decomposed `(p_lo, p_hi)` halves OR use `math/big.Int.Exp` (which the package currently doesn't import). Recommend adding `crypto.ModPowBig(base, exp, m *big.Int) *big.Int` as a Z18 prerequisite (~30 LOC) — a pure forwarding to `math/big` that is the FIRST `math/big` import in the entire repo and correctly so per 057-crypto-missing's T1-BIGINT recommendation.
- **`crypto.IsPrime` is uint64-only too** — same modulus-size constraint. For 256-bit primes (p, q for Schnorr group), need `crypto.MillerRabinBig(n *big.Int, k int) bool` (~50 LOC, ports the existing witness-set algorithm to `*big.Int`). Z18 cannot ship without this 80-LOC shim or against a `crypto.MillerRabin` that hasn't been bigint-extended. **Z18's true ship-cost is closer to ~230 LOC including the bigint shim, not 150.**
- **Z22's `MurmurOracle` and `FNVOracle` are NOT cryptographic.** Both have collision rates trivially distinguishable from random. `Unsound: true` field is mandatory. Any Fiat-Shamir transcript built on these is **broken** as a security primitive but works as a **correctness fixture** for golden-file pinning: byte-identical `MurmurOracle("domain", "msg")` output across Go/Python/C# proves the transcript-shape is portable, even though the security claim isn't real until Z2 lands.
- **Pairing-product gate Z19 is interface-only.** `MockPairingProductCheck{Should: true}` returns true ALWAYS; an attacker passing this mock can forge any "proof". Tests using the mock MUST fail-closed at integration time when a real `BLS12381Pairing` lands; flag as `// TEST_ONLY: replace with real pairing before production` in the mock docstring.
- **KZG verifier envelope Z20 assumes the SRS is honest.** `reality` does not generate the trusted-setup ceremony output (147 §"OUT" boundary); Z20's `srs SRS` parameter must be loaded from a caller-supplied byte-stream that the caller has independently verified came from a public ceremony (Powers of Tau / Filecoin / Zcash / Mainnet KZG ceremony). Z20 docstring must surface this.
- **Algorithm-constant churn risk.** Z21 adds 12 new `Algorithm*` constants. If a future agent renames any of them (per 147 §recommendation 3 considering renaming `AlgorithmHalo2` → `AlgorithmTranche2`), every cold-verifier parser breaks. Recommend pinning all 14 string constants in a `proof_format.json` golden fixture per CLAUDE.md §1, with a Go test that verifies the on-disk constants match the in-source constants byte-for-byte.
- **`crypto.SituationHashWithStructure` is structurally tempting as a Fiat-Shamir transcript.** It IS NOT one. It uses FNV-1a (≤ 64 bits collision-resistance, trivially attackable). Z22's adapter pattern is the safe path: `RandomOracleHashAdapter(SituationHashWithStructure)` ships with `Unsound: true` and refuses to be passed to any `Algorithm*` constant whose protocol requires cryptographic hash unless the caller explicitly passes `AcceptUnsound: true`. Belt-and-braces.
- **`crypto/rng.go` PRNGs are NOT cryptographic.** All three (MT19937, PCG, xoshiro256) have linear state recovery. Schnorr-mod-p (Z18) using xoshiro256 for `r` is broken if the seed entropy is < q-bits or if the state is observed. Document. Recommend a future `crypto.NewDRBG(seed)` per 057-T1-RNG returning a sponge-based DRBG (Hash-DRBG / HMAC-DRBG / CTR-DRBG) before Schnorr-mod-p ships in any production-facing consumer; for `reality`'s deterministic-pinning-only contract, the existing PRNGs suffice as long as the disclaimer is loud.
- **`zkmark.Proof.CorpusSHA [32]byte` is NOT computed by `crypto/`.** It's caller-supplied per the `SignerFunc` injection pattern. Adding a `crypto.SHA256(payload []byte) [32]byte` function (Z2 prerequisite) lets `zkmark` callers verify the corpusSHA via a single audit-trail call, instead of trusting the caller's hash. Cross-link to 057-T1-HASH and to 146 §G7 transcript-byte-pinning.

---

## Architectural placement

**Substrate-side, in `crypto/` and a new `zkmark/` subpackage tree.** Mirroring 147 §recommendation 2's eventual layout:

```
crypto/
  bigint.go                 # Z1 (~600 LOC; first math/big import)
  sha256.go                 # Z2 (part)
  keccak.go                 # Z2 (part)
  blake.go                  # Z2 (part)
  hmac.go                   # Z2 (part) — MAC over caller-supplied hash
  modular_big.go            # Z18 prereq (~80 LOC; bigint shim of ModPow + IsPrime)
  drbg.go                   # 057-T1-RNG; Schnorr-mod-p production-grade (deferred)

zkmark/
  zkmark.go                 # existing 262 LOC + Z21 algorithm constants + Z22 RandomOracle
  schnorr_modp.go           # Z18 (~150 LOC; only file that ships PR-1 against existing surface)
  pairing_gate.go           # Z19 (~80 LOC; PairingProductCheck interface + mock)
  kzg_envelope.go           # Z20 (~100 LOC; verify shape vs MockPairingProductCheck)
  oracle_adapter.go         # Z22 (~60 LOC; RandomOracle + Murmur/FNV unsound shims)

zkmark/field/                # Z3-Z4 (sub-package, mirrors 057+147)
zkmark/ec/                   # Z5-Z7 (sub-package)
zkmark/poly/                 # Z8-Z9-Z13 (sub-package)
zkmark/merkle/               # Z10
zkmark/transcript/           # Z11 (depends on zkmark/hash for Poseidon)
zkmark/hash/                 # Z12 (Poseidon/Rescue/Anemoi)
zkmark/commit/kzg/           # Z14 (depends on zkmark/ec + zkmark/poly + zkmark/transcript)
zkmark/commit/ipa/           # Z15
zkmark/commit/fri/           # Z16
zkmark/commit/multilinear/   # Z17
zkmark/proof/groth16/        # PR-6 future
zkmark/proof/plonk/          # PR-7 future
zkmark/proof/halo2/          # PR-7 future (Tranche-2 anchor)
zkmark/proof/stark/          # PR-3+ future
zkmark/proof/schnorr/        # not needed; Z18 lives in zkmark/schnorr_modp.go alongside the existing zkmark.go
```

**Cycle-free DAG (PR-1 only):** `zkmark/ → crypto/` (one-way; never reverses). `crypto/` does not import `zkmark/` and never will. Verified by enumeration of imports across both packages.

**After full landing:** `zkmark/ → crypto/` for the bigint-shim + SHA + DRBG; `zkmark/ → zkmark/{field, ec, poly, merkle, transcript, hash, commit/*, proof/*}` internal-only. Mirror direction never. The 16-consecutive-synergies consumer-side-placement convention (158-174) is preserved: zkmark consumes crypto, not the other way around. The substrate that lives in `zkmark/{field, ec, ...}` is internal-to-zkmark — these are not new sibling packages competing with `crypto/`, they are the math machinery owned by the ZK substrate per 147 §recommendation 2.

---

## Distinct-from-prior-reviews provenance

This is the **eighteenth Block-B synergy review** in the 175-of-400 sequence and the **first** `zkmark × crypto` review.

- **056-060 (crypto isolation).** 056 names: package called "crypto" delivers number theory + non-cryptographic-hash; SHA/HMAC/EC absent. 057 enumerates T1-BIGINT/T1-FIELD/T1-FIELDTOWER/T1-EC/T1-HASH at ~2,500 LOC. 058 is sota. 059 is API. 060 is perf. **THIS review consumes 057's foundation enumeration verbatim** and adds the four ZK-only T1 layers (T1-NTT/T1-MSM/T1-POLY/T1-MERKLE/T1-TRANSCRIPT/T1-RS/T1-POSEIDON) that 057 doesn't cover, exactly mirroring 147's Tier-1 list — but THIS review is the first to name the *cross-package composition consequence*, namely that without 057's T1-* and 147's T1-* both landing, nothing in the topic prompt composes against the existing surface.
- **146-150 (zkmark isolation).** 146 numerics finds the package is a 262-LOC interface stub. 147-zkmark-missing enumerates 11 topic-prompt primitives across 3 tiers + 8 ZK-only T1 layers (~7,750 LOC total). 148 is sota. 149 is API. 150 is perf. **THIS review's Z14-Z17 directly cite 147's T2-KZG / T2-FRI / T2-IPA / T2-MLE-COMMIT enumerations** and adds nothing to them; the contribution of this review is the **Z18-Z22 ship-today envelope** (~470 LOC: SchnorrModP + PairingGate + KZGEnvelope + AlgorithmRegistry + RandomOracleAdapter) that the isolation reviews didn't name because they were each scoped to a single package.
- **155-synergy-crypto-prob.** Composes `crypto/`'s three deterministic PRNGs with `prob/`'s seven distributions across the leftover-hash-lemma + min-entropy + extractor + DP-noise axis. **Orthogonal axis** — 155 is the entropy-meter side of `crypto/`, this is the cryptographic-substrate side. Both reviews find `crypto/`'s package name promises more than the surface delivers, but 155's twelve primitives ship over the existing surface (entropy is 060 LOC of math-on-arrays); THIS review's twenty-two primitives mostly DON'T ship over existing surface.
- **174-synergy-gametheory-optim.** Same wave (16 prior synergies all consumer-side-placed). 174 lands 20 primitives at ~2,580 LOC of pure connective tissue with ONLY 1 new abstraction (40-LOC `OnlineLearner`) — the canonical `well-stocked-A × well-stocked-B → composition` shape. **THIS review is the canonical "stub × stub → dependency map" inversion**: the same 20-primitive count, but 17 of 22 are substrate, 5 of 22 are connective. Worth flagging as a **structural counter-example** to the 158-174 well-stocked pattern — Block-B is not always a composition catalog; sometimes (when both packages are deliberately staged), it is a dependency-order reveal.
- **151-signal-prob, 154-chaos-timeseries, 156-topology-prob, 157-graph-linalg, 158-color-signal, 159-em-signal, 160-fluids-signal, 161-control-prob, 162-graph-prob, 163-optim-autodiff, 164-orbital-optim, 165-sequence-prob, 166-acoustics-signal, 167-audio-signal, 169-prob-optim, 170-info-prob, 171-graph-topology, 172-changepoint-timeseries, 173-queue-prob.** All sixteen are `well-stocked-A × well-stocked-B` compositions with 12-20 primitives shipping over existing surface. **None gates on missing substrate the way THIS review does.** Cross-link only: the consumer-side-placement convention applies; the zero-cross-edges-today empirical fact applies; the cycle-free-DAG verification applies. The structural shape (substrate-heavy) is the differentiator.

**Why this synergy is not yet reviewed:** unlike the 16 prior synergies where both packages had stable existing surface, both `zkmark/` and `crypto/` are explicit pre-foundation stubs (zkmark's "Tranche-1 substrate-only ship" + crypto's "number theory + non-crypto hash + PRNGs only" footprint). The synergy review is unavoidable in the 400-agent sweep because the topic-prompt items (Groth16/PlonK/KZG/etc.) are real and the cross-package boundary is real. But the review reveals the dependency map, not a composition catalog. Bottom line: this is the **structural-counter-example synergy** of the 400-sweep — five primitives ship today (Z18-Z22, ~470 LOC), seventeen primitives gate on substrate owned by 057-T1-* and 147-T1-* totalling ~7,500 LOC across ~5 engineer-months. The honest one-day shippable artifact today is **Z18 SchnorrLikeOverModP + Z21 AlgorithmRegistry** (~230 LOC including bigint shim) which provides a **third** zkmark Algorithm constant ("schnorr-modp") that is non-zero-knowledge but sound, ships golden-file cross-language vectors, and demonstrates the cross-edge `zkmark/ → crypto/` works at all — the smallest possible cross-edge against v0.10.0.

## Two-line summary (repeat)

`zkmark/` is a 262-LOC interface stub with literally zero crypto dependencies and `crypto/` is an 880-LOC number-theory + non-cryptographic-hash + deterministic-PRNG library (zero SHA, zero EC, zero pairings, zero math/big across the entire repo) — every topic-prompt item (Groth16/PlonK/Marlin/KZG/IPA/FRI/hash-to-curve/Pedersen/Schnorr-on-curve/BLS-aggregate/Fiat-Shamir/Poseidon/sumcheck/HyperPlonk/Brakedown/WHIR/lattice-PCS) gates on 17 substrate primitives Z1-Z17 totalling ~7,500 LOC owned by 057-T1-* and 147-T1-*, and only 5 connective primitives Z18-Z22 totalling ~470 LOC ship today against v0.10.0 (Z18 SchnorrLikeOverModP via crypto.ModPow+ModInverse+IsPrime as a third zkmark Algorithm constant, Z19 PairingProductCheck interface, Z20 KZG verify-envelope vs mock pairing, Z21 12-new-Algorithm-constants + 12 forward-compat sentinels, Z22 RandomOracleHashAdapter wrapping the existing FNV/Murmur as `Unsound: true` shims that cleanly upgrade once SHA-256 lands). The structural finding is that **this is the canonical "stub × stub → dependency map" inversion** of the 16 prior `well-stocked-A × well-stocked-B → composition catalog` synergies (158-174); the 1.5-engineer-day Z18+Z21+Z22 PR is the cheapest honest cross-edge `zkmark/ → crypto/` and is the shippable artifact, gating no substrate of its own and clean-aligned with the existing zkmark forward-compatibility envelope contract.

## Progress

- 2026-05-08 agent-175 synergy-zkmark-crypto: structural-counter-example synergy reveals that unlike the 16 prior consecutive consumer-side-placed Block-B synergies (158-174) where both packages are well-stocked and 12-20 primitives ship as pure composition over existing surface, THIS pair is stub-on-stub: zkmark/ is 262 LOC of zero-crypto-deps Tranche-1 envelope and crypto/ is 880 LOC of number-theory + non-cryptographic-hash + deterministic-PRNGs (zero SHA, zero EC, zero pairings, zero math/big in the entire repo); only 5 of 22 enumerated primitives (Z18-Z22 ~470 LOC) ship today against v0.10.0 — Z18 SchnorrLikeOverModP via crypto.ModPow+ModInverse+IsPrime+Xoshiro256 (~150 LOC + ~80 LOC bigint shim) as a third zkmark Algorithm constant ("schnorr-modp", non-zero-knowledge but sound; flagged per 147 §recommendation 1 deterministic-vs-ZK mismatch), Z19 PairingProductCheck interface + ErrPairingNotYetWired sentinel forward-binding the eventual Z7 pairing, Z20 KZG verify envelope-vs-mock-pairing (algorithm logic golden-pinned before Z14 substrate lands), Z21 ZkmarkAlgorithmRegistry adding 12 new Algorithm* constants (Groth16/Groth16BN254/KZGOpening/Plonk/HaloIPA/FRIGoldilocks/StarkGoldilocks/HyperPlonk/BLSAggregate/PedersenCommit/HashToCurve/SchnorrModP) + 12 matching not-yet-wired sentinels (forward-compat per 147 §rec3 add-siblings-don't-rename), Z22 RandomOracleHashAdapter wrapping existing FNV1a64+MurmurHash3_32 as Unsound:true shims that upgrade cleanly once Z2 SHA-256 lands; remaining 17 primitives Z1-Z17 (~7,500 LOC) are substrate already owned by 057 (T1-BIGINT/T1-FIELD/T1-FIELDTOWER/T1-EC/T1-PAIRING/T1-HASH) and 147 (T1-NTT/T1-MSM/T1-POLY/T1-MERKLE/T1-TRANSCRIPT/T1-POSEIDON/T1-RS) plus T2-KZG/T2-IPA/T2-FRI/T2-MLE-COMMIT — every topic-prompt item gates on these (BLS12-381/BN254/BLS12-377+pairings -> Z3+Z4+Z5+Z7, Groth16/PlonK/Marlin -> Z14, KZG -> Z14, IPA -> Z15, FRI/STARK -> Z16, Merkle -> Z2+Z10, hash-to-curve Wahby-Boneh -> Z2+Z5, Pedersen -> Z5, classical Schnorr -> already Z18 ships, BLS signatures -> Z5+Z7, Fiat-Shamir -> Z2+Z11, Poseidon/Rescue/Anemoi/Tip5 -> Z3+Z12, lookups Plookup/Caulk/cq -> Z14+Z9, sumcheck -> Z3+Z9 multilinear+Z11, HyperPlonk -> Z17, Brakedown/Orion/WHIR/Ligero+ -> Z3+Z13+Z10, lattice -> separate ring-arithmetic substrate); 8-PR sequence PR-1 ship-today envelope ~470 LOC 1.5 days saturating R-MUTUAL-CROSS-VALIDATION 3/3 on Schnorr-mod-p (Go ModPow x Python pow x C# BigInteger.ModPow byte-identical golden vectors), PR-2 substrate Z2+Z3+Z8+Z9+Z10+Z11+Z13 ~3500 LOC 4-6 weeks (gated on 057-T1-BIGINT), PR-3 FRI anchor Z16+Z12 ~600 LOC 1 week (closes 147-§6 cheapest-real-Tranche-2 recommendation, no curve no pairing required), PR-4 curve substrate Z4+Z5+Z6+Z7 ~2000 LOC 1 month, PR-5 KZG+Pedersen+hash-to-curve ~700 LOC 1-2 weeks, PR-6 Groth16+BLS-aggregate ~800 LOC 1-2 weeks (R-MUTUAL pin against arkworks-rs+gnark on canonical SHA256 preimage circuit), PR-7 Plonk+IPA-Pasta+Halo2 ~2500 LOC 4-6 weeks (closes zkmark.go S62-S63 Tranche-2 commitment), PR-8+ multilinear Z17+HyperPlonk+Brakedown/Orion/WHIR ~2500 LOC; total PR1-7 ~10500 LOC source ~10500 LOC tests ~3-4 engineer-months matching 147 §"~7750 LOC of math + equal test surface across ~8 sprints" estimate; precision hazards documented Schnorr-mod-p NOT zero-knowledge in reality per deterministic-by-construction posture (must flag in package docs same caveat 147 §rec1 makes), modular reduction %q bias 2^-64 per chunk for 256-bit q standard rejection-sampling ~q/2^64 rejection probability, crypto.ModPow uint64-only and crypto.IsPrime uint64-only Z18 needs ~80 LOC bigint shim (modular_big.go) FIRST math/big import in repo correctly per 057-T1-BIGINT, Z22 Murmur+FNV NOT cryptographic Unsound:true field mandatory but byte-identical for golden-file fixture purposes, Z19 MockPairingProductCheck{Should:true} TEST_ONLY must fail-closed at integration time when real BLS12381Pairing lands, Z20 KZG SRS must be loaded from caller-supplied byte-stream caller-verified to come from public ceremony (Powers of Tau / Filecoin / Zcash / Mainnet KZG) trusted-setup-generation OUT per 147 §"OUT" boundary, Z21 algorithm-constant churn risk pin all 14 strings in proof_format.json golden fixture per CLAUDE.md §1, crypto.SituationHashWithStructure structurally-tempting but linear-collision-resistance MUST be wrapped via Z22 with Unsound:true belt-and-braces, crypto/rng.go three PRNGs all NOT cryptographic linear-state-recovery Schnorr-mod-p Z18 broken if seed entropy < q-bits or state observed (recommend future crypto.NewDRBG sponge-DRBG before any production-facing consumer), zkmark.Proof.CorpusSHA[32]byte caller-supplied no SHA in crypto/ today recommend crypto.SHA256 as Z2 prerequisite for audit-trail single-call verification cross-link to 057-T1-HASH and 146 §G7; architectural placement crypto/bigint.go+sha256.go+keccak.go+blake.go+hmac.go+modular_big.go+drbg.go (Z1+Z2+Z18-prereq+future-DRBG) and zkmark/{schnorr_modp.go (Z18) + pairing_gate.go (Z19) + kzg_envelope.go (Z20) + oracle_adapter.go (Z22)} alongside existing zkmark.go + Z21-extended-constants, then sub-packages zkmark/{field,ec,poly,merkle,transcript,hash,commit/{kzg,ipa,fri,multilinear},proof/{groth16,plonk,halo2,stark}} mirroring 147 §rec2 substrate-internal-to-zkmark NOT new sibling packages competing with crypto/; cycle-free DAG zkmark/ -> crypto/ one-way-never-reverses verified by enumeration mirroring 158-174 sixteen-consecutive-synergy consumer-side-placement convention; this synergy is FIRST zkmark x crypto in 400-sequence and structural-counter-example to 158-174 well-stocked-A x well-stocked-B composition-catalog pattern (174-gametheory-optim 20 primitives at 2580 LOC pure-connective vs THIS 22 primitives at 7950 LOC where 95% is missing-substrate 5% is connective) — distinct from 056-060 (crypto isolation 057 enumerates 5-layer foundation THIS review consumes verbatim and adds the cross-package composition consequence neither was scoped to name), 146-150 (zkmark isolation 147 enumerates 11 topic primitives across 3 tiers + 8 ZK-only T1 layers THIS review's Z14-Z17 cite 147 verbatim and adds the Z18-Z22 ship-today envelope 147 didn't name as it was scoped to single package), 155-synergy-crypto-prob (orthogonal axis composes crypto PRNGs x prob distributions across leftover-hash-lemma+min-entropy+extractor+DP-noise 12 primitives ship over existing surface vs THIS 17/22 don't ship over existing surface — both find crypto/ promises more than surface delivers but at different axes), 174-synergy-gametheory-optim (same wave; canonical well-stocked composition pattern 1 new abstraction 19 compositions of existing surface vs THIS canonical stub-on-stub dependency-map structural-counter-example); cross-edges zkmark/ -> crypto/ (Z18 ModPow+ModInverse+IsPrime+NextPrime+Xoshiro256, Z22 wraps FNV1a64+MurmurHash3_32, future Z18-prereq math/big shim) zero edges before PR-1 cycle-free verified. Report at agents/175-synergy-zkmark-crypto.md, ~390 lines.
