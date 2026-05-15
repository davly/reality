# 324 — dive-msm (Pippenger / Bos-Coster / GLV / GLS / Straus / batch-normalize MSM audit)

## Headline
Reality v0.10.0 ships ZERO MSM surface and zero EC point arithmetic to compose against (verified by repo-wide grep on `Pippenger|MultiScalar|ScalarMul|wNAF|BosCoster|GLV|GLS|Straus|bucket.method|endomorphism` across all 22 packages — no callable hits in any `.go` file outside coincidental three-letter matches in `autodiff/doc.go`, `prob/copula/{archimedean,t}.go`, `orbital/orbital.go` for "EC" / "elliptical orbit"); since slot 292 owns the EC substrate keystone (BLS12-381 G_1/G_2 + Fp/Fp²/Fp⁶/Fp¹² tower) and slot 214 owns the pairing layer, slot 324 owns the **scalar-multiplication-optimization layer** sitting between them — the `ec/scalar/` family of `wNAF`, `Straus`, `Pippenger`, `GLV-split`, `GLS-split`, `Montgomery-batch-inverse` — and the singular highest-leverage primitive is **T2 Pippenger bucket-method MSM (~250 LOC)** because it is what every production ZK prover (Groth16, Plonk, Marlin, KZG, BLS-aggregate, batch-ECDSA-verify) bottlenecks on (~70-90% of prover time at N=2²⁰), giving 30-100× speedup over naive at zkmark scale; the singular cheapest day-1 PR is **T0 wNAF single-scalar + T1 Straus multi-scalar (~270 LOC)** which composes slot 292's `EllipticCurve.Add/Double` and gives ~5× speedup at small batch sizes while Pippenger is being landed.

## Findings

### State at HEAD (v0.10.0, 2026-05-09)

Repo-wide grep on `Pippenger|MultiScalar|MultiScalarMul|ScalarMul|wNAF|window.NAF|Bos.Coster|Straus|GLV|GLS|endomorphism|bucket.method` against `*.go`: **zero callable matches** in any source file; only review-document hits. Repo-wide grep on `elliptic|EC.point|BLS12|secp256k1|jacobian|projective.coord|curve25519|edwards`: only 4 files (`autodiff/doc.go`, `prob/copula/{archimedean,t}.go`, `orbital/orbital.go`) and all are coincidental ("EC" as a variable in copula log-likelihood, "elliptical orbit" docstring in orbital). **No EC point group exists; no scalar multiplication exists; no MSM exists.** This means slot 324 cannot ship today — the substrate (slot 292 T1-EC `EllipticCurve` + `Point.Add/Double/Negate` + `Fp.Mul/Inv`) must land first.

| Surface | Path | Status | Slot-324 relevance |
|---|---|---|---|
| `EllipticCurve` / `Point.Add` / `Point.Double` | absent (slot 292 T0-T1) | — | Hard prereq — every primitive in this slot is a wrapper over Add/Double |
| `Fp` field with constant-time `Mul / Square / Inverse` | absent (slot 292 T1-FIELDTOWER) | — | Hard prereq for batch-Montgomery-inverse + GLV-lattice-reduce |
| Big-integer arithmetic (`math/big` or shim) | absent (slot 292 T1-BIGINT) | — | BLS12-381 r is 255 bits, scalar lattice is 2-D over Z, GLV needs short-vector solve |
| `crypto/scalar.go` window-NAF / fixed-base comb | absent | — | T0 single-scalar mult, the substrate this slot adds |
| `crypto/msm.go` Pippenger bucket method | absent | — | T2 keystone of this slot |
| `crypto/glv.go` BLS12-381 cube-root-of-unity endomorphism scalar split | absent | — | T3 GLV — depends on slot 292 T6 `Frobenius` + `λ-eigenvalue` |
| `crypto/batch_invert.go` Montgomery's batch-inversion trick | absent | — | T5 — saves N-1 of N inversions in any affine-coord batch |

### Cross-slot orientation

| Adjacent slot | Overlap | Resolution |
|---|---|---|
| **292-new-elliptic-curves T1-EC + T1-FIELDTOWER** | Owns `Point.Add/Double/Negate` and `Fp/Fp²/Fp⁶/Fp¹²`. Without these, slot 324 has nothing to wrap. | Slot 324 is **strictly downstream** of slot 292 T0-T1. Cannot land independently. |
| **292-new-elliptic-curves T6 `frobenius.go`** | Owns `[β]P` cubic-root-of-unity endomorphism + `λ`-eigenvalue computation. | Slot 324 T3 GLV consumes 292-T6 verbatim. **GLV scalar-split logic (lattice reduction over Z[λ])** lives in 324; the **endomorphism map evaluation** lives in 292. |
| **214-new-pairings P10 optimal-Ate Miller-loop** | Miller-loop inner operation = scalar multiplication by Miller-scalar `|x| = 0xd201000000010000` for BLS12-381. | Slot 214's Miller-loop scalar is *fixed* (Hamming weight 6, NAF representation hardcoded), so it does NOT need general wNAF — but it DOES benefit from line-function evaluation hoisting that this slot's Pippenger loop discipline informs. **Independent but sibling**. |
| **175-synergy-zkmark-crypto Z6 = T1-MSM** | Already names "Multi-scalar-multiplication via Pippenger-bucket method; window size c ≈ log₂(n) − 2; GLV-endomorphism scalar split halves bit length on BLS12-381" at ~400 LOC, owned by 147-T1-MSM. | Slot 324 is the **deep-dive SOTA review** for what 175-Z6 / 147-T1-MSM enumerates. This slot is the canonical reference; 175/147 are the ship-line-items. |
| **150-zkmark-perf** | Pins the Pippenger window-size choice (c ≈ log₂(N) − 2) and per-op cycle budget for prover. | Slot 324 owns the algorithmic choices; slot 150 owns the cycle budget. **Co-author the `BENCHMARK_BUDGET.md` for MSM**. |
| **200-synergy-zkmark-info** | Names MSM as the Groth16/Plonk prover bottleneck (~70-90% of prover time at N=2²⁰). | Slot 324 is the SOTA references for 200's claim. |
| **307-dive-discrete-log** | BSGS / Pollard-rho / kangaroo all use repeated EC scalar mult — they would consume slot 324 T0 wNAF for inner-loop walks. | Independent: discrete-log has no `MSM`; it has *iterated single-scalar-mult* with state. |
| **323-dive-hash-to-curve** | RFC 9380 hash-to-curve emits a point P; subsequent operations (BLS-sign = sk·P, Verify = e(sig, g_2) == e(H(m), pk)) call scalar mult / pairing. | Independent but sibling: 323 emits the input to scalar-mul; 324 is the scalar-mul itself. |

### Distinct-from-292 disambiguation

Slot 292 (EC) owns: curve definition, Fp tower, point group law (Weierstrass / Edwards / Montgomery), Schoof / SEA / Velu / Miller / Tate / Weil / optimal-Ate, BLS12-381 / BN254 / secp256k1 parameter tables, j-invariant, division polynomials, isogenies.

Slot 324 (MSM) owns: **scalar-multiplication-optimization layer atop point group law**. wNAF representation, sliding-window scalar mult, fixed-base comb, Straus simultaneous mult, Yao's algorithm, Pippenger bucket method, Bos-Coster heap, GLV/GLS scalar decomposition (lattice reduction over Z[λ]), Montgomery's batch-inversion trick, batch-affine-add (Aranha-Faz-Hernández-López-Rodríguez 2013), constant-time vs variable-time variants.

Boundary: **`Point.Add(P, Q)`** is slot-292; **`MSM([k_1,…,k_n], [P_1,…,P_n])`** is slot-324. **`endomorphism map β: P ↦ (β·x_P, y_P)`** is slot-292 T6; **`GLV decompose: k ↦ (k_1, k_2) s.t. k = k_1 + k_2·λ mod r, ||k_i|| ≈ √r`** is slot-324. **`Miller-loop f_{r,P}(Q) inner double-and-add`** is slot-214 (it has fixed scalar shape; doesn't go through general MSM).

### Does math/big or stdlib cover this?

`crypto/elliptic` (Go stdlib) has unexported wNAF for P-256 specifically but no MSM, no GLV, no batch-invert. `golang.org/x/crypto/bn256` has small-batch BN254 verify but not Pippenger. `gnark-crypto` (Apache-2) has full Pippenger + GLV + batch-affine-add but is Apache-2 / Go-only / no cross-language golden. `blst` (Apache-2 + ASM) has hand-tuned Pippenger but ASM-coupled / not portable / single curve. `arkworks-rs` (MIT/Apache-2 dual) Rust-only. **There is no MIT pure-Go zero-dep deterministic-golden-file-validated MSM kernel in any library ecosystem.** Reality's positioning is identical to slot 292: the **only MIT pure-Go zero-dep cross-language deterministic-golden-file-validated MSM toolkit**. Narrow but real moat.

### SOTA references (web-search corroborated)

- **Pippenger 1976** "On the evaluation of powers and related problems" FOCS — original bucket method, asymptotically optimal.
- **Bos-Coster 1989** "Addition chain heuristics" — heap-based addition chains; faster than Pippenger at very small N (≤ 16) only; production libraries skip it.
- **Gallant-Lambert-Vanstone 2001** "Faster point multiplication on elliptic curves with efficient endomorphisms" — 2-D lattice scalar split.
- **Galbraith-Lin-Scott 2009** "Endomorphisms for faster ECC on a large class of curves" — 4-D split via Frobenius on Fp²-curves, 4× speedup.
- **Aranha-Faz-Hernández-López-Rodríguez 2013** "Faster implementation of scalar multiplication on Koblitz curves" — modern Pippenger + batch-affine-add.
- **Bernstein-Lange 2007** Explicit Formulas Database (https://hyperelliptic.org/EFD) — canonical Add/Double formula source.
- **Faz-Hernández-López 2014** "High-performance ECC implementations" — wNAF + GLV combined recipe.
- **EdMSM (eprint 2022/1400)** "Multi-Scalar-Multiplication for SNARKs and faster" — 2× speedup over gnark-crypto via signed-bucket signed-digit + bucket-set encoding; integrated in gnark-crypto v0.9.
- **OPTIMSM (eprint 2024/1827)** FPGA hardware accelerator for ZK-MSM; informs the layout assumptions an upstream pure-Go implementation should respect to be HW-amenable.
- **PipeMSM (eprint 2022/999)** hardware MSM pipeline — HW reference but algorithmic insights apply to SW (signed buckets, parallel bucket reduction).
- **ZPrize 2022/2023** multi-million-dollar MSM acceleration competition; ConsenSys/gnark won MSM-on-Mobile with 40-47% speedup over state-of-art Rust impl. Established that **GPU-MSM = 8-30× over CPU-MSM** at N=2²⁶, but pure-Go zero-dep is CPU-only.
- **Bandersnatch (eprint 2021/1152)** — fast curve over BLS12-381 scalar field with efficient endomorphism for Halo2-recursion-MSM contexts.

### Why MSM is THE bottleneck for ZK

Every slot-200 / 175 / 150 zkmark consumer (Groth16, Plonk, Marlin, KZG, BLS-aggregate-verify, batch-ECDSA-verify) reduces to *some* MSM. Specifically:

- **Groth16 prover**: 3 MSMs over G_1 (sizes ≈ N, N, N for [A]_1, [B]_1, [C]_1) + 1 MSM over G_2 (size ≈ N) where N = circuit size; for zkmark target N=2²⁰ that's 4 × 10⁶ scalar mults. Naive: ~3 hours. Pippenger: ~3 minutes. **30-60× speedup** is the difference between zkmark v1 being shippable and being a benchmark embarrassment.
- **Plonk prover**: ~9 MSMs over G_1 of varying sizes (committed wires, permutation, quotient, KZG-batching). Same Pippenger speedup applies.
- **KZG.commit(f)**: single MSM `[Σ f_i τ^i]_1 = Σ f_i [τ^i]_1` of size deg(f); the canonical "MSM with fixed bases" (here `[τ^i]_1` is the SRS, fixed). Allows extra **Pippenger fixed-base precomputation** (Lim-Lee 1994 comb method) — ~2× over plain Pippenger.
- **BLS aggregate-verify**: `e(sig_agg, g_2) == Π e(H(m_i), pk_i)`; the right-hand side reduces to two MSMs (one in G_1, one in G_2) plus 1 pairing. **For batch sizes N≥1000, this is 100× over individual verify.** Eth2 consensus uses this on ~1M validators per slot.
- **Batch-ECDSA-verify**: random-linear-combination of N verification equations folds into 2 MSMs + check. ~100× speedup at N=1000.

### What "MSM" means precisely

Given scalars (a_1, …, a_n) ∈ Fr and points (P_1, …, P_n) ∈ G, compute Q = Σ a_i · P_i. Standard algorithms:

| Algorithm | Cost (b = bit-length, n = batch) | When to use |
|---|---|---|
| **Naive** | n separate (b/log b)-doublings + n·log b adds → O(n·b/log b) | n ≤ 2; baseline only |
| **Yao 1976** | O(n·b/log b) but better constants — small-window precompute per scalar | n ≤ 8 |
| **Straus 1964** (also called Shamir trick) | O((n+b)·b/log b) — simultaneous double-and-add over interleaved windows | 4 ≤ n ≤ 64 |
| **Pippenger 1976 (bucket)** | O(n·b / log(n·b)) — windows of c ≈ log₂(n)−2 bits, n·b/c additions, 2^c·(b/c) bucket reductions | n ≥ 32; **gold standard for n ≥ 1024** |
| **Bos-Coster 1989** | O(n·b/log n) heap-of-scalars; theoretically beats Pippenger but only practical at very small n with adversarial scalar distributions | rarely used in production |
| **+ GLV / GLS** | halves (or quarters) effective scalar bit-length b on amenable curves | composes with all above on BLS12-381 / secp256k1 / BN254 |
| **+ batch-affine-Montgomery-trick** | replaces N inversions in batch-add with 1 inversion + 3N mults (Montgomery 1987) | composes with Pippenger bucket reduction |

Pippenger window size c: optimal is **c ≈ log₂(n) − 2** to balance #additions (n·b/c) against #bucket-reductions (2^c·b/c). For n = 2²⁰ → c ≈ 18. For n = 2²⁵ → c ≈ 23. This is the well-known "Pippenger sweet spot" — every production library tunes c per architecture.

### Constant-time vs variable-time

- **Public-input MSM (verifier side, KZG.verify, batch-verify)**: variable-time is fine; window-NAF is faster than constant-time ladder.
- **Secret-input MSM (signer side, BLS-sign, key-derivation)**: must be constant-time; use fixed-base comb / Montgomery-ladder fall-back; Pippenger naturally has scalar-dependent bucket access patterns and is **NOT constant-time** without padding.
- **For zkmark prover**: scalars come from *witness*, which is application-secret but circuit-known; standard practice is variable-time MSM with full witness-secrecy on transcript level (Fiat-Shamir hides per-mul timing). Reality should ship **both variants** with a public flag `MSM(scalars, points, ConstantTime: bool)`.

## Concrete recommendations

Tier numbering: T0 = single-scalar wNAF; T1 = Straus simultaneous; T2 = Pippenger keystone; T3 = GLV; T4 = GLS; T5 = batch-Montgomery-invert; T6 = fixed-base comb; T7 = signed-bucket EdMSM upgrade.

### T0 — `crypto/ec/scalar.go` ~150 LOC — DAY-1 substrate (depends on slot 292 T0-T1)

```go
// WNAF returns the width-w non-adjacent form of k as a slice of int8 in {-2^(w-1)+1, ..., 2^(w-1)-1, 0},
// least-significant-digit first. For w=4 the digit set is {-7,-5,-3,-1,1,3,5,7,0}; expected density of
// nonzero digits is ~1/(w+1) (Solinas 2000). Used as inner kernel of variable-time scalar mult.
func WNAF(k *big.Int, w int) []int8 { ... }

// ScalarMul computes [k]P via width-w wNAF; precomputes {±P, ±3P, ..., ±(2^(w-1)-1)P}, then sweeps wNAF
// digits MSB-first with a Double-or-DoubleAdd ladder. Variable-time in k.
func ScalarMul(P Point, k *big.Int, w int) Point { ... }
```

Day-1 PR composing slot 292 T0/T1; gives `ScalarMul` that the rest of the slot wraps. Pin: `WNAF(k, 4)` for `k = 0x123` matches Solinas-2000 Algorithm 4 example.

### T1 — `crypto/ec/straus.go` ~120 LOC — small-batch MSM

```go
// Straus computes Σ k_i · P_i via simultaneous interleaved-window double-and-add.
// For w-bit windows and n points, precomputes 2^(n·w) lookup table — ONLY tractable for small n
// (n ≤ 8 typical). For larger n, Pippenger dominates.
func Straus(scalars []*big.Int, points []Point, w int) Point { ... }
```

Pre-Pippenger ship; gives ~5× over naive at n ≤ 32 (typical batch-verify regime). Pin: random-vector regression vs T0 `ScalarMul`-then-add.

### T2 — `crypto/ec/pippenger.go` ~250 LOC — KEYSTONE (depends on T0)

```go
// Pippenger bucket-method MSM. Window size c is auto-chosen as max(2, floor(log2(n)) - 2).
// Algorithm: split each b-bit scalar into ceil(b/c) windows of c bits each; for window j:
//   (1) bucket each point into B[w_ij] for nonzero window value w_ij ∈ [1, 2^(c-1)] (signed)
//   (2) running-sum bucket reduction: S = B[1] + 2·B[2] + ... + (2^(c-1))·B[2^(c-1)]
//                                       computed via S = T_1 + T_2 + ... + T_{2^(c-1)}
//                                       where T_k = B[k] + B[k+1] + ... + B[2^(c-1)]
//   (3) accumulate Σ_j 2^(j·c) · S_j into final result via outer Horner double-and-add
// This is the canonical algorithm; gives O(n·b / log(n·b)) — the asymptotic optimum.
//
// For BLS12-381 G_1 with N = 2^20 scalars, this is ~30-60× faster than naive ScalarMul-then-Add.
// Workspace `PippengerScratch` per CLAUDE.md §3 (no allocations in hot path).
func Pippenger(scalars []*big.Int, points []Point, scratch *PippengerScratch) Point { ... }
```

The single most important ship in this slot. Without it, zkmark prover is unshippable. Ship with `signed-bucket` (negate half the buckets to halve the table) — modern best practice (EdMSM 2022 §3).

### T3 — `crypto/ec/glv.go` ~150 LOC — 2× speedup on BLS12-381 G_1, secp256k1, BN254 G_1

```go
// GLVDecompose splits scalar k ∈ Fr into (k_1, k_2) such that k = k_1 + k_2·λ (mod r) with
// ||k_i||_∞ ≤ ~√r ≈ 2^(b/2). Uses 2-D Babai-rounding lattice reduction with precomputed
// short-vector basis (a1, b1), (a2, b2) for the curve.
//
// For BLS12-381 G_1: λ = -(z² - 1) where z = -0xd201000000010000; β = cube-root-of-unity in Fp.
// Endomorphism is φ(P) = (β·x_P, y_P) and satisfies φ²(P) + φ(P) + P = 0.
//
// Caller then computes [k]P as MSM([k_1, k_2], [P, φ(P)]) — bit-length halved → ~2× speedup.
func GLVDecompose(k *big.Int, params GLVParams) (k1, k2 *big.Int) { ... }
```

Composes with T2 Pippenger: prover does `Pippenger([k_i], [P_i])` of size n; or **`Pippenger([k_{1,i}, k_{2,i}], [P_i, φ(P_i)])`** of size 2n with half-length scalars — net ~1.5-2× speedup. Production zkmark uses this universally.

### T4 — `crypto/ec/gls.go` ~200 LOC — 4× speedup on BLS12-381 G_2, BN254 G_2 (Fp²-curves)

```go
// GLSDecompose 4-way split for G_2 over Fp²: combines GLV (cube-root β) with Frobenius π
// (the p-power Frobenius restricted to G_2). Decomposes k into (k_1, k_2, k_3, k_4) with
// ||k_i||_∞ ≤ ~r^(1/4); MSM with bases [P, π(P), φ(P), φπ(P)].
//
// Galbraith-Lin-Scott 2009 §4. For BLS12-381 G_2 this is the production technique;
// gnark-crypto and blst both implement it.
func GLSDecompose(k *big.Int, params GLSParams) (k1, k2, k3, k4 *big.Int) { ... }
```

Less critical than T3 because (a) Groth16 G_2 MSM is only 1 of 4 MSMs, (b) implementation is fiddly (4-D lattice reduction). Defer until after T2+T3 are validated.

### T5 — `crypto/ec/batch_invert.go` ~80 LOC — Montgomery's trick

```go
// BatchInverse computes [a_1^{-1}, ..., a_n^{-1}] using 1 inversion + 3(n-1) mults.
// Algorithm: forward pass z_i = a_1·...·a_i; one final Inv(z_n); backward pass extracts each a_i^{-1}.
// Saves ~50× vs n separate inversions (Fp.Inv is 100-300× slower than Fp.Mul).
func BatchInverse(a []Fp) []Fp { ... }
```

Composes with batch-affine-add inside Pippenger bucket reduction (Aranha-Faz-Hernández-López-Rodríguez 2013) — ~3× extra speedup on Pippenger. Independent ship; useful in many other places (KZG.batch-open uses it).

### T6 — `crypto/ec/comb.go` ~180 LOC — fixed-base precomputation (KZG SRS)

```go
// FixedBaseComb precomputes a (w, h)-Lim-Lee 1994 comb table for fixed base P; subsequent
// scalar mults [k]P via comb table take only ceil(b/(w·h)) doublings + ceil(b/h) mixed adds.
// For h=8, w=8: 64-entry table, scalar-mult cost is ~16% of plain wNAF.
func PrecomputeComb(P Point, w, h int) *CombTable { ... }
func (c *CombTable) ScalarMul(k *big.Int) Point { ... }
```

For KZG.commit: `[τ^i]_1` SRS is fixed across calls; precomputing combs at SRS-load time gives ~2-3× over Pippenger. Gnark-crypto MSMConfig / blst `precomputed` API. Ship after T2 stabilizes.

### T7 — `crypto/ec/edmsm.go` ~120 LOC upgrade to T2 — EdMSM signed-bucket optimization (2022 frontier)

EdMSM (eprint 2022/1400) reduces bucket count from 2^(c-1) to ~2^(c-1)/2 via *signed-bucket-set encoding* — basically a NAF-like trick at the bucket level. Empirical 1.5-2× over plain Pippenger. Defer to v0.11+ after T2 lands; this is a "polish" tier.

### Day-1 cheapest PR

**T0 wNAF + T1 Straus (~270 LOC)** — gates only on slot 292 T0+T1 (`Point.Add/Double` + `Fp` field). Gives small-batch (n ≤ 32) `MSM` that's ~5× over naive — useful for batch-verify even before Pippenger lands. Ship as `crypto/ec/scalar.go` + `crypto/ec/straus.go`. **Ship JOINTLY with slot 292 T0-T1** as a unified PR ("EC + small-batch MSM, ~870 LOC"); zkmark consumers wait one tranche for Pippenger.

**Subsequent PR**: T2 Pippenger + T5 batch-invert (~330 LOC) — production zkmark prover unblocked.

**Then**: T3 GLV (~150 LOC) — 2× extra prover throughput on BLS12-381.

**Then**: T6 fixed-base comb (~180 LOC) — KZG SRS optimization.

**Defer**: T4 GLS, T7 EdMSM until v0.11+.

### R-MUTUAL-CROSS-VALIDATION 3/3 pin opportunities

1. **Naive ≡ Straus ≡ Pippenger** on identical input. For n = 1024 random scalars × 1024 random G_1 points: all three implementations must produce **bit-exact** identical output (point in projective coords normalized to affine). Regression: any implementation drift caught immediately. 3/3 saturation: (a) naive computes Σ k_i·P_i via T0 ScalarMul; (b) Straus T1; (c) Pippenger T2.
2. **GLV(k)·P ≡ k·P** verbatim. For 100 random scalars k ∈ Fr: `MSM([k_1, k_2], [P, φ(P)])` (using T3 GLV split) ≡ `ScalarMul(P, k)` (using T0 wNAF) bit-exact. Catches GLV lattice-reduction sign / eigenvalue / endomorphism-formula bugs (the historical bug class — Sage/Pari testbench is gold).
3. **Pippenger throughput scales as O(n·b / log(n·b))**. Benchmark regression at n ∈ {2¹⁶, 2¹⁸, 2²⁰}: time(n=2²⁰) / time(n=2¹⁶) ≤ 18 (theoretical 16 × 1.05 for log overhead). Catches O(n²) regressions where someone accidentally drops the bucket-reduction trick.

Bonus pin: **`Pippenger(rand_scalars, fixed_basis) ≡ KZG.commit(rand_poly)`** — the same MSM viewed as `commit` vs explicit MSM; cross-validates KZG implementation.

### Cross-link to consumers

- **zkmark Groth16 prover (slot 200, 175, 150, 147)**: 4 MSMs at N=2²⁰ → THE bottleneck. **Slot 324 T2 Pippenger is the single largest perf win in zkmark v1.** Coordinate ship-order: slot 292 T1-EC → slot 324 T0+T1+T2 → 175-Z14 KZG → 147 Groth16/Plonk.
- **BLS aggregate signatures (slot 214 P15-P18)**: aggregate-verify reduces to 2 MSMs over G_1/G_2. Eth2 consensus needs N=10⁶ batch — Pippenger gives 100× over per-sig verify.
- **KZG polynomial commitments (slot 175 Z14, 214 P25-P28)**: `commit(f) = MSM(coefs, SRS)`; with T6 fixed-base comb + T2 Pippenger, ~5× over plain Pippenger. KZG.batch-open uses T5 batch-invert.
- **batch-ECDSA-verify (slot 057 sig protocols)**: random-linear-combination + Pippenger gives 100× speedup at N=1000; relevant for blockchain client throughput.
- **Bulletproofs / IPA (slot 175 Z15)**: commit `c = Σ f_i · G_i` is the same MSM kernel; Pippenger applies as-is.
- **discrete-log attacks (slot 307 BSGS / Pollard rho)**: kangaroo/rho walks use single-scalar T0 wNAF inner loop, not MSM proper; orthogonal but shares the `ScalarMul` substrate.

### Risk / pitfalls

- **GLV lattice short-vector basis is curve-specific**; precomputing `(a_1, b_1, a_2, b_2)` correctly is the historical bug class (sign errors → 50% wrong outputs that pass small-N tests but fail at N=10⁴). Pin against gnark-crypto / blst golden vectors.
- **Pippenger window size c is empirical** — theoretical c = log₂(n) − 2 is wrong by ±1 on real CPUs due to cache; ship a config knob `MSMConfig.WindowSize int` (auto by default) and benchmark-tune.
- **Constant-time MSM is non-trivial** (bucket access pattern depends on scalars). Default to variable-time; offer constant-time variant only for signing paths.
- **Bos-Coster only beats Pippenger for n ≤ 16** with adversarial scalar distributions — DO NOT SHIP unless someone presents a concrete consumer; it adds maintenance burden for negligible win.
- **Subgroup membership tests must run BEFORE MSM** — accepting points in wrong subgroup leaks discrete-log info / breaks soundness. This is slot 292 P4-P5 territory but slot 324 should document the precondition in `Pippenger` doc-comment.

## Sources

### Repo files (all absolute, all confirmed at HEAD)
- `C:/limitless/foundation/reality/CLAUDE.md` — package list (no `crypto/ec/`, no `pairing/`, no `ec/`).
- `C:/limitless/foundation/reality/reviews/overnight-400/MASTER_PLAN.md:344` — slot 324 line.
- `C:/limitless/foundation/reality/reviews/overnight-400/agents/292-new-elliptic-curves.md` — EC keystone (T0 EllipticCurve, T1 Fp, T6 Frobenius hard prereq for slot 324 T3 GLV).
- `C:/limitless/foundation/reality/reviews/overnight-400/agents/214-new-pairings.md` — pairing layer; P10 optimal-Ate Miller-loop is sibling, not consumer of MSM.
- `C:/limitless/foundation/reality/reviews/overnight-400/agents/175-synergy-zkmark-crypto.md:81` — Z6 = T1-MSM 400 LOC line item.
- `C:/limitless/foundation/reality/reviews/overnight-400/agents/150-zkmark-perf.md` — Pippenger window-size budget pin (c ≈ log₂(n) − 2).
- `C:/limitless/foundation/reality/reviews/overnight-400/agents/200-synergy-zkmark-info.md` — MSM as ZK prover bottleneck.
- `C:/limitless/foundation/reality/reviews/overnight-400/agents/323-dive-hash-to-curve.md` — sibling slot.
- `C:/limitless/foundation/reality/reviews/overnight-400/agents/307-dive-discrete-log.md` — orthogonal consumer of T0 wNAF.
- Repo-wide grep `Pippenger|MultiScalar|ScalarMul|wNAF|BosCoster|GLV|GLS|Straus|bucket.method` over `*.go`: zero callable matches.
- Repo-wide grep `elliptic|EC.point|BLS12|secp256k1|jacobian|projective.coord` over `*.go`: 4 files all coincidental (autodiff/copula/orbital), zero EC surface.

### External sources (web-search corroborated 2026-05-09)
- Pippenger 1976, "On the evaluation of powers and related problems", FOCS — original bucket method.
- Bos-Coster 1989, "Addition chain heuristics" — heap-based addition chains.
- Yao 1976 / Straus 1964 — simultaneous-mult predecessors.
- Gallant-Lambert-Vanstone 2001, "Faster point multiplication on elliptic curves with efficient endomorphisms".
- Galbraith-Lin-Scott 2009, "Endomorphisms for faster ECC on a large class of curves".
- Lim-Lee 1994, "More flexible exponentiation with precomputation" — fixed-base comb.
- Solinas 2000, "Efficient arithmetic on Koblitz curves" — wNAF analysis.
- Aranha-Faz-Hernández-López-Rodríguez 2013, "Faster implementation of scalar multiplication on Koblitz curves".
- Bernstein-Lange Explicit Formulas Database — https://hyperelliptic.org/EFD .
- EdMSM (eprint 2022/1400, Botrel-El Housni 2022) — signed-bucket-set MSM, 2× over plain Pippenger.
- PipeMSM (eprint 2022/999) — HW-pipeline MSM.
- OPTIMSM (eprint 2024/1827) — FPGA MSM accelerator.
- ZPrize (https://www.zprize.io) — 2022/2023 multi-million-dollar MSM acceleration competition.
- gnark-crypto MSM (https://github.com/Consensys/gnark-crypto) — Apache-2 reference; production Go.
- blst (https://github.com/supranational/blst) — Apache-2 BLS12-381 ASM-tuned reference.
- arkworks-rs MSM — MIT/Apache-2 dual Rust reference.
- Bandersnatch (eprint 2021/1152) — fast curve over BLS12-381 scalar field with efficient endomorphism.
- zk-Bench (eprint 2023/1503) — comparative MSM/pairing benchmark across libraries.
- Galbraith-Paterson-Smart 2008, "Pairings for cryptographers" — taxonomy + endomorphism context.
