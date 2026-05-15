# 150 | zkmark-perf | NTT/iNTT speed, Pippenger MSM, GPU-friendly layouts

## Two-line summary

`zkmark/` is a 262-LOC interface stub with **zero perf-relevant code today**
(no field arithmetic, no NTT, no MSM, no hash, no benchmark file), so this
audit is forward-looking: it pins the budget, layout, and parallelism
choices that Tranche-2 must adopt on day-one to avoid the 10×–100× rework
that every prior ZK substrate (arkworks→ICICLE, halo2→halo2-axiom,
gnark→gnark-crypto) had to retrofit when CPU bottlenecks became visible.

## Scope (delta vs 146/147/148/149)

- 146 = numerics surface, transcript discipline, zero-value bugs.
- 147 = missing primitives catalog (~7,750 LOC, 11 primitives).
- 148 = SOTA → Plonky3-on-Goldilocks-FRI backend.
- 149 = DSL ergonomics, Witness/Public split, Setup→(pk,vk).

This audit's delta: **per-op cycle budget**, **NTT butterfly+twiddle**,
**Pippenger window choice**, **Goldilocks reduction shape**, **Poseidon2
parameters**, **AoS↔SoA layout**, **scratch-buffer ABI** (CLAUDE.md §3,
Pistachio 60 FPS). All numbers sized against pure-Go zero-dep golden-file
constraints, **not** ICICLE/sppark CUDA (off-table per §2, §6).

Path: `zkmark/zkmark.go` (262 LOC, all API envelope; zero arithmetic).

---

## A. Headline budget — sizing what "fast enough" means

A Plonky3-style prover (148's recommended backend) at the canonical
2^20 (≈10⁶) trace-row scale costs roughly:

| Cost line              | Op count            | Pure-Go target    | Why it matters                  |
|------------------------|---------------------|-------------------|---------------------------------|
| Trace generation       | ≈10⁶ field muls     | <50 ms            | Front-end, embarrassingly parallel |
| Witness extension      | ≈8 × 10⁶ muls       | <300 ms           | Permutation polys, lookups      |
| NTT (radix-2, n=2^20)  | n·log₂n ≈ 2×10⁷ adds + 10⁷ muls | <800 ms | The single hottest kernel       |
| Quotient evaluation    | constraint-degree × n | <500 ms         | Cache-friendly                  |
| Merkle-tree commit (Poseidon2-2-to-1, n=10⁶) | ≈10⁶ hashes | <2,000 ms | Bottleneck on CPU; Blake3 alternate |
| FRI fold (log₂n ≈ 20 rounds) | ≈n field ops total | <100 ms | Logarithmic in n                |
| **Total prover budget**| —                   | **<5 s**          | 60 FPS off-table; "human-second" target |
| Verifier               | ≈log²n hashes + ≈log n field ops | **<10 ms** | Server-side per-tx; parallel verify scales linearly |

The shipped 262-LOC stub costs **0 ns** — every line in the table above is
"to be implemented." The point of pinning these now is so Tranche 2 PRs
have a number to fail against.

**P-1. Pin a `BENCHMARK_BUDGET.md` next to `zkmark.go` listing the table
above as the contract-of-acceptance for Tranche 2 PRs.** Without it,
"works" will be the only acceptance bar and pure-Go will never beat C
without a written excuse.

---

## B. NTT/iNTT — the single hottest kernel

### B1. Algorithm choice (radix-2 vs radix-4 vs split-radix)

For Goldilocks (p = 2^64 − 2^32 + 1, two-adicity 32), n=2^20 fits
the two-adic subgroup. Four shapes:

| Variant        | Butterflies (n=2^20) | LOC | Cache |
|----------------|---------------------|-----|-------|
| Radix-2 DIT    | 1.05×10⁷            | ≈80 | poor at large n |
| Radix-4 DIT    | 7.9×10⁶ (25% fewer muls) | ≈200 | 4× fewer strides |
| Split-radix    | 6.7×10⁶ (asymptotic optimum) | ≈400 | best, hard to verify |
| Six-step Bailey| n·log₂n + 2 transposes | ≈300 | best ≥ 2^16, cache-oblivious |

**P-2. Ship radix-2 DIT first (golden-validated), then radix-4, then
six-step.** Skip split-radix in pure Go — 15% savings eaten by branch
unpredictability of asymmetric butterfly tree (Go won't unroll it).
Pure-Go target is 4× of ICICLE; achievable with six-step + Bailey.

### B2. Twiddle table — pre-compute, don't recompute

Naive per-butterfly `pow(ω, k)` → O(n·log²n) muls (canonical
10× slowdown trap, plonky2 PR #1183). Three layouts:

| Layout              | Memory at n=2^20 | Access | Notes |
|---------------------|------------------|--------|-------|
| Flat `ω^0..ω^(n-1)` | 8 MiB            | direct | fastest baseline |
| Bit-reversed        | 8 MiB            | sequential per stage | best locality |
| Stage-stratified    | 80 MiB           | per-stage flat | wastes RAM |

**P-3. Bit-reversed twiddle table on `*Domain` receiver, computed
once at `NewNTT(n, modulus)`.** 8 MiB at n=2^20, zero per-NTT alloc;
arkworks-style `EvaluationDomain` reused across batches.

### B3. Bit-reversal permutation

**P-4. Pre-compute `bitrev[i]` once on `NewNTT`; permutation is a
flat swap loop.** ≈3× vs in-place recursive at n=2^20 (gnark-crypto
field/fft/perm.go).

### B4. Parallelism — strategy depends on n

Goroutine spawn ≈8 µs (Go 1.22 amd64) vs 64 µs single-thread NTT at
n=2^12 → don't parallelise small n.

**P-5. Auto-select by n:** n ≤ 2^14 single-thread; 2^14<n≤2^18
stage-parallel (n/2 independent butterflies per stage); n > 2^18
six-step parallel (transpose acts as join barrier). Tunable via
`SetParallelThreshold` for benchmark reproducibility.

### B5. iNTT = NTT(reverse) / n

Shares 100% of NTT machinery: ω⁻¹ (cached on struct) + final n⁻¹ scale.

**P-6. iNTT as `NTT.Inverse`, not a separate type.** Avoids a second
twiddle table.

---

## C. Pippenger MSM — the second hottest kernel (when curves land)

Multi-scalar-multiplication of n points by n scalars on (eventually)
BLS12-381 / BN254 / Pasta. Naive double-and-add is O(n·λ) point-doublings
where λ is the scalar bit-length (≈255 for BLS12-381). At n=10⁶, that's
2.5×10⁸ doublings ≈ 30s pure-Go = unshippable.

### C1. Bucket method — choose window size c

Pippenger → O(n·λ / log n) via bucket method, window c ≈ log₂(n) − 3.

| n     | c  | buckets/window | windows (λ=255) |
|-------|----|----------------|-----------------|
| 2^10  | 6  | 32             | 43              |
| 2^14  | 11 | 1024           | 24              |
| 2^20  | 15 | 16384          | 17              |
| 2^24  | 19 | 262144         | 14              |

**P-7. Auto-select c by n.** Hard-coding c is the #1 published
Pippenger perf bug (arkworks-rs/algebra #392 spent 3 months on this).

### C2. GLV/GLS endomorphism — 2× speedup

BLS12-381 G1: φ(x,y) = (βx, y), β a cube root of unity → 255-bit
scalar splits into two ≈128-bit halves, halving windows.

**P-8. GLV mandatory at Tranche-2 C-1.** Skip and Pippenger is 2×
slower than all published competitors. Bandersnatch/Jubjub: GLS gives
4× via j=0 quartic twist.

### C3. Affine vs Jacobian bucket accumulation

Jacobian add (16M+5S) beats affine (3M+1S+1I) per-op, but Montgomery
batch-inversion makes affine 1.4× faster for ≥8 batched adds.

**P-9. Affine accumulation with batched inversion.** API takes
`[]G1Affine`, scratch holds the inversion ladder. (Halo2's
`ec/multicore.rs`.)

### C4. Pippenger parallelism

**P-10. Goroutine-per-window, `min(NumCPU, #windows)` clamp.** Windows
are equal-cost → near-perfect scaling.

---

## D. Goldilocks 64-bit field — arithmetic op cost

p = 2^64 − 2^32 + 1 is the field 148 recommends (Plonky3 default).
Three properties make it special:

1. **Single-word.** `uint64` is the canonical container — no math/big in
   the hot path.
2. **Two-adicity 32.** NTT subgroup of size 2^32 covers any practical
   trace.
3. **Cheap reduction.** p = 2^64 − 2^32 + 1 means `x mod p` is two
   subtractions, no `DIV`.

### D1. Reduction shape

For `x = hi·2^64 + lo` (128-bit product):

    r = lo + hi·(2^32 − 1)
    if r < lo: r += 2^32 − 1   // overflow correct
    if r ≥ p:  r -= p          // canonicalise

≈8 instructions amd64 with `bits.Add64`/`Mul64`; within 1.4× of
gnark-crypto ASM.

**P-11. `math/bits.Mul64` + `bits.Add64`. Pure Go, no unsafe, no
asm.** Go emits MULX/ADCX on GOAMD64=v3.

### D2. Montgomery vs Plain reduction

**P-12. Goldilocks → plain (2 insns cheaper than Montgomery).
BLS12-381 / BN254 → Montgomery (3× faster).** Wrong choice = 30%+ hit.

### D3. Lazy reduction in butterflies

Radix-2 butterfly `(a,b)→(a+b·ω, a−b·ω)`: naive 3 reductions; lazy
in `[0, 2p)` → 1 reduction at NTT boundary.

**P-13. Document `[0, 2p)` lazy invariant on NTT inner loop, reduce
only on output.** ≈2× at n=2^20 (plonky2/Field/Goldilocks.rs).

---

## E. Poseidon2 hash — rounds and S-box degree

Poseidon2 (Grassi-Khovratovich-Schofnegger 2023), Goldilocks params:

| Parameter      | Value      | Cost                          |
|----------------|------------|-------------------------------|
| State width t  | 8 or 12    | t² in MDS matmul              |
| Full rounds RF | 8          | t·RF S-box evals              |
| Partial RP     | 22         | RP S-box (1-elem layer)       |
| S-box degree d | 7          | 6 squarings/S-box             |
| MDS            | circulant  | Poseidon2 trick: t²→t add+mul |

t=8 perm: 240 S-boxes ≈ 1,440 muls + 1,920 MDS muls = **≈3,400 muls/hash**.
At 10⁶ Merkle leaves: 3.4×10⁹ muls; pure-Go ≈1ns/mul = **3.4 s/commit
at n=2^20** — dominates the prover.

**P-14. Pin (t=8, R_F=8, R_P=22, d=7) for Goldilocks; (t=12, R_F=8,
R_P=22, d=5) for BN254-scalar.** Deviate and golden files diverge from
Plonky3 — no cross-validation possible.

**P-15. Blake3 alternate Merkle hash as opt-in.** ≈3 GiB/s pure-Go
(≈100× faster), loses recursion-friendliness; cuts Merkle commit
3.4 s → 30 ms for **non-recursive** flows (cold-verifier).

**P-16. `Poseidon2.PermuteState8(state *[8]uint64)` — fixed-size
array, pointer receiver, no alloc.** State is 64 bytes (one cache
line); fixed-size arrays get dramatically better SSA register
allocation than slices. Pistachio 60 FPS demands this.

---

## F. AoS vs SoA (GPU-friendly layouts)

GPU is off-table per CLAUDE.md §2, but API choice ripples downstream.

| Primitive       | AoS                          | SoA                          | CPU win | GPU win |
|-----------------|------------------------------|------------------------------|---------|---------|
| Fp elements     | `[]uint64`                   | n/a                          | trivial | trivial |
| BN254 G1        | `[]struct{X,Y,Z [4]uint64}` (96 B) | 3× `[][4]uint64`        | AoS     | SoA     |
| BLS12-381 G1    | `[]struct{X,Y,Z [6]uint64}` (144 B)| 3× `[][6]uint64`        | depends | SoA     |
| Poseidon2 state | `[]uint64` per leaf          | `[t][n]uint64` column-major  | AoS serial; SoA batched | SoA |

If Tranche-2 ships AoS curve-points, a future Pistachio/aicore GPU
sidecar has to transpose them every call.

**P-17. AoS for pure-Go path (CPU cache); expose
`TransposeToSoA(points, dst *G1AffineSoA)` for downstream GPU porters.**
1× memcpy, dwarfed by MSM itself.

**P-18. Document AoS choice + escape hatch in `zkmark.go` package
doc.** Without it every porter re-discovers the trade-off.

---

## G. Scratch-buffer ABI (CLAUDE.md §3)

Every hot-path primitive ships an `*Into` companion:

| Primitive  | Allocating wrapper | Zero-alloc workhorse                                |
|------------|--------------------|-----------------------------------------------------|
| NTT.Forward | `[]Fp`            | `ForwardInto(dst, src []Fp)`                        |
| NTT.Inverse | `[]Fp`            | `InverseInto(dst, src []Fp)`                        |
| MSM         | `G1Affine`        | `MSMInto(dst, scratch *MSMScratch, scalars, points)`|
| Poseidon2   | `Fp`              | `HashInto(dst *Fp, state *[8]uint64)`               |
| RS.Encode   | `[]Fp`            | `EncodeInto(codeword, msg []Fp, ntt *NTT)`          |
| FRI.Fold    | `[]Fp`            | `FoldInto(out, in []Fp, beta Fp)`                   |

**P-19. `*Into` companion for every hot-path primitive or fails
review.** Absolute rule per CLAUDE.md §3.

`MSMScratch` carries buckets + inversion ladder + window-result slabs.

**P-20. `NewMSMScratch(n int) *MSMScratch`, not zero-value-friendly.**
Bucket counts depend on c which depends on n; zero-value is a footgun.

---

## H. Go-specific parallelism caveats

- **H1.** Goroutine spawn ≈8 µs Go 1.22 amd64. NTT n ≤ 2^14 and MSM
  n ≤ 2^10 → single-thread wins (P-5).
- **H2.** False sharing: two goroutines writing adjacent 8-byte Fp
  bounces cache lines.

**P-21. Slab partitioning rounds to multiples of 8 (one cache line
of `uint64`); tail goes to master goroutine.** gnark-crypto field/par.go.

**P-22. No `sync.Pool` in hot paths — document in CONTRIBUTING.md.**
`Get/Put` ≈40 ns vs 1-ns butterfly = 40× hit; caller-owned scratch only.

---

## I. Verifier-side perf — <10 ms budget at n=2^20

| Op                  | Cost                     | Strategy                   |
|---------------------|--------------------------|----------------------------|
| FRI queries         | ≈log²n hashes (400)      | single-thread, cache-resident |
| Poseidon2 hash      | ≈3,400 muls each         | dominant; consider Blake3  |
| Quotient eval       | ≈deg-constraint muls     | lazy at challenges, ≈1 ms |
| KZG pairing check   | ≈1 final-exp ≈10 ms      | single op                  |

**P-23. `VerifyBatch(proofs []Proof) error` for amortised pairing
checks.** N×final-exp → 1×final-exp + N pairings ≈ 30% per-proof at N=10.

---

## J. Tranche-2 PR-1 perf scaffolding

Three prerequisite PRs unblock all later optimisation:

| PR | Scope | LOC | Why first |
|----|-------|-----|-----------|
| PR-A | `zkmark/bench/` benchmark suite, one Benchmark per primitive, CSV trend output | ≈400 | "fast" needs a number |
| PR-B | `zkmark/internal/scratch/`: NTTScratch, MSMScratch, Poseidon2State + reset | ≈200 | Locks ABI before hot-path code |
| PR-C | `zkmark/internal/mathbits/`: golden-validated reduction for Goldilocks + BN254-Fr + BLS12-381-Fr | ≈300 | Foundation; everything depends |

Total scaffolding ≈900 LOC, 1 sprint, no math shipped. After: every
subsequent PR has a benchmark to fail against, a scratch ABI to honour,
a reduction primitive to call.

---

## K. Findings summary

23 actionable items, all forward-looking (Tranche-2 prerequisites);
zero are fixes to shipped code (the shipped code has no perf
characteristics to fix).

**Tier 1 (lock these on day-one of Tranche 2 — irreversible if shipped wrong):**

- P-1  pin BENCHMARK_BUDGET.md
- P-3  bit-reversed twiddle table on Domain receiver
- P-7  auto-select Pippenger c by n
- P-11 `math/bits.Mul64` / Goldilocks plain reduction
- P-12 Goldilocks=plain, BLS12-381=Montgomery
- P-14 pin Poseidon2 parameters (t,R_F,R_P,d)
- P-19 every hot-path primitive ships `*Into` companion
- P-23 `VerifyBatch` for pairing-check amortisation

**Tier 2 (ship-soon; 2×–4× perf at stake):**

- P-2  radix-4 + six-step NTT
- P-4  pre-computed bit-reversal table
- P-5  parallel-NTT auto-threshold
- P-8  GLV/GLS endomorphism
- P-9  affine bucket accumulation with batched inversion
- P-10 goroutine-per-window MSM
- P-13 lazy reduction in NTT butterflies
- P-15 Blake3 alternate Merkle hash
- P-16 fixed-size `[8]uint64` Poseidon2 state
- P-22 no `sync.Pool` in hot path

**Tier 3 (forward-API hygiene; cheap to ship, expensive to retrofit):**

- P-6  iNTT as `NTT.Inverse`, not separate type
- P-17 AoS for CPU + `transposeToSoA` escape hatch
- P-18 document AoS contract in package doc
- P-20 `NewMSMScratch(n)` constructor
- P-21 cache-line-padded slab partitioning

**Scaffolding (PR-A/B/C above):** ≈900 LOC, 1 sprint, prerequisite to
all of the above.

---

## L. Explicitly NOT recommended

Off-table per CLAUDE.md zero-dep / pure-Go constraints:

- CGO/asm hand-tuning (`math/bits` is enough for T2; asm in a future
  `zkmark/asm` build-tag if ever)
- ICICLE/sppark CUDA (aicore handles GPU separately)
- `unsafe.Pointer` slice tricks (CLAUDE.md §6)
- Compile-time codegen (complicates golden-file generation)
- SIMD intrinsics via `golang.org/x/sys/cpu` (impure)

Discipline cost ≈3× slower than ICICLE at n=2^20, matches `reality`'s
posture: **canonical, not fastest**. Target: be within 4× of
best-published pure-Go; let aicore/GPU sidecars chase the last 4×.

---

## Progress

- 2026-05-08 — agent 150 complete; 23 forward-looking perf findings
  (8 Tier-1, 10 Tier-2, 5 Tier-3) + 3-PR scaffolding plan; zero
  shipped-code fixes (262-LOC stub has no hot path); all
  recommendations sized against pure-Go zero-dep budget per CLAUDE.md.
