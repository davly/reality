# 148 — zkmark-sota

**Topic:** SOTA comparison: Arkworks, halo2, gnark, plonky3, sp1, risc0.

**Premise (delta vs 146/147):** Agent 146 audited the zkmark/ envelope (262-line stub, no math). Agent 147 enumerated 11 missing primitives (~7,750 LOC across 8 sprints). This agent does **not** re-enumerate primitives. Instead: profile each of the 6 SOTA libraries by (1) headline algorithm/system choice, (2) one engineering trick that defines them, (3) zero-dep portability for a `reality`-style Go reimplementation. Outcome: a backend-selection matrix the README/zkmark.go can use to retire its Halo2-only commitment.

---

## Comparison axes (locked once for all six)

For each library:

- **Stack** — field, curve (if any), commitment scheme, constraint system, hash.
- **Headline algorithm** — the single design choice that defines the library.
- **Engineering trick** — the one optimisation or architectural decision that gives the library its perf or composability edge in production.
- **Proof size / verifier cost** — order-of-magnitude only (real numbers depend on circuit; cited where the libraries themselves publish them).
- **Recursion** — supported? Native or wrapped? Cost?
- **Dep weight** — pure-language LOC / external-FFI LOC / pre-trusted-setup artifacts.
- **Reality portability** — what's the smallest subset that fits the zero-dep + golden-file + Go-canonical model? Is the canonical implementation the kind of code `reality` reimplements, or is it inseparable from a specific tool-chain (Rust nightly, intrinsic-bound MSM, GPU, etc.)?

---

## 1. Arkworks (Rust)

**Origin:** arkworks-rs/algebra ecosystem, started ~2020 by Stanford/Berkeley/iden3 cohort. Now the de-facto Rust crypto-substrate; ≈40 crates split by concern (`ark-ff`, `ark-ec`, `ark-poly`, `ark-bls12-381`, `ark-groth16`, `ark-marlin`, `ark-r1cs-std`, `ark-sponge`, `ark-relations`, ...).

**Stack:** field-agnostic (BLS12-381, BN254, BW6-761, MNT4/6 cycles, Pasta, Bandersnatch, Goldilocks); curve-agnostic; multiple commitment scheme crates (KZG, IPA, Hyrax, Ligero); multiple SNARK crates (Groth16, Marlin, PlonK is community-fork). **Library, not a system** — analogous to `reality` in posture.

**Headline algorithm:** *no single one*. Arkworks is a **trait-system** rather than a system: `Field`, `PrimeField`, `FftField`, `CurveGroup`, `PairingEngine`, `EvaluationDomain`, `PolynomialCommitment`, `SNARK` are all generic traits, and concrete protocols compose them. The closest single artifact is **Marlin** (Chiesa-Hu-Maller-Mishra-Vesely-Ward 2019) — universal-SRS PlonK predecessor, the showcase Arkworks SNARK.

**Engineering trick:** **monomorphisation-driven generics**. Arkworks ships `Fp<N: Limbs>` for arbitrary-prime fields with N-limb little-endian representations resolved at compile time; the Montgomery multiplication, NTT twiddle table, MSM bucket layout, and pairing miller-loop are all monomorphised per (field, curve, pairing) triple. End user pays no runtime dispatch and no boxing. Trade-off: the resulting `.rlib` files are huge (Arkworks-bls12-381 alone is ~30 MB of compiled code), and compile times are minutes — but the perf is within constant-factor of hand-rolled BLST.

**Proof size / verifier cost:** N/A at library level. For Marlin on BLS12-381: ~880 bytes proof, ~10 ms verifier (1M-gate circuit). Groth16 (also via `ark-groth16`): ~192 bytes, ~3 pairings, sub-millisecond verify.

**Recursion:** supports cycles-of-curves (MNT4/6, BLS12-377/BW6-761) for "trivial" recursion (verify previous proof in-circuit). No accumulation scheme native; community crates exist (Nova-on-arkworks). Halo2 is a separate ecosystem.

**Dep weight:** pure Rust, no FFI, depends on `rand_core`, `serde`, `digest` (the standard Rust ecosystem traits). `no_std` capable. Workspace ships ~600k LOC across all crates; a typical user pulls 3-5 crates ≈ 80k LOC.

**Reality portability:**
- **High** for the *trait shape*: `Field` / `PrimeField` / `FftField` / `Polynomial` / `PolynomialCommitment` / `SNARK` all map cleanly to Go interfaces; reality's existing `prob.Distribution`, `linalg.Matrix`, `geometry.Curve` interfaces show the pattern works in Go.
- **Low** for the *concrete code*: Arkworks lives in Rust trait-with-associated-type ergonomics that Go's interface system doesn't replicate cheaply (e.g. `Field::BasePrimeField` projection). Direct port loses ~10× expressiveness; idiomatic-Go rewrite is a clean-slate exercise.
- **Test-vector reuse:** **Yes, free win.** Arkworks ships JSON-friendly test fixtures for BLS12-381 (NIST-spec curve points) and the algebraic primitives have well-known reference vectors. `reality`'s golden-file infra can adopt arkworks vectors verbatim for any field/curve overlap.
- **The one piece worth porting first:** `ark-poly`'s `Radix2EvaluationDomain` is a clean ~400-line implementation of the Cooley-Tukey NTT + iNTT + coset-NTT that Agent 147's T1-NTT specifies. It is well-commented, free of Rust-specific tricks beyond limb-arithmetic, and would translate directly to `reality/zkmark/poly/` or a sibling.

---

## 2. halo2 (Zcash, Rust)

**Origin:** zcash/halo2, 2020-present. Production component of Zcash Orchard. Algorithm = "Halo" (Bowe-Grigg-Hopwood 2019) generalised to "Halo2" (ECC 2021).

**Stack:** Pasta cycle (Pallas + Vesta — half-pairing-friendly cycle, no pairings). IPA (Inner Product Argument) commitment. PlonK-style constraint system with custom gates + plookup. BLAKE2b transcript hash; Poseidon in-circuit.

**Headline algorithm:** **PlonK-over-IPA with recursive accumulation**. Halo2 commits to wire polynomials via Bulletproofs-IPA (no trusted setup), and instead of paying the IPA verifier's O(n) cost on-chain, the verifier check is *deferred* into an "accumulator" curve point that the next proof proves correct in its own circuit. The accumulator is a single curve point; recursive verification is O(log n) Poseidon hashes per recursion step instead of an O(n) MSM.

**Engineering trick:** **the `Region`/`Layouter` chip API**. Halo2 separates *constraint definition* (declarative: "this column at this row equals that column shifted") from *value assignment* (imperative: fill in the witness). The `Layouter` resolves regions to absolute row indices, which lets composable "chips" (range check, ECC ops, Poseidon, RSA) be built once and assembled into arbitrary circuits without manual row bookkeeping. This is the design choice that made Halo2 the dominant *application-developer* ecosystem despite Plonky2/3 being faster.

**Proof size / verifier cost:** ~1.5 KB / O(log n) curve ops + a final MSM. Recursive verifier: a single proof of size ~1.5 KB even after N levels of recursion (the accumulator absorbs the deferred work).

**Recursion:** **native and the headline feature**. Pasta cycle (Pallas verifies Vesta proof; Vesta verifies Pallas proof) drives every recursion step. No pairings.

**Dep weight:** Rust, ~50k LOC core (`halo2_proofs` crate) + ~30k LOC of standard chips (`halo2_gadgets`). Depends on `pasta_curves`, `ff`, `group`, `rand_core`. No FFI, no GPU, no nightly.

**Reality portability:**
- **Medium** for the *protocol math*: PlonK-over-IPA can be reimplemented in Go in ~2,500 LOC (T2-PLONK + T2-IPA from Agent 147). The Pasta cycle is a single-curve pair; the field arithmetic is non-Montgomery (Pasta is chosen specifically for fast non-Montgomery reductions on 64-bit hosts). Achievable.
- **Low** for the *chip API*: Halo2's `Layouter` is the value, not the protocol. Reimplementing it in Go is a 5,000-LOC engineering project orthogonal to the math `reality` ships. Reality should explicitly **not** ship a chip API; users compose primitives directly.
- **Test-vector reuse:** the Halo2 transcript byte format is unstable across versions (the protocol is `halo2_v1`, `halo2_v2`, ... and each bumps the transcript-tag bytes). Direct vector reuse is **fragile**; better to pin against a frozen Halo2 commit.
- **The recommendation Agent 147 §C-9 already pinned:** Halo2-as-the-Tranche-2-target is the *most* expensive choice in this six-library shoot-out. It buys you no-trusted-setup + recursion at the cost of pairing-free IPA (worse prover cost than KZG-PlonK) and the chip API (which `reality` does not ship). For a regulator-facing audit substrate, FRI/STARK over Goldilocks gives the same transparent-setup property at half the LOC.

---

## 3. gnark (ConsenSys, Go)

**Origin:** ConsenSys/gnark, 2020-present. The only major SOTA ZK library written in Go — directly relevant as a portability and idiom reference.

**Stack:** BLS12-377, BLS12-381, BN254, BW6-633/761, BLS24-315 (the cycle partner of BLS12-377). R1CS + PlonKish (Plonk + custom gates). KZG commitment for PlonK; pairing-based for Groth16. Mimc/Poseidon in-circuit; SHA-256/Keccak-256 transcript. Pure-Go field/curve arithmetic in `gnark-crypto`.

**Headline algorithm:** **two backends side-by-side**: `groth16` (per-circuit trusted setup, 192-byte proof, 3-pairing verifier) and `plonk` (universal trusted setup, ~1KB proof, larger verifier). The library ergonomically lets a developer compile a single circuit DSL (`frontend.Circuit`) to either backend.

**Engineering trick:** **the `frontend.API` DSL**. A developer writes their constraint logic as Go code calling `api.Mul`, `api.Add`, `api.AssertIsEqual`, etc., on `frontend.Variable`. The DSL traces execution to build R1CS or Plonkish constraints with **no separate front-end language** (unlike Circom or Noir). The same Go code is the circuit. This is gnark's ergonomic kill-shot vs Arkworks's macro-heavy `r1cs-std` and Halo2's chip API.

**Proof size / verifier cost:** Groth16 backend on BN254 — ~192 bytes / ~3 pairings (~3 ms). PlonK backend on BLS12-381 — ~1 KB / ~10 pairings (~10 ms).

**Recursion:** supported via BLS12-377/BW6-761 cycle (Aztec-style "in-circuit pairing"). Non-trivial — the BW6-761 prover is slow because the field is huge (761-bit). No accumulation scheme.

**Dep weight:** pure Go, two modules (`gnark` ~30k LOC, `gnark-crypto` ~80k LOC). No FFI, no GPU. Depends on `golang.org/x/crypto` and `consensys/bavard` (a code-generator for field-arithmetic templates — at compile time only, doesn't ship).

**Reality portability:**
- **Highest of the six**, by a large margin. gnark-crypto is the most direct precedent for what a `reality/zkmark/field/`, `reality/zkmark/curve/`, `reality/zkmark/poly/` would look like: ~80k LOC of pure Go, idiomatic, MIT-licensed, with NIST-validated test vectors.
- **Reality cannot just import gnark-crypto** because the CLAUDE.md §"Zero dependencies" rule forbids it (only language stdlib math). But:
  - gnark-crypto's *test vectors* can be lifted into reality's golden-file infra unchanged (JSON output; same precision discipline as reality's `testutil`).
  - gnark-crypto's *algorithms* (Karatsuba multiplication for 4-limb fields, Montgomery reduction for BN254, GLV decomposition for BLS12-381 MSM, Pippenger window-size selection table) are documented in code comments and translate 1:1 to a `reality` reimplementation.
  - gnark-crypto's *code-generation approach* (the `bavard` template engine) is the right model for `reality`'s eventual large-prime-field zoo — a single Go template generates per-field arithmetic; `reality` would use Go's `text/template` or `go generate`.
- **One concrete win:** gnark's `bavard`-generated `mulCIOS` Montgomery multiplication for 4×64-bit limbs is the canonical pure-Go reference; reality's T1-FIELD (per Agent 147) should benchmark against it and stay within 2× of its allocation/cycle count.

---

## 4. plonky3 (Polygon, Rust)

**Origin:** Polygon Zero, 2024-present. Successor to Plonky2 (which was StarkWare-style FRI-over-Goldilocks-with-Plonk-frontend).

**Stack:** **pluggable** at every layer. Field options: Goldilocks (p = 2^64 − 2^32 + 1), BabyBear (p = 2^31 − 2^27 + 1), Mersenne-31 (p = 2^31 − 1, Circle-STARK), KoalaBear (p = 2^31 − 2^24 + 1). Commitment options: FRI (default), Brakedown, Ligero, hybrid. Hash options: Poseidon2, Tip5, Monolith. Constraint system: AIR + RAP (randomised AIR with preprocessing).

**Headline algorithm:** **STARK over a small prime field with pluggable everything**. Plonky3 is the explicit "general-purpose toolkit" answer to the Halo2-vs-Plonky2-vs-RISC0 splintering: pick your field by latency target (Goldilocks = native u64 arith on x86-64; M31 = SIMD; BabyBear = best round-counts for Poseidon2), pick your commitment, pick your hash. Each choice is an independent crate with a stable trait boundary.

**Engineering trick:** **Mersenne-31 + Circle-STARK**. M31 has zero two-adicity (p − 1 = 2 · 3^2 · 5 · 7 · ...) — historically a deal-breaker for FFT-based STARKs. StarkWare's 2024 paper showed FFTs can be performed on the *circle group* of order p + 1 = 2^31 (which is 2-adic) instead of the multiplicative group. Plonky3 ships a production Circle-STARK implementation, opening the use of M31 (the fastest 32-bit prime field on every modern CPU because mod-Mersenne reduction is `(x >> 31) + (x & MASK)` with one carry-correction).

**Proof size / verifier cost:** typical AIR proof ~50-200 KB at λ=100 bits security; verifier ~5 ms in software, ~$0.01 on EVM with proof-aggregation.

**Recursion:** supported via STARK-aggregation-then-Groth16 (proof-recursion-as-a-service, one Groth16 wrapper at the top of a chain of STARK proofs to compress to ~200 bytes for L1 contracts).

**Dep weight:** Rust workspace, ~120k LOC across 30+ crates. No FFI, no GPU (CPU-only optimisations). Depends on `p3-field`, `p3-commit`, `p3-air`, `p3-fri`, `p3-poseidon2`, etc. — every layer is its own crate.

**Reality portability:**
- **Highest "math-purity"** of the six. Plonky3's design is the closest in spirit to `reality`'s posture: small prime fields, no curves, no pairings, no trusted setup, every primitive a separable trait. The Circle-STARK paper is open-access and references-only; the protocol is reimplementable in ~2,500 LOC of pure Go.
- **The Goldilocks subset** (drop M31, drop Circle-STARK) is the cheapest production STARK to ship in Go: T1-FIELD-Goldilocks (~150 LOC), T1-NTT (~300 LOC), T1-RS (~80 LOC), T1-MERKLE (~200 LOC), T1-TRANSCRIPT (~150 LOC), T1-POSEIDON2 over Goldilocks (~400 LOC), T2-FRI (~600 LOC), T2-STARK-AIR (~1,000 LOC). **Total ~2,880 LOC** for a production-shippable transparent-setup STARK system in Go. This is **half** of the Halo2 cost (Agent 147's ~5,250-LOC pairing-bound estimate).
- **Test-vector reuse:** Plonky3 ships per-crate test vectors for its 4 fields and Poseidon2 with public-domain round constants. Direct reuse is fine (no IP issues; MIT-licensed).
- **Recommendation reinforcement:** Agent 147 §C-3 said Tranche-2 should rename `AlgorithmHalo2` to backend-agnostic. **This audit goes further: the README should target Plonky3-style FRI-on-Goldilocks (or M31 once Circle-STARK is mature) as the explicit Tranche-2 backend.** It is cheaper to ship, doesn't gate on pairings, and matches `reality`'s zero-dep small-field aesthetic.

---

## 5. SP1 (Succinct Labs, Rust)

**Origin:** Succinct Labs, 2024. zkVM for RISC-V.

**Stack:** Plonky3 underneath (the field is BabyBear or Mersenne-31; the commitment is FRI; the hash is Poseidon2). Constraint system is RISC-V instruction tables expressed as AIR with lookup arguments (logUp). Proof-recursion-then-Groth16 wrapper for L1 verification (~200-byte final proof on EVM).

**Headline algorithm:** **RISC-V STARK with lookup-table-per-instruction (Lasso-style)**. Every RV32IM instruction is precomputed as an MLE table; the prover proves "the trace of my program is a sequence of valid `(opcode, operand1, operand2, result)` tuples" by lookup arguments into per-instruction tables. The cost per proven CPU cycle is roughly 1 lookup + 1 memory permutation argument.

**Engineering trick:** **per-instruction lookup tables via the `chip` macro**. Instead of building giant AIR transition polynomials for the entire ISA, SP1 generates one `Chip` per instruction (`AddChip`, `MulChip`, `LwChip`, ...). The Lasso/logUp lookup argument unifies them into a single proof. This is **the same pattern as Halo2's `Region`/`Layouter`** but applied at the ISA level rather than the application level — the application *is* a RISC-V program, so the chips are fixed once per ISA.

**Proof size / verifier cost:** raw STARK proof: ~5-50 MB per program (depending on cycle count); after recursion + Groth16 wrap: 192 bytes, ~3 pairings on EVM.

**Recursion:** Plonky3 STARK aggregation → Groth16 wrap. 3-stage compilation pipeline.

**Dep weight:** Rust workspace, ~200k LOC. Depends on plonky3, the `riscv-rt` runtime, and a custom RV32IM emulator. **Has a CUDA prover** (`sp1-cuda` crate) for GPU-accelerated trace generation — the first of the six libraries here to depend on GPU. CPU prover is ~10× slower.

**Reality portability:**
- **Library-as-a-whole: very low.** SP1 is a *system* (zkVM compiler + emulator + STARK proof generator + Groth16 wrapper + EVM contracts), not a library. Reimplementing it in Go is a multi-engineer-year project orthogonal to `reality`'s scope.
- **Components that fit `reality`'s model: same as Plonky3.** SP1 reuses Plonky3 wholesale, so the same Goldilocks/BabyBear/M31 + FRI + Poseidon2 layer described in §4 applies.
- **The `Chip`-per-instruction pattern is a code-organization trick, not math.** It is OUT of scope for `reality` (which ships pure math). A future consumer of `reality` (an L4 zkVM project) would be the right place for chip patterns; not here.
- **Note on GPU:** `reality` is CPU-only by CLAUDE.md §3 (no allocations, deterministic). SP1's CUDA prover is irrelevant to portability; the CPU code path is what matters, and it's the same Plonky3 code as §4.

---

## 6. RISC Zero (RISC Zero, Rust)

**Origin:** RISC Zero, 2022-present. zkVM for RISC-V (RV32IM, with custom precompiles).

**Stack:** BabyBear (p = 2^31 − 2^27 + 1) field, FRI commitment, Poseidon2 hash. **Custom STARK** (predates Plonky3 — they don't share code). Recursion via repeated STARK proofs, with optional Groth16 final-wrap.

**Headline algorithm:** **STARK over BabyBear with native recursion in BabyBear**. RISC Zero proved out the small-prime-field-recursion approach before Plonky3; Plonky3 then generalised. RISC0 still ships a slightly different proof shape (slightly larger, but with simpler verifier circuit) that they preserve for backward compatibility with deployed contracts.

**Engineering trick:** **the `zkVM` developer model**. A user writes a regular Rust program, marks the entrypoint with `#[risc0_zkvm::entry]`, the `cargo build` produces an ELF, and `risc0` proves the ELF executed correctly. No custom DSL, no custom front-end — the developer thinks they're writing a normal Rust program. This was the first such "guest binary" zkVM and remains the most polished UX.

**Proof size / verifier cost:** ~250 KB raw STARK proof; ~200 bytes after Groth16 wrap. Verifier on EVM: ~3 pairings post-wrap.

**Recursion:** BabyBear-native, two-stage (continuation proofs + lift-then-join + optional Groth16 wrap). Production-grade as of 2025.

**Dep weight:** Rust workspace, ~150k LOC. Includes a full RV32IM emulator, ELF parser, and `riscv0-zkvm-platform` runtime. CUDA-accelerated prover (similar to SP1).

**Reality portability:**
- **Same as SP1.** The library-as-a-whole is a zkVM system, OUT of scope for `reality`. The math substrate underneath (BabyBear field, FRI, Poseidon2) overlaps with Plonky3's BabyBear+FRI+Poseidon2 instance, so the §4 portability story applies.
- **Distinguishing detail:** RISC0's BabyBear arithmetic shipped before Plonky3's, and the round constants for Poseidon2-over-BabyBear are slightly different between the two libraries (different security analyses, different MDS matrix). Reality must pick one and pin it; **recommend Plonky3's choice** because Plonky3 is the more general-purpose substrate and has clearer round-constant derivation documents.

---

## Backend-selection matrix for `reality/zkmark/`

Pulling the six profiles together, scored against `reality`'s constraints:

| Library    | Field        | Curve req. | Pairing req. | Trusted setup | Recursion | Reality LOC est. (Tranche-2 backend) | Rec |
|------------|--------------|------------|--------------|---------------|-----------|--------------------------------------|-----|
| Arkworks   | any          | any        | optional     | per-protocol  | yes (cycles)| N/A (library not a system)          | partial port for traits |
| halo2      | Pasta        | yes        | no           | no (transparent IPA) | yes (Pasta cycle) | ~5,250            | NO (most expensive)  |
| gnark      | BN254/BLS12-* | yes       | yes (KZG)    | per-protocol  | yes (cycles, slow) | ~4,800             | NO (pairing-bound) |
| plonky3    | Goldilocks/M31/BabyBear | no | no | no (transparent FRI) | yes (small-field native) | **~2,880 (Goldilocks subset)** | **YES** |
| SP1        | BabyBear     | no         | no           | no (FRI)+Groth16 wrap | system-level | N/A (system not library) | borrow Plonky3 substrate |
| RISC0      | BabyBear     | no         | no           | no (FRI)+Groth16 wrap | system-level | N/A (system not library) | borrow Plonky3 substrate |

**Headline finding (delta vs 147):** Agent 147 estimated Halo2 = ~5,250 LOC of net-new math vs Plonky3-style Goldilocks-FRI-STARK = ~1,800 LOC. This audit refines that to **~2,880 LOC** for a production-shippable Goldilocks STARK in Go (adding T1-RAP + RISC-Zero/SP1-style lookup-arg layer that Agent 147 grouped under T2-LOGUP separately) — still **~45%** the cost of the Halo2 path with stronger small-field portability.

---

## What `reality/zkmark/` should adopt — three-axis recommendation

### 1. Backend (algorithm) — Plonky3-style, not Halo2

Already argued in Agent 147 §C-3 and reinforced here. Rename `AlgorithmHalo2` to `AlgorithmStarkGoldilocks` (or `AlgorithmTranche2` if backend choice still in flight). The math substrate to actually ship is:

- **Goldilocks field** (T1-FIELD instance, ~150 LOC; native u64 arithmetic, no Montgomery).
- **Cooley-Tukey radix-2 NTT** (port `ark-poly`'s `Radix2EvaluationDomain` shape; ~300 LOC).
- **Reed-Solomon encode** (one NTT + zero-pad; ~80 LOC).
- **Binary Merkle commit + path verify** (caller-supplied hash; ~200 LOC).
- **Fiat-Shamir transcript** (BLAKE2b-based, length-prefixed labels; ~150 LOC).
- **Poseidon2 over Goldilocks** (round-constant table from Plonky3, MDS matrix, full+partial rounds; ~400 LOC).
- **FRI low-degree-test** (commit + fold + query; ~600 LOC).
- **AIR + DEEP composition** (the actual STARK; ~1,000 LOC).

### 2. Library posture — gnark, not Arkworks

Arkworks's trait system is more expressive than Go can replicate idiomatically; gnark proves the same abstractions work fine in Go with simpler interfaces. Pattern after gnark:

- One package per primitive (`zkmark/field/goldilocks`, `zkmark/poly`, `zkmark/commit/fri`, `zkmark/proof/stark`, `zkmark/hash/poseidon2`).
- Code-generation for field-arithmetic templates (one `field.go.tmpl` parameterised on prime; `go generate`-driven).
- Lift gnark-crypto's test vectors verbatim where overlap exists.

### 3. Frontier-watch — Circle-STARK + Lasso

Two unfinished SOTA threads `reality` should *defer* but *track*:

- **Circle-STARK over Mersenne-31** (Plonky3 frontier) is the post-2026 zero-overhead-FFT story for 32-bit primes. Wait until the StarkWare 2024 paper has independent reimplementations (Plonky3's is one; sp1's M31 is another) and the protocol stabilises. Earliest reasonable port to `reality`: 2027.
- **Lasso/Jolt** (Setty-Thaler-Wahby 2023-2024) is the lookup-everything zkVM line. The math substrate is multilinear-extension-commit + sumcheck + logUp (covered by Agent 147's T2-MLE-COMMIT + T2-LOGUP). Worth shipping in `reality` *after* Goldilocks-FRI-STARK is stable, as a separate subpackage.

---

## Cross-library convergent design choices (signal for `reality`)

Across the six libraries:

1. **Poseidon2 has won as the default in-circuit hash** for small-prime-field protocols (Plonky3, SP1, RISC0 all use Poseidon2; Halo2 still uses Poseidon-1 but is converging). gnark uses Mimc (older); Arkworks supports both. **Reality should ship Poseidon2 first, treat Poseidon-1 as deprecated.** Round constants are public-domain (Grassi-Khovratovich-Lüftenegger-Rechberger-Roy-Schofnegger 2020/2023).

2. **Goldilocks is the default 64-bit prime** for "fast STARK without the M31 complexity" (Plonky2/3 default, RISC0 uses BabyBear which is similar but smaller; Plonky3 supports both). **Reality should ship Goldilocks first** (~150 LOC, native u64), defer M31 (Circle-STARK complexity) and BabyBear (overlaps Goldilocks with smaller prime → harder bounds analysis).

3. **FRI has won over IPA for transparent commitments** in the small-prime-field world; IPA only survives in Halo2 because Halo2 needed a *cycle of curves* (and pairing-friendly cycles don't exist, so curves require IPA). **For reality's curve-free ZK substrate, this is moot** — FRI is the only choice.

4. **Pippenger MSM with GLV decomposition is the standard** for curve-bound prover work (Arkworks, gnark, Halo2 all converge here). Reality only needs this if pairing-bound protocols are ever added; deferred.

5. **Recursion via small-prime-native STARK + Groth16 wrap** (SP1, RISC0 model) has displaced cycle-of-curves recursion (Halo2 model) for new systems. Reality, being curve-less in Tranche-2, defers the wrap to a downstream consumer (the `nexus` cold-verifier doesn't need EVM-deployable proofs; if it ever does, that's a future Tranche-3).

6. **No production library uses Marlin or Sonic** despite being the academic-PlonK predecessors. Universal-SRS PlonK (gnark, Arkworks) and FRI-STARK (everyone else) split the field. Reality should not ship Marlin/Sonic — there's no consumer.

7. **Every production library has a transcript-domain-separation tag with a per-protocol version suffix** (`b"halo2-v1"`, `b"plonky3-v0"`, `b"sp1-v3"`, etc.). Agent 146 §G7 flagged this as a forward-compat hazard already; this audit reinforces — pin `b"reality-zkmark-v1-stark-goldilocks"` (or analogous) before any byte-emitting code lands.

---

## Bottom line

Six libraries → one decision matrix → one recommendation: **`zkmark/`'s Tranche-2 backend should be Plonky3-style FRI-over-Goldilocks-with-Poseidon2-AIR (~2,880 LOC of pure-Go math), not Halo2 (~5,250 LOC of pairing-bound IPA-recursion math)**. The `Algorithm` constant should be renamed to be backend-agnostic *now* per Agent 147 §C-3, but the backend itself should be selected explicitly in favour of Plonky3's small-field-FRI-STARK pattern. gnark is the closest *posture* match for how the Go reimplementation should be organised (per-primitive subpackages, code-gen for field templates, test-vector reuse from gnark-crypto). Arkworks is the deepest *trait taxonomy* reference but is too Rust-flavoured for direct port; lift the trait shape, drop the macros. SP1 and RISC0 are not libraries `reality` competes with — they're systems *built on top* of the same Plonky3-shaped math substrate that `reality` should ship as primitives. Halo2's chip API is its real value-add and is firmly out of scope for `reality` (CLAUDE.md §"Reimplement from first principles" + the no-DSL-no-frontend posture). Circle-STARK and Lasso/Jolt are 2027+ frontier and should be tracked but not yet ported.

## Two-line summary

The six SOTA ZK libraries split into three groups: trait-systems (Arkworks, gnark — `reality` should pattern after gnark's per-package Go layout, lift gnark-crypto's test vectors, and treat Arkworks's trait taxonomy as a reference not a port target), application protocols (Halo2 = Pasta+IPA+recursion @ ~5,250 LOC, plonky3 = small-prime-field+FRI+Poseidon2 @ ~2,880 LOC for the Goldilocks subset — reality should explicitly pick Plonky3's pattern over Halo2's because it's curve-free and ~45% the LOC for equivalent regulator-grade transparent-setup soundness), and zkVMs (SP1, RISC0 — both build on Plonky3-shaped substrate so are not direct competitors but reinforce that the Goldilocks/BabyBear-FRI-Poseidon2 stack is the right Tier-2 anchor for reality's Tranche-2 PR per Agent 147's sprint ordering).

## Progress

- 2026-05-08 agent-148 zkmark-sota: profiled Arkworks/halo2/gnark/plonky3/sp1/risc0 across 7 axes (stack, headline-algorithm, eng-trick, proof-size, recursion, dep-weight, reality-portability); backend-selection matrix recommends Plonky3-style Goldilocks-FRI-Poseidon2-AIR (~2,880 LOC) over Halo2's pairing-IPA-recursion path (~5,250 LOC); gnark is closest Go-posture reference; SP1/RISC0 are systems-not-libraries built on Plonky3-shaped math substrate that reality should ship as primitives; flagged Poseidon2 / Goldilocks / FRI / domain-sep transcript-tag as cross-library convergent SOTA design choices.
