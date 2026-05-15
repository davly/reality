# 370 — research-zk-2026 (state of ZK proof systems May 2026)

## Headline
By May 2026, ZK production has consolidated around three families — Groth16/PLONK on BLS12-381/BN254 for compact one-shot proofs, Halo2/Plonky3 on Pasta/Goldilocks/KoalaBear for recursive folding, and FRI-based zkVMs (SP1, RISC Zero) for general-purpose RISC-V execution; reality's `zkmark`/`forge` slots should target the Plonky3 + KZG/FRI hybrid because it is the only stack with vendor-neutral spec, audited mainnet usage (Polygon, Succinct), and small-field arithmetic that maps cleanly to the existing `linalg`/`crypto`/`signal` (NTT) primitives.

## Survey

### 1. Groth16 (Groth 2016) — PRODUCTION (legacy)
SNARK with the smallest known proofs (3 group elements, ~192 bytes on BN254) and ~3 ms verification. Per-circuit trusted setup remains its Achilles heel — two of the first known live ZK exploits in 2024 stemmed from snarkjs Groth16 verifiers with malformed setup. Still the dominant scheme on Ethereum L1 (zk-rollup verification, Tornado Cash forks, Semaphore, RLN) because EVM precompiles for BN254 pairings make verification near-free in gas. Maturity: production-stable, widely audited, but considered "legacy" for new deployments. Use only when proof size dominates and the circuit is fixed.

### 2. PLONK (Gabizon-Williamson-Ciobotaru 2019) — PRODUCTION
Universal trusted setup (one ceremony, all circuits) with KZG polynomial commitments. ~500 byte proofs, ~10 ms verify. Adopted as the "default new-era SNARK" in Aztec, Mina (via Kimchi), zkSync's bellman-style stack, and as the wrap layer in SP1 → BN254 for Ethereum verification. Variants (TurboPLONK, UltraPLONK with custom gates, fflonk for batch openings) are the practical workhorse. Maturity: fully production-stable as of 2024; PLONK + KZG on BN254 is now the canonical "verify on Ethereum" target.

### 3. Halo2 (Bowe-Grigg-Hopwood, ZCash 2020) — PRODUCTION
PLONK-ish arithmetization + IPA (Bulletproofs-style) commitment over Pasta cycle (Pallas/Vesta) for recursion without a trusted setup. Adopted by ZCash Orchard, Scroll, Taiko, Privacy & Scaling Explorations (PSE), and Filecoin. PSE fork swaps IPA for KZG when targeting Ethereum verification. Kudelski's 2024 audit and zkSecurity's 2026 course material both cite Halo2 as the most mature recursive SNARK in the wild. Caveat: deeply complex API; some teams (Aztec/Payy) migrated to Noir/Barretenberg. Maturity: production-stable, widely deployed.

### 4. Plonky2 / Plonky3 (Polygon 2022 / 2024) — PRODUCTION (Plonky3 declared so April 2026)
PLONK-style IOP over the 64-bit Goldilocks field (Plonky2) and small-field set including KoalaBear/BabyBear (Plonky3) with FRI commitments. Plonky3 was officially declared production-ready by Polygon on 30 April 2026 and is the proving backend selected by Succinct's SP1, Lita's Valida, and Polygon zkEVM Type-1. AVX2/AVX-512/NEON vectorized field arithmetic. Universal verifier proposal (issue #511) is in flight. Maturity: production-stable as of late April 2026; the most actively-developed open ZK toolkit.

### 5. STARKs / ethSTARK / Stone / Stwo (Ben-Sasson 2018) — PRODUCTION
Transparent (no trusted setup), post-quantum (hash-based), large proofs (40–200 KB). StarkNet's Stone prover (Cairo VM) and the new Stwo prover (Circle STARKs over Mersenne-31 field, 2024–25) are the production reference. STARK proofs typically wrap into a SNARK for cheap on-chain verification. Stwo's small-field (M31) and Plonky3's KoalaBear move are the 2025-2026 trend. Maturity: production-stable; StarkNet has billions in TVL secured by Stone-derived proofs since 2022.

### 6. RISC Zero zkVM (Bowe et al. 2022) — PRODUCTION
RISC-V zkVM atop a custom STARK (BabyBear field) wrapped in Groth16 for on-chain verification. Audited (Veridise, NCC, Trail of Bits) and declared production-ready in 2024. Q1 2026 zkvm-benchmarks shows lower memory footprint than competitors on large programs but slower SHA-256 / Fibonacci than SP1. Mature SDK (Bonsai cloud prover, Boundless prover network). Maturity: production-stable; primary "ship today" zkVM for general Rust workloads.

### 7. SP1 (Succinct Labs 2024) — PRODUCTION
Open-source RISC-V zkVM built on Plonky3. Audited by Veridise, Cantina, Zellic, KALOS. Reported up to 28× faster than prior zkVMs on real workloads (pre-2026). Optimized BN254 / BLS12-381 precompiles cut Ethereum sync-committee verification from 6 B → 50 M cycles. PLONK+Groth16 wrap for on-chain. Used by Lido, Polygon AggLayer, Taiko's Raiko prover. Maturity: production-stable; the most aggressive performance entrant in 2025-2026 and the de-facto open zkVM standard.

### 8. Nova / SuperNova / HyperNova / ProtoStar (Kothapalli-Setty-Tzialla 2022 →) — LATE-STAGE RESEARCH
Folding schemes that batch incremental proofs via accumulation rather than full recursion. Nova on Pasta cycle is reference; SuperNova handles non-uniform IVC (different instructions per step → EVM/RISC-V); HyperNova generalizes to CCS; ProtoStar gives a generic accumulation scheme. zksecurity disclosed a soundness break of original Nova in 2024 ("Nova attack of the year") which has since been patched. LatticeFold (Boneh-Chen 2026) gives plausibly post-quantum folding over 64-bit fields, performance comparable to HyperNova. Maturity: late-stage research, real deployments in Lurk, Sonobe, but not yet at zkVM-grade audit coverage.

### 9. Spartan / Hyrax (Setty 2020) — LATE-STAGE RESEARCH (transparent SNARKs)
Sum-check-based zkSNARK with linear-time prover, transparent setup. PCS-agnostic — Hyrax (Pedersen), HyperKZG, Binius, Dory, BaseFold, WHIR all plug in. Microsoft's Spartan2 reference impl is the canonical codebase. Forms the engine inside Jolt zkVM (a16z). Slower verifier than KZG-PLONK but no trusted setup. Maturity: research-grade libraries with growing prod use via Jolt.

### 10. Bulletproofs (Bünz et al. 2018) — PRODUCTION (niche)
Logarithmic-size range proofs and arithmetic circuits, no trusted setup, discrete-log security. Linear-time verifier kills it for large circuits but remains the workhorse for confidential transactions: Monero (BP+) and Grin in mainnet, Mimblewimble. Spartan-Bulletproofs simulation-extractability proven for free in 2023. Maturity: production-stable for small circuits / range proofs; not competitive for general computation.

### 11. Lookup arguments — Plookup / Caulk / LogUp / Lasso — PRODUCTION
Plookup (Gabizon-Williamson 2020) embedded in Halo2/PLONK; Caulk (Zapico et al. 2022) gives sublinear preprocessing; LogUp (Habök 2022) uses logarithmic derivatives — 3-4× fewer commitments than Plookup. Lasso (Setty-Thaler-Wahby 2023) extends with sum-check; Jolt is built around it. Halo2's "lookup over arbitrary sets" is now standard in Scroll, Taiko, Polygon zkEVM, Plonky3. Maturity: production-stable; LogUp and Lasso are dominant for 2025-2026 zkVM lookups (memory tables, range checks, bitwise ops).

### 12. Binius (Irreducible 2024) — EARLY-STAGE RESEARCH
Proof system over binary tower fields (GF(2^k) extension towers). Carry-less addition (XOR), hardware-friendly. Vitalik's April 2024 endorsement marked broad interest. Polygon×Irreducible partnership announced for a Binius-based zkVM; Jolt is migrating from Lasso to Binius for recursion; Ingonyama prototyping FPGA acceleration. FRI Soundness Above the Johnson Bound (eprint 2026/858) addresses one of its remaining theoretical gaps. Maturity: early-stage research with strong industry momentum; not yet in production.

### 13. Curves & fields — what's stable
- **BN254** (alt-bn128): Ethereum precompile, ~100-bit security post-exTNFS. Used wherever you must verify on-chain in Ethereum gas budget. Production but security-marginal.
- **BLS12-381**: Ethereum 2.0 BLS sigs, ZCash Sapling, Filecoin, Algorand, Chia, Dfinity. ~120-bit security. The default for new pairing-based work. Production-stable.
- **Pasta (Pallas/Vesta)**: Halo2/Mina cycle, no pairing, designed for recursion. Production-stable.
- **Goldilocks (p = 2^64 - 2^32 + 1)**: Plonky2, ~64-bit. Production.
- **BabyBear (15·2^27 + 1, ~31-bit)**: RISC Zero, SP1. Production.
- **KoalaBear (~31-bit)**: Plonky3 default in 2025-2026, faster NTTs than BabyBear.
- **Mersenne-31 (2^31 - 1)**: Stwo/Circle STARKs. Production at StarkNet 2025+.
- **Binary tower fields (GF(2^128))**: Binius. Research.

### 14. Industry deployments by proof system (May 2026 snapshot)
- **Aleo** — varuna/Marlin (universal-setup PLONK variant) on BLS12-377. Mainnet, programmable privacy.
- **Aztec** — Honk + Barretenberg + Noir DSL. Migrated off Halo2. Production testnet → mainnet 2025-2026.
- **Polygon** — zkEVM Type-1 on Plonky3; AggLayer pessimistic proof on SP1.
- **zkSync** — Boojum (STARK for inner, Groth16 wrap) on Goldilocks. Production.
- **StarkNet** — Stone → Stwo prover (Circle STARK, M31). Production.
- **Mina** — Kimchi (PLONK variant) on Pasta. Production.
- **ZCash** — Halo2 on Pasta (Orchard) and Groth16 on BLS12-381 (Sapling, legacy). Production.
- **Scroll, Taiko, Linea** — Halo2-KZG / variants on BN254 for L1 verification. Production.

### 15. Stable picks vs experimental (consolidated tier)
- **PRODUCTION (stable, audited, mainnet):** Groth16, PLONK+KZG, Halo2 (IPA & KZG), Plonky2, Plonky3, RISC Zero zkVM, SP1, Bulletproofs (range proofs), Plookup/LogUp/Lasso lookups, BN254, BLS12-381, Pasta, Goldilocks, BabyBear.
- **LATE-STAGE RESEARCH (real libraries, not yet mainnet-grade):** Nova / SuperNova / HyperNova / ProtoStar, Spartan / Hyrax, Jolt zkVM, Honk, KoalaBear+M31 small-field STARKs.
- **EARLY-STAGE RESEARCH (papers + prototypes):** Binius / binary-tower-field SNARKs, LatticeFold, post-quantum folding, WHIR, BaseFold, sum-check-acceleration ASICs.

## Reality positioning

### Recommendation: target Plonky3 + FRI as the canonical backend, with a KZG verification adapter for Ethereum
`zkmark` (the benchmarking slot) and `forge` (the proof construction slot, if it exists in plan ≥350) should standardize on:
1. **Plonky3** as the proving toolkit because it is (a) the only fully-open Apache-2 stack declared production-ready in April 2026 with broad industry adoption (Polygon, SP1, Valida), (b) PCS-agnostic (FRI for transparent / KZG for compact), (c) field-agnostic (Goldilocks / KoalaBear / BabyBear / Mersenne-31), (d) vectorized arithmetic (AVX2/AVX-512/NEON) which matches reality's "no allocations in hot paths" rule.
2. **KoalaBear or Goldilocks** as the default field for golden-vector generation — both are 31/64-bit primes mappable to `uint32` / `uint64`, NTT-friendly (slot 293), and avoid the BN254 security fragility.
3. **BLS12-381 + KZG** for a verification-only adapter so reality consumers (aicore, Pistachio) can sink a proof onto Ethereum via the existing precompile.
4. **Reject** wrapping Halo2 — too much PSE-specific Rust API surface, IPA proof sizes too large, and the Pasta cycle is single-vendor (ZCash) compared to Plonky3's multi-vendor field portfolio.
5. **Defer** Binius / LatticeFold / Nova-family folding to a later research slot; not stable enough for golden-file commitments.

### Cross-links to other slots
- **slot 175 (synergy-zkmark-crypto):** pairing-based SNARK arithmetic — Groth16/PLONK proof verification depends on `crypto` exposing BN254/BLS12-381 pairings, hash-to-curve (RFC 9380), KZG polynomial commitments, FRI/Reed-Solomon over small fields. This slot should include `crypto.PairingBN254`, `crypto.PairingBLS12381`, `crypto.HashToCurve`, `crypto.KZGCommit`/`KZGOpen`, and `crypto.FRICommit` golden vectors.
- **slot 200 (synergy-zkmark-info):** knowledge-soundness lower bounds map directly to `prob.MutualInfo` and `prob.KLDivergence` for simulator-extractor entropy gap — formalize as `zkmark.SimulatorEntropy(pk, vk, π) → bits` golden case.
- **slot 292 (new-elliptic-curves):** SEA point-counting and complex-multiplication construction are exactly how Pasta, BLS12-381, and BN254 were built. Reality should ship `geometry.PointCountSEA`, `geometry.CMConstruct`, and golden vectors for the standard curves at low bit-counts (toy params validate impl, real-curve params are golden constants).
- **slot 293 (new-ntt):** Plonky3's hot path is small-field NTT (Cooley-Tukey 8-radix, Harvey butterflies). `signal.NTT` and `signal.NTT2D` over Goldilocks/KoalaBear/BabyBear are the primitive that `zkmark` will spend 60-80 % of its CPU in. Negacyclic NTT for KZG.
- **slot 325 (dive-poseidon):** every modern SNARK uses an algebraic hash (Poseidon2, Rescue-Prime, Griffin, Anemoi) for in-circuit Merkle and Fiat-Shamir. `zkmark` cannot exist without `crypto.Poseidon2` golden vectors keyed to the field choice. Plonky3 ships Poseidon2 with KoalaBear params; reality should mirror those exactly.

### Concrete `zkmark` package shape (proposal)
```
zkmark/
  groth16/   - BN254 verifier-only (gas budget reference)
  plonk/     - KZG, BN254, universal-setup verify
  plonky3/   - small-field FRI, KoalaBear, the canonical hot path
  fri/       - low-level FRI commit/query/fold (golden vectors)
  kzg/       - KZG commit/open (golden vectors)
  poseidon/  - re-export of crypto.Poseidon2 with Plonky3 params
```
Golden vectors keyed to the upstream Plonky3 reference suite to guarantee cross-language parity.

## Sources
- [Polygon Plonky3 is Production Ready (30 Apr 2026)](https://polygon.technology/blog/polygon-plonky3-the-next-generation-of-zk-proving-systems-is-production-ready)
- [zkMesh: April 2026 recap](https://zkmesh.substack.com/p/zkmesh-april-2026-recap)
- [On the Security of Halo2 Proof System — Kudelski 2024](https://research.kudelskisecurity.com/2024/09/24/on-the-security-of-halo2-proof-system/)
- [The Pasta Curves for Halo 2 and Beyond — Electric Coin](https://electriccoin.co/blog/the-pasta-curves-for-halo-2-and-beyond/)
- [Succinct Ships: Optimized bn254 & bls12-381 Precompiles in SP1](https://blog.succinct.xyz/succinctshipsprecompiles/)
- [GitHub Plonky3/Plonky3](https://github.com/Plonky3/Plonky3)
- [GitHub succinctlabs/sp1](https://github.com/succinctlabs/sp1)
- [GitHub risc0/risc0](https://github.com/risc0/risc0)
- [Stealth Cloud — ZK proof performance benchmarks (March 2026)](https://stealthcloud.ai/data/zero-knowledge-proof-performance-benchmarks/)
- [KZG vs IPA vs FRI — zksecurity](https://blog.zksecurity.xyz/posts/pcs-survey/)
- [Nova attack of the year — zksecurity](https://blog.zksecurity.xyz/posts/nova-attack/)
- [LatticeFold (Boneh-Chen 2026)](https://link.springer.com/chapter/10.1007/978-981-95-5099-9_11)
- [Binary Tower Fields — Irreducible](https://www.irreducible.com/posts/binary-tower-fields-are-the-future-of-verifiable-computing)
- [Vitalik — Binius highly efficient proofs over binary fields](https://vitalik.eth.limo/general/2024/04/29/binius.html)
- [Once Upon a Finite Field — Goldilocks/BabyBear](https://blog.icme.io/small-fields-for-zero-knowledge/)
- [a16z crypto — Zero Knowledge Canon](https://a16zcrypto.com/posts/article/zero-knowledge-canon/)
- [GitHub microsoft/Spartan2](https://github.com/microsoft/Spartan2)
- [Awesome Folding — lurk-lab](https://github.com/lurk-lab/awesome-folding)
- [zk-Bench (eprint 2023/1503)](https://eprint.iacr.org/2023/1503.pdf)
- [What Is a ZK Rollup — A 2026 Guide](https://eco.com/support/en/articles/10080409-what-is-a-zk-rollup-a-2026-guide-to-zero-knowledge-scaling)
- [FRI Soundness Above the Johnson Bound (eprint 2026/858)](https://eprint.iacr.org/2026/858.pdf)
