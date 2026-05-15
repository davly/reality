# 350 — dive-fp-precision-modes (bf16 / fp16 / fp8 / fp32 / fp64 conversion audit + Block-D close-out)

## Headline
Reality is correctly fp64-canonical and should stay so; add a small `numfmt` package providing pure round-to-nearest-even conversions to/from BF16, IEEE FP16, and FP8 (E4M3/E5M2) for ML interop — accept these as **explicit conversion**, never as input scalars.

## Findings

### Existing footprint
- Zero references to `BFloat16`, `Float16`, `Float8`, `FP8`, `BF16`, `Posit`, or `bfloat` in any `.go` file (Grep confirmed; the 22 hits were `Positive`/`PositivePopulations` false-substring matches in comments).
- CLAUDE.md is unambiguous: "Zero dependencies. Only the language's standard math library." and Go is canonical fp64. Adding reduced-precision *types* is consistent with this; pulling in `golang.org/x/exp/mmreflect`/external bf16 packages is not.
- `physics/`, `linalg/`, `signal/`, `prob/` all return `float64`. No internal codepath needs to *operate* in bf16/fp16; what's missing is a clean **boundary primitive** for shipping data to/from ML consumers (Pistachio audio → on-device ML; aicore embeddings stored as bf16).

### What ML consumers actually use (2026 baseline)
- **BF16** (Wang & Kanwar, Google 2018, hardware in TPUv2+, AVX-512 BF16, ARMv8.6-A): 1+8+7. Same exponent layout as fp32 → conversion = top-half truncation + RNE rounding bias. Standard for transformer training.
- **FP16 / IEEE binary16** (IEEE 754-2008): 1+5+10. Shorter exponent → easy to overflow; needs loss scaling. Still dominant for inference on Apple Neural Engine, ARM NEON.
- **FP8 E4M3** (Micikevicius et al. 2022, NVIDIA Hopper, Intel Gaudi 3): 1+4+3. Used for weights & activations. **Non-IEEE**: no infinities, single NaN bitpattern (`S.1111.111`), max finite = 448, subnormals supported.
- **FP8 E5M2** (same paper): 1+5+2. Used for gradients. IEEE-compliant: has Inf and NaN, max finite = 57344, exponent bias 15.
- **Posits** (Gustafson 2017): tapered precision, niche; not adopted by any major hardware vendor as of 2026. Skip.

### Algorithmic content (round-to-nearest-even)
- **fp32 → bf16** (RNE): treat fp32 as `uint32`, add rounding bias `0x7FFF + ((u >> 16) & 1)`, shift right 16 to keep top 16 bits. NaN preservation: if input is NaN, OR a sticky bit into mantissa (else NaN can become Inf). This is the standard "TPU rule." See [VDPBF16PS](https://www.felixcloutier.com/x86/vdpbf16ps) for hardware semantics.
- **bf16 → fp32**: zero-pad 16 low mantissa bits (lossless; bf16 ⊂ fp32).
- **fp64 → bf16**: do `fp64 → fp32 (RNE) → bf16 (RNE)` — single composite is harder because bf16 exponent range matches fp32, not fp64; the fp32 step handles both range clipping and exponent rebiasing in one well-tested operation. Document the double-rounding caveat (it adds ≤0.5 ulp at the bf16 level).
- **fp32 ↔ fp16**: full IEEE algorithm — exponent rebias (127→15), mantissa shift, denormal handling on both sides, Inf/NaN preservation, RNE rounding. ~60 LOC. The reference implementation is Mike Cowlishaw's table-free version; closed-form via bit ops is well documented.
- **fp32 ↔ fp8 E4M3**: exponent rebias (127→7), mantissa truncate from 23→3 bits with RNE, saturation to ±448 (no Inf), single-NaN canonicalization. ~60 LOC. Subnormals at the E4M3 end have exponent bias-7 = -6 → smallest normal 2⁻⁶ = 0.015625; smallest subnormal 2⁻⁹.
- **fp32 ↔ fp8 E5M2**: same shape as fp16 conversion in miniature; 1+5+2, IEEE-style, supports Inf/NaN. ~60 LOC.

### Why "accept reduced precision as input" is the wrong framing
Reality is the **truth source**. Letting `physics.KineticEnergy(mass, velocity Float16)` exist would force every primitive to choose: silently up-cast (a lie about precision), or carry a precision-tagged type (combinatorial explosion across 22 packages, breaks the zero-dependency-stdlib rule). Correct posture: reality functions take fp64, period. **Conversion is an explicit, separate primitive** the caller invokes at the boundary. This matches numpy's `astype` semantics and PyTorch's `.to(torch.bfloat16)` — both treat reduced precision as a storage/transport concern, not a computation concern.

### Round-trip preservation properties (the testable invariants)
- `bf16(fp64(bf16(x))) == bf16(x)` for all finite x (idempotency under widening). Provable: bf16→fp64 is lossless, fp64→bf16 with RNE is deterministic.
- For `x ∈ fp64` with |x| in bf16 normal range: the round-trip `fp64 → bf16 → fp64` agrees with `x` to ≤2⁻⁸ relative (7 mantissa bits + implicit). Easy regression pin.
- For fp16: same with relative error ≤2⁻¹¹ in normal range, ≤2⁻¹⁴ for subnormals.
- For E4M3 normal range: ≤2⁻⁴ relative (3 mantissa bits + implicit); for E5M2: ≤2⁻³.
- BF16-of-fp32-of-x equals direct fp32→bf16 truncation when round bias is correctly applied → cross-check property.

## Concrete recommendations

1. **Create package `numfmt/` (new, ~250 LOC across 4 files)**:
   - `bf16.go` (~70 LOC): `type BFloat16 uint16`; `BFloat16FromFloat32(f float32) BFloat16`; `BFloat16FromFloat64(f float64) BFloat16`; `(b BFloat16).Float32() float32`; `(b BFloat16).Float64() float64`. Document RNE rounding, NaN propagation, lack of subnormals on TPU but support them in software (we keep them — software conversion has no perf reason to flush).
   - `fp16.go` (~80 LOC): `type Float16 uint16` with full IEEE 754 binary16 conversion. Cite [riscv-bfloat16-zfbfmin](https://github.com/riscv/riscv-bfloat16/blob/main/doc/riscv-bfloat16-zfbfmin.adoc) for the conformant exponent/mantissa table; cross-check against Go's experimental `golang.org/x/text/cases` style table approach by reimplementing from scratch.
   - `fp8.go` (~120 LOC): `type FP8E4M3 uint8`; `type FP8E5M2 uint8`. Conversions go through `float32` (E4M3 cannot be losslessly represented as fp16 in all cases — fp16 has bigger range but coarser at the top). Cite Micikevicius 2022 [arXiv:2209.05433](https://arxiv.org/abs/2209.05433).
   - `numfmt.go` (~20 LOC): package doc with the precision/range table for each format. No `Posit` — niche, no hardware, no consumer pulling on it.
2. **Golden-file test vectors per CLAUDE.md rule (≥20 per direction)**:
   - For each format: zeros (+0,-0), smallest subnormal, largest subnormal, smallest normal, 1.0, fp64 representable exactly in target (e.g. 0.5, 0.25), fp64 NOT representable (e.g. 0.1, 1/3) — golden encodes the rounded bit pattern, NaN (with payload), +Inf, -Inf, denormal-flush-to-zero edge, RNE tie cases (last bit 0 vs 1 to verify "even" rounding).
   - 30 vectors per `Convert` direction × 6 directions (fp64↔bf16, fp64↔fp16, fp32↔E4M3, fp32↔E5M2) = ~180 vectors. Negligible.
3. **R-MUTUAL-CROSS-VALIDATION 3/3 pin opportunities** (all 3-way agreement candidates):
   - **(a)** fp64 → bf16 → fp64 round-trip relative error ≤2⁻⁸ for all generated normals; cross-check #1: against `(uint32(math.Float32bits(float32(x))) + bias) >> 16` formula; cross-check #2: against torch.from_numpy(np.float64(x)).bfloat16().float64() reference vectors generated once in CI.
   - **(b)** BF16(fp32(x)) ≡ truncation-with-RNE-bias of fp32 mantissa: bit-level identity, three implementations agreeing (the fast bias-add path, a straightforward "decompose-round-recompose" path, and `math/big` rational rounding).
   - **(c)** FP16 conversion via direct bit manipulation ≡ via cast through fp32: both must produce same bit pattern for all 65536 inputs (exhaustive test feasible since |Float16|=2¹⁶).
   - **(d)** E4M3 saturation: encoding of 449.0 must equal bit pattern of 448.0 (max finite); cross-check via three derivations (paper Table 2, NVIDIA cuBLAS spec, our impl).
4. **Documentation precision contract** in package doc:
   - "Reality computes in fp64. Use `numfmt` only at I/O boundaries with ML/embedded systems. Conversions use IEEE 754 round-to-nearest-even unless otherwise noted (FP8 E4M3 saturates instead of overflowing to Inf, per NVIDIA spec)."
   - State that bf16 has the same dynamic range as fp32 (~10⁻³⁸ to ~10³⁸); fp16 saturates around ±65504; FP8 E4M3 saturates around ±448; FP8 E5M2 around ±57344.
5. **Do NOT add Posits**: Gustafson 2017 niche; no Hopper/Blackwell/Apple/Qualcomm hardware; no consumer in the Limitless stack. Listed for completeness — defer to T3/never.
6. **Cheapest day-1 PR**: T0 BFloat16 + T1 Float16 (~150 LOC + 60 vectors). Immediately useful for aicore embedding storage and Pistachio→ML audio feature export. T2 FP8 follows when first Hopper-targeted code lands.

### Tier table (for the slot's deliverable schema)

| Tier | Primitive | LOC | Day-1? | Justification |
|---|---|---|---|---|
| T0 | `numfmt.BFloat16` + conversion | ~70 | YES | Single most common ML interchange format 2024–26; trivial implementation; same exponent range as fp32 means low risk of silent overflow when consumers downcast |
| T1 | `numfmt.Float16` + conversion | ~80 | YES | Apple Neural Engine, ARM NEON, ONNX inference; needed any time outputs ship to mobile ML |
| T2 | `numfmt.FP8E4M3` + `FP8E5M2` | ~120 | NO | Wait for first Hopper/Blackwell consumer; spec-stable since Sep 2022 (ratified by NVIDIA/Intel/Arm) so no deprecation risk |
| T3 | `numfmt.Posit` | ~200 | NO | Frontier, no hardware traction; revisit only if a consumer asks |

### Cross-link consumers (who actually pulls)
- **Pistachio audio → ML**: feature tensors (mel, MFCC, log-spectrogram from `signal/spectrogram`) currently fp64 → consumer must bf16-cast for on-device transformer inference. Today they roll their own; reality should own this conversion.
- **aicore embedding storage**: 768-dim/1536-dim embedding cache currently dense fp32. BF16 halves storage; reality's `numfmt.BFloat16` lets aicore do this without a third-party dep.
- **Embedded sensor firmware** (if any): may ingest fp16 IMU samples from a peripheral; reality should provide widening conversion to fp64 before `physics`/`orbital`/`control` consume.
- **GPU mixed-precision training**: not a Limitless workload today; T2 FP8 deferred until it is.

## Block-D close-out

Block D (slots 301–350) was 50 deep-dive audits. Collective output:

- **Numerical hygiene laid down end-to-end.** Stable summation (302), rel-error documentation (303), Chebyshev approx (346), double-double (347), interval arith (348), summation algorithm benchmark (349), and this slot (350) form a coherent precision story — from how reality represents numbers internally (fp64 + selective DD), to how it bounds error per function, to how it ships data across the ML boundary.
- **Existing-code bugs surfaced via deep audits.** Prime test bug (306), spectrogram bilinear lie (312), CRT overflow (291) — these were not theoretical: they were live correctness failures hiding behind plausible-looking code. Block D's audit method (read code → cross-check against authoritative source → run targeted regression) reliably finds them.
- **Gap audits identified high-value missing primitives.** Kalman family (308 square-root, 309 information form), MPC/QP (332), trajopt (333), MPPI (334), CMA-ES (335), Bayesian opt (336), HRTF (339), room IR via image-source method (340), Lambert-Izzo (343), orbital perturbations (344), quaternion SLERP (341). Each came with concrete LOC estimates and tier ranking.
- **Cross-cutting blockers documented.** Number-theoretic transform (293) gates polynomial multiplication speedups, eigenvector recovery (097) gates spectral methods, normal sampling (309) gates Kalman covariance updates, RK4 buffer-pool design (333/334) gates allocation-free trajectory optimization. These blockers cascade — fixing them unlocks 3–6 downstream primitives each.
- **Consistent posture: reality stays minimal, pure, fp64.** Across 50 slots no reviewer recommended adding a heavyweight dependency, switching language, or breaking the "reimplement from first principles" rule. The recurring pattern: small, well-tested numerical primitives (~80–200 LOC each), each with golden-file vectors, each with R-MUTUAL-CROSS-VALIDATION 3/3 pins where tractable.
- **Testing methodology converged on 3-way cross-validation.** Slots 304/306/312/337/347/350 each independently arrived at the same pattern: implement primitive ≥3 ways (closed-form, iterative, reference-library-as-fixture), pin agreement at function boundary, treat divergence as regression. Block C established this; Block D operationalized it.
- **No Block-D reviewer recommended a Posit, IEEE 754-2019 decimal, or stochastic-rounding type.** Convergent signal that reality's scope (universal truth, not numerical-format research) is correctly bounded.

## Sources

### Repo
- `C:\limitless\foundation\reality\CLAUDE.md` — fp64 canonical rule, zero-deps rule
- `C:\limitless\foundation\reality\reviews\overnight-400\MASTER_PLAN.md:370` — slot 350 line
- Grep over `*.go`: zero pre-existing bf16/fp16/fp8/Posit references — clean greenfield for `numfmt` package

### Web
- [bfloat16 floating-point format — Wikipedia](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format)
- [BFloat16: The secret to high performance on Cloud TPUs — Google Cloud](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus)
- [Bfloat16 Precision Conversion and Data Movement — CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__MISC.html)
- [VDPBF16PS — Dot Product of BF16 Pairs (Intel ISA reference)](https://www.felixcloutier.com/x86/vdpbf16ps)
- ["BF16" Extensions for BFloat16-precision Floating-point — RISC-V](https://docs.riscv.org/reference/isa/unpriv/bfloat16.html)
- [riscv-bfloat16-zfbfmin spec — RISC-V GitHub](https://github.com/riscv/riscv-bfloat16/blob/main/doc/riscv-bfloat16-zfbfmin.adoc)
- [FP32/FP16 to BF16 Model Conversion — AMD Quark documentation](https://quark.docs.amd.com/latest/supported_accelerators/ryzenai/tutorial_convert_fp32_or_fp16_to_bf16.html)
- [Micikevicius et al. 2022 — FP8 Formats for Deep Learning (arXiv:2209.05433)](https://arxiv.org/abs/2209.05433)
- [NVIDIA, Intel, Arm Release High-Performance FP8 Format for Interoperable Deep Learning](https://www.hackster.io/news/nvidia-intel-arm-release-high-performance-fp8-format-for-interoperable-deep-learning-work-e047a26d314b)
- [Efficient Post-Training Quantization with FP8 Formats — MLSys 2024](https://proceedings.mlsys.org/paper_files/paper/2024/file/dea9b4b6f55ae611c54065d6fc750755-Paper-Conference.pdf)
- IEEE 754-2019 §3.6 (binary16 spec); Higham 2002 *Accuracy and Stability of Numerical Algorithms* 2nd ed. ch. 2 (rounding modes)
- Gustafson 2017, *Beating Floating Point at its Own Game: Posit Arithmetic* — listed for tier-3 deferral, not adopted

### Cross-slot
- 302 (stable summation), 303 (rel-err bounds), 347 (double-double), 348 (interval arith), 349 (summation benchmark) — sibling precision slots; this slot's `numfmt` lives at the opposite boundary (downward to ML) of 347's DD (upward to extended precision)
