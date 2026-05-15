# 397 — meta-precision-modes (fp32 vs fp64 policy)

## Headline
Stay fp64-canonical; do NOT add an fp32 API surface or `[T constraints.Float]` generics. Slot 350's recommendation (fp64 + boundary-only conversion package `numfmt` covering bf16/fp16/fp8) is the correct posture and should be ratified as policy in CLAUDE.md.

## Confirmation
- `MASTER_PLAN.md:423` — `397 | meta-precision-modes      | Precision modes: should reality accept fp32 or stay fp64?`
- Repo-wide grep over `*.go`: **zero** `float32`, zero `Float32`, zero `constraints.Float`, zero generic-float function signatures. The codebase is uniformly `float64`. Greenfield decision, not a refactor.
- CLAUDE.md is unambiguous: "Go is canonical; Python/C++/C# validate against golden files." Golden files encode `float64` bit patterns at 256-bit `math/big` precision; an fp32 surface would require either a parallel set of golden files or accept tolerance-laundering (the same anti-pattern slot 394 flagged for cross-language drift).

## Slot 350 recap (verbatim posture)
Slot 350 (`350-dive-fp-precision-modes.md:4`) concluded:
> "Reality is correctly fp64-canonical and should stay so; add a small `numfmt` package providing pure round-to-nearest-even conversions to/from BF16, IEEE FP16, and FP8 (E4M3/E5M2) for ML interop — accept these as **explicit conversion**, never as input scalars."

And on why fp32-as-input is wrong (`350:29`):
> "Reality is the truth source. Letting `physics.KineticEnergy(mass, velocity Float16)` exist would force every primitive to choose: silently up-cast (a lie about precision), or carry a precision-tagged type (combinatorial explosion across 22 packages, breaks the zero-dependency-stdlib rule). Correct posture: reality functions take fp64, period. Conversion is an explicit, separate primitive the caller invokes at the boundary."

This slot (397) ratifies that recommendation as the explicit, top-level **precision-modes policy**. It is not just about exotic ML formats; it is also about fp32, the option this slot was specifically asked to evaluate.

## Options analysis

### (a) fp64 only — current state
- **Pros**: Single representation. One golden-file per function. Tolerances tractable (slot 303 rel-err discipline applies uniformly). Cross-language validation (slot 394) only has to specify one bit-width. No combinatorial test explosion.
- **Pros (Go-specific)**: `math` stdlib is fp64. `math/big.Float` and `math.Float64bits` are the only stdlib precision tools. fp32 would force per-function reimplementation of every transcendental that currently delegates to `math.*`.
- **Cons**: Mobile/embedded callers that internally store data fp32 must up-cast at the boundary (one float64 conversion per sample → fine for control-rate, potentially measurable for audio-rate hot loops in Pistachio).
- **Cons**: Cannot directly call into AVX2/NEON fp32 SIMD lanes. Reality already takes no SIMD posture, so this is theoretical.

### (b) Parallel fp32 API surface (`KineticEnergy32`, `Dot32`, ...)
- **Pros**: Explicit. Type-checked. No generic-instantiation cost.
- **Cons (fatal)**: Every one of the ~600+ exported functions across 22 packages doubles. Golden files double (or worse: fp32 golden files cannot be derived from fp64 ones by truncation — RNE rounding interacts non-trivially with composite operations like Kahan summation). Maintenance burden roughly 1.8x for a feature with no internal consumer.
- **Cons**: Documentation explosion. Every function's "valid input range, precision, failure modes" (CLAUDE.md rule 5) must be stated twice with different rel-err bounds.
- **Cons**: Cross-language validation (Python/C++/C#) becomes a Cartesian product: 4 languages × 2 widths.
- **Verdict**: Reject. The cost-to-value ratio is wrong; no current consumer in the Limitless stack pulls hard enough on fp32 to justify it.

### (c) Generics: `Float[T constraints.Float]`
- **Surface assessment**: Go 1.18+ supports `constraints.Float` (`~float32 | ~float64`). Some libraries (e.g. Cogent Core's `floats32`/`floats64` modules, fragments of `gonum/floats`) experiment with this approach.
- **Pros**: Single source of code, two precisions for free at compile time. Each callsite picks its width.
- **Cons (fatal-1)**: Numerical algorithms don't transport across precisions for free. **Iteration counts, stopping tolerances, and step sizes all depend on `eps`.** Newton's method, L-BFGS, simulated annealing (`optim/`), RK4 step control (`chaos/`, `calculus/`), Sabine RT60 convergence (`acoustics/`), Bode peak-finding (`control/`) — every iterative routine either needs a `T`-typed `eps` constant (achievable but verbose) or has a hard-coded fp64-style tolerance that silently fails for fp32. The first style works; the second is the bug factory.
- **Cons (fatal-2)**: Internal use of `math.*` is not generic. Every `math.Sin`, `math.Sqrt`, `math.Log` would need a polymorphic wrapper. Go has no generic `math` stdlib in 2026; we'd be reimplementing transcendentals or boxing through fp64. Reimplementing transcendentals from first principles for fp32 violates the spirit of "reimplement from first principles" (rule 6) — in practice we'd ship two copies, which is option (b) wearing a generic hat.
- **Cons (fatal-3)**: Golden-file tolerance per-instantiation. The same code instantiated at `T=float32` produces different bit patterns than `T=float64`; tolerance would have to be encoded per-(function, T)-pair. Slot 350's "per-function tolerance" guidance does not extend cleanly to per-precision tolerance without doubling the golden files.
- **Cons (fatal-4)**: Slot 394 (cross-language) found JSON test vectors lack a `bits` field today. Adding generics doubles the bit-width axis we don't yet specify.
- **Pros (only one I can defend)**: A *future* `linalg/floats32` companion package — pure vector ops, no transcendentals, no iterative tolerances — could plausibly use generics or simply ship as a parallel sub-package. This is a narrow exception, not a precedent for reality-wide generics.
- **Verdict**: Reject as the global policy. Permit a narrow `linalg/floats32` (BLAS-1 only, no decompositions, no transcendentals) if and only if a Pistachio-grade consumer materializes. Treat this as a separate, justified slot — not a default.

### (d) fp64 canonical + fp16/bf16 IO conversions only (slot 350 recommendation)
- **Pros**: Honors fp64 internal computation rule (CLAUDE.md). Adds a single ~250 LOC `numfmt/` package with bf16/fp16/fp8 round-to-nearest-even conversions.
- **Pros**: Conversion package has no internal API consumers — it is purely boundary I/O. Adding it does not perturb any existing function signature.
- **Pros**: Covers the *actual* ML-mobile use cases identified in slot 350 §"Cross-link consumers": Pistachio audio→ML feature export, aicore embedding storage as bf16, embedded sensor firmware ingestion. None of those consumers want fp32 — they want bf16 for storage compression and fp16 for Apple Neural Engine.
- **Pros**: fp32, when it appears, can pass through `numfmt`-style helpers (`Float32FromFloat64(x float64) float32` is a one-liner Go cast wrapped for symmetry) without committing the rest of the library to a parallel surface.
- **Cons**: Doesn't help a hypothetical caller who wants pure fp32 *computation* (e.g. running `fluids.Reynolds` on a GPU shader's fp32-only register file). Reality has no GPU posture; this is a non-consumer.
- **Verdict**: Adopt. This is the policy.

### (e) Boxed `Float` interface
- Performance hit (~5-10x for arithmetic). Method-call dispatch in tight loops kills 60-FPS hot paths (CLAUDE.md rule 3: "No allocations in hot paths"). Every interface method takes/returns a non-inlineable boxed value.
- Design-rule violation: introduces an indirection layer that contradicts "minimal, pure" posture (slot 350 close-out).
- **Verdict**: Reject. Not a credible option for a numerical truth library. Listed for completeness only.

## Peer survey (2026 baseline)

| Library | Native scalar | Reduced precision | Notes |
|---|---|---|---|
| **gonum** (Go) | fp64 | Limited fp32 BLAS via codegen | "float32 code is not tested in practice due to behavioral differences" — gonum-dev mailing list. Effectively fp64-canonical. |
| **statrs** (Rust) | fp64 | None | Pure fp64 distributions. No generic precision. |
| **SciPy** (Python) | fp64 default | fp32 supported per-array via `dtype` | Recent scipy-dev discussion: "scipy assumed default floating-point type is double-precision float64" — there is active *tension* (torch/jax default fp32) but SciPy's policy is still fp64 with fp32 opt-in via numpy dtype machinery, not a parallel API. |
| **NumPy** (Python) | fp64 default | All IEEE widths via dtype | Dtype is an array-level attribute, not an API-level split. Reality has no array-of-array runtime; this pattern doesn't transfer cleanly to a Go function-call API. |
| **PyTorch / JAX** | per-tensor dtype | fp32/fp16/bf16/fp8 first-class | Tensor dtype is *runtime*-dispatched. Their reality-equivalent is Triton/CUDA kernel codegen, not a hand-written numerical library. Not analogous. |
| **Eigen** (C++) | template `Scalar` | float, double, long double, complex | Template-driven (option (c) in C++). Pays the per-instantiation cost in compile time and binary size. Eigen has full BLAS, no transcendental-heavy iterative solvers above LU/QR — narrower scope than reality's 22 packages. |
| **Boost.Math** (C++) | policy-templated | All IEEE widths + multiprecision | Most aggressive precedent for option (c). Cited consequence: 30k+ LOC of policy machinery, slowest C++ template compile in the ecosystem. Wrong tradeoff for Go. |

**Synthesis**: The mainstream pattern for hand-written numerical libraries (gonum, statrs, SciPy core, Boost.Math defaults) is fp64-canonical. Per-tensor dtype dispatch is an array-runtime feature (NumPy/PyTorch/JAX), not an API-design pattern. Templated Scalar (Eigen, Boost) pays for it in compile time, binary size, and tolerance specification — an acceptable tradeoff in C++ where templates are idiomatic; less so in Go where generics are still narrowly used.

## Recommendation

**Adopt option (d) as policy. Document it explicitly.**

Concrete actions:

1. **Amend CLAUDE.md** — add a new "Precision Policy" section under "Key Design Rules":
   > "Reality computes in `float64`. `float64` is the only floating-point type accepted by any reality function as input or returned as output. Reduced-precision formats (bf16, fp16, fp8 E4M3/E5M2, fp32) are accessed via the `numfmt/` package as explicit, RNE-rounded boundary conversions. There is no fp32 API surface, no generic-float instantiation, and no boxed `Float` interface. Callers requiring reduced-precision storage or transport call `numfmt.BFloat16FromFloat64(x)` (etc.) at the boundary; callers requiring reduced-precision *computation* are out of scope for reality."
2. **Land slot 350's `numfmt/` package** (T0: bf16, T1: fp16, ~150 LOC + 60 golden vectors). This is the affirmative half of the policy — fp64 in, named-format-uint out. Independently justified by slot 350; this slot ratifies it.
3. **Add `numfmt.Float32FromFloat64(float64) float32` and `numfmt.Float64FromFloat32(float32) float64`** as ~5-LOC helpers. Symmetric with bf16/fp16. Documents that fp32 is treated as a transport format like the others, not a computation type. Closes the "what about fp32?" question explicitly rather than leaving it implicit.
4. **Reject any future PR that introduces** (a) a `*Float32` parallel function, (b) a `[T constraints.Float]` generic in a reality package outside `linalg/floats32` (if that ever lands as its own slot), or (c) a `Float` interface boxing layer. Cite this slot (397) and slot 350 in the rejection.
5. **Narrow exception, pre-blessed**: A `linalg/floats32` companion package may be added if a Pistachio-grade consumer requires BLAS-1 fp32 for SIMD/SIMT hot loops. Constraint: BLAS-1 only (vector ops, dot, axpy, norm). No transcendentals. No iterative solvers. No generics — parallel `float32`-typed functions, naming convention `Dot32` etc., living entirely in `linalg/floats32/`. This is a separate slot decision, not a policy default.
6. **Cross-language test vectors** (per slot 394): JSON schema continues to encode fp64 only. `numfmt` golden files encode the target format's bit pattern (uint16/uint8) directly, not as a float — sidesteps the cross-language float-equality problem.
7. **Documentation surface**: Each package's doc.go gains a one-line statement: "Inputs and outputs are `float64`. See `numfmt/` for boundary conversion to bf16/fp16/fp8/fp32." Once. Centralized.

## Why this is the right answer

- **Truth-source posture (CLAUDE.md preamble)**: "Universal truth encoded in code." Truth is precision-bounded; mixing precisions at the API surface is precisely what makes a numerical library *less* truthful, not more flexible.
- **Test-infrastructure posture (CLAUDE.md "Golden-File Testing")**: Golden files at fp64 + 256-bit `math/big` reference are tractable and already established. Doubling the precision axis multiplies maintenance without adding evidence of correctness.
- **Consumer reality (slot 350 §"Cross-link consumers")**: Pistachio + aicore want bf16/fp16 for **storage and transport**, not for computation. They compute in fp64 then ship in bf16. Option (d) serves this exactly. No consumer demands fp32-native computation today.
- **Industry pattern (peer survey)**: gonum / statrs / SciPy-core / mainstream Rust numerical crates are fp64-canonical. Reality's posture is the *consensus* posture for hand-written numerical libraries; adding fp32 would diverge from peers, not catch up to them.
- **Block F (meta) consistency**: Slot 391 (units), 392 (time), 393 (frame), 394 (cross-language), 395 (vendor-deps), 396 (build-targets) all converged on "narrow scope, document the convention, reject sprawl." This slot's recommendation matches that pattern.

## Sources

### Repo
- `C:\limitless\foundation\reality\CLAUDE.md` — fp64-canonical statement, golden-file test rule, "no allocations in hot paths" rule
- `C:\limitless\foundation\reality\reviews\overnight-400\MASTER_PLAN.md:423` — slot 397 line
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\350-dive-fp-precision-modes.md` — sibling deep-dive; this slot ratifies its recommendation
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\394-meta-cross-language.md` (per PROGRESS.md tail) — JSON vector schema lacks `bits` field; relevant to why doubling the precision axis is premature
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\395-meta-vendor-deps.md` — zero-deps confirmed; relevant to rejecting any third-party generic-float library
- Grep `float32|Float32|constraints.Float` over `**/*.go` in repo root: **0 hits**, confirming greenfield

### Web
- [Gonum for types other than float64 — gonum-dev mailing list](https://groups.google.com/g/gonum-dev/c/QKMmB76FV30) — "float32 code is not tested in practice due to behavioral differences"
- [gonum repo README](https://github.com/gonum/gonum) — fp64-canonical posture
- [Should scipy tests assume float64 as the default floating-point dtype? — scipy-dev forum](https://discuss.scientific-python.org/t/should-scipy-tests-assume-float64-as-the-default-floating-point-dtype/1606) — confirms SciPy treats fp64 as default; per-array dtype is the opt-in mechanism, not a parallel API surface
- [NumPy v2.4 Data types manual](https://numpy.org/doc/stable/user/basics.types.html) — fp64 default, dtype machinery
- [Go Generics tutorial — go.dev](https://go.dev/doc/tutorial/generics) — `constraints.Float` definition (`~float32 | ~float64`)
- IEEE 754-2019 §3.6 — binary precision specifications
- Higham 2002, *Accuracy and Stability of Numerical Algorithms* 2nd ed. ch. 2 — why iterative-tolerance constants don't transport across precisions (kills option (c))

### Cross-slot
- 350 (fp-precision-modes deep dive) — recommendation source for option (d)
- 303 (rel-err documentation), 347 (double-double), 348 (interval arith), 349 (summation benchmark) — sibling precision slots, all assume fp64 internal
- 391–396 (meta block F prior slots) — convergent "narrow scope, document convention, reject sprawl" pattern this slot continues
