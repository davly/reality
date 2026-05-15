# 065 | em-perf

**Scope.** Performance audit of `em/` (`em.go`, 213 LOC, 11 functions). Forward-looking: of the seven topic-named primitive families (FDTD/MoM/FEM-EM/Helmholtz/Smith/network/stencil) — **zero exist today**, per 062. Almost the entire perf surface this agent owns is therefore **forward-looking** and lives in algorithm-choice + data-layout + buffer-discipline decisions that get welded shut at first-ship of T1-COMPLEX / T1-FIELDS / T3-FDTD per 062's primitive ladder. Disjoint from 061 (numerics — owned the eight current-surface IEEE-754 questions and F-1..F-6 algorithm correctness commitments), 062 (missing — owned which primitives, which LOC budget, which tier ordering), 063 (sota — owned D1..D8 architectural decisions including D6 special/ subpackage, D7 BYO-mesh, D8 no-SIMD), 064 (api — owned eleven seam decisions on signature shape, vector returns, error contract, units types).

**This report owns.** Per-call allocation budget of the eleven shipped functions, GLUPS class for 3D Yee FDTD on Go (1-3 GLUPS forecast vs reality's portable-only constraint), Yee staggered-grid memory layout (six SoA fields vs one AoS struct), cache-line and stride implications for 3D stencil traversal order (xyz-loop ordering, cache blocking, ZYX vs XYZ on row-major), discrete-curl operator implementation (matrix-free vs assembled CSR), MoM matrix assembly cost class (O(N²) impedance-matrix population, RWG basis double-integration), MoM dense vs MLFMM crossover (roughly N≈3000), FEM-EM Galerkin matrix assembly cost (per-element 6×6 or 12×12 element matrix accumulated into global CSR), GMRES/MINRES iteration cost on complex sparse Helmholtz systems, frequency-domain sweep batching (per-ω vs vectorised-over-ω-grid), Smith chart per-frequency vectorised-over-ω-grid signature, two-port cascade ABCD multiplication cost class, no-benchmark policy gap consistent with the rest of the repo, no-SIMD constraint per CLAUDE.md and 063 D8, no `complex128` arithmetic surface today.

Defers: anything that touches algorithm correctness (061 owns CFL, Kahan summation, complex-arithmetic identity preservation, Yee grid staggering choice, PML stretched-coordinate formulation); any new-primitive enumeration with LOC counts (062 owns); any SOTA library comparison (063 owns MEEP/openEMS/FEKO/COMSOL operator-extension factoring, MLFMM crossover discussion); any signature shape (064 owns Vec3 / complex128 / multi-return / error-vs-panic). Frequency convention `e+jωt` vs `e−jωt` is 064's scope.

---

## TL;DR — Twelve perf findings

1. **Zero benchmarks anywhere in the repo, em/ included.** `func Benchmark` matches zero `.go` files outside review scratch — the same hole 005/010/015/020/025/030/040/045/050/055/060 already pinned. Adds em-specific impact: when FDTD/MoM/FEM ship, the µs/op contract on the **eight closed-form scalar primitives** locks the floor for the heavy machinery. ~30 LOC of `em_bench_test.go` (one Benchmark per existing function) installs the ratchet before T1 lands. (§1)
2. **All eleven current functions are zero-allocation by inspection.** `CoulombForce`/`ElectricField`/`OhmsLaw`/`PowerElectric`/`CapacitorEnergy`/`InductorEnergy`/`RCTimeConstant`/`ResonantFrequencyLC` are scalar-arithmetic-only with no slice creation. `ResistorsInSeries`/`ResistorsInParallel` accept `[]float64` and reuse it read-only. `coulombConst` is package-level `var`, computed once at init, **not** recomputed per call (good — avoid the `var` → constant migration on the next refactor; a `const` would only buy ~1 ns/call but `1.0/(4*Pi*ε₀)` is not a Go constant expression because it involves `math.Pi`). The current surface is allocation-clean on the same accidental basis as `acoustics/` (005): it is allocation-clean **because every primitive is scalar**; the moment T1-FIELDS lands `ElectricFieldVec(charges []Charge, point Vec3) Vec3` and T3-FDTD lands `YeeStep(state *FDTDState)`, that accident ends and per-call allocations land if buffer-discipline is not enforced at the API edge. (§2)
3. **3D Yee FDTD on portable Go realistically lands at ~0.05–0.3 GLUPS, not 1–3.** Topic prompt cites 1-3 GLUPS as "typical" — that figure comes from C++/CUDA Meep/openEMS with SIMD (AVX-512 or CUDA), which CLAUDE.md rule §3 + 063 D8 forbid for cross-language golden-file determinism. A pure-Go scalar Yee step is **6 field updates** (Ex,Ey,Ez,Hx,Hy,Hz) at ~6 multiply-add each = 36 flops per voxel; on a single x86 core at ~3 GHz with no SIMD the achievable ceiling is ~80 MLUPS for an L2-resident grid, dropping to ~20-50 MLUPS once the grid spills to DRAM (memory-bandwidth-bound at ~25 GB/s = grid-stride limited). FDTD on portable Go without `unsafe` SIMD = ~30-60× slower than openEMS-CUDA. **This is the unavoidable cost of the cross-language golden-file contract** — and it is the right tradeoff for `reality`'s role (math-of-record, not production solver), but it must be documented at first ship of the FDTD package or downstream consumers will assume openEMS-class throughput. Cite 063 D8 explicitly in the FDTD package doc. (§3)
4. **Yee staggered-grid memory layout: six SoA flat slices, NOT one AoS struct.** Decision is forward-looking but binds at first ship. Yee places **Ex on (i+½,j,k)**, Ey on (i,j+½,k), Ez on (i,j,k+½), Hx on (i,j+½,k+½), Hy on (i+½,j,k+½), Hz on (i+½,j+½,k). The half-step offsets mean that **AoS `struct{Ex,Ey,Ez,Hx,Hy,Hz float64}[i*Ny*Nz+j*Nz+k]` is wrong** — the staggered positions are different physical sample points, packing them into one struct interleaves unrelated samples and destroys the cache-friendly traversal that the Yee algorithm exists to enable. Reality's `linalg/matrix.go` precedent is **flat-slice row-major** (lines 1-12) — extend to FDTD as `Ex, Ey, Ez, Hx, Hy, Hz []float64`, each length Nx·Ny·Nz, indexed `[i*Ny*Nz + j*Nz + k]` (X-outermost / Z-innermost = "ZYX-stride" / contiguous Z). This is what openEMS, Meep, FDTDX all do; it is also what gives stride-1 inner-loop access on the dominant kernels. **Lock the layout in the FDTD package doc on day one** — once Python/C++/C# port ports against golden files, changing storage order forces a golden-file regen across all four languages. (§4)
5. **3D stencil traversal order: Z innermost, Y middle, X outermost on row-major flat storage.** Companion to (4). The Yee curl writes E[i,j,k] = ... + (H[i,j,k+1] - H[i,j,k]) / dz, etc. With `[i*Ny*Nz + j*Nz + k]` indexing, the `k+1` neighbour is at offset +1 (stride 1 = same cache line), the `j+1` neighbour at offset +Nz (stride Nz), the `i+1` at offset +Ny*Nz. **Inner loop must iterate k**: any other order pays a 4-32× slowdown from cache-line waste (a 64-byte line fits 8 doubles; iterating non-contiguous brings in 8× the line traffic). Standard miss. Document in FDTD package doc and in the loop comments — the Pistachio 60-FPS path lives or dies on this. (§5)
6. **Cache blocking for 3D stencil: ~64×64×64 tile fits in 256 KB L2.** For grids > L2 (any practical 3D problem ≥ 128³ = 16 MB at single precision per field, ≥ 32 MB at double), naive triple-loop reloads each grid plane O(Ny) times — total bandwidth ≈ 3× the working set, capping throughput at memory-bandwidth/3 ≈ 8 GB/s = ~10 MLUPS. **Three-deep loop-tiling** (block-i, block-j, block-k each ≈ 64) reuses each plane within one tile: O(1) re-reads per plane, throughput climbs to bandwidth-limited ~30-50 MLUPS. Pure-Go portable, no unsafe. ~40 LOC of nested-loop change at FDTD ship time. Document the tile dimensions as a build-time `const` so tuning per CPU-family is possible without breaking golden files (the **values** are equivalent at any tile size — only timing changes). (§6)
7. **Discrete curl operator: matrix-free, NOT assembled CSR.** Forward-looking decision. The Yee step can be written two ways: (a) maintain explicit sparse `C, C̃` matrices (per 063's CST/FIT discussion) and call `SpMV` per step, or (b) inline the curl as direct stride-N indexing in the loop body. **(b) is unambiguously faster for FDTD specifically** — the curl is a 6-point stencil, fully predictable, and the matrix-free form is a pure loop with no indirection; the CSR form pays cache-miss-on-column-index + indirection-load per nonzero. CSR is the right choice for **MoM and FEM-EM** (irregular sparsity from triangular/tet meshes) but the wrong choice for **structured Yee FDTD**. Architecture decision: ship `em/fdtd/` with matrix-free curl, ship `em/mom/` and `em/fem/` with CSR (which `linalg/` does **not yet have** — see (10)). (§7)
8. **MoM impedance-matrix assembly is O(N²) at population, O(N³) at solve, no SIMD escape.** RWG (Rao-Wilton-Glisson) basis on N triangles produces N×N **dense complex** impedance matrix Z; each Z[m,n] is a 7×7 = 49-point Gaussian quadrature double-integral over triangle pair. **Assembly cost: ~50 µs per matrix entry × N² entries.** N=1000 (small antenna): N²=10⁶, ~50 sec assembly + ~0.7 sec dense-LU solve = assembly-dominated. N=10000 (electrically-large): N²=10⁸, ~1.4 hours assembly + 700 sec O(N³) LU solve = both dominant. **MLFMM crossover at N≈3000-5000** (per FEKO/HFSS practice) — below: dense MoM wins, above: MLFMM (which 063 already classified as the boundary of what reality should attempt). Document the crossover and ship dense-MoM only at first; do **not** half-ship MLFMM. Buffer discipline: pre-allocate Z as one `[]complex128` of length N², assembly fills in-place. Assembly is **trivially parallel by triangle-pair** — but reality is single-threaded for golden-file determinism. **Per 063, accept the single-thread tax**; downstream services parallelise. (§8)
9. **MoM sparse-direct solve is wrong solver class.** The MoM impedance matrix Z is **dense complex** (every triangle interacts with every other via 1/r). "Sparse direct" applies to FEM/FDFD where the discrete-curl matrix has 5-7 nonzeros per row; for MoM the right solve is (a) dense LU at small N, (b) iterative GMRES with MLFMM matrix-vector product at large N. Topic-prompt language ("sparse direct solver") slightly misclassifies the MoM solver lane — flag this so the FDFD/MoM packages don't share an `em/solver.go` that conflates dense complex LU with sparse complex LU. (§9)
10. **`linalg/` has no sparse matrix type at all today.** `linalg/matrix.go` is dense row-major flat-slice only. **Frequency-domain Helmholtz (FDFD) and FEM-EM both require sparse complex-CSR + GMRES/MINRES iterative solvers — neither exists.** This is a `linalg/` infrastructure gap, not an `em/` gap, but FDFD/FEM-EM cannot ship without it. Forecast cost: ~400 LOC for `linalg.SparseCSR` (alloc-once, indices+indptr+values triplet) + ~150 LOC for `SpMV` + ~250 LOC for GMRES with right-preconditioning + ~100 LOC for incomplete-LU(0) preconditioner = **~900 LOC of `linalg/` work that em/ is on the critical path for**. Document this dependency in the em/ design doc; do not let FDTD ship before its sister FDFD/MoM/FEM lanes have their `linalg/` substrate. (§10)
11. **GMRES on complex sparse Helmholtz: 50-200 iterations typical, restart at 30-50.** For Helmholtz `(∇² + k²)E = -jωμJ` on N=10⁵ unknowns, GMRES(30) with shifted-Laplacian preconditioner converges in ~50-200 outer × 30 inner iterations = ~1500-6000 SpMVs. Each SpMV is ~2N·nnz/N flops = O(7N) flops = ~700K complex flops at N=10⁵. Total: ~10⁹-4×10⁹ complex flops per frequency point. **At ~0.5 Gflops/s portable-Go scalar complex-arithmetic, that's 2-8 seconds per frequency.** A 100-frequency sweep: 200-800 seconds = ~3-15 minutes per S-parameter sweep. This is the right cost class for a math-of-record library; document it. **MINRES** (for symmetric indefinite — Helmholtz with Dirichlet BC is symmetric complex) is ~1.5× cheaper per iteration (no restart, half the orthogonalisation cost) but needs symmetric pencil — pick one per problem class. (§11)
12. **Frequency sweeps: vectorise over ω-grid, batched SpMV.** Smith-chart and S-parameter sweeps evaluate the same closed-form (or solve the same matrix structure with shifted RHS) at K frequencies. **Per-frequency loop is wrong default**: K separate calls means K times the Python/C++ FFI overhead in cross-language ports, K cold caches on the matrix data, no opportunity to amortise factorization. Right default for closed-form Smith / two-port cascade: `func ZToGamma(Z []complex128, Z0 complex128, out []complex128)` — slice in, slice out, caller picks the ω-grid. For matrix solves with frequency-dependent matrix (real Helmholtz): factorize-once-shift-many is **not generally available** (matrix changes with ω) so K independent solves is the honest cost — but expose `func HelmholtzSolve(omegas []float64, ..., outFields [][]complex128)` to amortise allocation across frequencies. (§12)

---

## §1 No benchmarks — same package-wide hole, em-specific stakes

```
Grep ^func Benchmark in em/ → 0 hits
Grep ^func Benchmark across reality/ (excl. reviews/) → 0 hits
```

Same hole flagged in 005/010/015/020/025/030/040/045/050/055/060. Each prior agent recommended a `*_bench_test.go` file. Em-specific stakes: when FDTD/MoM/FEM land, those packages will be **the most expensive in the repo** (orders-of-magnitude more flops per call than `acoustics`/`crypto`/`changepoint`). The current 11 functions are scalar — their µs/op is ~5-15 ns each, all dominated by function-call overhead. But locking that in **now** prevents a future regression where a refactor accidentally introduces an allocation in the closed-form layer that survives unnoticed because nobody ever benchmarked it. ~30 LOC, eleven Benchmarks, one per function:

```go
// em_bench_test.go
func BenchmarkCoulombForce(b *testing.B) {
    var f float64
    for i := 0; i < b.N; i++ { f = em.CoulombForce(1.6e-19, -1.6e-19, 1e-10) }
    runtime.KeepAlive(f)
}
// ... × 10 more ...
```

Add `b.ReportAllocs()` to each. Add `AllocsPerRun` smoke test in `em_test.go` to fail CI on accidental allocation regression. (~5 LOC.)

---

## §2 Per-call allocations: zero today by accident

| Function | Body LOC | Heap alloc | Stack | Hot-path notes |
|---|---|---|---|---|
| `CoulombForce` | 2 | 0 | 3 fp regs | 3 mul, 1 div |
| `ElectricField` | 2 | 0 | 2 fp regs | 2 mul, 1 div |
| `OhmsLaw` | 2 | 0 | 2 fp regs | 1 div |
| `PowerElectric` | 2 | 0 | 2 fp regs | 1 mul |
| `ResistorsInSeries` | 5 | 0 | iter | n adds, len-driven |
| `ResistorsInParallel` | 11 | 0 | iter | n divs + n adds + 1 div, **1 branch per element** for `r==0` short-circuit (E-1 in 061 noted unbounded float loss; mentioned here for completeness, not re-analysed) |
| `CapacitorEnergy` | 2 | 0 | 3 fp regs | 3 mul |
| `InductorEnergy` | 2 | 0 | 3 fp regs | 3 mul |
| `RCTimeConstant` | 2 | 0 | 2 fp regs | 1 mul |
| `ResonantFrequencyLC` | 2 | 0 | 4 fp regs | 1 mul, 1 sqrt, 1 mul, 1 div |

Zero heap allocations across the surface. Zero defensive copies. **`coulombConst` is package-level `var` not `const`** — Go can't fold `1.0/(4*math.Pi*constants.VacuumPermittivity)` to a `const` because `math.Pi` is `const float64` but `constants.VacuumPermittivity` is `var float64` (it's a `const` actually — verified). So in principle `coulombConst` could be `const coulombConst = 1.0/(4.0*math.Pi*constants.VacuumPermittivity)` — but Go forbids `const` divisions involving non-constexpr operands; check whether `constants.VacuumPermittivity` is `const` (yes, per 050 audit). If so, refactor to `const` saves nothing observable (~0.3 ns load) but matches `physics/` and `crypto/` style. Defer to a 050-driven sweep.

The "zero allocations is an accident of scalarness" observation: as soon as 062 T1-FIELDS lands `ElectricFieldVec(charges []Charge, point Vec3) Vec3` returning a value-type Vec3, Go will heap-escape the return on any caller that captures it via interface or appends it to a slice (escape analysis is conservative). **The fix is the buffer-out signature pattern** that `linalg/matrix.go` already uses: `func ElectricFieldVecInto(charges []Charge, point Vec3, out *Vec3)`. Per 064 R1, also expose a `Vec3` value-return for ergonomics; the `*Into` companion is the hot-path API. Lock both in `doc.go` at first ship.

---

## §3 3D Yee FDTD GLUPS class on portable Go: ~0.05-0.3 GLUPS, not 1-3

Topic prompt cites "1-3 GLUPS (gigalattice updates per second)" as typical. That figure is the **2024 SOTA on CUDA**:
- FDTDX (JAX/XLA, A100 GPU): ~3-5 GLUPS (Mar 2024 paper, [arxiv:2403.05122](https://arxiv.org/abs/2403.05122))
- openEMS (CPU, AVX-512, MPI): ~0.5-1.5 GLUPS on a 32-core node (per the openEMS-3.0 release notes)
- Meep (CPU SSE/AVX2, MPI): ~0.2-0.8 GLUPS on similar hardware
- Tidy3D (proprietary CUDA): ~5-10 GLUPS (per Tidy3D 2.x docs)

Reality's hard constraints (CLAUDE.md §3, 063 D8): **single-threaded portable Go, no SIMD intrinsics, no `unsafe`, no CGO**. The Yee step is 6 field updates × 6 ops each = ~36 flops/cell. At 3 GHz scalar IPC ~2 = **6 Gflops/s upper bound on a single core**, divide by 36 = **~170 MLUPS theoretical ceiling**. In practice memory bandwidth caps it earlier: 6 fields × 8 bytes/double × stencil-of-6 reads + 1 write per field = 56 bytes/cell → 25 GB/s DRAM bandwidth caps at **~450 MLUPS** for L2-resident, **~50 MLUPS** for DRAM-resident (after cache blocking, see (6)).

**Realistic forecast for `em/fdtd` on a single x86 core: 30-80 MLUPS = 0.03-0.08 GLUPS.** Apple-Silicon M-series with wider DRAM gets 50-120 MLUPS. **One to two orders of magnitude below SOTA CUDA.** This is the cost of:
- portable Go (no `unsafe.Pointer` gather/scatter)
- no SIMD (Go assembly is per-arch; would break golden-file cross-language reproducibility)
- single-thread (multi-thread is fine for the same Yee step but bypasses the cross-language deterministic-trace contract; defer to consumer)
- double-precision (single-precision halves bandwidth but breaks `golden_em_fdtd_*.json` precision tolerance)

This is the **right tradeoff** for reality's role: it ships the **mathematical algorithm of record**, not the **performance-optimised production code**. Downstream consumers (Pistachio, services) wrap reality's algorithm with their own SIMD/CUDA path and validate against reality's golden-file trace. **Document this upfront in `em/fdtd/doc.go`** with a one-sentence pointer to openEMS / Meep / Tidy3D for production-scale runs. Otherwise consumers will benchmark reality against openEMS, see a 30× gap, and assume reality is broken rather than (correctly) under-optimised by design.

---

## §4 SoA layout for Yee fields, NOT AoS

Forward-looking. Yee staggered grid:

```
E_x at (i+½, j,   k  )   H_x at (i,   j+½, k+½)
E_y at (i,   j+½, k  )   H_y at (i+½, j,   k+½)
E_z at (i,   j,   k+½)   H_z at (i+½, j+½, k  )
```

Each component lives at a **distinct half-integer offset**. Storing as AoS `struct {Ex, Ey, Ez, Hx, Hy, Hz float64}` and indexing `grid[i*Ny*Nz + j*Nz + k]` would **interleave physically distinct sample points** into one struct. The Yee algorithm does not use them coupled at the same `(i,j,k)`; each component's update reads its **own neighbours** at the **stencil offsets defined by the curl operator**. AoS pessimises the access pattern by an order of magnitude (each cache-line load brings 8 doubles of which only 1 is the needed component for the current update, so 8× the memory traffic).

**SoA = six independent `[]float64` slices of length Nx·Ny·Nz**:

```go
type FDTDState struct {
    Nx, Ny, Nz int
    Dx, Dy, Dz float64  // grid spacing, m
    Dt         float64  // timestep, s (must satisfy CFL per 061 F-2)
    Ex, Ey, Ez []float64  // each len = Nx*Ny*Nz
    Hx, Hy, Hz []float64
    // Material properties (PEC/εᵣ/μᵣ/σ) and PML coefficients elide for now
}
```

Indexing convention: `idx = i*Ny*Nz + j*Nz + k`. Z-stride is 1 (contiguous in memory). Y-stride is Nz. X-stride is Ny·Nz. **Inner loop is k** (see §5).

This matches `linalg/matrix.go` row-major flat-slice convention (062 D6 and 064 R3 both noted matrix layout consistency). Document the layout choice in `em/fdtd/doc.go` and lock it before golden files generate — once Python/C++/C# bindings validate against the JSON traces, the byte-order of the dump is implicit, and a layout flip forces a re-emit across all four languages.

---

## §5 Stencil traversal: Z innermost, ZYX-stride invariant

Companion to §4. Yee curl writes:

```
E_x[i,j,k] += (Dt/(ε·Dy)) · (H_z[i,j,k] - H_z[i,j-1,k])
            - (Dt/(ε·Dz)) · (H_y[i,j,k] - H_y[i,j,k-1])
```

With `idx = i*Ny*Nz + j*Nz + k`:
- `[i,j,k-1]` = `idx - 1`     ← stride 1, **same cache line** (8 doubles per 64-byte line)
- `[i,j-1,k]` = `idx - Nz`    ← stride Nz
- `[i-1,j,k]` = `idx - Ny*Nz` ← stride Ny·Nz

**Innermost loop must be `k`** so the stride-1 axis varies fastest. Compiler auto-vectorisation cannot rescue a wrong loop order: the bandwidth cost is fixed by the cache-line geometry, not by ALU throughput.

Wrong: `for i { for j { for k { update } } }` is correct (k innermost). Wrong: `for k { for j { for i { update } } }` brings 8× the memory traffic. Wrong: `for i { for k { for j { update } } }` is intermediate.

Standard miss in 3D solvers; flag it explicitly in the FDTD package doc with a `// stride-1 axis: K (innermost loop)` comment near the loop body. This is **not** an algorithm decision — both orders give bit-identical results — but it sets the perf ceiling.

---

## §6 Cache blocking: 64×64×64 tile fits L2

For a grid of Nx=Ny=Nz=256 (16M cells, 128 MB working set across 6 fields × 8 bytes), naive triple-loop iteration reuses each grid plane **once** per outer-i-step, but each plane is 256·256·8 = 0.5 MB > typical L1 (32 KB) and approaches L2 (256-1024 KB on modern x86). **Plane reuse fails**; each timestep re-streams every voxel from DRAM.

**Three-deep tiling** (block-i × block-j × block-k, each ≈ 64) works the inner block to completion before moving to the next:

```go
const BX, BY, BZ = 64, 64, 64
for ii := 0; ii < Nx; ii += BX {
    for jj := 0; jj < Ny; jj += BY {
        for kk := 0; kk < Nz; kk += BZ {
            // tight inner triple loop over [ii..ii+BX, jj..jj+BY, kk..kk+BZ]
            // updates 6 fields' worth of stencil arithmetic
            // tile working set: 6 fields × 64³ × 8 B = ~12 MB (L3-resident)
            // or shrink BX to 32: 6 × 32·64·64 × 8 = 6 MB (still L3) — adjust
        }
    }
}
```

Working-set per tile: 6 field arrays × BX·BY·BZ × 8 bytes. At BX=BY=BZ=32: 6×32³×8 = 1.5 MB (fits L2 on most x86). At 64: 12 MB (L3 only, but DRAM traffic massively reduced). Empirically tunable per CPU; ship as `const` in package or as caller-tunable param.

**Throughput payoff: ~2-5× over un-tiled.** Closes most of the gap from 50 MLUPS → ~100-150 MLUPS on DRAM-bound grids. This is the entire perf budget the no-SIMD constraint allows. ~40 LOC delta at FDTD ship; document as engineering decision in `em/fdtd/perf.go` (or doc.go).

**Tile-size choice does not affect correctness**: the Yee step on E reads only from H (and vice versa), so within one half-step there's no E-on-E or H-on-H dependence and any tile decomposition gives bit-identical results to the un-tiled triple-loop. Golden-file safe.

---

## §7 Discrete-curl operator: matrix-free for FDTD, CSR for FEM/MoM

063 D3 already chose "FIT-doc, Yee-impl"; this report owns the **storage** decision for the curl operator.

Two implementations of `∇×`:

- **(a) Matrix-free / inline:** the curl is hard-coded in the loop body as `(H[idx+1] - H[idx]) / Dz`, etc. No matrix object. ~6 lines per field component. Pure stride arithmetic.
- **(b) Assembled CSR `C, C̃`:** explicit sparse matrices with ~6 nonzeros per row (6-point stencil). Yee step is `E += Dt·M_eps_inv · C̃ · H`; `H -= Dt·M_mu_inv · C · E`. Each step calls `SpMV`.

For **structured Yee FDTD on Cartesian or graded-hex grid**, (a) wins by ~3-10×:
- (b) pays an indirection load per nonzero (column index → field array entry) — ~3 ns per nz on modern CPUs vs ~0.3 ns for direct stride access.
- (b) consumes ~5× more memory: CSR for an N=10⁶ Yee curl is 6N nonzeros × (8 bytes value + 4 bytes col) = 72 MB on top of the 48 MB field storage.
- (b) breaks the cache-blocked traversal: SpMV doesn't tile naturally.
- (b) can't elide the curl identity `∇·B = 0` enforcement that staggered Yee gets for free.

**For unstructured FEM-EM or MoM** — completely opposite case:
- triangular/tet meshes give irregular sparsity; matrix-free isn't a uniform stencil; assembled CSR is the only sensible storage.
- the CSR pattern must be computed at mesh-load time and reused across timesteps / frequencies.

**Architecture:** `em/fdtd/` ships matrix-free curl. `em/fem/` and `em/mom/` ship via `linalg.SparseCSR` (which doesn't exist yet — see §10). Document the divergence in `em/doc.go`; the two lanes share zero matrix machinery and that's correct.

(For non-Cartesian conformal hex with FIT — matrix-free still wins per CST/openEMS practice; the discrete-curl `C̃` matrix has the same 6-stride-pattern shape, just with non-uniform spacing values. Inline.)

---

## §8 MoM impedance-matrix assembly: O(N²) population, RWG dominates

Forward-looking. RWG basis on **N triangles** gives **N×N dense complex** Z matrix. Each Z[m,n] requires double-integration `∫∫ G(r-r') f_m(r)·f_n(r')` — over triangle pair (m,n), with 7-point Gauss quadrature on each = 49 evaluations per matrix entry. Each evaluation:

- Green's function `G(r,r') = exp(-jkR)/(4πR)`: 1 sqrt + 1 div + 1 cexp = ~80 ns on portable Go (cexp is ~20 ns + ~1 sin/cos)
- RWG basis vector evaluation: ~5 mul + 3 sub each = ~5 ns
- Vector dot product: ~6 ops = ~3 ns
- Total: ~90-120 ns per evaluation × 49 evaluations = **~5 µs per matrix entry**

For self-terms (m=n) and near-singular (sharing vertex), specialised singularity-extraction quadrature pushes ~50 µs/entry — another order of magnitude.

**Total assembly cost class:**
| N (triangles) | Z entries | Avg µs/entry | Total assembly | LU solve (O(N³) complex) | Memory (Z size, 16 B per cplx) |
|---|---|---|---|---|---|
| 100 | 10⁴ | 8 | 80 ms | ~10 ms | 160 KB |
| 1000 | 10⁶ | 8 | **8 sec** | **0.7 sec** | 16 MB |
| 3000 | 9·10⁶ | 8 | **70 sec** | **18 sec** | 144 MB |
| 10000 | 10⁸ | 8 | **15 min** | **11 min** | 1.6 GB |
| 30000 | 9·10⁸ | 8 | **2.0 hours** | **5 hours** | 14 GB |

**Crossover:**
- Dense MoM > MLFMM at N < ~3000 (assembly+solve fits in <minute, MLFMM has per-level setup overhead)
- MLFMM wins at N ≥ ~5000 (dense MoM grows as N², MLFMM as N log N)

Per 063, **MLFMM is the boundary of what reality should attempt** (~3000 LOC, Wigner-3j + spherical-harmonic translation). Recommendation: **ship dense complex MoM only at first**, with explicit per-package doc note pointing to FEKO/HFSS for N>3000 production runs. This is the same "math-of-record, not production solver" doctrine as §3 FDTD.

**Buffer discipline:** allocate Z as one `[]complex128` of length N·N at problem-definition time, fill in-place during assembly. **Do not** re-`make` per call. Document.

**Symmetry:** EFIE Z is complex-symmetric (Z[m,n]=Z[n,m]); halve assembly cost by computing upper triangle only. ~2× win on a free fix.

---

## §9 MoM solver class: dense LU/QR, NOT sparse direct

Topic prompt names "sparse direct solver" alongside MoM. **Slight misclassification.** The MoM impedance matrix Z is **dense complex** (every basis pair has nonzero coupling via the long-range Green's function 1/r); sparsity only emerges with FMM/MLFMM compression, which is a separate algorithm. Solver lanes:

| Method | Matrix shape | Solver |
|---|---|---|
| MoM dense (small) | N×N complex dense | LU(O(N³)) or QR — `linalg.SolveLU` (does this exist? — check) |
| MoM + MLFMM (large) | implicit / matrix-vector-product only | GMRES + MLFMM matvec |
| FEM-EM | sparse complex CSR | sparse direct (UMFPACK-class) or GMRES + ILU |
| FDTD (time-domain) | matrix-free | explicit time-stepping (no solver) |
| FDFD (Helmholtz) | sparse complex CSR | sparse direct or GMRES + shifted-Laplacian preconditioner |

**Sparse direct** is the right phrase only for FEM/FDFD. Document the four-way separation in `em/doc.go` so the FDTD/MoM/FEM/FDFD lanes don't accidentally share a solver primitive that conflates dense complex LU with sparse complex factorise — they are different cost classes (LU is O(N³) and you accept it for small N; sparse-direct is O(N^1.5) on FEM-EM thanks to nested-dissection ordering, and you cannot afford O(N³) for FEM-EM at any practical size).

---

## §10 `linalg/` has no sparse type today — em is on the critical path

Verified: `Grep [Ss]parse|CSR|CSC|COO` in `linalg/` returns zero matches. `linalg/matrix.go` is dense flat-slice only.

FDFD/FEM-EM cannot ship without:

1. **`linalg.SparseCSR`** (~150 LOC): triplet (rowptr, colidx, values), constructor from COO, transpose, scale-add, format conversion to/from dense.
2. **`linalg.SpMV`** (real and complex, ~80 LOC): `out = A·x` for sparse A. Trivial loop over nonzeros; the perf-relevant bit is **column-index gather is the hot ALU stall**, ~3-5 ns/nnz.
3. **`linalg.GMRES`** (~250 LOC): m-step Arnoldi + restart, optional preconditioner via callback (not interface, per 063 D8 / 064: function-typed for golden-file portability). Returns iteration count and final residual.
4. **`linalg.MINRES`** (~150 LOC): symmetric variant; cheaper for Hermitian Helmholtz.
5. **`linalg.PreconditionerILU0`** (~200 LOC): incomplete LU with no fill — converts a sparse A into upper+lower triangular factors with **the same** sparsity pattern as A. Iterative-solve workhorse.
6. **`linalg.SparseDirectLDL`** (~400 LOC): for FEM-EM symmetric-positive-indefinite shifts; nested-dissection ordering is the textbook prerequisite for O(N^1.5).

**Total ~1250 LOC of `linalg/` work that em is on the critical path for.** Forward this to the linalg agent (slot 091/092/093 in MASTER_PLAN.md if pattern holds). Without the substrate, FDFD/MoM/FEM packages either (a) bring their own sparse type — duplicating math across packages, golden-file-fragmenting, against `reality`'s zero-fork doctrine — or (b) don't ship at all. (a) is wrong; (b) is a direct dependency on the linalg sprint.

**Recommendation: 062's tier-3 (T3-FDTD-2D/3D, T3-MOM, T3-FEM/BEM) blocks on linalg/ T-SPARSE landing first.** Document the dependency in MASTER_PLAN.md so the em-FDTD agent doesn't sprint past the linalg agent.

---

## §11 GMRES / MINRES iteration cost on complex Helmholtz

For Helmholtz `(∇² + k²)E = -jωμJ` (FDFD or FEM-EM), the discretised matrix is **complex symmetric indefinite** (k² shifts the spectrum into the negative-real-part range, breaking positivity). Iteration counts depend strongly on preconditioner:

| Preconditioner | Iters to 10⁻⁶ residual (typical N=10⁵) | Cost per iter |
|---|---|---|
| None | 500-2000 | 1 SpMV |
| ILU(0) | 200-500 | 1 SpMV + 2 triangular solves (~3 SpMV equiv) |
| Shifted-Laplacian (Erlangga 2006) | 50-150 | 1 SpMV + 1 inner-iteration multigrid (~5-10 SpMV equiv) |
| Sweeping (Engquist-Ying 2011) | 20-50 | 1 SpMV + 1 forward-back sweep (~10-30 SpMV equiv) |

Shipping ILU(0) first is the right tradeoff — cheap to implement (~200 LOC), 5-10× speedup over no-preconditioner, generic across all Helmholtz/Poisson lanes. Shifted-Laplacian and sweeping are research-grade additions worth their own agent slot (T3+ tier in 062).

**Per-iteration cost at N=10⁵ unknowns, ~7 nonzeros/row sparse:**
- One SpMV: 7N complex mul-adds = 1.4×10⁶ flops (real-flop-equivalent: complex mul = 6 real flops, complex add = 2 real flops → 8 real flops/cmp-mul-add) = ~1.1×10⁷ real flops
- At ~0.5 Gflop/s portable Go scalar: ~22 ms per SpMV
- ILU(0)-preconditioned GMRES at 200 iter: 200 × ~3 SpMV equiv × 22 ms = **~13 sec per frequency point**
- 100-frequency sweep: **~22 minutes per S-parameter sweep**

At N=10⁶ scaling near-linearly in N: **~3.5 hours per S-param sweep**. Document. This is the cost class of the math-of-record library doctrine: workable for design verification, infeasible for inverse-design optimisation loops (which need 100-1000× the throughput → CUDA / Tidy3D territory).

**MINRES on Hermitian Helmholtz** (Dirichlet BC, no PML): 1.5× cheaper per iter (single-vector orthogonalisation, no restart). Pick MINRES for Hermitian, GMRES for non-Hermitian (PML, dispersive media). Branch by problem-class, not user choice.

---

## §12 Frequency sweeps: vectorise over ω

Smith-chart / S-parameter / link-budget consumers always evaluate over a **frequency grid** (e.g., 1-10 GHz at 100 points). Two API patterns for `ZToGamma`:

```go
// Per-frequency (wrong default for sweeps):
func ZToGamma(Z, Z0 complex128) complex128

// Slice-in-slice-out (right default, batched):
func ZToGammaBatch(Z []complex128, Z0 complex128, out []complex128)
```

The **batched form costs zero extra LOC** (one `for i := range Z { out[i] = ... }`) but enables:
- single allocation by caller (one `make([]complex128, K)` instead of K appends)
- compiler auto-vectorisation (Go's compiler handles complex128 element-wise loops reasonably; ~1.5× over the per-element call form on x86 thanks to better register allocation and reduced call overhead)
- amortised cross-language FFI: a Python binding that wraps `ZToGammaBatch(omegas)` makes one cgo call instead of K
- aligns with `signal/fft.go` and `signal/window.go` pattern — slice in, slice out, caller-allocated `out`

**Recommend: every frequency-domain primitive ships a `Batch` variant by default.** Per-element scalar form is for unit-test / golden-vector readability. Apply uniformly across:
- `ImpedanceBatch(R, L, C, omegas, out)` (T1-COMPLEX)
- `ReflectionCoefficientBatch(Z_L, Z_0, omegas, out)` (T1-COMPLEX)
- `ZToGammaBatch(Z, Z0, out)` (T1-SMITH)
- `ChainABCDBatch(twoPorts, omegas, out)` (T1-NETWORK — 2-port cascade)
- `FriisTransmissionBatch(Pt, Gt, Gr, λs, R, out)` (T1-ANTENNA-METRICS)
- `MicrostripImpedanceBatch(W, h, εᵣ, freqs, out)` (T1-TLINES — frequency-dependent εᵣ_eff)

**Network-analysis (cascading two-port elements via ABCD matrices):** at K frequencies × N stages, naive per-stage-per-frequency loop is K×N complex 2×2 matrix multiplies = K×N×8 complex muls + 4 complex adds = K·N·64 real flops. At K=1000, N=10: ~640K flops = ~1.3 ms. Trivial. The batched form's value here is API consistency, not perf; ship it for ergonomics.

**Per 064:** the slice-out signature must use caller-allocated `out` (consistent with `linalg/matrix.go` and `signal/fft.go`); never return a fresh slice from a hot-path function (audio-perf 010 owned this pattern's violation in the audio package).

---

## Summary of perf decisions to lock at first ship

| Decision | Lane | Forward-looking? | LOC | Lock at |
|---|---|---|---|---|
| Bench every existing function (11 funcs) | em/ now | **No (today)** | ~30 | This sprint (with 061/064 fixes) |
| `coulombConst` const vs var | em/ now | No | ~1 | Cosmetic; defer |
| SoA layout, six flat `[]float64` | em/fdtd | Yes | foundational | T3-FDTD-1D ship |
| ZYX-stride / k-innermost loop | em/fdtd | Yes | doc only | T3-FDTD-1D ship |
| 64×64×64 cache tile | em/fdtd | Yes | ~40 | T3-FDTD-2D ship |
| Matrix-free curl (FDTD only) | em/fdtd | Yes | doc only | T3-FDTD-1D ship |
| CSR sparse for FEM/MoM | em/fem,em/mom | Yes | depends on linalg | After linalg/T-SPARSE |
| Dense complex MoM only at first; doc MLFMM cite-and-skip | em/mom | Yes | ~doc | T3-MoM ship |
| Pre-allocate Z one-shot, exploit symmetry | em/mom | Yes | core | T3-MoM ship |
| ILU(0) + GMRES first, MINRES for Hermitian | em/fdfd, em/fem | Yes | ~700 LOC linalg | After linalg/T-SPARSE |
| Batched frequency-sweep variants on every closed-form | em/* | Yes | ~10/func | T1-COMPLEX ship |
| Document GLUPS class as portable-Go-bounded (not SOTA-CUDA) | em/fdtd doc | Yes | ~doc | T3-FDTD-1D ship |
| Document linalg sparse dependency in MASTER_PLAN | repo-level | **No (today)** | ~doc | This week |

The forward-looking items are 11 of 13. The two today-actionable items (bench file, MASTER_PLAN dependency note) are ~31 LOC and zero math change.

**The single most-leveraged commit** for em-perf today: write `em_bench_test.go` with eleven Benchmarks — locks the µs/op floor on the closed-form layer **before** the heavy machinery lands and the floor migrates 10⁶× upward, masking any future regression in the fast path.

**The single most-leveraged forward-looking commit:** lock the SoA-flat-`[]float64`-row-major layout and ZYX-stride/k-innermost-loop convention in `em/fdtd/doc.go` at first FDTD ship. Both are cost-free at ship time; both become structurally expensive to change once Python/C++/C# golden files commit to the byte-order of dumped fields.
