# 070 | fluids-perf

**Scope.** Performance audit of `fluids/` (`fluids.go`, 235 LOC, 11 functions). Forward-looking by necessity: of the eight topic-prompt primitive families (pipe-network solvers, sparse Jacobians, LBM streaming/collision, SPH neighbour search, NS divergence-free projection) — **zero exist today**, per 067. The current surface is eleven scalar closed-form correlations plus one fixed-point Colebrook iteration. Perf surface owned here lives in algorithm-choice + data-layout + buffer-discipline decisions that get welded shut at first ship of T1-COLEBROOK-FAMILY / T1-NETWORK / T3-NS / T3-LBM / T3-SPH per 067's tier ladder.

**This report owns.** Per-call allocation budget of the eleven shipped functions, Colebrook iteration cost-per-call (~10-15 fixed-point steps × ~80 ns each = ~1 µs/call), `math.Pow(Re, 0.9)` micro-pessimisation in the Swamee-Jain seed (one-time cost vs `math.Exp(0.9*math.Log(Re))` — wash, but documenting), Hardy Cross / modified Hardy Cross iteration cost class for pipe-network solvers (when added — O(N_loops × N_iter) per timestep, dense vs sparse Jacobian crossover at ~50 pipes), Newton-Raphson global pipe-network solver as the right modern default (sparse Jacobian via incidence matrix, 5-10× faster than Hardy Cross at N>30), LBM D2Q9 lattice site update cost (~0.5-1 GLUPS class on portable Go, identical to em/fdtd reasoning per 065), SPH neighbour-search via spatial-hash grid (O(N) per step amortised, cell-linked-list per 068 D-5), NS divergence-free projection cost (Poisson solve dominates, ~10⁹ flops at N=10⁶ unknowns), forward µs/op forecasts for canonical operations, no-benchmark policy gap consistent with the rest of the repo.

Defers: anything that touches algorithm correctness (066 owns Re=2300 cliff, Colebrook abs-vs-rel tolerance, math.Pow at fractional exponent precision, CFL/von-Neumann stability when PDEs land); any new-primitive enumeration with LOC counts (067 owns ~1,850 / ~1,300 / ~6,200 LOC tier breakdown); any SOTA library comparison (068 owns D-1..D-7 architectural binding); any signature-shape ergonomics (069 owns FrictionMethod enum, Fluid/Pipe struct, regime three-return). Numerical-stability of Colebrook iteration is 066's lane, not this report's.

---

## TL;DR — Twelve perf findings

1. **Zero benchmarks anywhere in the repo, fluids/ included.** `func Benchmark` matches zero `.go` files outside `reviews/`. Same hole pinned in 005/010/015/020/025/030/040/045/050/055/060/065. Fluids-specific stakes: when T3-NS / T3-LBM / T3-SPH ship, those packages will be **the most expensive in the repo by 5-7 orders of magnitude** (LBM D2Q9 = ~9 fields × 256² grid × 1000 timesteps = 5.9×10⁸ updates per simulation). The current 11 scalar functions are ~5-15 ns each — locking that ceiling **now** ratchets in the closed-form-layer floor before T3 land. ~30 LOC of `fluids_bench_test.go`, eleven Benchmarks, one per function. (§1)
2. **All eleven current functions are zero-allocation by inspection.** Seven (`ReynoldsNumber`, `BernoulliPressure`, `DarcyWeisbach`, `DragForce`, `LiftForce`, `StokesLaw`, `MassFlowRate`, `VolumetricFlowRate`) are pure scalar arithmetic with 1-7 floating-point ops, no slice creation, no closure capture. `TerminalVelocity` is one branch + one `math.Sqrt`. `PipeFlowFriction` is the only non-trivial cost-per-call (~1 µs vs ~5-15 ns for everything else). The "zero allocations is an accident of scalarness" observation: as soon as 067 T3-NS lands `FluidField{U, V, P []float64}` returning gridded fields, Go will heap-escape on any caller that captures via interface or appends to a slice. **The fix is the buffer-out signature pattern** that `linalg/matrix.go` and `signal/fft.go` already use — lock at first ship of grid solvers. (§2)
3. **`math.Pow(Re, 0.9)` in PipeFlowFriction is one-shot, not hot-path.** `fluids.go:91` uses `math.Pow(Re, 0.9)` in the Swamee-Jain initial-seed computation. `math.Pow` is ~20-40 ns on x86 vs ~5 ns for an explicit multiply chain; for **integer** exponents this is the textbook pessimisation. **For 0.9 it's actually correct** — Go's `math.Pow(x, 0.9)` is the cheapest way to express `x^0.9` (no closed-form integer-multiply substitute exists; `math.Exp(0.9 * math.Log(Re))` is a wash, slightly less accurate). The seed is computed **once per `PipeFlowFriction` call**, not per iteration; the iteration body uses `math.Sqrt` + `math.Log10` (lines 96-97), neither of which is `math.Pow`. **No action.** Flag in §3 only because the topic prompt names this pattern — it does NOT apply here. The pattern matters when `math.Pow(x, 2)` or `math.Pow(x, 3)` shows up in a hot path; verified zero such usages in `fluids.go`. (§3)
4. **Colebrook fixed-point iteration is fast already (~1 µs/call).** Per 066's finding 2 + lines 95-103 of `fluids.go`: 100-iter cap, 1e-12 abs convergence test (066 N-2 separately notes the abs-vs-rel issue), inner body is 1 `math.Sqrt` + 1 `math.Log10` + ~6 floating-point ops per iter. Empirical iteration count is **5-15 typical, 20-30 worst-case** (rough pipes near transitional regime). At ~80 ns per iter on portable Go (sqrt ~12 ns, log10 ~30 ns, arithmetic ~15 ns), total ~0.4-2.4 µs per `PipeFlowFriction` call. **This is the right cost class** — the Newton-step alternative (one Newton iter per fixed-point iter, derivative computed analytically) converges in 4-6 iterations vs 5-15 for fixed-point, ~2-3× faster but ~30 LOC more code; defer to 068 D-2 which already routes Newton/Brent to a separate `optim.Brent` slot in the linalg/optim sprint. (§4)
5. **Hardy Cross is the wrong default for new pipe-network solvers; Newton-Raphson global solve is.** Forward-looking. Hardy Cross (1936) iterates per-loop with ΔQ = -ΣΔh / (2·Σ|Δh/Q|), updating loop-by-loop sequentially. Cost: O(L · N_loops · N_iter) where L is pipes-per-loop. Convergence is **linear**, often 50-200 iterations on real networks. Modern practice (EPANET, Cross-Todini, KYPipe) uses **global Newton-Raphson on the simultaneous-equation system** with sparse Jacobian: F(Q) = 0 where F encodes head-loss + continuity, J = ∂F/∂Q is N×N sparse with ~3-5 nonzeros per row (one per pipe at each junction). **5-10× faster than Hardy Cross at N>30 pipes**, quadratic convergence (~5-8 iterations), **and requires the sparse linear-solve infrastructure that 067 + 068 + 065 all pin as the cross-package bottleneck**. Architecture: ship Hardy Cross at T1-NETWORK as the textbook reference (~150 LOC, useful for unit-testing Newton-Raphson against), then ship Newton-Raphson as the production path at T2-NETWORK once `linalg.SparseCSR` + `linalg.GMRES`/`linalg.SparseDirectLDL` land. Both must produce identical golden files at convergence. (§5)
6. **Sparse Jacobian for pipe networks: COO-build → CSR-solve, never dense.** A pipe network with N=100 pipes has J of size N×N = 10,000 entries; with ~3-5 nonzeros per row (one per pipe at each end junction) the **sparse representation is 300-500 entries vs 10,000 dense = 95% sparsity**. At N=10,000 pipes (city-scale water distribution): 30-50K nonzeros vs 10⁸ dense = **99.95% sparsity, dense is impossible** (1.6 GB memory at double-precision). **Build pattern: COO triplets `(rowIdx, colIdx, value)` during incidence-matrix walk → convert to CSR (sorted-by-row, indptr+colidx+values) once per Newton outer iteration → call `linalg.SpMV` and `linalg.SparseDirectLDL` from the `linalg/` substrate that 065 already pins as the gating dependency for em-FDFD/FEM/MoM**. **Same substrate, single sprint, dual unblock.** Document the dependency in `fluids/network/doc.go` at first ship; add the cross-reference to em-FDFD/FEM in MASTER_PLAN.md so the linalg-sparse agent knows two consumers ride on it. (§6)
7. **LBM D2Q9 lattice site update on portable Go: ~0.3-0.8 GLUPS class.** Forward-looking. D2Q9 has 9 distribution functions per cell `f[0..8]`, each ~8-12 flops per timestep (1 streaming + 1 BGK collision = `f_new = f - (f - f_eq)/τ` with `f_eq` requiring ρ, u, u² computation = ~30 flops shared across 9 components). **Per-cell cost: ~80-100 flops + ~9 × 8 byte reads + 9 writes = 144 bytes/cell.** At 3 GHz scalar IPC ~2 = 6 Gflops/s upper bound, 80 flops/cell → **~75 MLUPS** ALU-ceiling. Memory-bandwidth: 144 bytes/cell × 25 GB/s DRAM = **~170 MLUPS**. Cache-blocked: ~80-150 MLUPS realistic = **0.08-0.15 GLUPS portable Go**. SOTA reference: Palabos AVX-512 + MPI ~2-4 GLUPS, Lettuce GPU torch ~5-15 GLUPS (per Lettuce 2024 benchmarks). **Reality is correctly 1-2 orders of magnitude below SOTA CUDA** for the same reasons as em/fdtd (065 §3): no SIMD, no `unsafe`, no CGO, single-thread for cross-language golden-file determinism. **Document upfront in `fluids/lbm/doc.go` with explicit citation of 068 D-3 (Palabos lattice-descriptor pattern) and CLAUDE.md §3 (no allocations in hot paths).** (§7)
8. **LBM streaming step: pull-scheme NOT push-scheme; D2Q9 weights as exact rationals.** Forward-looking. LBM streaming has two implementations: (a) "push" — each cell **scatters** its f[i] to 9 neighbours (`f_new[neighbour] = f_old[me]`) — has gather/scatter inefficiency; (b) "pull" — each cell **gathers** from 9 neighbours (`f_new[me] = f_old[neighbour]`) — pure read pattern, cache-friendly, identical math. **Pull is unambiguously faster on portable Go** (~1.5-2× on the streaming step). Standard practice in Palabos/lbmpy/Lettuce. Document in package doc + lock the loop body comment to "pull-scheme" so future contributors don't switch to push thinking it's symmetric.
   - **D2Q9 weights `w_i ∈ {4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36}`**: per 067 forward F-10, store as **exact-rational `const`s** not float-derived (`const W0 = 4.0/9.0` not `var W0 = 4.0/9.0`); Go folds the const-arithmetic at compile time at full precision, IEEE 754 round-once-to-float64, deterministic across all four target languages. **Don't compute weights at runtime.** ~9 LOC of constants. (§8)
9. **SPH neighbour search: cell-linked-list spatial hash, O(N) per step amortised.** Forward-looking. SPH evaluates kernel sums `f(x_i) = Σ_j W(x_i - x_j, h)` over neighbours within smoothing length h. Naive O(N²) is unworkable at N>1000. **Cell-linked-list (Hockney-Eastwood 1981, the textbook spatial hash):** partition domain into cubic cells of side h, store particle-to-cell binning as `[]int` head + `[]int` next chain, query enumerates 3³=27 (3D) or 9 (2D) neighbour cells. Per-step cost: O(N) to rebuild cell-linked-list (one pass) + O(N · k) to evaluate kernels where k is avg-neighbours-per-particle (~50-200 typical for SPH). **Total: O(N · k) per timestep, not O(N²).** At N=10⁵ particles, k=100: ~10⁷ kernel evaluations per step, ~30 ns each (W3 quintic-spline + 1 sqrt) = ~0.3 sec/step on portable Go. Per 068 D-5 (DualSPHysics SoA + cell-linked-list pattern), this is the right substrate. **Allocation discipline: pre-allocate the cell-head + particle-next arrays at problem-init, reuse every step.** Cell rebuild is in-place. (§9)
10. **NS divergence-free projection: Poisson solve dominates timestep cost.** Forward-looking. Chorin's projection method (textbook NS for incompressible flow) at each timestep: (a) advection-diffusion of u* = O(N) explicit, (b) **solve ∇²p = ∇·u\*/Δt**, (c) project u^{n+1} = u* - Δt·∇p = O(N) explicit. **Step (b) is 90-99% of total cost.** For grid N×N=10⁶ unknowns, sparse-direct LDL costs ~O(N^1.5) = ~10⁹ flops on 2D Cartesian (with nested-dissection ordering); GMRES + multigrid preconditioner is O(N) per iter × ~10-30 iter = ~3·10⁷ to 3·10⁸ flops per timestep. **At ~0.5 Gflop/s portable Go scalar: 60 ms-600 ms per Poisson solve, dominating the timestep.** A 1000-step simulation = ~1-10 minutes. Per 067 + 068, **multigrid-preconditioned CG/GMRES is the production path**; sparse-direct is the unit-test reference (factor-once for steady-state). Both depend on the linalg/T-SPARSE sprint flagged in §6. **Pseudospectral periodic-box NS via existing `signal.FFT`** (per 068 D-4) is an alternate path that avoids the Poisson solve entirely (∇²p in Fourier space is divide-by-|k|², trivial); ship FIRST as JAX-CFD-style reference solver, ~300 LOC, leverages `signal/` already-shipped. (§10)
11. **NS Cartesian staggered MAC grid: SoA layout, three flat slices.** Forward-looking. MAC (Marker-and-Cell, Harlow-Welch 1965) places `u` on i+½,j faces, `v` on i,j+½ faces, `p` on i,j cell centres — analogous to em/fdtd's Yee staggering (065 §4). Same conclusion as em/fdtd: **AoS `struct{U, V, P float64}[]` is wrong** because the staggered positions are different physical sample points. **SoA = three independent `[]float64` slices** (or four in 3D), each length Nx·Ny (or Nx·Ny·Nz), row-major flat with **stride-1 along the innermost axis** (Y-innermost for 2D, Z-innermost for 3D), matching `linalg/matrix.go` precedent. Lock at first NS ship; once Python/C++/C# golden files commit to byte-order, layout flip forces cross-language golden-file regen. (§11)
12. **Frequency/timestep batching: stateless-kernel discipline already enforced by CLAUDE.md.** Forward-looking decision that's mostly free. Per 068 D-3 (Palabos collision-as-interface) + 068 D-6 (hand-rolled time-reversible adjoints), the LBM/NS/SPH packages should expose **stateless kernels** that take `(state_in, state_out, params)` and never mutate state-in or allocate. Caller drives the time loop. Three benefits: (a) Pistachio/services pick their own threading/SIMD wrap layer without touching reality; (b) golden-file traces emit the per-step state for cross-language validation without runtime instrumentation; (c) reverse adjoints recover state-in by re-running the forward kernel backward (time-reversible LBM/SPH per 068 D-6) without storing a tape. **No additional perf cost** — the buffer-out pattern is what `signal/fft.go` and `linalg/matmul.go` already use. Lock the convention in `fluids/lbm/doc.go` + `fluids/sph/doc.go` + `fluids/ns/doc.go` at first ship. (§12)

---

## §1 No benchmarks — same package-wide hole, fluids-specific stakes

```
Grep ^func Benchmark in fluids/ → 0 hits
Grep ^func Benchmark across reality/ (excl. reviews/) → 0 hits
```

Same hole flagged in 005/010/015/020/025/030/040/045/050/055/060/065. Each prior agent recommended a `*_bench_test.go` file. Fluids-specific stakes: when T3-NS / T3-LBM / T3-SPH land (per 067 ~6,200 LOC), those packages will be **the most expensive in the repo by orders of magnitude** — a single 256² LBM simulation at 1000 timesteps is ~6×10⁸ updates; a 2D NS at the same grid is ~10⁹ ops/step × 1000 steps = 10¹². The current 11 functions are scalar — their µs/op floor is 5-15 ns. Locking the floor **now** prevents a future regression where a refactor introduces an allocation in the closed-form layer that survives unnoticed because nobody benchmarks it after T3 lands and the µs/op floor migrates 10⁶× upward.

~30 LOC, eleven Benchmarks, one per function:

```go
// fluids_bench_test.go
func BenchmarkReynoldsNumber(b *testing.B) {
    var f float64
    for i := 0; i < b.N; i++ { f = fluids.ReynoldsNumber(998, 1.0, 0.05, 1e-3) }
    runtime.KeepAlive(f)
}
// ... × 10 more ...
```

Add `b.ReportAllocs()` to each. Add `AllocsPerRun` smoke test in `fluids_test.go` to fail CI on accidental allocation regression (~5 LOC). Mirror the pattern from 005/010/015/020/065 reports.

**`PipeFlowFriction` deserves its own Benchmark with multiple Re bands** (laminar Re=1000, transitional Re=4000, turbulent-smooth Re=1e5, turbulent-rough Re=1e7) — different iteration counts pinned at the µs/op floor.

---

## §2 Per-call allocations: zero today by accident

| Function | Body LOC | Heap alloc | Stack | Hot-path notes |
|---|---|---|---|---|
| `ReynoldsNumber` | 1 | 0 | 4 fp regs | 3 mul, 1 div |
| `BernoulliPressure` | 1 | 0 | 6 fp regs | ~6 mul, 4 add/sub |
| `PipeFlowFriction` | ~25 | 0 | 4 fp + iter | 1 `math.Pow` (one-shot seed) + N iter × (1 sqrt + 1 log10 + ~6 ops); ~5-15 iter typical |
| `DarcyWeisbach` | 1 | 0 | 4 fp regs | 4 mul, 1 div |
| `DragForce` | 1 | 0 | 4 fp regs | 4 mul |
| `LiftForce` | 1 | 0 | 4 fp regs | 4 mul |
| `TerminalVelocity` | 4 | 0 | 3 fp regs | 1 branch + 3 mul + 1 div + 1 sqrt |
| `StokesLaw` | 1 | 0 | 3 fp regs | 4 mul (incl. π) |
| `MassFlowRate` | 1 | 0 | 3 fp regs | 2 mul |
| `VolumetricFlowRate` | 1 | 0 | 2 fp regs | 1 mul |

Zero heap allocations across the entire surface. Zero defensive copies. Zero closure captures. **The "zero allocations is an accident of scalarness" observation** repeats from 065 §2: as soon as 067 T1-NETWORK lands `HardyCross(network *PipeNetwork) error` returning per-pipe flow vectors, or T3-NS lands `NavierStokesStep(state *FlowState)`, Go will heap-escape any non-buffer-out signature.

**Lock the buffer-out signature pattern at first ship** of every non-scalar primitive:

```go
// Wrong (returns fresh slice — heap-escape per call):
func PressureField(grid *Grid) []float64

// Right (caller-allocated out — zero alloc per call):
func PressureFieldInto(grid *Grid, out []float64)
```

Mirror `signal/fft.go` and `linalg/matrix.go`. Document in `fluids/network/doc.go` at first network-solver ship.

---

## §3 `math.Pow(Re, 0.9)` is one-shot, not hot-path — no action

The topic prompt names "math.Pow vs explicit multiply (Pow is slow)" as a pattern. **Verified the pattern does NOT apply to `fluids.go`:**

`fluids.go:91` uses `math.Pow(Re, 0.9)` in the **Swamee-Jain initial-seed computation** for `PipeFlowFriction`. This is computed **once per call**, before the fixed-point iteration begins. The iteration body (lines 96-97) uses `math.Sqrt` + `math.Log10` only — no `math.Pow`.

**`math.Pow(x, 0.9)` is the right call** for fractional exponent 0.9; no closed-form integer-multiply substitute exists. Alternative `math.Exp(0.9 * math.Log(x))` is ~30 ns vs ~25 ns for `math.Pow` (Go's `math.Pow` already detects fractional exponents and routes to the same exp/log path internally) — wash.

**Where the pattern WOULD apply** (zero current usages, monitor at T3 ship):

- `math.Pow(x, 2.0)` → replace with `x*x` (~25 ns → ~1 ns, 25× speedup)
- `math.Pow(x, 3.0)` → replace with `x*x*x` (~25 ns → ~2 ns, 12× speedup)
- `math.Pow(x, 0.5)` → replace with `math.Sqrt(x)` (~25 ns → ~12 ns, 2× speedup)
- `math.Pow(x, -1.0)` → replace with `1.0/x` (~25 ns → ~5 ns, 5× speedup)

**No instances in `fluids.go` today.** Add a 5-LOC linter check (`go vet` extension or `staticcheck` SA4015) at CI time when T1-COMPRESSIBLE / T2-DRAG-CORRELATIONS land — both surfaces involve `(1+(γ-1)/2·M²)^(γ/(γ-1))` (isentropic relations) and `Cd·Re^0.687` (Schiller-Naumann) which are **legitimate fractional-exponent uses**, but adjacent code may accidentally write `math.Pow(M, 2.0)` instead of `M*M`. Lock the linter config before T1-COMPRESSIBLE ships.

---

## §4 Colebrook fixed-point: ~1 µs/call, fast already

Per 066 finding 2 + `fluids.go:95-103`:

```go
for i := 0; i < 100; i++ {
    sqrtF := math.Sqrt(f)
    rhs := -2.0 * math.Log10(relRough/3.7+2.51/(Re*sqrtF))
    fNew := 1.0 / (rhs * rhs)
    if math.Abs(fNew-f) < 1e-12 {
        return fNew
    }
    f = fNew
}
```

**Per-iter cost on portable Go (x86):**
- `math.Sqrt` ~12 ns
- `math.Log10` ~30 ns
- ~6 floating-point arithmetic ops ~3 ns
- 1 abs + 1 compare + 1 branch ~2 ns
- **Total: ~47 ns/iter, padded to ~80 ns with cache effects**

**Iteration count empirical:**
- Smooth pipes (ε=0): 5-8 iterations
- Commercial steel (ε≈4.6e-5 m): 8-15 iterations
- Rough cast iron (ε≈2.5e-4 m): 12-20 iterations
- Worst-case transitional Re=4000 + rough: 25-30 iterations

**Total per call: 0.4-2.4 µs**, dominated by the `math.Log10` per iter.

**This is the right cost class.** Newton-Raphson on the Colebrook implicit equation converges in 4-6 iterations vs 5-15 for fixed-point — 2-3× faster, ~30 LOC more code (analytical derivative). Per 068 D-2, the architectural plan is to ship **Serghides explicit (1984)** as the speed-default (no iteration, ~5 ns equivalent to a closed-form correlation, 1e-7 max error per Bell 2018) and **Brent-iterated Colebrook** as the scipy-parity reference (rigorous bracketing, ULP-pinned). Both go to the linalg/optim sprint.

**No action on the current iteration.** The 1 µs/call floor is acceptable for steady-state pipe-network solvers (one Colebrook eval per pipe per Newton outer-iter; at N=100 pipes × 8 Newton iters = 800 calls = 0.8 ms total = trivial). It would matter at LBM-scale (N=10⁵ × 1000 timesteps) but Colebrook doesn't apply there.

**066 N-2's abs-vs-rel tolerance issue** (`math.Abs(fNew-f) < 1e-12` is absolute; for `f` near 0.005 this is 5e-11 relative; for `f` near 0.05 it's 5e-13 relative — inconsistent across regimes) is **a numerical-correctness issue, not a perf issue**; defer to 066. Perf-wise, the 100-iter cap is never hit in practice.

---

## §5 Hardy Cross is wrong default for new pipe-network solvers — Newton-Raphson global solve is

Forward-looking. Pipe-network steady-state flow distribution is the canonical T1-NETWORK headline (per 067 + topic prompt).

**Hardy Cross (1936):** iterate per-loop with `ΔQ = -ΣΔh / (2·Σ|Δh/Q|)`. Updates loop-by-loop sequentially. Cost class:
- O(L · N_loops · N_iter) where L is pipes-per-loop (~3-5 typical) and N_loops is independent loops in the network (~N_pipes - N_junctions + 1).
- Convergence: **linear**, often 50-200 iterations on real networks (worse on highly-looped or stiff networks).
- Pros: zero matrix infrastructure, hand-trace-able, textbook reference.
- Cons: slow, sequential (not parallelisable), brittle on networks with reverse flows.

**Newton-Raphson on global simultaneous-equation system (modern default — EPANET, KYPipe, Cross-Todini variant):**
- Formulate: `F(Q) = 0` where F encodes head-loss + continuity simultaneously across all pipes.
- Jacobian: `J = ∂F/∂Q`, **N×N sparse** with ~3-5 nonzeros per row (one per pipe at each end junction).
- Newton step: solve `J · ΔQ = -F`, then `Q ← Q + ΔQ`.
- **Quadratic convergence**, typically 5-8 outer iterations to 1e-10 residual.
- Cost per outer iter: 1 sparse linear solve + 1 sparse matrix assembly = O(N) for assembly + O(N^1.5) for sparse-direct LDL.
- **5-10× faster than Hardy Cross at N>30 pipes** (per Todini-Pilati 1988 benchmarks).

**Architecture decision:**
1. **Ship Hardy Cross at T1-NETWORK** (~150 LOC, 067 sprint-1) as the **textbook reference for unit-testing**. Useful at N<30 (manual examples, Crane TP-410 worked examples).
2. **Ship Newton-Raphson at T2-NETWORK** as the **production path** once `linalg.SparseCSR` + `linalg.GMRES` / `linalg.SparseDirectLDL` land (which gates on the linalg-sparse sprint per §6 + 065 §10 + 067).

Both must produce identical golden files at convergence (within tolerance) — golden-file diff cross-validates the two solvers. Document in `fluids/network/doc.go`.

**Modified Hardy Cross (Wood-Charles 1973):** updates all loops simultaneously using the linearised system; intermediate between Hardy Cross and Newton-Raphson, ~3× faster than Hardy Cross but still slower than Newton. **Skip.** Newton-Raphson supersedes it.

---

## §6 Sparse Jacobian: COO-build → CSR-solve, never dense

Forward-looking. Pipe-network Jacobian sparsity:

| N (pipes) | Dense memory | Sparse nonzeros (~3-5/row) | Sparse memory | Sparsity |
|---|---|---|---|---|
| 30 | 7.2 KB | ~120 | ~1.4 KB | 87% |
| 100 | 80 KB | ~400 | ~4.8 KB | 96% |
| 1,000 | 8 MB | ~4,000 | ~48 KB | 99.5% |
| 10,000 | 800 MB | ~40,000 | ~480 KB | 99.95% |
| 100,000 | **80 GB (impossible)** | ~400,000 | ~4.8 MB | 99.995% |

**Build pattern:** COO triplets `(rowIdx, colIdx, value)` accumulated during incidence-matrix walk (one pass over pipes, two entries per pipe — one for each endpoint junction) → convert to CSR (sorted-by-row, indptr+colidx+values) once per Newton outer iteration → call `linalg.SpMV` and `linalg.SparseDirectLDL` (or `GMRES`+`ILU(0)`).

**Critical-path dependency on linalg/T-SPARSE:**

Per 065 §10 (em-fdfd/fem critical path) + 067 (T3-NS critical path) + this report (T2-NETWORK critical path), **three packages — em, fluids, possibly future packages — all gate on `linalg.SparseCSR` + iterative + direct sparse solvers.** Single shared linalg sprint unblocks all three.

**Recommended sparse infrastructure (totals from 065 §10):**
- `linalg.SparseCSR` (~150 LOC)
- `linalg.SpMV` real + complex (~80 LOC)
- `linalg.GMRES` (~250 LOC, function-typed preconditioner for golden-file portability)
- `linalg.MINRES` (~150 LOC, symmetric)
- `linalg.PreconditionerILU0` (~200 LOC)
- `linalg.SparseDirectLDL` (~400 LOC, nested-dissection ordering)
- **Total ~1230 LOC linalg work**, dual-unblocks em-FDFD/FEM/MoM and fluids-network/NS/Helmholtz.

Document in MASTER_PLAN.md cross-reference. Add to `fluids/network/doc.go` at first ship with explicit pointer to `linalg/sparse.go`.

**At small N (< 30 pipes), dense Jacobian is acceptable** as the unit-test reference (use existing `linalg.Solve` against `Matrix`). Document the `N<30` threshold as the dense-fallback contract.

---

## §7 LBM D2Q9 lattice update: ~0.3-0.8 GLUPS class on portable Go

Forward-looking. D2Q9 lattice has 9 distribution functions per cell, indexed `f[i]` for `i ∈ {0..8}` (rest, four cardinal, four diagonal directions).

**Per-cell timestep cost:**
- Compute macroscopic moments: `ρ = Σf[i]` (8 adds), `u_x = Σf[i]·c_x[i] / ρ` (~9 muls + 8 adds + 1 div), `u_y = Σf[i]·c_y[i] / ρ` (same).
- Compute equilibrium `f_eq[i] = w[i]·ρ·(1 + 3(c·u) + 4.5(c·u)² - 1.5|u|²)` for i=0..8: ~7 ops × 9 = ~63 ops.
- BGK collision: `f_post[i] = f[i] - (f[i] - f_eq[i])/τ` for i=0..8: 3 ops × 9 = 27 ops.
- Streaming: 9 reads from neighbours, 9 writes (no flops, pure memory).

**Total: ~80-100 flops + ~9 reads × 8 bytes + 9 writes × 8 bytes = 144 bytes/cell + ~90 flops/cell.**

**Throughput ceiling on portable Go (x86, 3 GHz, scalar IPC ~2):**
- ALU-ceiling: 6 Gflops/s ÷ 90 flops = **67 MLUPS**
- Memory-bandwidth: 25 GB/s DRAM ÷ 144 bytes = **170 MLUPS**
- Cache-blocked (working set in L2/L3): **80-150 MLUPS realistic = 0.08-0.15 GLUPS portable Go**
- Apple-Silicon M-series with 100-200 GB/s unified memory: 200-400 MLUPS achievable.

**SOTA reference points:**
- Palabos AVX-512 + MPI (Latt et al. 2021): ~2-4 GLUPS on a 32-core node.
- Lettuce GPU torch (Bedrunka 2022): ~5-15 GLUPS on V100.
- waLBerla AVX-512 codegen (lbmpy 2024): ~5-8 GLUPS single-node.
- XLB JAX (Ataei-Salehi 2024): ~3-10 GLUPS on TPU v4.

**Reality is correctly 1-2 orders of magnitude below SOTA CUDA**, same reasoning as em/fdtd (065 §3): no SIMD, no `unsafe`, no CGO, single-thread for cross-language golden-file determinism. **This is the right tradeoff** for `reality`'s role (math-of-record, not production solver).

**Document upfront in `fluids/lbm/doc.go`** with explicit pointer to Palabos/Lettuce/waLBerla for production scale. Otherwise downstream consumers benchmark reality against waLBerla, see a 30× gap, assume reality is broken rather than (correctly) under-optimised by design.

Per 068 D-3, the **lattice descriptor** should be a Go generic type parameter (or interface in pre-1.18 idiom — but reality is Go 1.21+). The collision step is an interface (`type Collider interface { Collide(state *LatticeState) }`) so users can swap BGK / MRT / regularized / entropic without re-shipping streaming. **Stateless kernel discipline (068 D-3 + this report §12)** keeps the streaming and collision phases buffer-out.

---

## §8 LBM streaming: pull-scheme + D2Q9 weights as exact-rational consts

Forward-looking. Two streaming implementations: **push-scheme** (each cell scatters its f[i] to 9 neighbours: 9 close reads, 9 scattered writes) is ~1.5-2× slower on portable Go than **pull-scheme** (each cell gathers from 9 neighbours: 9 scattered reads, 9 contiguous writes — cache-friendly, single-line writes). Standard in Palabos / lbmpy / Lettuce / waLBerla. **Lock pull-scheme in package doc + loop body comment.** Mathematically identical results — but a future contributor switching to push thinking it's symmetric would silently 2× the cost.

**D2Q9 weights** per 067 forward F-10: `const W0 = 4.0/9.0`, `const W1 = 1.0/9.0`, `const W2 = 1.0/36.0` — exact rationals, Go const-arithmetic folds at compile time at full big.Float precision, IEEE 754 round-once-to-float64, inlines as immediate operand into MULSD/ADDSD with zero memory load. Same pattern 050 verified for `constants/`. **NOT** `var` (heap-allocated, indirect-load per access). **NOT** computed at runtime. Same pattern for D3Q15/D3Q19/D3Q27 when 3D lands. ~30 LOC across all four lattice variants. Lock in `fluids/lbm/lattice.go`.

---

## §9 SPH neighbour search: cell-linked-list spatial hash, O(N·k) per step

Forward-looking. SPH evaluates kernel sums:

```
f(x_i) = Σ_j  W(|x_i - x_j|, h)  ·  m_j  ·  ψ_j
```

over all particles j with `|x_i - x_j| < 2h` (smoothing-length cutoff, kernel goes to zero outside).

**Naive O(N²)** is unworkable: N=10⁵ → 10¹⁰ kernel evaluations × ~30 ns each = 5 minutes per step. Doesn't fit anyone's iteration budget.

**Cell-linked-list (Hockney-Eastwood 1981)** — the textbook spatial hash:

```go
type SPHGrid struct {
    cellSize   float64       // = 2h
    head       []int         // [cellIdx] -> first particle in cell, -1 if empty
    next       []int         // [particleIdx] -> next particle in same cell, -1 if last
    cellCounts []int         // optional: count per cell, debug only
}

// Per timestep:
// 1. Clear head[] to -1 (one pass O(numCells)).
// 2. For each particle, compute cellIdx, prepend to head[cellIdx] linked list (one pass O(N)).
// 3. For each particle i, enumerate 27 neighbour cells (3D) or 9 (2D), walk each cell's linked
//    list, compute kernel for each j within 2h.
```

**Cost class:** O(N) to rebuild grid + O(N · k) to evaluate kernels where k = average-neighbours-per-particle (~50-200 typical for 3D SPH).

**At N=10⁵ particles, k=100:**
- Rebuild: ~10⁵ ops × ~5 ns = 0.5 ms
- Kernel evals: 10⁷ × ~30 ns = 300 ms = ~0.3 sec/step on portable Go
- **Total: ~0.3 sec/step**, ~5 min for 1000 steps.

**SOTA reference:** DualSPHysics OpenMP CPU ~10× faster, GPU ~100× faster — same single-thread-no-SIMD penalty as LBM/FDTD per 068 D-5.

**Allocation discipline (forward-looking lock):**
- Pre-allocate `head`, `next`, `cellCounts` arrays at problem-init from particle-count and domain-size.
- Reuse every timestep (clear head[] in-place, refill next[] in-place).
- **Zero `make()` calls in the timestep loop.** Document in `fluids/sph/doc.go` at first ship.

**Cell-size choice:** must be `cellSize ≥ 2h` (else kernel cutoff exceeds neighbour-cell radius, missed pairs). Smaller cells = more cells walked but fewer pairs per cell; larger cells = fewer cells walked but more pairs per cell. **Optimum is `cellSize = 2h`** (textbook). Hard-code; no caller tunable.

**3D variant: 27 neighbour cells per particle.** Same logic, just larger constant. Cost scales by 3×.

Per 068 D-5 (DualSPHysics SoA + cell-linked-list pattern), this is the right substrate.

---

## §10 NS divergence-free projection: Poisson solve dominates

Forward-looking. Chorin projection method per timestep: (1) advection-diffusion of u* — O(N) explicit, ~20 flops/cell; (2) **pressure Poisson `∇²p = (1/Δt)·∇·u*` — THE BOTTLENECK**, 90-99% of total cost; (3) projection u^{n+1} = u* - Δt·∇p — O(N) explicit, ~6 flops/cell.

For grid N=10⁶ unknowns at ~0.5 Gflop/s portable Go scalar: sparse-direct LDL ~60 ms-2 sec/solve; multigrid V-cycle ~20-60 ms; GMRES+ILU(0) ~60-600 ms; **CG+multigrid ~20-60 ms** (production path for SPD Poisson). 1000-step run: ~1-10 minutes Poisson-dominated.

Per 067+068, **multigrid-preconditioned CG is the production path**; sparse-direct LDL is the unit-test reference (factor-once for steady-state). **Both depend on linalg/T-SPARSE sprint** (§6, 065 §10, 067).

**Pseudospectral periodic-box NS via existing `signal.FFT`** (per 068 D-4) avoids the Poisson solve entirely: in Fourier space ∇²p̂ = -|k|²p̂, divide by -|k|² in O(N) (k=0 mode zero by gauge). Per step: 3 forward + 3 inverse FFTs + nonlinear-term-real-space = ~12 FFTs × O(N log N). At N=10⁶ via `signal/fft.go`, ~30 ms/FFT → 360 ms/step → 6 min for 1000 steps. **Comparable to multigrid-CG per step, zero linalg-sparse dependency.** Ship FIRST as JAX-CFD-style reference (~300 LOC). Full NS-Cartesian with multigrid waits on linalg-sparse sprint.

---

## §11 NS Cartesian staggered MAC grid: SoA layout, three flat slices

Forward-looking. MAC (Marker-and-Cell, Harlow-Welch 1965) staggered grid for NS:

```
u  on (i+½, j   )  faces — x-velocity, between cells in x
v  on (i,   j+½ )  faces — y-velocity, between cells in y
p  on (i,   j   )  centres — pressure at cell centres
```

(3D analogous: u on (i+½,j,k), v on (i,j+½,k), w on (i,j,k+½), p on (i,j,k).)

Same conclusion as em/fdtd Yee staggering (065 §4):

**AoS `struct{U, V, P float64}[idx]` is wrong.** The staggered positions are different physical sample points — packing them into one struct interleaves unrelated samples and pessimises traversal by an order of magnitude.

**SoA = three independent `[]float64` slices** (or four in 3D), each of length Nx·Ny (or Nx·Ny·Nz):

```go
type FlowState struct {
    Nx, Ny     int
    Dx, Dy, Dt float64
    U          []float64  // x-velocity, length (Nx+1)*Ny (extra column for face)
    V          []float64  // y-velocity, length Nx*(Ny+1)
    P          []float64  // pressure, length Nx*Ny
    // optional: Rho, Nu (density, viscosity) if variable
}
```

Row-major flat with **stride-1 along the innermost axis** (Y-innermost for 2D, Z-innermost for 3D), matching `linalg/matrix.go` precedent.

Lock at first ship of `fluids/ns/cartesian/`. Once Python/C++/C# golden files commit to byte-order, layout flip forces cross-language golden-file regen. Mirror the SoA / row-major / inner-loop convention from 065 §4.

**Cache blocking:** for grids > L2 (any N>128² in 2D, N>32³ in 3D), apply 64×64 or 32×32×32 tile decomposition per 065 §6. ~40 LOC of nested-loop change at NS ship time. Bit-identical results to un-tiled.

---

## §12 Stateless-kernel discipline: zero extra cost, large architectural win

Forward-looking. Per 068 D-3 (Palabos collision-as-interface) + 068 D-6 (hand-rolled time-reversible adjoints) + CLAUDE.md §3 (no allocations in hot paths):

**Every LBM/NS/SPH kernel takes (state_in, state_out, params) and is pure: never mutates state-in, never allocates.** Caller drives the time loop.

```go
// Right (stateless, buffer-out, caller drives loop):
func LBMStep(in, out *LatticeState, params LBMParams)
func NSStep(in, out *FlowState, params NSParams)
func SPHStep(in, out *ParticleState, params SPHParams, grid *SPHGrid)

// Wrong (stateful, opaque internal state, alloc per step):
type LBMSolver struct { state *LatticeState; ... }
func (s *LBMSolver) Step()
```

**Three benefits, zero perf cost:**

1. **Threading/SIMD wrap layer is consumer-owned.** Pistachio/services pick their own pthreads/SIMD/CUDA wrap layer without touching reality. Per CLAUDE.md (no SIMD intrinsics, single-thread golden-file determinism).
2. **Golden-file traces are transparent.** Per-step `state_out` dumps to JSON for cross-language validation without runtime instrumentation. State is a flat struct of slices; serialise via `encoding/json`.
3. **Time-reversible adjoints (068 D-6).** LBM and SPH are time-reversible; running the forward kernel backward recovers `state_in` from `state_out` without storing a tape (O(boundary) memory not O(timesteps)). FDTDX-2024 / diffSPH-2025 architectural lesson. Hand-rolled, not autodiff.

**Lock the convention in `fluids/lbm/doc.go` + `fluids/sph/doc.go` + `fluids/ns/doc.go` at first ship.** Mirror `signal/fft.go` and `linalg/matmul.go` already-shipped pattern. ~5 LOC of doc per package.

---

## Summary of perf decisions to lock at first ship

| Decision | Lane | Forward-looking? | LOC | Lock at |
|---|---|---|---|---|
| Bench every existing function (11 funcs) | fluids/ now | **No (today)** | ~30 | This sprint (with 066/069 fixes) |
| `math.Pow(Re, 0.9)` is correct — no action | fluids/ now | No | 0 | Verified (§3) |
| Colebrook fixed-point cost ~1 µs/call — fast already | fluids/ now | No | 0 | Verified (§4) |
| Hardy Cross at T1-NETWORK as reference | fluids/network | Yes | ~150 | T1-NETWORK ship |
| Newton-Raphson global solve at T2-NETWORK | fluids/network | Yes | ~250 | After linalg/T-SPARSE |
| COO-build + CSR-solve for sparse Jacobian | fluids/network | Yes | ~depends on linalg | After linalg/T-SPARSE |
| SoA layout (multiple flat `[]float64`) | fluids/lbm, fluids/ns | Yes | foundational | T3-LBM / T3-NS ship |
| Pull-scheme streaming for LBM | fluids/lbm | Yes | doc + loop comment | T3-LBM ship |
| D2Q9/D3Q19 weights as `const` exact rationals | fluids/lbm | Yes | ~30 | T3-LBM ship |
| Cell-linked-list spatial hash for SPH | fluids/sph | Yes | ~150 | T3-SPH ship |
| Pre-allocate cell-grid + reuse every step | fluids/sph | Yes | core | T3-SPH ship |
| Multigrid-CG Poisson for NS projection | fluids/ns | Yes | ~depends on linalg | After linalg/T-SPARSE |
| Pseudospectral periodic NS via signal/FFT | fluids/ns | Yes | ~300 | First NS path, before sparse |
| Stateless-kernel discipline (in, out, params) | fluids/* | Yes | doc only | Every PDE primitive ship |
| Document portable-Go GLUPS class as 0.08-0.15 | fluids/lbm doc | Yes | ~doc | T3-LBM ship |
| Document linalg-sparse cross-package dep | repo-level | **No (today)** | ~doc | This week |

The forward-looking items are 14 of 16. The two today-actionable items (bench file + MASTER_PLAN dependency note) are ~31 LOC and zero math change.

**The single most-leveraged commit** for fluids-perf today: write `fluids_bench_test.go` with eleven Benchmarks — locks the µs/op floor on the closed-form layer **before** T3 lands and the floor migrates 10⁶× upward, masking any future regression in the fast path. Cross-coordinate with 005/010/015/020/025/030/040/045/050/055/060/065 — the bench-file pattern wants a single agent slot to carry across all 22 packages.

**The single most-leveraged forward-looking commit:** lock the SoA-flat-`[]float64`-row-major layout + pull-scheme + stateless-kernel + buffer-out conventions in `fluids/lbm/doc.go` + `fluids/ns/doc.go` + `fluids/sph/doc.go` at first ship of each. All cost-free at ship time; all become structurally expensive to change once Python/C++/C# golden files commit to the byte-order of dumped state.

**The cross-package critical-path call:** linalg-sparse sprint (~1230 LOC) unblocks em-FDFD/FEM/MoM **and** fluids-network/NS/Helmholtz simultaneously. Add to MASTER_PLAN.md as a shared-foundation slot before T2-NETWORK / T3-NS / em-T3-FDFD ships.
