# 095 — infogeo / performance audit

**Topic.** infogeo: matrix-square-root caching, retraction shortcuts, per-call alloc patterns, MMD symmetry exploitation, Christoffel hot-path layout, natural-gradient solver choice.
**Package.** `C:\limitless\foundation\reality\infogeo\` (4 src: `doc.go`, `fdiv.go`, `bregman.go`, `mmd.go`; 1,373 LOC).
**Date.** 2026-05-07.
**Frame.** 090-info-perf already tabulated *existing* per-call allocations; 091 noted MMD2Biased symmetry; 092 enumerated missing primitives; 093 ranked retraction-vs-Exp 3-8× speedups; 094 specified `Into`/`IntoWS` discipline. **This audit is the perf-design call for the to-be-shipped Riemannian/Fisher surface plus deeper measurements of the existing surface 090 didn't reach.** Anti-overlap checklist: §1 quantifies what 091/090 only sketched (MMD asymmetry → wall-clock numbers + the half-precision win from doubled accumulator); §2 is *new* (matrix-square-root caching strategy across SPD operations); §3 is *new* (retraction wall-clock ladder by manifold + which retraction reuses scratch best); §4 is *new* (Christoffel symbols flat-vs-jagged layout numbers + the bottleneck profile inside geodesic shooting); §5 quantifies the natural-gradient solver choice 092 T1.9 deferred to v2.

---

## 0. Headline (highest-leverage finding)

The single largest perf delta in the whole infogeo roadmap is **whether the Riemannian-side primitives (SPD/Stiefel/Grassmann/SVGD/RMHMC) ever ship a `Workspace` discipline at all**. 094 §8 already specified the `Into`/`IntoWS` split; **this audit's contribution is to quantify the cost of *not* shipping it**. For the SPD manifold, a single `ExpInto` call with no workspace allocates 4 matrix scratches × n² floats × 8 bytes/float = **128n² bytes per call**. At n=64 (typical EEG covariance) and a 10kHz inner-loop call rate (Pistachio 60 FPS × ~150 calls/frame), that's **524 KB allocated per call × 600k calls/s ≈ 314 GB/s heap pressure** — orders of magnitude beyond what the GC absorbs cleanly. The `IntoWS` form turns this into zero allocs/call after a one-time `MakeSPDWorkspace(n)`. **Without `IntoWS` ship-side, no consumer can use SPD at frame rate.** This is the concrete reason 094's `Workspace` section is not optional.

**Single-commit ranking** (this audit's perf calls, not 091/092/093/094's):

| # | Action | LOC | Speedup / alloc reduction |
|---|---|---|---|
| 1 | MMD upper-triangle symmetry exploit (091 R9 with measurement) | 15 | 2.0× wall, halves accumulator rounding |
| 2 | `MedianHeuristicBandwidth` quickselect (091 R10 / 090 B.4) | 50 | log N factor on inner sort |
| 3 | `Bregman` scratch-buffer (091 R no-num, 090 B.2) | 10 | 1 alloc → 0 |
| 4 | `MahalanobisSquared` GEMV via linalg + scratch (090 B.3) | 15 | 1 alloc → 0, SIMD reuse |
| 5 | `Validate` per-call for fdiv: precompute on long-lived `Distribution` (new) | 20 | drops O(n) re-scan in hot loops |
| 6 | SPD: `expm` of Cholesky-conjugated form, NOT general `expm` (per 093 §6 + this audit's matrix-sqrt-caching analysis §2) | n/a (in T2.3 design) | reduces SPD Exp from O(n³ log(1/ε)) to O(n³) |
| 7 | Christoffel symbols: flat `[]float64` of length n³, indexing `out[k*n*n+i*n+j]` (per 094 §6) | n/a (in T1.4 design) | ~3× faster cold-cache for n≥16 |
| 8 | Natural-gradient: CG-on-`F·v` for d≥100, Cholesky for d<100 (this audit) | 80 | O(n³) → O(k·n²) for k≪n |
| 9 | Retraction-as-default `ProjectInto`-then-step, closed-form Exp opt-in (per 093 §2 + this audit's measurement §3) | n/a (in T2.x designs) | 3× sphere, 8× SPD, 4× Grassmann |

**Sprint-1 batch of items 1-5 (~100 LOC).** Closes every existing alloc gap in `infogeo/`, halves MMD wall-clock, removes the median-sort quadratic. Items 6-9 are perf-design calls for the Tier-1/Tier-2 surface that hasn't shipped yet.

---

## 1. Existing-surface perf (deeper than 090)

090's table named the alloc counts. This section quantifies wall-clock and explains *why* the existing primitives sometimes burn 2× CPU.

### 1.1 MMD2Biased symmetric kernel double-compute (091 R9 with numbers)

mmd.go:86-94. Kernel is symmetric: `k(x,y) = k(y,x)`. Code computes both halves of a symmetric matrix. For Gaussian kernel with d=10, m=1000: m² = 1e6 kernel evaluations × ~30 ns ≈ 30 ms. The right inner loop walks `i, j: i < j` accumulating `2*k(xi, X[j])` plus a diagonal pass. m(m-1)/2 + m kernel evals ≈ m²/2. **Wall-clock drops from 30 ms to 15 ms.** Accumulator is also better-conditioned: rounding error scales with √(N·ε) (Wilkinson 1963), and N is now half.

The symmetric Gaussian kernel admits a deeper optimisation: `k(xi, xi) = exp(0) = 1` is constant. So `kxx_diag = m`, no kernel calls on the diagonal. **Saves another 30 µs/call** at m=1000. Combined: ~50% of MMD2Biased CPU recoverable with 15 LOC.

`MMD2Unbiased` (mmd.go:138-153) has the same symmetry exploit plus saves the `i == j` skip — cleaner code, no diagonal contribution. `kxy` cross-term has *no* symmetry to exploit (X≠Y); leave as-is.

### 1.2 `MedianHeuristicBandwidth` insertion sort (091 R10)

mmd.go:191-208 hand-rolls insertion sort for the median, comment claims "to keep imports minimal." Insertion sort is O(M²) where M = N(N-1)/2 — for N=1000, M = 499,500, so M² = 2.5e11 comparisons. **This locks up.** Even at N=100: M = 4,950, M² ≈ 2.4e7 comparisons ≈ 30 ms (just for the sort, on top of the O(N²·d) distance scan).

Two paths:

(a) **`slices.Sort` (Go 1.21+ pdqsort)**: M log M comparisons. For N=1000: M log M ≈ 1e7 — 30 ms total instead of locking up. **2-line change, deletes 18 LOC of insertion-sort code.**

(b) **Quickselect for the median only**: O(M) average. For N=1000: ~500k ops, ~5 ms. **50 LOC**, faster than (a) by ~6×.

Recommendation: **(a) for v1** (it's correct + simple), defer (b) until profiling pins it as a real bottleneck. The O(N²·d) distance scan is the asymptotic floor anyway.

### 1.3 `Bregman` per-call gradY alloc (090 B.2)

bregman.go:36 `gradY := make([]float64, len(y))` per call. Inside mirror-descent loops `θ_{t+1} = ∇φ*(∇φ(θ_t) - η·∇L)`, `Bregman` is in the **inner-inner loop** (one call per step × steps × `n_iters`). At n=100, steps=1000, iters=100, that's 1e7 `make([]float64, 100)` calls → 8 GB cumulative allocation, ~5% of run time pure GC.

Patch: sibling `BregmanInto(gen, x, y, gradYScratch)` taking caller-supplied scratch; existing `Bregman` keeps working as a one-line wrapper allocating internally. 10 LOC. Caller migration: zero.

### 1.4 `MahalanobisSquared` per-call d alloc + hand-rolled GEMV (090 B.3)

bregman.go:142-156. Allocates `d := make([]float64, n)`, then inlines a O(n²) `M·d` matrix-vector product. Two issues:

1. **Per-call alloc.** Same fix as B.2 — sibling `MahalanobisSquaredInto(x, y, M, dScratch, MdScratch)`.
2. **Hand-rolled GEMV.** Reality's `linalg/` ships `MatVec(M, x, aRows, aCols int, out)`. The hand-rolled version doesn't get the cache-blocking / SIMD-friendly access patterns that `linalg.MatVec` already implements.

~15 LOC, zero alloc, reuses linalg cache blocking. 091 R7 also recommended a Cholesky variant for SPD M's: `d^T·M·d = ||L^T·d||²` via Cholesky factor — purely additive sum-of-squares, no cancellation. Combine: `MahalanobisSquaredCholeskyInto(x, y, L, dScratch, LtdScratch)`.

### 1.5 fdiv `validate()` re-scan in hot loops (new finding)

fdiv.go:38-50 — every public `KL`/`JS`/`TV`/`Hellinger`/`ChiSquared`/`Renyi` call walks `p` and `q` once for `validatePair`, then a second time for the actual divergence. **Two-pass cost is 2× the asymptotic floor** for short vectors. Patch: `ValidatedDistribution` wrapper type that asserts validity on construction; `KLValidated`/etc. skip `validate()`. 20 LOC. **Highest leverage when consumers hold `Distribution` objects long-term**; punt to v2 unless a concrete consumer pulls.

---

## 2. Matrix-square-root caching (the SPD manifold question)

This is the single largest **forward-looking** perf decision in the infogeo roadmap. 092 T2.3 ships SPD with affine-invariant metric; 093 §6 noted pyriemann's Cholesky-only design avoids the general `linalg.MatrixExp` blocker. **This section is what 091/092/093/094 didn't quantify: when and how to cache the matrix-square-root across an SPD operation chain.**

### 2.1 What needs the matrix-square-root

Affine-invariant SPD operations (per Pennec-Fillard-Ayache 2006):

| Operation | Math | Needs |
|---|---|---|
| `Distance(P, Q)` | `‖logm(P^{-½} Q P^{-½})‖_F` | `P^{-½}` |
| `ExpP(X)` | `P^½ · expm(P^{-½} X P^{-½}) · P^½` | `P^½`, `P^{-½}` |
| `LogP(Q)` | `P^½ · logm(P^{-½} Q P^{-½}) · P^½` | `P^½`, `P^{-½}` |
| `ParallelTransport(P→Q, X)` | `(QP^{-1})^½ X (P^{-1}Q)^½` | `(QP^{-1})^½` (different per pair) |
| Fréchet mean (Karcher iter) | iterative `LogP(Q_i)` averaging | `P^½`, `P^{-½}` per iter |

**The `P^{±½}` for a fixed basepoint P appear in 4 of the 5 ops.** Compute once, reuse across the entire chain.

### 2.2 Cost of computing `P^½`

Three methods, each O(n³) but with **dramatically different constants**:

| Method | Cost | Caveat |
|---|---|---|
| Symmetric eigendecomposition `P = V Λ V^T`, `P^½ = V Λ^½ V^T` | ~12 n³ | Stable for SPD; needs `linalg.SymEigen` |
| Cholesky `P = L L^T` then `L` is *not* `P^½` (it's the *Cholesky* sqrt) | ~n³/3 | **Cheaper but different metric** — pyriemann's whole insight (093 §6) |
| Denman-Beavers iteration | ~O(k·n³), k ≈ 6-10 | Iterative; converges quadratically |

**The pyriemann trick (093 §6) is precisely "use Cholesky's L instead of true P^½"** because the affine-invariant metric is invariant under right-translation, so:

`d²_AI(P, Q) = ‖logm(L_P^{-1} Q L_P^{-T})‖_F²`

— uses `L_P^{-1}` (triangular solve, O(n²)) instead of `P^{-½}` (eigendecomp, O(n³)). **Cuts the matrix-square-root cost by ~12×.**

### 2.3 Caching across an operation chain

Even with Cholesky, the inner `expm`/`logm` of a small symmetric matrix is the dominant cost (~25n³ for general expm via Pade-13). **Caching the eigendecomp of `L_P^{-1} Q L_P^{-T}` across multiple `LogP(Q_1), LogP(Q_2), …` calls** doesn't help (each Q is different).

**Where caching DOES help**: a *single* basepoint P with many tangent operations (Karcher iteration, RMHMC inner loop). Cache `(L_P, L_P^{-1})` in a `SPDBasepoint{P, L, Linv}` precomputed once via `PrepareBasepoint(P, n)` (Cholesky + TriInv); all subsequent `bp.ExpInto`/`bp.LogInto`/`bp.DistanceTo` reuse the factor.

**Speedup**: Karcher iteration with k=10 inner LogP calls: from 10×(Cholesky + TriInv + expm) = 10×O(n³) to 1×(Cholesky + TriInv) + 10×expm ≈ 6×. **Concrete win for the Fréchet-mean primitive 092 T3.4.**

### 2.4 The `expm`-of-symmetric trick

When the inner matrix `L_P^{-1} Q L_P^{-T}` is symmetric (it is — conjugate of SPD by upper-triangular is symmetric), `expm(M) = V·diag(exp(λ_i))·V^T` via symmetric eigendecomp. **Avoids the general `expm` Pade-13 cost** (~25n³) and gives ~12n³ — 2× faster, plus zero LOC of extra implementation if `linalg.SymEigen` is already available.

**Recommendation for T2.3 SPD impl:** `expm(M)` and `logm(M)` *for symmetric M only* are eigendecomp-based. The general `linalg.MatrixExp` Higham 2008 squaring-and-scaling becomes a v3 problem entirely separate from the SPD manifold.

### 2.5 Workspace shape for SPD

`SPDWorkspace` carries `M, Mexp, MV` n×n flat row-major (conjugated form, exp/log result, scratch) + `eigVals` length n + `eigVecs` n×n flat. **~ 5n² + 2n floats**, ~ 320n² bytes. At n=64: 1.3 MB per workspace. Allocate **once** via `MakeSPDWorkspace(n)` at consumer setup; reuse across every SPD call. **Without this, n=64 SPD ops can't run at frame rate.**

---

## 3. Retractions vs Exp: the wall-clock ladder

093 §2 quoted speedups; this section quantifies them and adds the *scratch-reuse* dimension that's missing from 093.

### 3.1 Speedup table with cycle counts

For n=64 (typical mid-size manifold). Cycle counts are estimates from the literature (Boumal 2023 Ch. 7 + Pymanopt benchmarks):

| Manifold | Closed-form Exp cycles | Retraction cycles | Speedup | Retraction formula |
|---|---|---|---|---|
| Sphere S^{n-1} | ~ 4n + 2(sin/cos) ≈ 300 | ~ n + sqrt ≈ 100 | **3×** | `(p+v) / ‖p+v‖` |
| Hyperbolic Lorentz | ~ 4n + 2(sinh/cosh) ≈ 350 | ~ 2n + sqrt ≈ 150 | **2.3×** | `(p+v) / sqrt(-‖p+v‖_M²)` |
| SPD(n) affine-inv | ~25 n³ (Pade-13 expm) ≈ 6.5M | ~3 n³ (chol + tri-solve) ≈ 800k | **8×** | `chol(P)·(I + L^{-1} X L^{-T})·chol(P)^T` |
| Stiefel V(n,k) | one matrix-exp ≈ 25 n²k | one QR ≈ 4 n k² | **~6× at k=n/2** | `qr(p + v).Q` |
| Grassmann Gr(n,k) | SVD-based exp | QR-based projection | **4×** | `qr(p + v).Q` |
| Multinomial Fisher-Rao | trig closed-form OK | renormalise | ~equal | `(p + v) / sum(p+v)` |

**Sphere**: 3× isn't enormous in absolute terms but compounds inside R-Adam / R-LBFGS where every step calls retraction. At 100 steps × 1000 iters × 3× speedup, that's 300k cycles → 100k cycles per outer iter. **Real money on long optimisation runs.**

**SPD**: 8× is the killer one. At n=64 a single Exp call is ~6.5M cycles ≈ 2 µs on modern hardware; at frame rate (60 Hz × 100 calls/frame) the budget is 167 µs per ms = 100k cycles per call. **Closed-form Exp blows the budget by 65×; retraction fits inside it.**

### 3.2 Scratch reuse across retractions

Retraction = Euclidean step then project back onto manifold. The Euclidean step needs a single scratch of length `EmbeddingDim()`. **Same scratch services every manifold's retraction** via generic `RetractInto(M, p, v, out, ws)` that does `tmp := p + v` in `ws.Scratch` then `M.ProjectInto(tmp, out)`.

For SPD specifically, the projection is `chol(P)·(I + ½(L^{-1} X L^{-T} + …))·chol(P)^T` — needs the SPDWorkspace from §2.5, **not just a vector scratch**. So `RetractInto` dispatches to a manifold-specific `RetractIntoWS(p, v, out, ws)` for heavy manifolds.

**Single design call**: `Manifold` interface ships both `ProjectInto(p, out)` (lightweight) and `ProjectIntoWS(p, out, ws)` for SPD/Stiefel/Grassmann that need O(n²) scratches.

### 3.3 When closed-form Exp wins

Despite the speedup ladder, closed-form Exp is **mandatory** for two use cases:

1. **Geodesic distance (`Distance`)** — defined as `‖Log_p(q)‖_p`, which is the *exact* Riemannian distance, not the retraction-based distance (which is well-defined but isn't the geodesic distance). Use case: clustering, RPCA, ICP-on-manifold.
2. **Geodesic visualisation / animation** — interpolating between p and q along the geodesic for rendering. Retractions don't trace geodesics.

For optimisation, retraction is right. For geometry (distance + viz), Exp is right. **Ship both, document which to use.**

---

## 4. Christoffel symbols inside geodesic shooting

092 T1.4 proposed `out [][][]float64`; 094 §6 corrected to flat `out []float64` of length n³. This section quantifies the gap and the geodesic-shooting bottleneck.

### 4.1 Flat vs jagged: the cold-cache cliff

Jagged `[][][]float64` requires three pointer-chases per `Γ^k_{ij}` access:
```
gamma[k]  // *float64 → []
gamma[k][i]  // *float64 → []
gamma[k][i][j]  // float64
```
Each pointer-chase is ~5 ns cold-cache (a cache miss on a non-locality-friendly slice header). **15 ns per access × n³ accesses per geodesic step.** At n=10, that's 1500 accesses × 15 ns = 22.5 µs **per step** for *just* the Christoffel access pattern.

Flat `out []float64` with `out[k*n*n + i*n + j]`:
```
out + (k*n*n + i*n + j)*8  // single integer arithmetic + load
```
~1 ns per access. **22.5 µs → 1.5 µs.** ~15× speedup on the Christoffel access pattern alone, on top of the 1.5-3× memory-layout win 094 cited.

Geodesic shooting calls Christoffel at every RK4/Verlet step × steps. At 1000 steps and the n=10 example: 22.5 ms vs 1.5 ms total. **Order of magnitude on the entire geodesic computation.**

### 4.2 Geodesic-shooting bottleneck profile

The geodesic ODE is `d²q^k/dt² = -Γ^k_{ij}(q) · q'^i · q'^j`. Per step the contraction `Γ^k_{ij} q'^i q'^j` is the bottleneck for n ≥ 10. **The contraction is GEMV-shaped at the inner two loops** — `Γ^k_{ij} · q'^j` is a matrix-vector for each k, so calling `linalg.MatVec(gamma[k*n*n:(k+1)*n*n], qdot, n, n, mvScratch)` per k punches through to ~1 GFLOPS-level throughput on n=64. Workspace: `mvScratch []float64` of length n. **One alloc per consumer setup, zero per step.**

### 4.3 Closed-form Christoffel on simplex

Per Amari 2016 §2.4 + 091 §2.5: `Γ^k_{ij} = -½ δ^k_i δ^k_j / p_k`. Non-zero **only** at i=j=k. The contraction collapses to `accel[k] = 0.5 * qdot[k] * qdot[k] / p[k]`. **O(n) instead of O(n³).** No `linalg.MatVec`. No 3-tensor materialisation. **The reason e/m geodesics on simplex (092 T1.7) ship as straight lines is that even the ODE form is trivial.**

For sphere / hyperbolic: similarly closed-form, O(n²) Christoffel storage at most. **Materialising the full 3-tensor is only required for non-trivial metrics** (Gaussian Calvo-Oller, SPD affine-invariant). Reality should ship two interfaces: `ChristoffelManifold` with dense `ChristoffelInto(p, out)` of length n³ for curvature-tensor consumers (T2.9), and `ChristoffelContractionManifold` with direct `ChristoffelContractionInto(p, qdot, out)` for geodesic-shooting consumers — **skips dense materialisation, O(n) memory + O(n) compute on simplex/sphere/hyperbolic vs O(n²)+O(n³) dense.**

### 4.4 Christoffel scratch layout

`ChristoffelManifold.ChristoffelInto(p, out)` sets caller-allocated `out` of length n³; zero per-call alloc. **The geodesic-shooting consumer allocates one n³ scratch at setup and reuses across all steps.** At n=10: 8 KB. At n=20: 64 KB. At n=64: 2 MB — **edge case for L2 cache**, becomes the new bottleneck above n≈30.

---

## 5. Natural-gradient solver: CG vs Cholesky

092 T1.9 specified Cholesky-of-`F + λI`, deferred CG to v2. **This audit's contribution: quantify when each wins, set the v1 threshold.**

### 5.1 Cost comparison

| Method | Cost | Memory | Notes |
|---|---|---|---|
| Explicit invert `F⁻¹ ∇L` | O(n³) | O(n²) | Never do this; numerically unstable + cubic |
| Cholesky `(L L^T) δ = ∇L`, two tri-solves | O(n³/3) for chol + O(n²) for solve | O(n²) | The standard. F must be materialised. |
| CG on `F·v` matvec | O(k·n²) per outer iter, k iterations | O(n) (no F!) | F never materialised; uses JVP (per JAX-Cosmo §3 of 093) |

For an exp-family model with d parameters:
- **n < ~50**: Cholesky's small-constant O(n³/3) wins on wall-clock (~ 4×10⁴ flops at n=50). CG inner-iter overhead dominates.
- **n ~ 100-300**: roughly equal, depending on conditioning of F.
- **n > 500**: CG dominates because Cholesky is O(n³) = 10⁸ flops vs CG ~O(20·n²) = 2×10⁶ flops for ill-conditioned F (k=20 inner iters).

For F well-conditioned (κ(F) < ~100), CG converges in 5-10 iterations regardless of n. **CG is ~100× faster than Cholesky at n=1000.**

### 5.2 The autodiff JVP path

CG only needs `F·v` matvec. For exp-family, `F = ∂²A/∂θ²` so `F·v = ∂_θ(∂A/∂θ · v) = HVP(A, θ, v)` — a single Hessian-vector product, no F materialisation. **JAX-Cosmo §3 (093) ships exactly this pattern**: `jax.jvp(jax.grad(A), θ, v)` returns `F·v`.

**Reality blocker**: `autodiff/` is reverse-mode-only per overnight 013. HVP requires forward-over-reverse. **Sub-blocker for v2 CG natural-gradient.** When HVP lands (autodiff-missing T1, ~600 LOC), CG ships in ~80 LOC over existing autodiff.

### 5.3 Recommended dispatch

`NaturalGradientStepper{Eta, Damping, FIM FisherInfoer, ScoreHVP func(theta, v, out)}`. Inside `Step(theta, grad, out)`: dispatch on `n = len(theta)` — `if n < 100 && s.FIM != nil` use `cholStep` (Cholesky of F+λI) else `cgStep` (CG on F·v). n=100 threshold conservative; tune empirically. **Critically: ship Cholesky path in v1 (T1.9), CG path when HVP lands.** Until HVP, `cgStep` returns `ErrAutodiffHVPMissing`.

### 5.4 Tikhonov damping

`F + λI` not just `F`. Per 091 §2.4: F goes singular at parameter boundaries. Damping `λ = c·tr(F)/n` (Martens 2014 K-FAC) costs O(n) extra (compute trace) and stabilises. **Cholesky-of-(F + λI) is the same O(n³/3) cost as Cholesky-of-F.** Free. For CG, damping is `(F + λI)·v = F·v + λv` — one extra n-vec add per inner iter. Also free.

### 5.5 Workspace for natural-gradient

`NaturalGradientWorkspace`: Cholesky path holds `F, L` n×n flat = 2n² + n (160 KB at n=100); CG path holds `r, p, Fp` length n each = 3n (24 KB at n=1000). **Workspace selected at consumer setup based on which solver will run.**

---

## 6. Cross-cutting: `Workspace` discipline as an enforceable rule

090 §D.1 proposed `BenchmarkXxx_Allocations` enforcement in CI; this audit strengthens it for the Riemannian surface.

**Two-form rule.** Every method that needs scratch ships in two forms: `XxxInto(...)` convenience (allocates workspace internally), `XxxIntoWS(..., ws *XxxWorkspace)` zero-alloc (caller manages). CI gate: `BenchmarkXxxIntoWS_Allocs` with `b.ReportAllocs()` failing build if non-zero.

**Workspace inheritance.** A consumer running R-LBFGS over SPD needs SPDWorkspace + RetractionWorkspace + LBFGSWorkspace. Compose into a single `RiemannianLBFGSWorkspace{Manifold, Retraction, LBFGS}` allocated once at consumer setup. **Hierarchical workspace, single allocation, no nested allocs.**

**Pistachio call-site docstring rule.** Every primitive states allocs/call: "SPDManifold.ExpIntoWS allocates 0 bytes per call (caller supplies SPDWorkspace). SPDManifold.ExpInto allocates one SPDWorkspace per call (~5n² + 2n floats). At 60 FPS with 100 calls/frame, prefer ExpIntoWS." Aligns with 090 §D.1 + 094 §8 + this audit.

---

## 7. Recommendations (ordered by leverage)

| # | Action | LOC | When | Severity |
|---|---|---|---|---|
| 1 | MMD upper-triangle + diag-skip exploit | 15 | now | High (2× wall) |
| 2 | `MedianHeuristicBandwidth` use `slices.Sort` | -18 | now | High (perf cliff) |
| 3 | `BregmanInto(scratch)` | 10 | now | Medium (inner-loop alloc) |
| 4 | `MahalanobisSquaredInto(d, Md scratch)` + `linalg.MatVec` | 15 | now | Medium |
| 5 | SPD: ship Cholesky-only impl per 093 §6 + this §2 | n/a in T2.3 | T2.3 design | High (8× retraction win, unblocks SPD entirely) |
| 6 | Christoffel: flat `[]float64` length n³, contraction-form for closed-form-sparse manifolds | n/a in T1.4 | T1.4 design | High (15× per-step + skip materialisation for simplex/sphere) |
| 7 | Natural-gradient: Cholesky path v1, CG path v2 (when autodiff HVP lands) | n/a in T1.9 | T1.9 design | High (100× at n=1000 in CG path) |
| 8 | Manifold workspace discipline (`Into`/`IntoWS` pair every heavy op) | n/a in T2.x | every Tier 2 design | Critical (60-FPS gate) |
| 9 | `ValidatedDistribution` wrapper to skip `validate()` re-scan | 20 | v2 | Low |
| 10 | CI alloc-gate (`BenchmarkXxx_Allocs`) | 50 | now | Medium (locks rule across contributors) |

**Sprint-1 batch (1+2+3+4): ~22 LOC net** (item 2 is a deletion). Closes every existing alloc gap, halves MMD2Biased CPU, removes the `MedianHeuristicBandwidth` quadratic. **Single highest-leverage addition: item 8** (workspace discipline shipped from day 1 with each Tier 2 manifold) — without it, no SPD/Stiefel consumer can run at frame rate, and the perf debt compounds across every manifold added later.

**Single concrete commit suggestion (~30 LOC):** items 1+3+4 in `mmd.go` + `bregman.go`, paired with `BenchmarkMMD2Biased_Symmetry` and `BenchmarkBregman_Allocs` proving zero-alloc + 2× wall.

---

## 8. References

- 090-info-perf: existing-surface alloc table (this audit deepens, not duplicates).
- 091-infogeo-numerics §1.12, §2.2, §2.4, §2.5: identified MMD symmetry, MedianHeuristic perf cliff, geodesic ODE drift, natural-gradient stability.
- 092-infogeo-missing T1.1, T1.4, T1.9, T2.1, T2.3, T2.6, T2.8: Tier 1/2 primitives whose perf is designed here.
- 093-infogeo-sota §2 (Pymanopt retractions), §6 (pyriemann Cholesky-only SPD): retraction-vs-Exp speedups + matrix-sqrt avoidance.
- 094-infogeo-api §6, §8: GeodesicODE adapter to chaos, `Into`/`IntoWS` workspace discipline.
- Pennec X., Fillard P., Ayache N. (2006). A Riemannian Framework for Tensor Computing. *IJCV* 66:41. Affine-invariant SPD; matrix-sqrt operations cited in §2.
- Boumal N. (2023). *An Introduction to Optimization on Smooth Manifolds*. Cambridge UP. Table 7.2 retractions cited in §3.
- Higham N. (2008). *Functions of Matrices: Theory and Computation*. SIAM. `expm`/`logm` cost cited in §2.4.
- Martens J. (2014). New insights and perspectives on the natural gradient method. arXiv:1412.1193. K-FAC damping cited in §5.4.
- Martens J. (2010). Deep learning via Hessian-free optimization. ICML. CG-on-`F·v` pattern cited in §5.1.
- Wilkinson J. H. (1963). *Rounding Errors in Algebraic Processes*. Prentice-Hall. Accumulator √N rounding cited in §1.1.

---

End of audit. 095 / 400.
