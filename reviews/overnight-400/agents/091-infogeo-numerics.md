# 091 — infogeo / numerical correctness

**Topic:** Fisher information matrix conditioning, geodesic ODE drift.
**Package:** `C:\limitless\foundation\reality\infogeo\` (4 src: doc, fdiv, bregman, mmd).
**Date:** 2026-05-07.
**Verdict:** *the requested numerics do not exist in the package.* The audit therefore
(a) documents that absence as the headline finding and (b) audits what is present:
f-divergences, Bregman divergences, MMD.

---

## 0. Scope reality check (highest-leverage finding)

MASTER_PLAN asks about Fisher information conditioning, geodesic ODE drift,
exp/log map precision, natural-gradient stability, Christoffel symbols,
α-connections. **None are implemented in `infogeo/`.** doc.go:55-74 is explicit:

> "MVP scope: f-divergences (KL, reverse-KL, JS, TV, Hellinger, χ², Rényi-α);
> Bregman divergences (sq Euclidean, gen-KL, Itakura-Saito, generic phi/grad);
> empirical (MMD² biased + unbiased)."

No `Fisher`, no `Geodesic`, no `Exp`/`Log`, no `Christoffel`, no
`NaturalGradient`, no α-connection apparatus. The package name is `infogeo`
but the contents are *information-theoretic divergences on the simplex*, not
*information geometry on a Riemannian manifold*. This is a **naming-vs-scope
hazard**: downstream consumers seeing `infogeo` will assume the
manifold/Fisher machinery is present.

**Recommendation 1 (5 LOC, no behaviour change):** amend doc.go:1 —

> "Package infogeo implements information-theoretic divergences on discrete
> probability vectors. Despite its name, it does NOT (yet) implement the
> Riemannian-manifold side — Fisher info, geodesics, exp/log, natural
> gradient, α-connections are out of scope for v1."

Single highest-leverage commit: 30 seconds, prevents months of misdirected
imports.

---

## 1. What IS implemented — numerical audit

### 1.1 `validate()` / `validatePair()` (fdiv.go:25-50)

- `probTol = 1e-9` is fine for n ≤ 10⁶; tight for larger n (uniform pmf of
  length 10⁹ stored as float64 sums to 1 ± ~1.1e-7, fails the gate). Doc the
  upper bound.
- Single-pass naive sum, no Kahan. Adequate for n < 10⁶; document.
- `v < 0` rejects negatives but allows zero (required for `0·log 0 = 0`).
  Verified.
- NaN check ordered before `<` comparison — safe.

### 1.2 `KL` (fdiv.go:65-81)

- `pi/qi` then `Log` is correct ordering (avoids `log pi − log qi`
  cancellation in the `pi ≈ qi` regime where users actually look).
- `0·log(0/q) = 0` is a branch (fdiv.go:71-73), not IEEE — correct
  (without it, `0 · -Inf = NaN` would propagate).
- `pi > 0 && qi == 0 → +Inf` returned, not errored — defensible (matches
  scipy `entropy`); doc-note for consumers expecting finite range.
- **Latent cancellation regime:** `p = (0.5+ε, 0.5−ε)`, `q = (0.5−ε, 0.5+ε)`,
  KL ≈ 4ε² + O(ε⁴). Two summands cancel from O(ε) to O(ε²). For ε=1e-7 the
  answer is ~4e-14, within an OOM of float64 ε. **Recommendation 2:** add a
  golden vector in this regime (1 line) so future refactors can't silently
  regress; consider Kahan if downstream complains.

### 1.3 `JS` (fdiv.go:98-117)

- Computes `m = 0.5(p+q)` on the fly (zero-alloc). Good.
- `mi == 0` (i.e., pi=qi=0) skipped; `pi`, `qi` checked separately for
  one-sided zeros. Bounded by log 2 means no cancellation concern.
- TestJS_BoundedByLog2 confirms saturation at log 2 for disjoint support.

### 1.4 `Renyi` (fdiv.go:196-227)

The most numerically interesting function.

- Uses `math.Pow(pi, alpha) * math.Pow(qi, 1-alpha)`. Textbook but **not
  stable**. For α=10 and pi=1e-6, `pi^10 = 1e-60`; the product can underflow
  to 0 prematurely. If every term underflows the function returns `+Inf` via
  the `sum <= 0` branch — **WRONG**: correct answer is `(LSE)/(α-1)` where
  LSE is over `α·log pi + (1-α)·log qi`. **Latent bug** for large-α tail
  regimes.
- **Recommendation 3:** rewrite via log-sum-exp:

  ```
  xs[i] = α·log(pi) + (1-α)·log(qi)   // skip pi=0
  Renyi = LSE(xs) / (α - 1)
  ```

  ~15 LOC. Closes the underflow hole; also faster (one `exp` + one `log`
  per term vs. two `Pow`).

- `alpha == 1.0` rejected (correct). TestRenyi_LimitMatchesKL uses α=1.001
  with 1e-2 tolerance — appropriate for that α (residual is O(α-1) = 1e-3),
  but masks lack of pinning at α-1=1e-6 where consumers actually call Rényi.
- `alpha <= 0` rejected. The doc-comment lists α=0 (Hartley) and α=∞
  (max-divergence) limits but the function rejects them — **docs/code
  mismatch**. **Recommendation 4:** implement the limits OR tighten doc to
  "α ∈ (0,1)∪(1,∞), open at both ends; α=0/α=∞/α=1 not yet implemented." 5
  LOC option.
- `sum <= 0 → +Inf` conflates underflow with absolute-continuity failure;
  fine for "answer is unbounded" semantics, loses diagnostic distinction.

### 1.5 `Hellinger` (fdiv.go:145-155)

`sqrt(0.5·sum(sqrt pi - sqrt qi)²)`. Numerically excellent: `sqrt` monotone,
squared-difference of square-roots is well-conditioned. No issues. Bounded
in [0,1], saturates at 1 for disjoint support (verified). The cleanest
function in the file.

### 1.6 `ChiSquared` (fdiv.go:163-180)

`(pi-qi)²/qi`. `d == 0` skip avoids 0/0 when pi=qi=0. `pi>0, qi=0 → +Inf`
correctly handled (the `qi==0` branch fires only after `d==0` skip, so when
hit, pi>0 necessarily). Single-pass, no Kahan.

### 1.7 `TotalVariation` (fdiv.go:125-134)

`0.5·sum|pi-qi|`. Bounded by 1, saturates at 1, no numerical concerns.

### 1.8 `Bregman` framework (bregman.go:26-45)

- `gradY` allocated per call (bregman.go:36) — known scratch hot spot
  inside mirror-descent inner loops (per perf audit 090). Worth a
  `BregmanInto(gen, x, y, scratch)` companion, ~10 LOC.
- `phiX - phiY - dot` (bregman.go:44): **catastrophic cancellation when
  x ≈ y.** All three terms ~equal in magnitude; answer is the small
  surviving quantity. For `x = y + ε`, the Taylor of D_φ is
  `0.5·ε^T·H_φ(y)·ε + O(ε³)` (quadratic in ε), but the formula computes
  it as O(ε) differences; ~16 bits of precision lost per coord at
  ε=1e-8. **Recommendation 5:** doc-note that for `||x-y|| < 1e-6` the
  formula is cancellation-prone; caller should use a Taylor formulation
  (mid-point Hessian for symmetric Bregman). Out of scope for v1, but
  flag.

### 1.9 `GeneralisedKL` (bregman.go:75-95)

`xi·log(xi/yi) - xi + yi`. The `-xi+yi` is what makes this Bregman (not
just KL on non-simplex). `xi == 0` → `sum += yi` (limit `lim_{x→0}
x·log(x/y) - x + y = y`). `yi == 0` (with xi>0) → +Inf. No issues.

### 1.10 `ItakuraSaito` (bregman.go:109-123)

`x/y - log(x/y) - 1`. Minimum at x=y (value 0). For `x ≈ y` the Taylor
is `0.5·((x-y)/y)² + O³`. The formula computes this as
`(small) - log(1+small) ≈ small - (small - small²/2) = small²/2` — OK,
but `math.Log(x/y)` for x/y very near 1 loses precision. **Recommendation
6:** for `|x-y|/y < 1e-8`, use `math.Log1p((x-y)/y)`. ~3 LOC. Today's
TestItakuraSaito_ZeroAtEqual passes only because x==y exactly; off-by-1ULP
not exercised. Scale-invariance test at bregman_test.go:82 is good.

### 1.11 `MahalanobisSquared` (bregman.go:133-156)

- No PD check on M (caller responsibility); for κ(M)>1e8, ~half
  precision lost; no warning.
- `d^T·M·d` computed as nested-sum bilinear form. Symmetric M's
  numerically better path is Cholesky `M = L·L^T`; then
  `d^T·M·d = ||L^T·d||²` is purely additive sum-of-squares — no
  cancellation. **Recommendation 7:** `MahalanobisSquaredCholesky(x, y, L)`
  variant (~20 LOC). Faster + more accurate when caller already has L.

### 1.12 MMD (mmd.go)

- **`MMD2Biased`** (mmd.go:64-103): three O(N²) double loops. The
  estimator IS theoretically non-negative (squared RKHS norm) but
  the implementation does NOT clamp at 0; rounding could push
  ~N²·ε below 0. For N=10⁶, ~1e-4 rounding. **Recommendation 8:**
  Kahan-compensated summation (~30 LOC) — N² adds is exactly the
  regime where Kahan pays off.

- **`MMD2Unbiased`** correctly does NOT promise non-negativity (doc
  states slight negativity allowed on finite samples). Good.

- **Inner loop calls `k(xi, xj)` for both (i,j) and (j,i).** Gaussian
  kernel is symmetric, so this is **2× wasted work**. Compute upper
  triangle, double it, single diagonal pass for `k(xi, xi) = 1`.
  **Recommendation 9:** ~15 LOC, 2× speedup, halves accumulator
  rounding (also flagged by perf audit 090).

- **`GaussianKernel`**: `exp(-||x-y||²/(2σ²))`. Large arg underflows
  cleanly to 0. Near-zero arg `exp(-tiny) ≈ 1 - tiny` loses precision;
  `Expm1` formulation rarely worth it for MMD (signal is in
  off-diagonal terms).

- **`MedianHeuristicBandwidth`** (mmd.go:169-189): full O(N²)
  pairwise distances, then in-place **insertion sort** (median.go:191-208).
  Insertion sort for the median is **astronomically wrong for N>100**
  (O(N⁴/4) when called via O(N²) distances); `slices.Sort` (or
  quickselect for the median only) gives O(N²·log N) total.
  **Recommendation 10:** quickselect (~50 LOC) or `slices.Sort`. Run
  once per MMD call so absolute time may not show up in profile, but
  the cliff is at N≈1000.

### 1.13 Autodiff parity test (`autodiff_test.go`)

The strongest numerical assertion in the package:

> `∇_θ KL(p || softmax(θ)) = softmax(θ) − p` to 1e-9 per coordinate.

R-CLOSED-FORM-PINNED-TO-AUTODIFF pattern's second consumer (saturation
2/3). The 1e-9 per-coord tolerance is appropriate: ~5 elementary AD
ops per coord, cumulative ~5ε ≈ 5e-16 × gradient magnitude (≤1) — 7
OOM safety margin to 1e-9. The 1e-12 value-mismatch tolerance is also
appropriate. **Indirectly validates every code path in KL** except
the pi=0 / qi=0 boundary branches (test inputs are all strictly
interior). **Recommendation 11:** add a fourth case at θ=(0,0,0,-50)
where q[3]=exp(-50)/Z is denormal-adjacent — tests softmax-via-
subtract-max robustness at boundary. ~10 LOC.

---

## 2. The "Fisher / geodesic / α-connection" gap (forward-looking)

Numerical landmines worth flagging now:

### 2.1 Fisher information matrix

- **Exponential family** `p(x;θ) = h(x)·exp(η(θ)·T(x) - A(θ))`: FIM = ∂²A/∂θ².
  No numerical drift; closed form.
- **Non-exponential**: `E[∂_θ log p · ∂_θ log p^T]` via MC/quadrature. **Becomes
  singular at parameter boundaries** (variance→0 in Gaussian; mixture weight→0
  in Bernoulli mixture); κ(F) → ∞.
- **Mitigation**: Tikhonov `F + λI` with `λ = c·tr(F)/n` (Martens 2014 K-FAC),
  empirical c. Cleaner: Fisher-Rao trust region refusing boundary steps. Ship
  FIM with a `cond(F)` query; let downstream pick the regulariser.

### 2.2 Geodesic ODE drift

- `d²θ^k/dt² + Γ^k_{ij}·dθ^i/dt·dθ^j/dt = 0`. Naive RK4 (which `chaos/` has)
  drifts off the manifold for hyperbolic geometries (Poincaré disk: ||θ||<1
  constraint, RK4 happily steps past 1).
- **Symplectic alternatives**: Verlet/leapfrog on the Hamiltonian
  `H = 0.5·g^{ij}·p_i·p_j` preserves H = const to ~ε per step (vs. RK4's
  ε·t drift). Reality has no symplectic integrator yet; recommend
  `chaos.Verlet` cross-cutting (~80 LOC) before geodesics land.
- **Manifold projection retraction**: cheaper alternative — after each Euler
  step, project back (renormalise simplex). ~20 LOC. Use for v1; symplectic
  is v2.

### 2.3 Exp / log map precision

- **Near identity**: `exp_p(v) = p + v + O(||v||²)`. For Fisher-Rao on
  multinomial, closed form `(sqrt(p_i)·cos(||v||) + (v_i/2||v||)·sin(||v||))²`
  (Amari 2016). At v=0, `cos→1-||v||²/2`, `sin/||v||→1`, both well-conditioned.
- **Near antipode** (||v||→π): `cos(π)→-1`, `sin(π)/π→0`. Catastrophic
  cancellation between `sqrt(p_i)·cos(||v||)` and `(v_i/2π)·sin(||v||)`. Use
  `cos(π-x) = -cos(x)` reduction when `||v|| > π/2`. ~5 LOC.

### 2.4 Natural gradient G⁻¹∇L stability

- `θ_{t+1} = θ_t - η·G⁻¹·∇L`. `G⁻¹` itself is the hazard: at boundary G is
  singular.
- **Standard fix**: **conjugate gradient** for `δ = G⁻¹∇L` rather than explicit
  inversion (Martens 2010 Hessian-Free). CG needs only `G·v` matvec
  products, doable as JVPs in autodiff without forming G. ~50 LOC over
  existing `linalg/iterative` (if present) or ~150 LOC.
- **Damping**: `(G + λI)·δ = ∇L` with adaptive λ via Levenberg-Marquardt
  trust region — `optim/` already has trust-region machinery; reuse.

### 2.5 Christoffel symbols

- **Closed form for Fisher-Rao on simplex**: `Γ^k_{ij} = -0.5·δ^k_i·δ^k_j·(1/p_k)`.
  Pointwise; only issue is boundary `p_k → 0` (same singularity as natural-
  gradient blowup).
- **By autodiff of metric**: `Γ^k_{ij} = 0.5·g^{kl}·(∂_i g_{jl} + ∂_j g_{il} -
  ∂_l g_{ij})`. Each `∂g` is second-order autodiff (Hessian). Reality's
  `autodiff/` is reverse-mode-only (per overnight 013 SOTA); forward-over-
  reverse for Hessians is a known forward direction.

### 2.6 α-divergences and α-connections

- **α-divergence** `D_α(p||q) = (1/(α(1-α)))·(1 - sum p^α·q^{1-α})`. Limits
  α=0 → KL(q||p), α=1 → KL(p||q), α=1/2 → 4·(1 - Hellinger affinity).
  Algebraically related to Rényi-α, distinct. **Same `p^α·q^{1-α}` underflow
  hazard** — fix in Recommendation 3 should be lifted into α-divergence at
  the same time.
- **α-connection** `∇^{(α)} = (1-α)/2·∇^{(m)} + (1+α)/2·∇^{(e)}` interpolates
  mixture (m, α=-1) and exponential (e, α=+1) connections. Christoffels
  follow.
- **Dual flatness**: α=±1 are flat in their respective coordinates;
  **geodesics are straight lines** there (no ODE, no drift): `θ_t = θ_0 + tv`.
  **Ship α=±1 geodesics as v1 minimum** — trivial numerically, covers
  majority of practical use (exp-family natural-parameter geodesic IS the
  α=1 geodesic).

---

## 3. Cross-package observations

- **No tie-in to `prob/`'s `Distribution` interface.** Per overnight 089-info-api,
  natural home for Fisher is on `prob.Distribution` (FIM is from log-likelihood).
  Closed-form FIM as a method on concrete types `(d *NormalDist) FisherInfo()`;
  generic `FisherInfoNumerical(d, θ)` fallback. Mirrors existing
  `KLDivergence` / `KLDivergenceNumerical` split.
- **No tie-in to `linalg/` Cholesky for FIM solves.** Natural-gradient
  `G⁻¹∇L` should route through `linalg.CholeskySolve`, not explicit inversion.
- **No tie-in to `chaos/` ODE solvers for geodesics.** `chaos.RK4` exists;
  symplectic Verlet does not. Geodesics need symplectic.

---

## 4. Recommendations (ordered by leverage)

| # | Item | LOC | Severity |
|---|------|-----|----------|
| 1 | Doc-comment narrowing package's stated scope | 5 | High (prevents misimport) |
| 2 | Golden vector for `p≈q` cancellation regime in KL | 1 | Low |
| 3 | LSE-based Renyi-α inner loop | 15 | Medium (latent underflow bug) |
| 4 | Implement Renyi α=0,∞ limits OR tighten doc | 5-30 | Low (docs/code mismatch) |
| 5 | Doc on Bregman cancellation for ||x-y||→0 | 3 | Low |
| 6 | `Log1p` in Itakura-Saito for x≈y | 3 | Low |
| 7 | `MahalanobisSquaredCholesky` variant | 20 | Medium (perf+precision) |
| 8 | Kahan summation in MMD double loops | 30 | Medium (large-N) |
| 9 | Symmetry exploit in MMD inner loop | 15 | High (2× speedup, halves rounding) |
| 10 | Quickselect/`slices.Sort` median in `MedianHeuristicBandwidth` | 50 | High (perf cliff) |
| 11 | Boundary-θ test in autodiff KL gradient pin | 10 | Medium |

**Sprint-1 batch (1+9+10):** ~70 LOC, closes one perf cliff, doubles MMD
throughput, removes the largest documentation-vs-implementation gap. **Single
highest-leverage:** #1 — 5 LOC, prevents the entire class of "where is Fisher?"
bug reports the package name otherwise generates.

**Forward-looking:** when Fisher/geodesics/α land — exp-family closed-form
Fisher first (no drift); CG-based natural-gradient solves not explicit
inverses; symplectic Verlet for non-trivial geodesics; α=±1 geodesics first
(straight lines in the right coordinates).

---

End of audit. 091 / 400.
