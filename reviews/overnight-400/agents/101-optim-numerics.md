# 101 — optim: Numerical Correctness Audit

**Scope.** Files audited (top-level `optim/`):
- `gradient.go` (GradientDescent, LBFGS, lbfgsLineSearch)
- `gradient_validated.go` (GradientDescentValidated, LBFGSValidated)
- `linear.go` (SimplexMethod, InteriorPoint)
- `rootfind.go` (BisectionMethod, NewtonRaphson, GoldenSectionSearch, LinearInterpolateRoot)
- `genetic.go` (GeneticAlgorithm)
- `metaheuristic.go` (SimulatedAnnealing)
- `interpolate.go` (LinearInterpolate, CubicSplineNatural)

Subpackages `proximal/` and `transport/` are out of scope here (separate review topics).

Severity legend: **CRIT** = wrong result / silent failure; **HIGH** = numerical robustness / classical-method violation; **MED** = missing safeguard / API contract; **LOW** = cosmetic or doc.

---

## Findings

### F1 — `lbfgsLineSearch` does not satisfy any classical Wolfe condition (HIGH)
File: `gradient.go:194-220`.

The doc-comment at `gradient.go:62` says LBFGS uses "Wolfe line search" and the line-search header says "Armijo condition", but the routine implements **only the sufficient-decrease (Armijo) test** and never checks the curvature condition `|g(x+αd)·d| ≤ c2|g·d|`. Consequences:

1. The doc comment at line 62 is **wrong** ("Wolfe" vs "Armijo backtracking").
2. Without curvature, the curvature pair `s·y` is **not guaranteed positive** even on convex functions; the code at line 167 has the "skip update if `sy ≤ 0`" guard, so positive-definiteness of the implicit `H_k` is preserved by *skipping* updates rather than by line-search design. On ill-conditioned problems with steep descent steps, many updates can be skipped, degrading L-BFGS to scaled steepest descent.
3. There is no descent-direction guard: if numerical noise makes `dg = g·d ≥ 0` (e.g., near a saddle, or when the implicit `H` becomes nearly indefinite from skipped pairs), the Armijo test `f_trial ≤ fx + c1·step·dg` becomes a *non-decrease* condition (because `c1·step·dg ≥ 0`), so the line search can return a step that *increases* `f`. Recommend: if `dg ≥ -tiny`, fall back to `d = -g`.
4. The line search returns the last shrunken step (`step *= shrink` for 40 iterations → `2^-40`) **without signalling failure**; the outer loop then takes that microscopic step and continues, producing ghost convergence.

The **`grad` parameter is captured but never used** (`_ = grad` at line 195) — explicit proof Wolfe is not implemented.

Action: rename the doc to "backtracking Armijo" *or* implement strong-Wolfe via cubic interpolation (Nocedal & Wright Alg. 3.5/3.6) and report failure status.

---

### F2 — L-BFGS skips the **damped BFGS** update on `sy ≤ 0` instead of damping (HIGH)
File: `gradient.go:166-170`, `gradient_validated.go:218-221`.

Standard practice for non-convex problems (Nocedal & Wright §18.3, "Damped BFGS Update"; Powell 1978) is to **damp** `y_k` toward `Bs_k` so the curvature condition holds, e.g.:

```
θ = 1                                 if s·y ≥ 0.2 s·Bs
θ = 0.8 s·Bs / (s·Bs - s·y)            otherwise
y_k ← θ y_k + (1-θ) B s_k
```

Skipping is also acceptable for L-BFGS but is **silent** here: the iterate continues, the user does not know the Hessian model was rejected. At minimum, log/return a counter; ideally implement Powell damping.

There is also no guard against `|s·y|` being denormally tiny on a very flat problem: `rhok = 1/sy` can blow to `+Inf` and the next two-loop recursion will produce `NaN`s. Add `if math.Abs(sy) < 1e-300 { continue }`.

---

### F3 — L-BFGS initial scaling `H_0 = γ I` uses fragile fallback (MED)
File: `gradient.go:120-129`, `gradient_validated.go:184-191`.

```go
gamma := 1.0
if k > 0 {
    sy := vecDot(sHist[k-1], yHist[k-1])
    yy := vecDot(yHist[k-1], yHist[k-1])
    if yy > 0 { gamma = sy / yy }
}
```

Issues:
1. When `yy = 0` (zero gradient change), `γ = 1` is used. This is a *step size of 1 in raw gradient units* — for a poorly scaled problem this can be 10⁶ off and the resulting first step explodes. Better: use the *previous* `γ`, or scale by `1/||g||`.
2. `sy/yy` is the well-known Shanno-Phua scaling and is generally non-negative when `sy > 0` (which is enforced by the curvature gate above), but it is **not bounded away from zero** — on a near-linear ridge, `sy → 0` and `γ → 0`, producing direction `d = -γq ≈ 0`, which causes the line search to do 40 backtracks on the zero direction and return the dead step. There is no termination on `||d|| < eps`.

---

### F4 — `SimplexMethod` Bland-rule tie-break has out-of-bounds access (CRIT)
File: `linear.go:113`.

```go
if ratio < minRatio || (ratio == minRatio && basis[i] < basis[leaving]) {
```

When the *first* candidate (with `tableau[i][entering] > 1e-10`) is encountered, `leaving == -1` from the initialisation at line 108, and `basis[leaving]` is `basis[-1]` — **panic, runtime out-of-range**. The path is taken whenever the first row with positive pivot has `ratio == minRatio` (which equals `+Inf` initially, so this branch never fires the first time — but the second condition is short-circuited by `||`, so the bug is **latent**, not active). Still, the code is wrong as written: any future refactor that changes `minRatio = math.Inf(1)` to `minRatio = math.MaxFloat64` will unmask the panic. Fix:

```go
if leaving < 0 || ratio < minRatio || (ratio == minRatio && basis[i] < basis[leaving]) {
```

Also note the tie-break is **incomplete Bland's rule**: Bland requires the *smallest index in the basis* among rows that *attain* the min ratio, not just the candidate compared against the current best. Because the loop goes in order `i = 0..m-1` and replaces only on strict improvement *or* lex-smaller basis index, the result is correct only because the ratio test is exact-equal; with floats, two equal ratios that should be a tie may differ by an ULP and the tie-break is skipped → cycling possible. Add an epsilon: `math.Abs(ratio-minRatio) < 1e-12`.

---

### F5 — `SimplexMethod` reduced-cost loop is O(m·n) per iteration but rebuilt each time (HIGH numerics, perf)
File: `linear.go:88-101`.

The "revised simplex" doc-comment is **misleading** — the implementation is the *full tableau* simplex (it carries the entire `m × (n+m)` tableau and pivots all of it). Numerical issue: the reduced cost at line 93-96 is recomputed from the *running pivoted tableau* via `c[j] - c_B' * tableau[:, j]`. Since `tableau` accumulates pivot error, the reduced costs drift. After many iterations on ill-conditioned `A`, the entering-variable test `rc < -1e-10` fires on numerical noise, causing **fake non-optimality** and infinite loops up to `maxIter = 10000`.

The classic fix is **periodic LU re-factorisation of the basis** (revised simplex) or, for the tableau form, periodic reset by re-solving `B x_B = b` from the original `A`. Neither is present.

The unbounded check at line 119-121 is correct; the infeasibility detection is **missing entirely** — for a problem where `b ≥ 0` is enforced by row-flipping (lines 58-63), the slack-variable initial basis is always feasible, so no Phase-I is needed *for `Ax ≤ b, x ≥ 0`*. Good. But this means the function silently solves only this one form; the doc says so, but a user with `Ax = b` will get a nonsensical answer because there is no Big-M / two-phase support.

---

### F6 — `SimplexMethod` ignores zero entering variable when `c[j] = 0` for an artificial slack (MED)
File: `linear.go:77-78`.

`cost = [c, 0, 0, ..., 0]` (slacks have zero cost). Reduced cost of a slack column equals `0 - c_B' * (column)`. When the basis is *itself* slacks, this is `-c_B[i]·1 = 0` for the slack-in-basis and zero for not-in-basis. So at iteration 0, `rc < -1e-10` only fires for true `x` columns with negative reduced cost — fine. But after pivoting, if a slack leaves and a real `x` enters, the new reduced costs of *remaining slacks* can be negative, and Bland's rule (first index) re-enters them, undoing work. This is *valid* for Bland (it cannot cycle), but it is wasteful. Pure Dantzig (most-negative rc) is faster and the cycling theorem only requires Bland on degeneracy. Recommend: Dantzig's rule, with Bland fallback after `k` degenerate iterations.

---

### F7 — `InteriorPoint` is **not a primal-dual interior-point method** (CRIT — wrong algorithm)
File: `linear.go:172-316`.

The doc comment says "primal-dual interior-point (barrier) method ... Wright 1997". The implementation is none of the standard variants (Mehrotra predictor-corrector, MPC; long-step path-following; affine-scaling). Specifically:

- **No KKT system**: there is no Newton step solving
  ```
  [ 0  Aᵀ  I  ] [Δx]     [r_d]
  [ A  0   0  ] [Δλ]  =  [r_p]
  [ S  0   X  ] [Δs]     [r_c]
  ```
  Instead, line 270-274 uses the heuristic
  ```
  dx[j] = -rd[j] * x[j]^2 / (mu + x[j]*x[j])
  ```
  which is a **diagonal scaling of the dual residual** — this is *gradient-style* update, not Newton. Convergence is not even Q-linear; the geometric `mu *= 0.2` schedule will hit `mu < 1e-12` long before primal feasibility is achieved on any non-trivial problem.
- **Lambda is nudged by `lambda += 0.1 * rp`** (line 278) — this is dual-ascent on the primal residual, completely separate from the Newton system. The two updates (`x` and `λ`) do not satisfy any joint linearisation.
- **Centring parameter `sigma = 0.2` is fixed** — Mehrotra's adaptive σ is the whole point.
- **Step length** at line 282-296 ensures positivity but uses **the same α for both x and slacks**; primal-dual interior-point uses *separate* primal/dual step lengths.
- **Dual residual is `-λᵢ - μ/sᵢ`** (line 247-251) which would be the *complementarity* residual `s·λ - μ` if `λ` had the right sign — it does not, so the equations being "solved" are not the KKT system of the LP.

**Empirically** this routine returns reasonable values for tiny, well-conditioned LPs only by luck (the heuristic gradient happens to point downhill on the barrier surface). On any real LP it will return a far-from-optimal interior point.

The function should either be:
- (a) **renamed** `BarrierGradientHeuristic` with a clear "not for production LP" disclaimer, *or*
- (b) replaced with an actual MPC (≈200 lines, possible without dependencies; reuse `linalg` Cholesky for the normal-equations form `AΘAᵀΔλ = ...`).

This is the single biggest correctness gap in `optim/`.

---

### F8 — `BisectionMethod` does not validate sign change (MED)
File: `rootfind.go:22-38`.

If the user calls `BisectionMethod(f, 0, 1, tol)` with `f(0) = 1, f(1) = 1` (no sign change), the code bisects forever (well, until `b-a < tol`) and returns midpoint of the original interval — **silently wrong**. Add an upfront check: `if math.Signbit(fa) == math.Signbit(f(b)) && fa != 0 && f(b) != 0 { return NaN, errors.New("no sign change") }`.

Also: the loop uses `math.Signbit(fa) == math.Signbit(fm)` which incorrectly classifies `+0.0` and `-0.0` (signbit of `-0.0` is true). For roots crossing through exact zero, this can choose the wrong sub-interval. Use `(fa < 0) == (fm < 0)` and handle `fm == 0` separately (already done at line 27).

---

### F9 — `NewtonRaphson` derivative-zero fallback returns mid-iterate silently (HIGH)
File: `rootfind.go:59-61`.

```go
if fpx == 0 {
    return x // derivative vanished — return best guess
}
```

No error signal, no fallback to bisection. If `f'` is just *very small* (not exactly zero), the next iterate `x - f/f'` overflows to `±Inf` and subsequent `f(Inf)` evaluations are usually `NaN` — the function then loops with `x = NaN`, `math.Abs(NaN) < tol` is **false** (good), but `NaN - NaN/NaN = NaN` propagates and the function returns `NaN` after `maxIter` with no diagnostic. Recommend:

- guard `math.Abs(fpx) < tinyDerivative` and return error/fallback;
- detect non-finite `x` and exit with an error.

The function signature `func(...) float64` cannot return an error — propose a `NewtonRaphson2` that returns `(float64, error)` or use `(x, ok)` style. This matches the pattern already in use in `linear.go`.

---

### F10 — `NewtonRaphson` has **no globalisation** (HIGH, by design)
Doc says "may diverge for bad initial guesses" — accepted. But the audit baseline is "Newton-Raphson with derivative=0 fallback"; the project does not provide a **damped Newton** or **Newton-with-bisection** safeguarded variant. For a foundation library that promises correct answers, providing only pure Newton is a gap. Recommend: add `SafeNewton(f, fp, a, b, tol, maxIter)` that maintains a bracket and falls back to bisection when Newton steps outside `[a, b]` (Numerical Recipes §9.4 `rtsafe`).

There is **no Brent's method** (`rtsafe` / `zbrent`), no **Halley's**, no **secant**. `LinearInterpolateRoot` is a single secant *step*, not the iterative method. For a "classical numerical methods" library this is a notable omission.

---

### F11 — `GoldenSectionSearch` does not accept `a > b` and is silent on non-unimodal (LOW)
File: `rootfind.go:78-103`.

If `a > b`, the loop condition `b - a > tol` is false on first iteration and the function returns the (wrong) midpoint without error. Add `if a > b { a, b = b, a }`.

If `f` is not unimodal, golden-section converges to a **local** minimum without warning. Doc says "unimodal" — acceptable, but add a sanity check on the final probe pattern (e.g., warn if the residuals at boundaries are smaller than the returned minimum).

There is no **parabolic interpolation** layer, i.e., **Brent's 1-D minimiser** (golden-section + inverse parabolic interpolation, Numerical Recipes §10.3 `brent`) is missing. This is a standard companion to golden-section — strongly recommended.

---

### F12 — `LinearInterpolateRoot` returns `NaN` on `y0 = y1` but does not signal "not a root" vs "horizontal line" (LOW)
File: `rootfind.go:112-118`. Acceptable but the doc says "horizontal line — no unique root"; the case `y0 = y1 = 0` is *every* point being a root. Doc wording is fine; behaviour is fine.

---

### F13 — `GeneticAlgorithm` hard-codes search domain `[-5, 5]^dim` (HIGH usability, MED correctness)
File: `genetic.go:67-76`, doc `genetic.go:36-39`.

The doc explicitly tells callers to "internally transform coordinates" inside their fitness function. This is fragile: the *mutation Gaussian noise* (`sigma * normalRand()`) and *BLX-α extrapolation* (`α = 0.5`, extends interval by 50%) operate in the *raw* coordinate space, so the user's transform is bypassed by mutation. Result: probability mass concentrates near `[-5, 5]` even when the user's transform maps to `[0, 10⁶]`. Fix: take an explicit `bounds [][2]float64` parameter, or normalise to `[0, 1]` and let the user transform.

Mutation `sigma = 1.0 * (1 - gen/(gens+1))` ramps from 1.0 to ~0.01 — fixed schedule, not adaptive to fitness landscape. No diversity preservation. **No stopping criterion** — `gens` is an iteration budget, not a convergence test. This is acknowledged GA practice but the doc claims "global optimization": for non-trivial functions, fixed budgets give no convergence guarantee. Recommend documenting "stochastic best-so-far estimator" rather than "global optimizer".

Box-Muller at line 58-65 uses `cos(2π u₂)` only — discards the `sin` half (loses 50% of normal samples per call). Cheap fix: cache the sin half across calls.

---

### F14 — `GeneticAlgorithm` BLX-α generation off-by-design when parents identical (LOW)
File: `genetic.go:130-140`.

When `pop[p1][j] == pop[p2][j]`, `lo = hi`, `span = 0`, `cLo = cHi = lo`, child `c1 = c2 = lo` for that gene — **no crossover diversity**. Combined with mutation rate `0.1` per gene, premature convergence is likely. Standard fix: when `span < eps`, sample uniformly from `[lo - σ, lo + σ]`.

---

### F15 — `SimulatedAnnealing` cooling schedule is multiplicative-only (LOW)
File: `metaheuristic.go:38-93`.

Geometric cooling `T ← 0.999·T` is implemented; **no logarithmic schedule** (`T_k = T_0 / log(k+1)`, the only one with theoretical convergence-in-probability to global optimum, Hajek 1988). Doc claim of being "a metaheuristic" is fine, but the **Reference** to Kirkpatrick implies the user might infer convergence; spell out "no theoretical global-optimum guarantee under geometric cooling".

There is **no stopping criterion** beyond `maxIter` — no *equilibrium detection*, no *acceptance-rate* monitoring. Practical SA implementations stop when the rolling acceptance rate drops below ~5%.

`if !accept && temp > 0` on line 72 — once `temp` underflows to `0` (with cooling 0.999 over 16,384 iterations, `temp ≈ temp0 · 1e-7`; with 100k iterations it's 1e-43, denormal), the worse-move acceptance becomes hard rejection. Acceptable; document the effective `iter * log(1/cooling) ≈ ln(temp0 / minDouble)` cutoff.

---

### F16 — `CubicSplineNatural` does not check `n == 2` separately (MED)
File: `interpolate.go:44-156`.

When `n == 2`, the conditional at line 70 (`if n > 2`) skips the tridiagonal solve entirely, so `c[] = [0, 0]`. The polynomial coefficients at line 119-121 then degenerate to `bᵢ = (y₁-y₀)/h`, `cc = 0`, `d = 0` — i.e., a **straight line**. This is the *correct* natural-spline answer for two points, so it works. Good. Worth a unit test pin, though, if not present.

The closure at line 132-155 does a **binary search** every call. For monotone query patterns this is wasteful; a Pistachio-style 60-FPS caller would benefit from a "hint" parameter (last index). Not a correctness issue.

---

### F17 — `LinearInterpolate` at `interpolate.go:18-20` does not panic on `x0 == x1` despite doc (MED)
The doc says "Panics if x0 == x1 (degenerate interval)". Implementation just divides:
```go
return y0 + (y1-y0)*(x-x0)/(x1-x0)
```
Yields `±Inf` or `NaN` (if also `x == x0`). Either implement the panic the doc promises, or update the doc to say "returns Inf/NaN".

---

### F18 — Missing classical algorithms (informational, not bugs)
Algorithms named in the audit topic list that are **not present**:
- **Brent's 1-D root finder (`zbrent`)** — has bisection + Newton + golden-section, no Brent.
- **Brent's 1-D minimiser** — same, no parabolic interpolation.
- **Levenberg–Marquardt** — completely absent. (For a foundation library that includes L-BFGS, omitting LM for nonlinear least-squares is surprising; LM is the workhorse for parameter fitting.)
- **CMA-ES** — absent.
- **Conjugate gradient (Fletcher–Reeves / Polak–Ribière)** — absent. The closest is `GradientDescent`. This is a notable gap because CG is *the* classical method for unconstrained smooth optimisation when a Hessian-vector product is available.
- **Trust-region methods** — absent.
- **Augmented Lagrangian / KKT solvers** — absent. There is no constrained nonlinear optimisation at all.
- **Newton with cubic interpolation line search** — absent.

Strict ranking by frequency-of-need for a foundation library:
1. Conjugate gradient (FR/PR with restart)
2. Brent's 1-D minimiser (companion to `GoldenSectionSearch`)
3. Levenberg–Marquardt
4. Brent's `zbrent` root finder
5. CMA-ES
6. Augmented Lagrangian / SQP / KKT

---

### F19 — `vecAddScaled` aliasing comment at `gradient.go:244-249` is partially wrong (LOW)
The comment says "dst may alias a". It does **not** explicitly say whether `dst` may alias `b`. The body
```go
dst[i] = a[i] + scale*b[i]
```
is safe for **both** aliasings (each iteration reads index `i` before writing index `i`). The doc could simply say "any aliasing is safe".

---

### F20 — Validated variants share the same line-search and L-BFGS issues (HIGH, transitive)
`gradient_validated.go` reuses `lbfgsLineSearch` and copies the same two-loop / curvature-skip logic. Every finding F1–F3 applies to `LBFGSValidated` verbatim. The R123 validity-check is added but the *underlying numerical robustness* is unchanged. Worth recording so that fixes to `gradient.go` are kept in lockstep with the validated variants.

---

## Severity summary

| # | Topic | Severity |
|---|---|---|
| F1 | LBFGS line-search is Armijo-only, doc says Wolfe | HIGH |
| F2 | Damped BFGS missing; silent skip on `sy ≤ 0` | HIGH |
| F3 | L-BFGS init scaling fragile near degeneracy | MED |
| F4 | Simplex Bland-rule out-of-bounds (latent) | CRIT |
| F5 | Simplex tableau drift; no refactorisation | HIGH |
| F6 | Bland vs Dantzig: works but slow | LOW |
| F7 | InteriorPoint is not a primal-dual IPM | **CRIT** |
| F8 | Bisection: no sign-change validation | MED |
| F9 | Newton: f'=0 fallback returns silently | HIGH |
| F10 | Newton: no globalisation; no Brent | HIGH (gap) |
| F11 | GoldenSection: no swap, no parabolic | LOW + gap |
| F12 | LinearInterpolateRoot: fine | (none) |
| F13 | GA: hard-coded `[-5,5]` domain bypasses transform | HIGH |
| F14 | BLX-α loses diversity when parents equal | LOW |
| F15 | SA: geometric-only cooling; no stopping | LOW |
| F16 | Spline n=2 path correct, untested | MED (test) |
| F17 | LinearInterpolate doesn't panic as documented | MED |
| F18 | Missing CG / Brent-min / LM / CMA-ES / SQP | gap |
| F19 | vecAddScaled aliasing doc | LOW |
| F20 | Validated variants inherit F1–F3 | HIGH |

**Top fixes for immediate action:**
1. **F7 (InteriorPoint):** rename or replace; current implementation does not solve LPs reliably.
2. **F4 (Simplex):** trivial `leaving < 0 ||` guard prevents future panic.
3. **F1+F2 (LBFGS):** either correct the doc to "Armijo backtracking, curvature-skip" or implement strong-Wolfe + Powell damping.
4. **F8+F9 (Newton/Bisection):** add error-returning variants for safety.
5. **F18 (gaps):** prioritise Conjugate Gradient and Brent's minimiser — these are 1-day adds with high reuse value.

---

## Files referenced (absolute paths)

- `C:\limitless\foundation\reality\optim\gradient.go`
- `C:\limitless\foundation\reality\optim\gradient_validated.go`
- `C:\limitless\foundation\reality\optim\linear.go`
- `C:\limitless\foundation\reality\optim\rootfind.go`
- `C:\limitless\foundation\reality\optim\genetic.go`
- `C:\limitless\foundation\reality\optim\metaheuristic.go`
- `C:\limitless\foundation\reality\optim\interpolate.go`
- `C:\limitless\foundation\reality\optim\optim_test.go` (tests cited)
