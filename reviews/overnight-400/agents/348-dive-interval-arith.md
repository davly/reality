# 348 — dive-interval-arith (rounded arithmetic / affine / Krawczyk verification audit)

## Headline
Reality has zero interval arithmetic, zero Kahan/compensated summation, and zero robust geometric predicates — interval is a small, tier-0-feasible primitive (~120 LOC) but a niche product whose value is unlocked only by downstream verified consumers (rigorous root-find, robust orient2D, certificate ODE), so ship it gated behind one explicit consumer.

## Findings

### Existence audit (greenfield)
- No `interval/` sub-package; no `Interval` type anywhere in repo.
- No `Krawczyk`, `Hansen`, `AffineArith`, `RoundedArith` symbols (grep result: zero).
- No `math.Nextafter` / `Nextafter` calls anywhere in `*.go` (the rounding primitive interval needs).
- No `Kahan`, `TwoSum`, `FastTwoSum`, `TwoProd`, `compensated` summation anywhere — note: this means even error-bounded *floating* arithmetic (Higham §4) is absent, not just intervals.
- No robust geometric predicates: no `Orient2D`, `InCircle`, `Predicate`, no Shewchuk-style adaptive determinants in `geometry/` (just `curves.go`, `polygon.go`, `quaternion.go`, `sdf.go`).
- Existing "interval" hits (e.g. `WilsonConfidenceInterval` at `prob/prob.go:239`, `SplitInterval` in `prob/conformal/`) are *statistical* confidence intervals, unrelated to validated/rigorous interval arithmetic.

So the design space is fully open. Nothing to retrofit, nothing to integrate with on day 1.

### What interval arithmetic buys you
1. **Verified bounds.** `f([a,b])` returns `[lo,hi]` such that `lo ≤ f(x) ≤ hi` for *every* `x ∈ [a,b]`, modulo correctly-rounded primitives. Bounds are conservative (often pessimistic) but never wrong.
2. **Verified zero-finding.** Krawczyk operator `K(x,X) = x - C·f(x) + (I - C·f'(X))·(X - x)` proves existence-and-uniqueness of a zero in `X` when `K(x,X) ⊂ int(X)` (Moore-Kioustelidis 1980; refined Rump 2010). Hansen-Sengupta (1981) extends this to systems via interval-Gauss-Seidel.
3. **Robust geometric predicates.** Sign of a determinant computed in interval arithmetic; if `0 ∉ result`, the sign is certified, otherwise fall through to exact arithmetic (Shewchuk-style filter; Brönnimann-Melquiond-Pion 2006 Boost.Interval is the C++ reference).
4. **Affine arithmetic** (Comba-Stolfi 1993): represent `x̃ = x₀ + Σ xᵢ·εᵢ`, `εᵢ ∈ [-1,1]` shared across affine forms. Captures linear correlations that naive interval cannot — fixes the `X − X = [-w,w]` overestimation problem. Costs O(n) per op where n = symbol count; symbol count grows with non-affine ops, so periodic condensation is required.

### The Go rounding-mode problem
Go does not expose IEEE 754 rounding modes. Three options for safe-rounding outward:

- **Naive widen-by-one-ulp (recommended T0).** For `z = x ⊕ y` (round-to-nearest), the exact result is within `½ ulp(z)`, so `[Nextafter(z,-Inf), Nextafter(z,+Inf)]` is a guaranteed enclosure. Cost: ~3× naive arithmetic. *Slack of 1 ulp on each side*, vs `½ ulp` if we had directed rounding. Acceptable for almost every consumer; unacceptable for tightest-possible bounds research.
- **Dekker DD trick** (Dekker 1971; `TwoSum`/`TwoProd`): exact result represented as `hi+lo` pair, then bounds `hi ⊕ Nextafter(lo,±Inf)`. Tighter (~½ ulp), ~2× cost over naive widen.
- **CGO to fesetround.** Violates "zero deps" rule and breaks cross-language golden-file parity.

Recommendation: T0 ships with naive widen (correct, simple, ~120 LOC). T1 transcendentals can adopt Dekker only where ulp-tightness matters.

### Transcendentals
- Monotonic on `[a,b]`: `Sin/Cos/Exp/Log/Sqrt`-style — apply to endpoints, sort, widen by 1 ulp.
- Sin/Cos: split `[a,b]` at `π/2 + kπ` extrema; piecewise-monotonic.
- Tan: poles at `π/2 + kπ` → return `[-Inf,+Inf]` (or error if "no infinities" mode).
- Exp/Log: monotonic globally, easy.
- All assume the underlying `math.Sin` etc. is correctly rounded — Go's `math.Sin` is NOT correctly rounded (it's near-1-ulp), so widen by 2-3 ulps to be safe (Muller §11). This is a real subtlety to document.

### Krawczyk verified zero-find — the marquee primitive
- Input: `f, f' (interval-extension), X = [a,b], x = mid(X)`.
- Compute `C ≈ 1/f'(x)` (point inverse), `K = x - C·f(x) + (1 - C·f'(X))·(X-x)`.
- If `K ⊂ int(X)`: zero exists in K, is unique in X. Done.
- Else: bisect, recurse.
- ~150 LOC including bisection driver. Cross-check: when `X` is tight around a true zero, K converges quadratically — same rate as Newton, but with a *certificate*. This is the natural R-MUTUAL-CROSS-VALIDATION 3/3 pin.

### Affine arithmetic (T2)
- Carry vector of `(εᵢ, coefᵢ)` plus a residual error `±r`.
- Linear ops exact (add, scale). Multiplication: `(x₀ + Σxᵢεᵢ)(y₀ + Σyᵢεᵢ) = x₀y₀ + (x₀yᵢ + xᵢy₀)εᵢ + (Σxᵢεᵢ)(Σyᵢεᵢ)`; the last term is non-affine, condensed to a fresh symbol with bound `(Σ|xᵢ|)(Σ|yᵢ|)`.
- Big win: SDF evaluation, ray-marching tight enclosures, Lipschitz bounds on neural surrogates. Slot 077 (geometry SDF) is the natural consumer.
- Pessimism: still grows on long computations. Modern variants (Messine 2002 reduced affine, Goubault 2013 zonotopes) are research-grade.

### Hansen-Sengupta linear solve (T4)
- Given interval matrix `[A]`, interval RHS `[b]`, find interval enclosure of `{x : Ax = b, A ∈ [A], b ∈ [b]}`.
- Hansen-Sengupta = preconditioned interval-Gauss-Seidel with `C ≈ mid([A])⁻¹`.
- ~200 LOC; foundation for verified-everything (verified eigenvalues, verified Newton on systems).
- This is where Rump's INTLAB makes its money. For reality, only justified if a verified-ODE or verified-optimizer is funded.

## Concrete recommendations

1. **Defer until a real consumer commits.** Interval arithmetic without a verified-something downstream is library code with no users. Right now reality has zero verified-numeric APIs. Do NOT ship a stand-alone `interval/` package that nothing imports — that's the worst form of architecture astronaut work. Wait until one of (a) robust geometric predicates, (b) verified Lorenz/Lorenz-like ODE (Tucker 1999 cousin, slot 220), or (c) certificate-providing root-finder is on the roadmap, then ship interval as the *enabling* primitive in the same PR.

2. **If shipped, structure as `interval/` with strict tiering.**
   - **T0 (~120 LOC, day-1 PR):** `Interval{Lo, Hi float64}`, `Add/Sub/Mul/Div/Neg/Sqr`, `Width/Mid/Mag/Mig/Contains/Overlaps`, all using `math.Nextafter` widening. Edge cases: empty interval (`Lo > Hi`), unbounded (±Inf endpoints), NaN propagation. Constants: `Empty`, `Entire = [-Inf,+Inf]`. Source: Moore 1966, Kulisch-Miranker 1981.
   - **T1 (~200 LOC):** transcendentals — `Sin/Cos/Tan/Exp/Log/Sqrt/Pow` with monotonicity-based decomposition. Document 2-3 ulp slack from non-correctly-rounded `math.*`.
   - **T2 (~200 LOC):** affine arithmetic `AAForm{x0 float64, eps []EpsTerm, r float64}` with `New/Add/Sub/Mul/Sqr/Reciprocal/Bound`. Include a `Condense(maxTerms)` for long-running uses. Source: Comba-Stolfi 1993, de Figueiredo-Stolfi 2004 survey.
   - **T3 (~150 LOC):** `KrawczykZero(f, df, X) (Interval, Verified bool, error)` — the actual marquee. Source: Moore-Kioustelidis 1980, Rump 2010.
   - **T4 (~200 LOC):** Hansen-Sengupta `LinSolve([]A,[]b)`. Source: Hansen-Sengupta 1981, Rump 1999 INTLAB.

3. **Pure-Go safe rounding via `math.Nextafter`.** Do NOT take CGO + `fesetround`. The `Nextafter`-widen approach gives correct (if slightly pessimistic) bounds, stays in pure Go, preserves the cross-language golden-file invariant. Document the 1-ulp slack explicitly in the package doc — research users (rigorous-numerics academics) will need this disclosure to know whether reality's interval is suitable for their use case.

4. **R-MUTUAL-CROSS-VALIDATION 3/3 pin opportunities.**
   - **(a) Containment for arithmetic.** Generate 1000 random `(x, y, X∋x, Y∋y)` tuples; assert `(x op y) ∈ (X op Y)` for op ∈ {+,−,×,÷}. Regression-pin the count; assert 1000/1000.
   - **(b) Tightness for arithmetic.** For point intervals (`X.Lo == X.Hi == x`), assert `Width(X op Y) ≤ 4·ulp(|x op y|)` (the constant accounts for our widening). Pin the width-vs-ulp ratio histogram.
   - **(c) Krawczyk ≡ Newton on convergent inputs.** When `X` is tight around a true zero of `f`, Krawczyk's midpoint trajectory must match Newton's iterates to within interval width. Cross-pin against `optim.Newton` (if it exists) or a hand-coded Newton.
   - **(d) Affine ≤ naive interval.** For any computation with linear correlations (e.g., `x − x` where `x ∈ X`), `Bound(AffineForm)` must be ⊆ naive `Interval` result. Width ratio < 1 in the dependency case, = 1 otherwise. Pin the ratio for a fixed test set.

5. **Cross-link consumers (none exist today; design for them).**
   - `geometry/`: a future `RobustOrient2D(a,b,c) Sign` should evaluate determinant in `Interval`, fall through to exact arithmetic only when `0 ∈ result`. (Slot 077 cousin.)
   - `chaos/`: a future verified Lorenz solver wraps RK4 in `Interval` for rigorous trajectory enclosures (Tucker 1999 used Taylor-model + interval to prove the Lorenz attractor exists).
   - `optim/`: a `VerifiedRoot(f, X)` API on top of Krawczyk gives certificate-providing root-finding — qualitatively different from `optim/bisection.go` because it returns *proof of existence*, not just a small residual.
   - `prob/`: rigorous bounds on integrals (verified quadrature) is a niche but real use case (Bayesian posterior bounds).

6. **Day-1 cheapest PR if greenlit: T0 only (~120 LOC + tests).** Single file `interval/interval.go`, golden-file vectors checking containment+width on the n=1000 random pair generator. Document explicitly: "this is a foundation primitive; no consumer in reality v0.10 imports it; it exists to enable future verified APIs." If no consumer materializes by v0.12, *delete it*.

7. **Watch out for the Go-`math` precision floor.** The interval `Sin/Cos/Exp/Log` quality is bounded by Go's `math.Sin/Cos/Exp/Log` quality, which is ~1 ulp on most inputs but worse near reduction-sensitive args (`Sin(2^53)`). For bulletproof bounds you'd need a correctly-rounded math library (CRlibm, Daramy-Boldo-Muller 2003) — out of scope. Document this honestly: "interval transcendentals are conservative-up-to-Go-math-quality; for bit-exact rigorous numerics, this package is not sufficient."

8. **Affine arithmetic is the most under-appreciated win.** For consumers doing long iterative computations (ODE rollout, optimization paths) naive interval explodes (Moore 1966 wrapping effect); affine reduces this dramatically when the path is dominated by linear dependencies. If the SDF/ray-marcher in `geometry/sdf.go` ever needs verified bounds, affine is the right tool.

9. **Skip Boost.Interval-style policy templates.** Brönnimann-Melquiond-Pion 2006 use C++ template policies for rounding/checking/comparison strategies. Go has no equivalent; keep the API monomorphic with one rounding strategy (Nextafter-widen) and one comparison semantics (cset / Kulisch-Miranker — interval-as-set, not interval-as-number).

## Sources

### Repo files audited
- `C:/limitless/foundation/reality/geometry/` (curves.go, polygon.go, quaternion.go, sdf.go) — no robust predicates.
- `C:/limitless/foundation/reality/prob/prob.go:224-239` — `WilsonConfidenceInterval` (statistical, unrelated).
- `C:/limitless/foundation/reality/prob/conformal/split.go` — `SplitInterval` (statistical, unrelated).
- Top-level grep for `Kahan|TwoSum|TwoProd|compensated|Nextafter|Krawczyk|AffineArith|RoundedArith|Interval(?!Confidence|Signed)` — zero matches in non-statistical context.
- `C:/limitless/foundation/reality/CLAUDE.md` — design rules (zero deps, golden-file testing, no allocations in hot path).

### Canonical literature (cited, not web-fetched — these are textbook references)
- Moore, R.E. 1966. *Interval Analysis*. Prentice-Hall. The founding work — defines interval arithmetic, dependency / wrapping effect, interval Newton.
- Kulisch, U.W., Miranker, W.L. 1981. *Computer Arithmetic in Theory and Practice*. Academic Press. Formalizes directed rounding semantics; foundation for IEEE 754's later directed-rounding modes.
- Comba, J.L.D., Stolfi, J. 1993. "Affine Arithmetic and its Applications to Computer Graphics." SIBGRAPI '93. The affine-arithmetic original.
- de Figueiredo, L.H., Stolfi, J. 2004. "Affine Arithmetic: Concepts and Applications." *Numerical Algorithms* 37. The 2004 survey paper everyone cites.
- Hansen, E., Sengupta, S. 1981. "Bounding Solutions of Systems of Equations Using Interval Analysis." *BIT* 21:203-211. Hansen-Sengupta interval-Gauss-Seidel.
- Moore, R.E., Kioustelidis, J.B. 1980. "A Simple Test for Accuracy of Approximate Solutions to Nonlinear (or Linear) Systems." *SIAM J. Numer. Anal.* 17. Krawczyk-style verification predates this; refined here.
- Tucker, W. 1999. "The Lorenz Attractor Exists." *C. R. Acad. Sci. Paris* 328. Famous application: rigorous interval+Taylor model proof of Lorenz attractor existence.
- Rump, S.M. 2010. "Verification Methods: Rigorous Results Using Floating-Point Arithmetic." *Acta Numerica* 19:287-449. The modern survey; INTLAB author.
- Brönnimann, H., Melquiond, G., Pion, S. 2006. "The Design of the Boost Interval Arithmetic Library." *Theoretical Computer Science* 351. Reference C++ design — informs API decisions even though we won't copy the policy templates.
- Higham, N.J. 2002. *Accuracy and Stability of Numerical Algorithms*, 2nd ed. SIAM. Ch.3-4 for floating-point error bounds (the alternative to interval for forward error analysis).
- Muller, J.-M. et al. 2018. *Handbook of Floating-Point Arithmetic*, 2nd ed. Birkhäuser. Ch.11 for correctly-rounded transcendentals; Ch.4-5 for TwoSum/TwoProd; explains why Go's `math.Sin` is ~1 ulp not 0.5 ulp.
- Stahl, V. 1995. "Interval Methods for Bounding the Range of Polynomials and Solving Systems of Nonlinear Equations." PhD, Linz. Tightening Krawczyk for polynomial systems.
- Daramy, C., Boldo, S., Muller, J.-M. 2003. CRlibm. Correctly-rounded libm — the upstream solution to the Go-math precision floor.
